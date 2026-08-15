from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable

from google.genai.errors import APIError

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

_MIN_RETRY_DELAY = 0.05

# Errors containing these patterns represent quota exhaustion that is unlikely
# to recover during the current evaluation request.
_QUOTA_EXHAUSTION_PATTERNS = (
    "generate_content_free_tier_requests",
    "perdayperprojectpermodel-freetier",
    "quota exceeded for metric",
    "quotaexceeded",
    "resource_exhausted",
)

# Errors that should never be retried.
_NON_RETRY_KEYWORDS = (
    "api key",
    "apikey",
    "unauthorized",
    "authentication",
    "auth failed",
    "credential",
    "invalid key",
    "not found",
    "does not exist",
    "not exist",
    "bad request",
    "invalid_argument",
    "invalid argument",
    "validation",
    "model not found",
)

# Genuine temporary infrastructure/API failures.
_TRANSIENT_KEYWORDS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "503",
    "502",
    "504",
    "500",
    "temporary",
    "temporarily unavailable",
    "service unavailable",
    "gateway",
    "timeout",
    "time out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "server disconnected",
    "try again",
)


# ============================================================================
# ERROR CLASSIFICATION
# ============================================================================

def _is_quota_exhaustion_error(
    message: str,
) -> bool:
    """
    Detect quota exhaustion.

    Examples from Gemini:

        GenerateRequestsPerDayPerProjectPerModel-FreeTier

        quota exceeded for metric:
        generativelanguage.googleapis.com/generate_content_free_tier_requests

    These are deliberately NOT retried inside the evaluation request because
    waiting a few seconds will not restore a daily/model quota.
    """

    normalized = (
        message
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )

    for pattern in _QUOTA_EXHAUSTION_PATTERNS:

        normalized_pattern = (
            pattern
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        if normalized_pattern in normalized:
            return True

    return False


def _extract_status_code(
    exc: Exception,
) -> int | None:
    """
    Extract an HTTP/API status code from common exception shapes.
    """

    candidates: list[Any] = [
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(exc, "code", None),
    ]

    for value in candidates:

        if value is None:
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return None


def is_transient_error(
    exc: Exception,
) -> bool:
    """
    Determine whether an exception should be retried.

    Retry:
        - 429 rate limiting
        - 5xx server errors
        - connection failures
        - timeouts
        - temporary service failures

    Do NOT retry:
        - authentication/API-key errors
        - invalid model/request errors
        - validation errors
        - exhausted daily/model/project quota

    Important:
        Gemini quota exhaustion is often returned as HTTP 429, but a 429 is
        not automatically retryable. We distinguish ordinary rate limiting
        from quota exhaustion.
    """

    exc_name = (
        exc.__class__.__name__
        .lower()
    )

    message = str(exc).lower()

    # ------------------------------------------------------------------------
    # Pydantic/schema validation errors are deterministic.
    # ------------------------------------------------------------------------

    if (
        "validationerror" in exc_name
        or "pydanticusererror" in exc_name
        or "validation" in message
    ):
        return False

    # ------------------------------------------------------------------------
    # Quota exhaustion MUST be checked before generic 429 handling.
    # ------------------------------------------------------------------------

    if _is_quota_exhaustion_error(
        message
    ):
        logger.warning(
            "Detected exhausted Gemini quota. "
            "This error will not be retried: %s",
            str(exc),
        )
        return False

    # ------------------------------------------------------------------------
    # Google GenAI APIError.
    # ------------------------------------------------------------------------

    if isinstance(
        exc,
        APIError,
    ):

        code = _extract_status_code(
            exc
        )

        if code is not None:

            # 429 can mean either rate limiting OR quota exhaustion.
            # Exhaustion was already handled above.
            if code == 429:
                return True

            if 500 <= code < 600:
                return True

            return False

        # Fallback to message inspection.
        if any(
            keyword in message
            for keyword in _NON_RETRY_KEYWORDS
        ):
            return False

        return any(
            keyword in message
            for keyword in _TRANSIENT_KEYWORDS
        )

    # ------------------------------------------------------------------------
    # Generic exceptions carrying HTTP status information.
    # ------------------------------------------------------------------------

    status_code = _extract_status_code(
        exc
    )

    if status_code is not None:

        if status_code == 429:
            return True

        if 500 <= status_code < 600:
            return True

        return False

    # ------------------------------------------------------------------------
    # Exception class checks.
    # ------------------------------------------------------------------------

    transient_exception_names = (
        "timeout",
        "timeouterror",
        "connectionerror",
        "connectionreseterror",
        "connectionabortederror",
        "connecterror",
    )

    if any(
        name in exc_name
        for name in transient_exception_names
    ):
        return True

    # ------------------------------------------------------------------------
    # Explicit non-retryable message checks.
    # ------------------------------------------------------------------------

    if any(
        keyword in message
        for keyword in _NON_RETRY_KEYWORDS
    ):
        return False

    # ------------------------------------------------------------------------
    # Generic transient-message checks.
    # ------------------------------------------------------------------------

    if any(
        keyword in message
        for keyword in _TRANSIENT_KEYWORDS
    ):
        return True

    return False


# ============================================================================
# OPTIONAL RETRY-AFTER EXTRACTION
# ============================================================================

def _extract_retry_after_seconds(
    exc: Exception,
) -> float | None:
    """
    Try to extract Gemini's suggested retry delay.

    Example:

        Please retry in 59s.

    This is only useful for genuine rate-limit/server failures.
    """

    message = str(exc)

    patterns = (
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        r"retryDelay.*?([0-9]+(?:\.[0-9]+)?)s",
    )

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            flags=re.IGNORECASE,
        )

        if match:

            try:
                value = float(
                    match.group(1)
                )

                if value >= 0:
                    return value

            except (
                TypeError,
                ValueError,
            ):
                pass

    return None


# ============================================================================
# DEADLINE HELPERS
# ============================================================================

def _remaining_time(
    deadline: float | None,
) -> float | None:
    """
    Return remaining evaluation time.

    Returns None when there is no deadline.
    """

    if deadline is None:
        return None

    return max(
        0.0,
        deadline - time.time(),
    )


def _can_wait_for_retry(
    delay: float,
    deadline: float | None,
) -> bool:
    """
    Determine whether there is enough request budget to perform the retry
    delay.

    We intentionally require a small amount of time to remain after sleeping.
    """

    if deadline is None:
        return True

    remaining = _remaining_time(
        deadline
    )

    if remaining is None:
        return True

    return (
        remaining
        > delay + 0.1
    )


# ============================================================================
# RETRY EXECUTION
# ============================================================================

def execute_with_retry(
    func: Callable[..., Any],
    evaluator: str,
    framework: str,
    conversation_id: str,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    deadline: float | None = None,
    *args,
    **kwargs,
):
    """
    Execute a function with deadline-aware exponential backoff.

    Behavior
    --------
    - Executes immediately.
    - Retries only genuinely transient errors.
    - Never retries exhausted Gemini daily/model/project quota.
    - Never sleeps beyond the remaining request deadline.
    - Honors Gemini's RetryInfo delay when available.
    - Stops immediately when the deadline is exhausted.

    `max_retries=3` means at most three attempts total.
    """

    # Protect against invalid configuration.
    if max_retries <= 0:
        max_retries = 1

    if initial_delay < 0:
        initial_delay = 0.0

    if backoff_factor < 1.0:
        backoff_factor = 1.0

    attempt = 1
    delay = max(
        initial_delay,
        _MIN_RETRY_DELAY,
    )

    while True:

        # --------------------------------------------------------------------
        # Check deadline BEFORE starting the attempt.
        # --------------------------------------------------------------------

        if deadline is not None:

            remaining = _remaining_time(
                deadline
            )

            if (
                remaining is None
                or remaining <= 0
            ):

                exc = TimeoutError(
                    (
                        "Deadline exceeded before execution of "
                        f"{evaluator} ({framework})."
                    )
                )

                logger.warning(
                    (
                        "Evaluation attempt FAILED: "
                        "conversation_id=%s | "
                        "evaluator=%s | "
                        "framework=%s | "
                        "attempt=%d | "
                        "duration=0.000s | "
                        "status=failed | "
                        "transient=False | "
                        "error=%s"
                    ),
                    conversation_id,
                    evaluator,
                    framework,
                    attempt,
                    str(exc),
                )

                raise exc

        start_time = time.time()

        try:

            result = func(
                *args,
                **kwargs,
            )

            duration = (
                time.time()
                - start_time
            )

            logger.info(
                (
                    "Evaluation attempt SUCCESS: "
                    "conversation_id=%s | "
                    "evaluator=%s | "
                    "framework=%s | "
                    "attempt=%d | "
                    "duration=%.3fs | "
                    "status=success | "
                    "transient=None"
                ),
                conversation_id,
                evaluator,
                framework,
                attempt,
                duration,
            )

            return result

        except Exception as exc:

            duration = (
                time.time()
                - start_time
            )

            transient = is_transient_error(
                exc
            )

            logger.warning(
                (
                    "Evaluation attempt FAILED: "
                    "conversation_id=%s | "
                    "evaluator=%s | "
                    "framework=%s | "
                    "attempt=%d | "
                    "duration=%.3fs | "
                    "status=failed | "
                    "transient=%s | "
                    "error=%s"
                ),
                conversation_id,
                evaluator,
                framework,
                attempt,
                duration,
                str(transient),
                str(exc),
            )

            # ----------------------------------------------------------------
            # Do not retry deterministic failures.
            # ----------------------------------------------------------------

            if not transient:

                raise

            # ----------------------------------------------------------------
            # Retry limit reached.
            # ----------------------------------------------------------------

            if attempt >= max_retries:

                logger.warning(
                    (
                        "Retry limit reached: "
                        "conversation_id=%s | "
                        "evaluator=%s | "
                        "framework=%s | "
                        "attempt=%d | "
                        "max_retries=%d"
                    ),
                    conversation_id,
                    evaluator,
                    framework,
                    attempt,
                    max_retries,
                )

                raise

            # ----------------------------------------------------------------
            # Calculate remaining request budget.
            # ----------------------------------------------------------------

            remaining = _remaining_time(
                deadline
            )

            if (
                remaining is not None
                and remaining <= 0
            ):

                logger.warning(
                    (
                        "Skipping retry because the "
                        "evaluation deadline has already expired: "
                        "conversation_id=%s | "
                        "evaluator=%s | "
                        "framework=%s"
                    ),
                    conversation_id,
                    evaluator,
                    framework,
                )

                raise

            # ----------------------------------------------------------------
            # Prefer provider-suggested retry delay when available.
            # ----------------------------------------------------------------

            retry_after = (
                _extract_retry_after_seconds(
                    exc
                )
            )

            retry_delay = (
                retry_after
                if retry_after is not None
                else delay
            )

            retry_delay = max(
                _MIN_RETRY_DELAY,
                retry_delay,
            )

            # ----------------------------------------------------------------
            # Do not wait longer than the remaining request deadline.
            # ----------------------------------------------------------------

            if not _can_wait_for_retry(
                retry_delay,
                deadline,
            ):

                logger.warning(
                    (
                        "Skipping retry: remaining budget "
                        "is insufficient for retry delay "
                        "(%.1fs). "
                        "conversation_id=%s | "
                        "evaluator=%s | "
                        "framework=%s"
                    ),
                    retry_delay,
                    conversation_id,
                    evaluator,
                    framework,
                )

                raise

            logger.info(
                (
                    "Retrying in %.1fs due to transient error: "
                    "conversation_id=%s | "
                    "evaluator=%s | "
                    "framework=%s | "
                    "next_attempt=%d"
                ),
                retry_delay,
                conversation_id,
                evaluator,
                framework,
                attempt + 1,
            )

            time.sleep(
                retry_delay
            )

            attempt += 1

            # ----------------------------------------------------------------
            # Exponential backoff for subsequent attempts.
            #
            # If Gemini supplied RetryInfo, use that retry time for this
            # attempt, but still maintain exponential fallback behavior.
            # ----------------------------------------------------------------

            if retry_after is not None:

                delay = max(
                    delay * backoff_factor,
                    retry_after,
                )

            else:

                delay *= backoff_factor