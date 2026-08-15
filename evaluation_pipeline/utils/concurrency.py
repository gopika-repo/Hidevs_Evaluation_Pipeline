import logging
import os
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ============================================================================
# GLOBAL CONCURRENCY STATE
# ============================================================================

_semaphore = None
_current_limit = None

_semaphore_lock = threading.Lock()

_active_calls = 0
_active_calls_lock = threading.Lock()

_max_observed_concurrency = 0


# ============================================================================
# CONFIGURATION
# ============================================================================

_DEFAULT_CONCURRENCY_LIMIT = 10
_DEFAULT_SEMAPHORE_WAIT_TIMEOUT = 30.0
_MIN_WAIT_TIMEOUT = 0.05


# ============================================================================
# HELPERS
# ============================================================================

def _read_concurrency_limit() -> int:
    """
    Read the current concurrency limit from the environment.

    Invalid or non-positive values fall back to the default.
    """

    try:
        limit = int(
            os.getenv(
                "EVALUATION_MAX_CONCURRENCY",
                str(_DEFAULT_CONCURRENCY_LIMIT),
            )
        )
    except (TypeError, ValueError):
        limit = _DEFAULT_CONCURRENCY_LIMIT

    if limit <= 0:
        limit = _DEFAULT_CONCURRENCY_LIMIT

    return limit


def _read_semaphore_wait_timeout() -> float:
    """
    Read the maximum semaphore acquisition wait time.

    This is separate from the Gemini/API timeout.
    """

    try:
        timeout = float(
            os.getenv(
                "EVALUATION_SEMAPHORE_WAIT_TIMEOUT",
                str(_DEFAULT_SEMAPHORE_WAIT_TIMEOUT),
            )
        )
    except (TypeError, ValueError):
        timeout = _DEFAULT_SEMAPHORE_WAIT_TIMEOUT

    if timeout <= 0:
        timeout = _DEFAULT_SEMAPHORE_WAIT_TIMEOUT

    return timeout


# ============================================================================
# SEMAPHORE
# ============================================================================

def get_semaphore():
    """
    Return the shared semaphore and its configured limit.

    IMPORTANT:
    A semaphore is NEVER replaced while evaluator calls are actively using it.

    This prevents the old implementation's bug where changing
    EVALUATION_MAX_CONCURRENCY could create multiple independent semaphore
    pools at the same time.
    """

    global _semaphore
    global _current_limit

    requested_limit = _read_concurrency_limit()

    with _semaphore_lock:

        # First initialization.
        if _semaphore is None:
            _semaphore = threading.Semaphore(
                requested_limit
            )

            _current_limit = requested_limit

            logger.info(
                (
                    "Concurrency semaphore initialized: "
                    "limit=%d"
                ),
                requested_limit,
            )

            return (
                _semaphore,
                requested_limit,
            )

        # No change.
        if requested_limit == _current_limit:
            return (
                _semaphore,
                _current_limit,
            )

        # --------------------------------------------------------------------
        # Configuration changed.
        #
        # Do NOT replace the semaphore while calls are active.
        #
        # Existing calls continue under the current semaphore. Once the
        # active count reaches zero, the next call can safely create a new
        # semaphore with the requested limit.
        # --------------------------------------------------------------------

        with _active_calls_lock:
            active_calls = _active_calls

        if active_calls == 0:

            logger.info(
                (
                    "Concurrency limit changing: "
                    "%d -> %d"
                ),
                _current_limit,
                requested_limit,
            )

            _semaphore = threading.Semaphore(
                requested_limit
            )

            _current_limit = requested_limit

        else:

            logger.info(
                (
                    "Concurrency limit change deferred: "
                    "current=%d requested=%d active_calls=%d"
                ),
                _current_limit,
                requested_limit,
                active_calls,
            )

        return (
            _semaphore,
            _current_limit,
        )


# ============================================================================
# CONCURRENCY METRICS
# ============================================================================

def get_max_observed_concurrency() -> int:
    """
    Return the highest number of simultaneous evaluator calls observed.
    """

    global _max_observed_concurrency

    with _active_calls_lock:
        return _max_observed_concurrency


def reset_max_observed_concurrency():
    """
    Reset the maximum observed concurrency counter.
    """

    global _max_observed_concurrency

    with _active_calls_lock:
        _max_observed_concurrency = 0


def get_active_calls() -> int:
    """
    Return the current number of active evaluator calls.
    """

    global _active_calls

    with _active_calls_lock:
        return _active_calls


# ============================================================================
# CONTROLLED CONCURRENCY
# ============================================================================

@contextmanager
def controlled_concurrency(
    evaluator: str,
    framework: str,
    conversation_id: str,
    deadline: float | None = None,
):
    """
    Enforce the shared evaluation concurrency limit.

    Parameters
    ----------
    evaluator:
        Evaluator name, e.g. "groundedness".

    framework:
        Framework name, e.g. "Gemini API".

    conversation_id:
        Conversation being evaluated.

    deadline:
        Optional absolute Unix timestamp for the evaluation request.

        Backward compatible:
            Existing callers that do not provide deadline continue to work.

    Behavior
    --------
    1. Acquire the shared semaphore.
    2. Track active calls.
    3. Track maximum observed concurrency.
    4. Release the semaphore reliably in finally.
    5. Never wait longer than the remaining request deadline.
    """

    global _active_calls
    global _max_observed_concurrency

    semaphore, limit = get_semaphore()

    start_wait = time.time()

    logger.info(
        (
            "Evaluation execution state: START | "
            "conversation_id=%s | evaluator=%s | framework=%s | "
            "max_limit=%d | state=waiting_for_semaphore"
        ),
        conversation_id,
        evaluator,
        framework,
        limit,
    )

    # ------------------------------------------------------------------------
    # Calculate semaphore wait timeout.
    # ------------------------------------------------------------------------

    configured_wait_timeout = (
        _read_semaphore_wait_timeout()
    )

    wait_timeout = configured_wait_timeout

    if deadline is not None:

        remaining = (
            deadline
            - time.time()
        )

        if remaining <= 0:

            duration = (
                time.time()
                - start_wait
            )

            logger.error(
                (
                    "Evaluation execution state: FAILED | "
                    "conversation_id=%s | evaluator=%s | framework=%s | "
                    "duration=%.3fs | state=deadline_exceeded | "
                    "error=Evaluation request deadline already exceeded"
                ),
                conversation_id,
                evaluator,
                framework,
                duration,
            )

            raise TimeoutError(
                (
                    f"Evaluation request deadline exceeded "
                    f"before acquiring concurrency slot for "
                    f"{evaluator} ({framework})."
                )
            )

        wait_timeout = min(
            configured_wait_timeout,
            remaining,
        )

    wait_timeout = max(
        _MIN_WAIT_TIMEOUT,
        wait_timeout,
    )

    # ------------------------------------------------------------------------
    # Acquire semaphore.
    # ------------------------------------------------------------------------

    acquired = semaphore.acquire(
        timeout=wait_timeout
    )

    if not acquired:

        duration = (
            time.time()
            - start_wait
        )

        if deadline is not None:

            remaining = (
                deadline
                - time.time()
            )

            if remaining <= 0:

                logger.error(
                    (
                        "Evaluation execution state: FAILED | "
                        "conversation_id=%s | evaluator=%s | framework=%s | "
                        "duration=%.3fs | state=deadline_exceeded | "
                        "error=Evaluation request deadline reached "
                        "while waiting for concurrency slot"
                    ),
                    conversation_id,
                    evaluator,
                    framework,
                    duration,
                )

                raise TimeoutError(
                    (
                        f"Evaluation request deadline exceeded "
                        f"while waiting for concurrency slot for "
                        f"{evaluator} ({framework})."
                    )
                )

        logger.error(
            (
                "Evaluation execution state: FAILED | "
                "conversation_id=%s | evaluator=%s | framework=%s | "
                "duration=%.3fs | state=semaphore_timeout | "
                "error=Could not acquire concurrency slot"
            ),
            conversation_id,
            evaluator,
            framework,
            duration,
        )

        raise TimeoutError(
            (
                f"Could not acquire concurrency slot "
                f"for {evaluator} ({framework}) "
                f"within {wait_timeout:.2f} seconds."
            )
        )

    # ------------------------------------------------------------------------
    # Semaphore acquired.
    # ------------------------------------------------------------------------

    wait_duration = (
        time.time()
        - start_wait
    )

    if wait_duration > 0.1:

        logger.info(
            (
                "Evaluation execution state: WAITED | "
                "conversation_id=%s | evaluator=%s | framework=%s | "
                "wait_duration=%.3fs"
            ),
            conversation_id,
            evaluator,
            framework,
            wait_duration,
        )

    # ------------------------------------------------------------------------
    # Track active calls.
    # ------------------------------------------------------------------------

    with _active_calls_lock:

        _active_calls += 1

        if (
            _active_calls
            > _max_observed_concurrency
        ):
            _max_observed_concurrency = (
                _active_calls
            )

        current = _active_calls

    logger.info(
        (
            "Evaluation execution state: ACQUIRED | "
            "conversation_id=%s | evaluator=%s | framework=%s | "
            "active_slots=%d/%d"
        ),
        conversation_id,
        evaluator,
        framework,
        current,
        limit,
    )

    start_execution = time.time()

    try:

        yield

        execution_duration = (
            time.time()
            - start_execution
        )

        logger.info(
            (
                "Evaluation execution state: SUCCESS | "
                "conversation_id=%s | evaluator=%s | framework=%s | "
                "duration=%.3fs | state=completed"
            ),
            conversation_id,
            evaluator,
            framework,
            execution_duration,
        )

    except Exception as exc:

        execution_duration = (
            time.time()
            - start_execution
        )

        logger.error(
            (
                "Evaluation execution state: FAILED | "
                "conversation_id=%s | evaluator=%s | framework=%s | "
                "duration=%.3fs | state=exception | error=%s"
            ),
            conversation_id,
            evaluator,
            framework,
            execution_duration,
            str(exc),
        )

        raise

    finally:

        # --------------------------------------------------------------------
        # Always update active count and release semaphore.
        # --------------------------------------------------------------------

        with _active_calls_lock:

            _active_calls -= 1

            current = _active_calls

        semaphore.release()

        logger.info(
            (
                "Evaluation execution state: RELEASED | "
                "conversation_id=%s | evaluator=%s | framework=%s | "
                "active_slots=%d/%d"
            ),
            conversation_id,
            evaluator,
            framework,
            current,
            limit,
        )