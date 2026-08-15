"""
Shared LLM Judge client for the evaluation pipeline.

Wraps Google Gemini via LangChain with:
  • Native structured JSON output when available
  • Controlled fallback to raw JSON only for schema/structured-output
    compatibility errors
  • Robust JSON extraction
  • Deadline-aware Gemini timeouts
  • Retry logic for transient failures
  • Secure LLM call logging
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENVIRONMENT / CONFIGURATION
# ============================================================================

_api_key_configured = (
    "yes"
    if os.getenv("GOOGLE_API_KEY")
    else "no"
)

_gemini_model = os.getenv(
    "GEMINI_MODEL_NAME",
    "gemini-3.5-flash-lite",
)

print(
    f"Gemini API key configured: {_api_key_configured}",
    flush=True,
)

print(
    f"Gemini model: {_gemini_model}",
    flush=True,
)


# ============================================================================
# DEFAULTS
# ============================================================================

_DEFAULT_MODEL = "gemini-3.5-flash-lite"

_DEFAULT_TEMPERATURE = 0.0

_DEFAULT_MAX_TOKENS = 4096

_DEFAULT_GEMINI_TIMEOUT = 30.0

_MAX_RETRIES = 3

_RETRY_BACKOFF_BASE = 2.0

_MIN_API_TIMEOUT = 0.05


# ============================================================================
# LLM JUDGE
# ============================================================================

class LLMJudge:
    """
    LLM-as-a-judge client backed by Google Gemini via LangChain.

    Environment variables:
        GEMINI_MODEL_NAME
        GOOGLE_API_KEY
        GEMINI_TIMEOUT
        GEMINI_MAX_OUTPUT_TOKENS
        ENABLE_RAW_LLM_LOGS

    Usage:
        judge = LLMJudge()

        parsed, raw = judge.call_with_json(
            system_prompt,
            user_prompt,
        )
    """

    def __init__(self) -> None:

        self.model_name = os.getenv(
            "GEMINI_MODEL_NAME",
            _DEFAULT_MODEL,
        )

        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is required. "
                "Set it in your .env file or export it in your shell."
            )

        self.api_key = api_key

        self.gemini_timeout = self._read_float_env(
            "GEMINI_TIMEOUT",
            _DEFAULT_GEMINI_TIMEOUT,
            minimum=0.1,
        )

        self.max_output_tokens = self._read_int_env(
            "GEMINI_MAX_OUTPUT_TOKENS",
            _DEFAULT_MAX_TOKENS,
            minimum=256,
        )

        logger.info(
            (
                "LLM Judge initialized: "
                "model=%s temperature=%.1f timeout=%.1fs "
                "max_output_tokens=%d"
            ),
            self.model_name,
            _DEFAULT_TEMPERATURE,
            self.gemini_timeout,
            self.max_output_tokens,
        )

    # ========================================================================
    # ENV HELPERS
    # ========================================================================

    @staticmethod
    def _read_float_env(
        name: str,
        default: float,
        minimum: float,
    ) -> float:

        try:
            value = float(
                os.getenv(
                    name,
                    str(default),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            value = default

        if value < minimum:
            value = default

        return value

    @staticmethod
    def _read_int_env(
        name: str,
        default: int,
        minimum: int,
    ) -> int:

        try:
            value = int(
                os.getenv(
                    name,
                    str(default),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            value = default

        if value < minimum:
            value = default

        return value

    # ========================================================================
    # LLM CREATION
    # ========================================================================

    def _build_llm(
        self,
        timeout: float,
    ) -> ChatGoogleGenerativeAI:
        """
        Build one Gemini client for the current call.

        We intentionally create it with the current deadline-aware timeout
        rather than using a stale timeout from initialization.
        """

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=_DEFAULT_TEMPERATURE,
            max_output_tokens=self.max_output_tokens,
            timeout=timeout,
        )

    def _calculate_api_timeout(
        self,
        deadline: float | None,
    ) -> float:
        """
        Determine Gemini timeout for this individual call.

        When a request deadline is supplied, Gemini can never receive more
        time than the remaining request budget.
        """

        configured_timeout = self.gemini_timeout

        if deadline is None:
            return configured_timeout

        remaining = (
            deadline
            - time.time()
        )

        if remaining <= 0:
            raise TimeoutError(
                "Evaluation request deadline exceeded "
                "before Gemini call."
            )

        return max(
            _MIN_API_TIMEOUT,
            min(
                configured_timeout,
                remaining,
            ),
        )

    # ========================================================================
    # CONTENT CONVERSION
    # ========================================================================

    @staticmethod
    def _convert_to_string(
        content: Any,
    ) -> str:
        """
        Convert LangChain response content into plain text.
        """

        if isinstance(
            content,
            list,
        ):

            parts: list[str] = []

            for part in content:

                if (
                    isinstance(
                        part,
                        dict,
                    )
                    and "text" in part
                ):
                    parts.append(
                        str(
                            part["text"]
                        )
                    )

                elif isinstance(
                    part,
                    str,
                ):
                    parts.append(
                        part
                    )

                elif hasattr(
                    part,
                    "text",
                ):
                    parts.append(
                        str(
                            part.text
                        )
                    )

                elif hasattr(
                    part,
                    "get",
                ):
                    value = part.get(
                        "text"
                    )

                    if value:
                        parts.append(
                            str(value)
                        )

                else:
                    parts.append(
                        str(part)
                    )

            return "".join(parts)

        if isinstance(
            content,
            str,
        ):
            return content

        return str(
            content
        )

    # ========================================================================
    # SECURE LOGGING
    # ========================================================================

    def _log_to_file(
        self,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        evaluator: str = "unknown",
        conversation_id: str = "unknown",
        duration: float = 0.0,
        status: str = "success",
        error_type: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """
        Write LLM metadata to logs.

        Raw prompts/responses are written only when
        ENABLE_RAW_LLM_LOGS=true.
        """

        log_dir = "logs"

        try:
            os.makedirs(
                log_dir,
                exist_ok=True,
            )
        except Exception as exc:
            logger.warning(
                "Failed to create log directory: %s",
                exc,
            )
            return

        log_file = os.path.join(
            log_dir,
            "llm_calls.log",
        )

        timestamp = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        enable_raw = (
            os.getenv(
                "ENABLE_RAW_LLM_LOGS",
                "false",
            ).lower()
            == "true"
        )

        try:

            with open(
                log_file,
                "a",
                encoding="utf-8",
            ) as f:

                f.write(
                    f"=== LLM CALL AT {timestamp} ===\n"
                )

                f.write(
                    f"CONVERSATION ID: {conversation_id}\n"
                )

                f.write(
                    f"EVALUATOR: {evaluator}\n"
                )

                f.write(
                    f"MODEL: {self.model_name}\n"
                )

                f.write(
                    f"DURATION: {duration:.3f}s\n"
                )

                f.write(
                    f"STATUS: {status}\n"
                )

                f.write(
                    "ERROR TYPE: "
                    f"{error_type or 'None'}\n"
                )

                f.write(
                    "INPUT TOKENS: "
                    f"{input_tokens if input_tokens is not None else 'N/A'}\n"
                )

                f.write(
                    "OUTPUT TOKENS: "
                    f"{output_tokens if output_tokens is not None else 'N/A'}\n"
                )

                if enable_raw:

                    def redact(
                        text: str,
                    ) -> str:

                        google_key = os.getenv(
                            "GOOGLE_API_KEY"
                        )

                        if google_key:
                            text = text.replace(
                                google_key,
                                "[REDACTED_API_KEY]",
                            )

                        text = re.sub(
                            r"sk-[a-zA-Z0-9]{20,}",
                            "[REDACTED_API_KEY]",
                            text,
                        )

                        text = re.sub(
                            r"mongodb(?:\+srv)?://\S+",
                            "[REDACTED_MONGO_URI]",
                            text,
                        )

                        text = re.sub(
                            r"(?i)(api[_-]?key\s*[:=]\s*)\S+",
                            r"\1[REDACTED_API_KEY]",
                            text,
                        )

                        text = re.sub(
                            r"(?i)(password\s*[:=]\s*)\S+",
                            r"\1[REDACTED_PASSWORD]",
                            text,
                        )

                        return text

                    f.write(
                        "--- SYSTEM PROMPT ---\n"
                    )

                    f.write(
                        redact(
                            system_prompt
                        )
                    )

                    f.write(
                        "\n--- USER PROMPT ---\n"
                    )

                    f.write(
                        redact(
                            user_prompt
                        )
                    )

                    f.write(
                        "\n--- RESPONSE ---\n"
                    )

                    f.write(
                        redact(
                            response_text
                        )
                    )

                    f.write(
                        "\n"

                    )

                f.write(
                    "=" * 80
                    + "\n\n"
                )

        except Exception as exc:
            logger.warning(
                "Failed to write LLM log file: %s",
                exc,
            )

    # ========================================================================
    # ERROR CLASSIFICATION
    # ========================================================================

    @staticmethod
    def _exception_text(
        exc: Exception,
    ) -> str:

        return str(
            exc
        ).lower()

    @classmethod
    def _is_quota_error(
        cls,
        exc: Exception,
    ) -> bool:

        text = cls._exception_text(
            exc
        )

        return (
            "429" in text
            or "resource_exhausted" in text
            or "quota exceeded" in text
            or "too many requests" in text
            or "rate limit" in text
        )

    @classmethod
    def _is_deadline_error(
        cls,
        exc: Exception,
    ) -> bool:

        text = cls._exception_text(
            exc
        )

        return (
            isinstance(
                exc,
                TimeoutError,
            )
            or "deadline" in text
            or "timed out" in text
            or "timeout" in text
        )

    @classmethod
    def _is_auth_error(
        cls,
        exc: Exception,
    ) -> bool:

        text = cls._exception_text(
            exc
        )

        return (
            "401" in text
            or "403" in text
            or "unauthorized" in text
            or "permission denied" in text
            or "invalid api key" in text
        )

    @classmethod
    def _is_structured_schema_error(
        cls,
        exc: Exception,
    ) -> bool:
        """
        Identify errors where switching from native structured output to
        regular text JSON is actually useful.

        This intentionally does NOT classify quota/timeouts/authorization
        errors as fallback candidates.
        """

        text = cls._exception_text(
            exc
        )

        if (
            cls._is_quota_error(
                exc
            )
            or cls._is_deadline_error(
                exc
            )
            or cls._is_auth_error(
                exc
            )
        ):
            return False

        schema_indicators = (
            "response_schema",
            "response schema",
            "json_schema",
            "structured output",
            "structured_output",
            "schema validation",
            "invalidargument",
            "invalid argument",
            "unknown name",
            "additional_properties",
            "additionalproperties",
        )

        return any(
            indicator in text
            for indicator in schema_indicators
        )

    # ========================================================================
    # TOKEN USAGE
    # ========================================================================

    @staticmethod
    def _extract_usage(
        response: Any,
    ) -> tuple[
        int | None,
        int | None,
    ]:
        """
        Extract token usage from LangChain/Gemini response metadata.
        """

        input_tokens: int | None = None
        output_tokens: int | None = None

        usage_metadata = getattr(
            response,
            "usage_metadata",
            None,
        )

        if usage_metadata:

            if isinstance(
                usage_metadata,
                dict,
            ):

                input_tokens = usage_metadata.get(
                    "input_tokens"
                )

                output_tokens = usage_metadata.get(
                    "output_tokens"
                )

            else:

                input_tokens = getattr(
                    usage_metadata,
                    "input_tokens",
                    None,
                )

                output_tokens = getattr(
                    usage_metadata,
                    "output_tokens",
                    None,
                )

        if (
            input_tokens is None
            or output_tokens is None
        ):

            response_metadata = getattr(
                response,
                "response_metadata",
                None,
            )

            if isinstance(
                response_metadata,
                dict,
            ):

                token_usage = response_metadata.get(
                    "token_usage",
                    {},
                )

                if isinstance(
                    token_usage,
                    dict,
                ):

                    if input_tokens is None:

                        input_tokens = token_usage.get(
                            "prompt_tokens"
                        )

                    if output_tokens is None:

                        output_tokens = token_usage.get(
                            "completion_tokens"
                        )

        return (
            input_tokens,
            output_tokens,
        )

    # ========================================================================
    # PUBLIC API — STRUCTURED JSON
    # ========================================================================

    def call_with_json(
        self,
        system_prompt: str,
        user_prompt: str,
        evaluator: str = "unknown",
        conversation_id: str = "unknown",
        response_schema: type[BaseModel] | None = None,
        deadline: float | None = None,
    ) -> tuple[
        dict[str, Any],
        str,
    ]:
        """
        Call Gemini and parse structured JSON.

        Primary path:
            Native JSON schema output.

        Fallback path:
            Raw text JSON extraction.

        IMPORTANT:
            Raw fallback is used only for structured-schema compatibility
            failures. Quota, deadline, timeout, authentication, and similar
            failures are NOT followed by another Gemini request.
        """

        from evaluation_pipeline.utils.concurrency import (
            controlled_concurrency,
        )
        from evaluation_pipeline.utils.retry_utils import (
            execute_with_retry,
        )

        start_time = time.time()

        # --------------------------------------------------------------------
        # API timeout
        # --------------------------------------------------------------------

        api_timeout = self._calculate_api_timeout(
            deadline
        )

        local_llm = self._build_llm(
            api_timeout
        )

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=user_prompt
            ),
        ]

        # --------------------------------------------------------------------
        # 1. Native structured output
        # --------------------------------------------------------------------

        if response_schema is not None:

            try:

                structured_llm = (
                    local_llm.with_structured_output(
                        response_schema,
                        method="json_schema",
                    )
                )

                def _invoke_structured():

                    with controlled_concurrency(
                        evaluator,
                        "Gemini API (Structured)",
                        conversation_id,
                    ):
                        return structured_llm.invoke(
                            messages
                        )

                response_obj = execute_with_retry(
                    _invoke_structured,
                    evaluator=evaluator,
                    framework="Gemini API (Structured)",
                    conversation_id=conversation_id,
                    max_retries=_MAX_RETRIES,
                    initial_delay=_RETRY_BACKOFF_BASE,
                    deadline=deadline,
                )

                if response_obj is not None:

                    if isinstance(
                        response_obj,
                        BaseModel,
                    ):
                        parsed_dict = (
                            response_obj.model_dump()
                        )
                    elif isinstance(
                        response_obj,
                        dict,
                    ):
                        parsed_dict = (
                            response_obj
                        )
                    else:
                        raise ValueError(
                            "Structured Gemini response "
                            "was neither a Pydantic model "
                            "nor a dictionary."
                        )

                    duration = (
                        time.time()
                        - start_time
                    )

                    self._log_to_file(
                        system_prompt,
                        user_prompt,
                        json.dumps(
                            parsed_dict,
                            default=str,
                        ),
                        evaluator=evaluator,
                        conversation_id=conversation_id,
                        duration=duration,
                        status="success",
                    )

                    return (
                        parsed_dict,
                        json.dumps(
                            parsed_dict,
                            default=str,
                        ),
                    )

            except Exception as structured_exc:

                # ------------------------------------------------------------
                # CRITICAL FIX
                # ------------------------------------------------------------
                #
                # Do NOT immediately send another Gemini request for:
                #
                #     429 quota
                #     deadline exceeded
                #     timeout
                #     authentication
                #     permission
                #
                # Doing that only burns more request time/quota.
                # ------------------------------------------------------------

                if (
                    not self._is_structured_schema_error(
                        structured_exc
                    )
                ):
                    logger.error(
                        (
                            "[%s] Native structured output failed for %s "
                            "without a safe fallback condition. "
                            "Error: %s"
                        ),
                        evaluator,
                        conversation_id,
                        structured_exc,
                    )

                    duration = (
                        time.time()
                        - start_time
                    )

                    self._log_to_file(
                        system_prompt,
                        user_prompt,
                        "",
                        evaluator=evaluator,
                        conversation_id=conversation_id,
                        duration=duration,
                        status="failed",
                        error_type=(
                            structured_exc.__class__.__name__
                        ),
                    )

                    raise

                logger.warning(
                    (
                        "[%s] Native structured output failed for %s "
                        "because of a structured-schema compatibility "
                        "issue. Falling back to raw JSON extraction. "
                        "Error: %s"
                    ),
                    evaluator,
                    conversation_id,
                    structured_exc,
                )

        # --------------------------------------------------------------------
        # 2. Raw text fallback
        # --------------------------------------------------------------------

        # Recalculate the timeout because the structured attempt may have
        # consumed part of the shared request deadline.

        api_timeout = self._calculate_api_timeout(
            deadline
        )

        local_llm = self._build_llm(
            api_timeout
        )

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=user_prompt
            ),
        ]

        def _invoke_api():

            with controlled_concurrency(
                evaluator,
                "Gemini API",
                conversation_id,
            ):
                return local_llm.invoke(
                    messages
                )

        try:

            response = execute_with_retry(
                _invoke_api,
                evaluator=evaluator,
                framework="Gemini API",
                conversation_id=conversation_id,
                max_retries=_MAX_RETRIES,
                initial_delay=_RETRY_BACKOFF_BASE,
                deadline=deadline,
            )

            raw_text = self._convert_to_string(
                response.content
            )

            (
                input_tokens,
                output_tokens,
            ) = self._extract_usage(
                response
            )

            duration = (
                time.time()
                - start_time
            )

            self._log_to_file(
                system_prompt,
                user_prompt,
                raw_text,
                evaluator=evaluator,
                conversation_id=conversation_id,
                duration=duration,
                status="success",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            parsed = self._extract_json(
                raw_text
            )

            if response_schema is not None:

                validated_obj = (
                    response_schema.model_validate(
                        parsed
                    )
                )

                parsed = (
                    validated_obj.model_dump()
                )

            return (
                parsed,
                raw_text,
            )

        except Exception as exc:

            duration = (
                time.time()
                - start_time
            )

            self._log_to_file(
                system_prompt,
                user_prompt,
                "",
                evaluator=evaluator,
                conversation_id=conversation_id,
                duration=duration,
                status="failed",
                error_type=(
                    exc.__class__.__name__
                ),
            )

            raise

    # ========================================================================
    # PUBLIC API — RAW TEXT
    # ========================================================================

    def call_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        evaluator: str = "unknown",
        conversation_id: str = "unknown",
        deadline: float | None = None,
    ) -> str:
        """
        Call Gemini and return raw text.
        """

        from evaluation_pipeline.utils.concurrency import (
            controlled_concurrency,
        )
        from evaluation_pipeline.utils.retry_utils import (
            execute_with_retry,
        )

        start_time = time.time()

        api_timeout = self._calculate_api_timeout(
            deadline
        )

        local_llm = self._build_llm(
            api_timeout
        )

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=user_prompt
            ),
        ]

        def _invoke_api():

            with controlled_concurrency(
                evaluator,
                "Gemini API",
                conversation_id,
            ):
                return local_llm.invoke(
                    messages
                )

        try:

            response = execute_with_retry(
                _invoke_api,
                evaluator=evaluator,
                framework="Gemini API",
                conversation_id=conversation_id,
                max_retries=_MAX_RETRIES,
                initial_delay=_RETRY_BACKOFF_BASE,
                deadline=deadline,
            )

            raw_text = self._convert_to_string(
                response.content
            )

            (
                input_tokens,
                output_tokens,
            ) = self._extract_usage(
                response
            )

            duration = (
                time.time()
                - start_time
            )

            self._log_to_file(
                system_prompt,
                user_prompt,
                raw_text,
                evaluator=evaluator,
                conversation_id=conversation_id,
                duration=duration,
                status="success",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            return raw_text

        except Exception as exc:

            duration = (
                time.time()
                - start_time
            )

            self._log_to_file(
                system_prompt,
                user_prompt,
                "",
                evaluator=evaluator,
                conversation_id=conversation_id,
                duration=duration,
                status="failed",
                error_type=(
                    exc.__class__.__name__
                ),
            )

            raise

    # ========================================================================
    # JSON EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_json(
        text: str,
    ) -> dict[str, Any]:
        """
        Extract a JSON object from Gemini output.

        Handles:
          • Markdown ```json fences
          • Plain ``` fences
          • Entire response being JSON
          • Nested JSON objects
        """

        # --------------------------------------------------------------------
        # Strategy 1: Markdown code fences
        # --------------------------------------------------------------------

        fence_pattern = (
            r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        )

        fence_matches = re.findall(
            fence_pattern,
            text,
            re.DOTALL,
        )

        for match in fence_matches:

            try:
                parsed = json.loads(
                    match.strip()
                )

                if isinstance(
                    parsed,
                    dict,
                ):
                    return parsed

            except json.JSONDecodeError:
                continue

        # --------------------------------------------------------------------
        # Strategy 2: Entire response
        # --------------------------------------------------------------------

        try:

            parsed = json.loads(
                text.strip()
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except json.JSONDecodeError:
            pass

        # --------------------------------------------------------------------
        # Strategy 3: Find outermost JSON object
        # --------------------------------------------------------------------

        start_idx = text.find(
            "{"
        )

        if start_idx != -1:

            depth = 0
            in_string = False
            escape_next = False

            for i in range(
                start_idx,
                len(text),
            ):

                ch = text[i]

                if escape_next:
                    escape_next = False
                    continue

                if ch == "\\":
                    escape_next = True
                    continue

                if ch == '"':
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if ch == "{":

                    depth += 1

                elif ch == "}":

                    depth -= 1

                    if depth == 0:

                        candidate = text[
                            start_idx : i + 1
                        ]

                        try:

                            parsed = json.loads(
                                candidate
                            )

                            if isinstance(
                                parsed,
                                dict,
                            ):
                                return parsed

                        except json.JSONDecodeError:
                            # Continue searching for another possible object
                            pass

        # --------------------------------------------------------------------
        # Failed
        # --------------------------------------------------------------------

        raise ValueError(
            (
                "Could not extract valid JSON "
                "from LLM response. "
                f"Response preview: {text[:300]}..."
            )
        )