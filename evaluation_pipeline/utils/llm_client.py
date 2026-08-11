"""
Shared LLM Judge client for the evaluation pipeline.

Wraps Google Gemini via LangChain with:
  • Robust JSON extraction (handles markdown fences, nested objects)
  • Retry logic for transient API errors
  • Consistent temperature=0 for deterministic evaluation
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Environment Validation
_api_key_configured = "yes" if os.getenv("GOOGLE_API_KEY") else "no"
_gemini_model = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
print(f"Gemini API key configured: {_api_key_configured}", flush=True)
print(f"Gemini model: {_gemini_model}", flush=True)

# Defaults
_DEFAULT_MODEL = "gemini-3.5-flash"
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 4096
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0  # seconds


class LLMJudge:
    """
    LLM-as-a-judge client backed by Google Gemini via LangChain.

    Initializes from environment variables:
      - GEMINI_MODEL_NAME  (default: gemini-3.5-flash)
      - GOOGLE_API_KEY     (required)

    Usage
    -----
    >>> judge = LLMJudge()
    >>> parsed, raw = judge.call_with_json(system_prompt, user_prompt)
    """

    def __init__(self) -> None:
        self.model_name = os.getenv("GEMINI_MODEL_NAME", _DEFAULT_MODEL)
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is required. "
                "Set it in your .env file or export it in your shell."
            )

        gemini_timeout = float(os.getenv("GEMINI_TIMEOUT", "30.0"))
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=_DEFAULT_TEMPERATURE,
            max_output_tokens=_DEFAULT_MAX_TOKENS,
            timeout=gemini_timeout,
        )
        logger.info(
            "LLM Judge initialized: model=%s, temperature=%.1f, timeout=%.1fs",
            self.model_name,
            _DEFAULT_TEMPERATURE,
            gemini_timeout,
        )

    def _convert_to_string(self, content: Any) -> str:
        """Convert response content block list or other types to a single string."""
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
                elif hasattr(part, "text"):
                    parts.append(part.text)
                elif hasattr(part, "get") and part.get("text"):
                    parts.append(part.get("text"))
                else:
                    parts.append(str(part))
            return "".join(parts)
        elif isinstance(content, str):
            return content
        return str(content)

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
        output_tokens: int | None = None
    ) -> None:
        """Log LLM call metadata securely, with optional raw logging if configured."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "llm_calls.log")
        timestamp = datetime.now(timezone.utc).isoformat()
        enable_raw = os.getenv("ENABLE_RAW_LLM_LOGS", "false").lower() == "true"
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"=== LLM CALL AT {timestamp} ===\n")
                f.write(f"CONVERSATION ID: {conversation_id}\n")
                f.write(f"EVALUATOR: {evaluator}\n")
                f.write(f"MODEL: {self.model_name}\n")
                f.write(f"DURATION: {duration:.3f}s\n")
                f.write(f"STATUS: {status}\n")
                f.write(f"ERROR TYPE: {error_type or 'None'}\n")
                f.write(f"INPUT TOKENS: {input_tokens if input_tokens is not None else 'N/A'}\n")
                f.write(f"OUTPUT TOKENS: {output_tokens if output_tokens is not None else 'N/A'}\n")
                
                if enable_raw:
                    def redact(text: str) -> str:
                        key = os.getenv("GOOGLE_API_KEY")
                        if key:
                            text = text.replace(key, "[REDACTED_API_KEY]")
                        # Redact credentials/URIs
                        text = re.sub(r"sk-[a-zA-Z0-9]{20,}", "[REDACTED_API_KEY]", text)
                        text = re.sub(r"mongodb(?:\+srv)?://\S+", "[REDACTED_MONGO_URI]", text)
                        return text
                    
                    f.write(f"--- SYSTEM PROMPT ---\n{redact(system_prompt)}\n")
                    f.write(f"--- USER PROMPT ---\n{redact(user_prompt)}\n")
                    f.write(f"--- RESPONSE ---\n{redact(response_text)}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            logger.warning("Failed to write to raw LLM log file: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call_with_json(
        self,
        system_prompt: str,
        user_prompt: str,
        evaluator: str = "unknown",
        conversation_id: str = "unknown",
        response_schema: type[BaseModel] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """
        Call the LLM and parse structured JSON from its response.
        Uses native structured output if response_schema is provided, with fallback to manual JSON extraction.
        """
        from evaluation_pipeline.utils.concurrency import controlled_concurrency
        from evaluation_pipeline.utils.retry_utils import execute_with_retry

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        start_time = time.time()

        # 1. Native Structured Output Attempt
        if response_schema:
            try:
                # Bind response schema to LLM
                structured_llm = self.llm.with_structured_output(response_schema, method="json_schema")
                
                def _invoke_structured():
                    with controlled_concurrency(evaluator, "Gemini API (Structured)", conversation_id):
                        return structured_llm.invoke(messages)

                response_obj = execute_with_retry(
                    _invoke_structured,
                    evaluator=evaluator,
                    framework="Gemini API (Structured)",
                    conversation_id=conversation_id,
                    max_retries=_MAX_RETRIES,
                    initial_delay=_RETRY_BACKOFF_BASE
                )
                
                if response_obj is not None:
                    parsed_dict = response_obj.model_dump()
                    duration = time.time() - start_time
                    
                    self._log_to_file(
                        system_prompt, user_prompt, str(parsed_dict),
                        evaluator=evaluator, conversation_id=conversation_id,
                        duration=duration, status="success",
                    )
                    return parsed_dict, str(parsed_dict)
            except Exception as structured_exc:
                logger.warning(
                    "[%s] Native structured output failed for %s. Error: %s. Falling back to old JSON extractor.",
                    evaluator, conversation_id, structured_exc
                )

        # 2. Fallback to raw text + regex JSON extraction
        def _invoke_api():
            with controlled_concurrency(evaluator, "Gemini API", conversation_id):
                return self.llm.invoke(messages)

        try:
            response = execute_with_retry(
                _invoke_api,
                evaluator=evaluator,
                framework="Gemini API",
                conversation_id=conversation_id,
                max_retries=_MAX_RETRIES,
                initial_delay=_RETRY_BACKOFF_BASE
            )
            raw_text = self._convert_to_string(response.content)
            
            input_tokens = None
            output_tokens = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = response.usage_metadata.get("input_tokens")
                output_tokens = response.usage_metadata.get("output_tokens")
            elif hasattr(response, "response_metadata") and response.response_metadata:
                token_usage = response.response_metadata.get("token_usage", {})
                if token_usage:
                    input_tokens = token_usage.get("prompt_tokens")
                    output_tokens = token_usage.get("completion_tokens")

            duration = time.time() - start_time
            self._log_to_file(
                system_prompt, user_prompt, raw_text,
                evaluator=evaluator, conversation_id=conversation_id,
                duration=duration, status="success",
                input_tokens=input_tokens, output_tokens=output_tokens
            )
            parsed = self._extract_json(raw_text)
            
            if response_schema:
                # Validate the fallback parsed dict against schema
                validated_obj = response_schema(**parsed)
                parsed = validated_obj.model_dump()
                
            return parsed, raw_text
        except Exception as exc:
            duration = time.time() - start_time
            self._log_to_file(
                system_prompt, user_prompt, "",
                evaluator=evaluator, conversation_id=conversation_id,
                duration=duration, status="failed", error_type=exc.__class__.__name__
            )
            raise exc

    def call_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        evaluator: str = "unknown",
        conversation_id: str = "unknown",
    ) -> str:
        """
        Call the LLM and return the raw text response (no JSON parsing).
        """
        from evaluation_pipeline.utils.concurrency import controlled_concurrency
        from evaluation_pipeline.utils.retry_utils import execute_with_retry

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        def _invoke_api():
            with controlled_concurrency(evaluator, "Gemini API", conversation_id):
                return self.llm.invoke(messages)

        start_time = time.time()
        try:
            response = execute_with_retry(
                _invoke_api,
                evaluator=evaluator,
                framework="Gemini API",
                conversation_id=conversation_id,
                max_retries=_MAX_RETRIES,
                initial_delay=_RETRY_BACKOFF_BASE
            )
            raw_text = self._convert_to_string(response.content)

            input_tokens = None
            output_tokens = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = response.usage_metadata.get("input_tokens")
                output_tokens = response.usage_metadata.get("output_tokens")
            elif hasattr(response, "response_metadata") and response.response_metadata:
                token_usage = response.response_metadata.get("token_usage", {})
                if token_usage:
                    input_tokens = token_usage.get("prompt_tokens")
                    output_tokens = token_usage.get("completion_tokens")

            duration = time.time() - start_time
            self._log_to_file(
                system_prompt, user_prompt, raw_text,
                evaluator=evaluator, conversation_id=conversation_id,
                duration=duration, status="success",
                input_tokens=input_tokens, output_tokens=output_tokens
            )
            return raw_text
        except Exception as exc:
            duration = time.time() - start_time
            self._log_to_file(
                system_prompt, user_prompt, "",
                evaluator=evaluator, conversation_id=conversation_id,
                duration=duration, status="failed", error_type=exc.__class__.__name__
            )
            raise exc

    # ------------------------------------------------------------------
    # JSON extraction (robust)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """
        Extract a JSON object from LLM output, handling:
          • Markdown ```json ... ``` fences
          • Plain ``` ... ``` fences
          • Bare JSON in the text
          • Nested braces

        Raises ValueError if no valid JSON can be found.
        """
        # Strategy 1: Extract from markdown code fences
        fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        fence_matches = re.findall(fence_pattern, text, re.DOTALL)
        for match in fence_matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Strategy 2: Try the entire text as JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 3: Find the outermost { ... } with proper nesting
        start_idx = text.find("{")
        if start_idx != -1:
            depth = 0
            in_string = False
            escape_next = False

            for i in range(start_idx, len(text)):
                ch = text[i]

                if escape_next:
                    escape_next = False
                    continue

                if ch == "\\":
                    escape_next = True
                    continue

                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start_idx : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

        raise ValueError(
            f"Could not extract valid JSON from LLM response. "
            f"Response preview: {text[:300]}..."
        )
