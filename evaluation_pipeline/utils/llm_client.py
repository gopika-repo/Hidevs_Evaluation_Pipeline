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

# Defaults
_DEFAULT_MODEL = "gemini-1.5-flash"
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 4096
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0  # seconds


class LLMJudge:
    """
    LLM-as-a-judge client backed by Google Gemini via LangChain.

    Initializes from environment variables:
      - GEMINI_MODEL_NAME  (default: gemini-1.5-flash)
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

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=_DEFAULT_TEMPERATURE,
            max_output_tokens=_DEFAULT_MAX_TOKENS,
        )
        logger.info(
            "LLM Judge initialized: model=%s, temperature=%.1f",
            self.model_name,
            _DEFAULT_TEMPERATURE,
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

    def _log_to_file(self, system_prompt: str, user_prompt: str, response_text: str) -> None:
        """Log the raw LLM prompt and response to a local log file."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "llm_calls.log")
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"=== LLM CALL AT {timestamp} ===\n")
                f.write(f"MODEL: {self.model_name}\n")
                f.write(f"--- SYSTEM PROMPT ---\n{system_prompt}\n")
                f.write(f"--- USER PROMPT ---\n{user_prompt}\n")
                f.write(f"--- RESPONSE ---\n{response_text}\n")
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
    ) -> tuple[dict[str, Any], str]:
        """
        Call the LLM and parse structured JSON from its response.

        Parameters
        ----------
        system_prompt : str
            System-level instructions for the judge.
        user_prompt : str
            The evaluation prompt with conversation data.

        Returns
        -------
        tuple[dict, str]
            (parsed_json, raw_response_text)

        Raises
        ------
        ValueError
            If JSON cannot be extracted after all retries.
        RuntimeError
            If the LLM API call fails after all retries.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.llm.invoke(messages)
                raw_text = self._convert_to_string(response.content)

                logger.debug(
                    "LLM response (attempt %d, %d chars): %s...",
                    attempt,
                    len(raw_text),
                    raw_text[:200],
                )

                self._log_to_file(system_prompt, user_prompt, raw_text)
                parsed = self._extract_json(raw_text)
                return parsed, raw_text

            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "JSON parse failed on attempt %d/%d: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                last_error = exc
                # Retry with backoff — LLM may produce valid JSON next time
                if attempt < _MAX_RETRIES:
                    sleep_time = _RETRY_BACKOFF_BASE ** attempt
                    logger.info("Retrying in %.1fs...", sleep_time)
                    time.sleep(sleep_time)

            except Exception as exc:
                logger.warning(
                    "LLM API call failed on attempt %d/%d: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                last_error = exc
                if attempt < _MAX_RETRIES:
                    sleep_time = _RETRY_BACKOFF_BASE ** attempt
                    logger.info("Retrying in %.1fs...", sleep_time)
                    time.sleep(sleep_time)

        raise RuntimeError(
            f"LLM call failed after {_MAX_RETRIES} attempts. Last error: {last_error}"
        )

    def call_raw(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Call the LLM and return the raw text response (no JSON parsing).

        Useful when you need free-form text rather than structured output.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.llm.invoke(messages)
                raw_text = self._convert_to_string(response.content)
                self._log_to_file(system_prompt, user_prompt, raw_text)
                return raw_text
            except Exception as exc:
                logger.warning(
                    "LLM API call failed on attempt %d/%d: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_BASE ** attempt)

        raise RuntimeError(
            f"LLM call failed after {_MAX_RETRIES} attempts. Last error: {last_error}"
        )

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
