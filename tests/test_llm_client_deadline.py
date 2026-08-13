"""
Tests for LLMJudge deadline consistency (Final Targeted Patch).

Verifies that BOTH call paths (call_with_json, call_raw) in llm_client.py:
  - Cap api_timeout to min(configured, max(0.05, remaining))
  - NEVER allow api_timeout > remaining request budget
  - Raise TimeoutError immediately when deadline has already expired
  - Never start a Gemini call that would violate the request deadline
"""

import time
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Shared context manager: freeze time, patch LangChain, capture timeout
# ---------------------------------------------------------------------------

@contextmanager
def _intercept_llm_timeout(now: float, deadline: float, gemini_timeout_env: str = "30.0"):
    """Freeze time.time(), patch ChatGoogleGenerativeAI, yield captured dict."""
    captured = {}

    def _fake_llm(**kwargs):
        captured["timeout"] = kwargs.get("timeout")
        m = MagicMock()
        # make with_structured_output chain work
        m.with_structured_output.return_value.invoke.return_value = MagicMock(
            model_dump=lambda: {
                "detected_true_intent": "technical",
                "intent_accuracy": {"score": 5, "reasoning": "ok"},
                "clarification_handling": {"score": 5, "reasoning": "ok"},
                "was_misclassified": False,
                "explanation": "ok",
            }
        )
        m.invoke.return_value = MagicMock(
            content="{}",
            usage_metadata=None,
            response_metadata=None,
        )
        return m

    env_patch = {"GEMINI_TIMEOUT": gemini_timeout_env, "GOOGLE_API_KEY": "mock-key"}

    with patch("evaluation_pipeline.utils.llm_client.time.time", return_value=now), \
         patch("langchain_google_genai.ChatGoogleGenerativeAI", side_effect=_fake_llm), \
         patch.dict("os.environ", env_patch):
        yield captured, deadline


def _make_judge_bare():
    """Create a bare LLMJudge without triggering real API calls."""
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "mock-key"}), \
         patch("langchain_google_genai.ChatGoogleGenerativeAI"):
        from evaluation_pipeline.utils.llm_client import LLMJudge
        j = LLMJudge.__new__(LLMJudge)
        j.model_name = "gemini-3.5-flash"
        return j


# ---------------------------------------------------------------------------
# call_with_json deadline tests
# ---------------------------------------------------------------------------

class TestCallWithJsonDeadline(unittest.TestCase):

    def _captured_timeout(self, remaining: float) -> float | None:
        """Run call_with_json with given remaining budget; return captured api_timeout."""
        now = 10000.0
        deadline = now + remaining
        j = _make_judge_bare()

        captured = {}

        def _fake_llm(**kwargs):
            captured["timeout"] = kwargs.get("timeout")
            m = MagicMock()
            m.with_structured_output.return_value.invoke.return_value = MagicMock(
                model_dump=lambda: {
                    "detected_true_intent": "technical",
                    "intent_accuracy": {"score": 5, "reasoning": "ok"},
                    "clarification_handling": {"score": 5, "reasoning": "ok"},
                    "was_misclassified": False,
                    "explanation": "ok",
                }
            )
            m.invoke.return_value = MagicMock(
                content="{}", usage_metadata=None, response_metadata=None
            )
            return m

        with patch("evaluation_pipeline.utils.llm_client.time.time", return_value=now), \
             patch("langchain_google_genai.ChatGoogleGenerativeAI", side_effect=_fake_llm), \
             patch("evaluation_pipeline.utils.retry_utils.execute_with_retry",
                   side_effect=lambda f, *a, **kw: f()), \
             patch("evaluation_pipeline.utils.concurrency.controlled_concurrency"), \
             patch.dict("os.environ", {"GEMINI_TIMEOUT": "30.0", "GOOGLE_API_KEY": "mock-key"}):
            try:
                j.call_with_json("sys", "usr", deadline=deadline)
            except Exception:
                pass

        return captured.get("timeout")

    def test_5s_remaining_timeout_le_5s(self):
        """5.0s remaining → api_timeout ≤ 5.0 (bounded by budget, not blown up to 30s)."""
        t = self._captured_timeout(5.0)
        self.assertIsNotNone(t, "ChatGoogleGenerativeAI was not instantiated")
        self.assertLessEqual(t, 5.0 + 1e-9)

    def test_0_5s_remaining_timeout_le_0_5(self):
        """0.5s remaining → api_timeout ≤ 0.5s, never 1.0s."""
        t = self._captured_timeout(0.5)
        self.assertIsNotNone(t)
        self.assertLessEqual(t, 0.5 + 1e-9)
        self.assertLess(t, 1.0, "timeout must NOT be inflated to 1.0 when only 0.5s remain")

    def test_0_1s_remaining_timeout_le_0_1(self):
        """0.1s remaining → api_timeout ≤ 0.1s, never 1.0s."""
        t = self._captured_timeout(0.1)
        self.assertIsNotNone(t)
        self.assertLessEqual(t, 0.1 + 1e-9)
        self.assertLess(t, 1.0)

    def test_expired_deadline_raises_timeout_not_invoked(self):
        """Expired deadline (remaining ≤ 0) → TimeoutError raised; Gemini never called."""
        now = 10000.0
        deadline = now - 0.1  # already past
        j = _make_judge_bare()
        invoked = []

        def _fake_llm(**kwargs):
            invoked.append(1)
            return MagicMock()

        with patch("evaluation_pipeline.utils.llm_client.time.time", return_value=now), \
             patch("langchain_google_genai.ChatGoogleGenerativeAI", side_effect=_fake_llm), \
             patch.dict("os.environ", {"GEMINI_TIMEOUT": "30.0", "GOOGLE_API_KEY": "mock-key"}):
            with self.assertRaises(TimeoutError):
                j.call_with_json("sys", "usr", deadline=deadline)

        # The first ChatGoogleGenerativeAI call is the one inside __init__ (mocked away in _make_judge_bare).
        # Any NEW call from call_with_json would be appended to `invoked`.
        self.assertEqual(len(invoked), 0, "Gemini must NOT be instantiated after deadline expiry")

    def test_no_child_timeout_exceeds_remaining_budget(self):
        """For various budgets, api_timeout must stay within remaining + tiny epsilon."""
        for remaining in [0.06, 0.3, 0.5, 1.0, 5.0, 15.0, 29.0]:
            t = self._captured_timeout(remaining)
            if t is not None:
                self.assertLessEqual(
                    t, remaining + 1e-6,
                    f"api_timeout {t} exceeded remaining {remaining}"
                )


# ---------------------------------------------------------------------------
# call_raw deadline tests
# ---------------------------------------------------------------------------

class TestCallRawDeadline(unittest.TestCase):

    def _captured_timeout_raw(self, remaining: float) -> float | None:
        now = 10000.0
        deadline = now + remaining
        j = _make_judge_bare()

        captured = {}

        def _fake_llm(**kwargs):
            captured["timeout"] = kwargs.get("timeout")
            m = MagicMock()
            m.invoke.return_value = MagicMock(
                content="hello", usage_metadata=None, response_metadata=None
            )
            return m

        with patch("evaluation_pipeline.utils.llm_client.time.time", return_value=now), \
             patch("langchain_google_genai.ChatGoogleGenerativeAI", side_effect=_fake_llm), \
             patch("evaluation_pipeline.utils.retry_utils.execute_with_retry",
                   side_effect=lambda f, *a, **kw: f()), \
             patch("evaluation_pipeline.utils.concurrency.controlled_concurrency"), \
             patch.dict("os.environ", {"GEMINI_TIMEOUT": "30.0", "GOOGLE_API_KEY": "mock-key"}):
            try:
                j.call_raw("sys", "usr", deadline=deadline)
            except Exception:
                pass

        return captured.get("timeout")

    def test_5s_remaining_call_raw_timeout_le_5s(self):
        """call_raw: 5.0s remaining → api_timeout ≤ 5.0."""
        t = self._captured_timeout_raw(5.0)
        self.assertIsNotNone(t)
        self.assertLessEqual(t, 5.0 + 1e-9)

    def test_0_5s_remaining_call_raw_timeout_le_0_5(self):
        """call_raw: 0.5s remaining → api_timeout ≤ 0.5s, never 1.0s."""
        t = self._captured_timeout_raw(0.5)
        self.assertIsNotNone(t)
        self.assertLessEqual(t, 0.5 + 1e-9)
        self.assertLess(t, 1.0)

    def test_0_1s_remaining_call_raw_timeout_le_0_1(self):
        """call_raw: 0.1s remaining → api_timeout ≤ 0.1s, never 1.0s."""
        t = self._captured_timeout_raw(0.1)
        self.assertIsNotNone(t)
        self.assertLessEqual(t, 0.1 + 1e-9)
        self.assertLess(t, 1.0)

    def test_expired_deadline_raises_timeout_call_raw(self):
        """call_raw: expired deadline → TimeoutError; Gemini never instantiated."""
        now = 10000.0
        deadline = now - 0.5
        j = _make_judge_bare()
        invoked = []

        def _fake_llm(**kwargs):
            invoked.append(1)
            return MagicMock()

        with patch("evaluation_pipeline.utils.llm_client.time.time", return_value=now), \
             patch("langchain_google_genai.ChatGoogleGenerativeAI", side_effect=_fake_llm), \
             patch.dict("os.environ", {"GEMINI_TIMEOUT": "30.0", "GOOGLE_API_KEY": "mock-key"}):
            with self.assertRaises(TimeoutError):
                j.call_raw("sys", "usr", deadline=deadline)

        self.assertEqual(len(invoked), 0)

    def test_no_child_timeout_exceeds_remaining_budget_raw(self):
        """call_raw: for various budgets, api_timeout stays within remaining + epsilon."""
        for remaining in [0.06, 0.3, 0.5, 1.0, 5.0, 15.0, 29.0]:
            t = self._captured_timeout_raw(remaining)
            if t is not None:
                self.assertLessEqual(
                    t, remaining + 1e-6,
                    f"call_raw api_timeout {t} exceeded remaining {remaining}"
                )


if __name__ == "__main__":
    unittest.main()
