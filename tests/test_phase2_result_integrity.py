"""
Phase 2 — Evaluator Result-State Integrity Tests.

Verifies:
1. classify_exception maps exception types to correct statuses.
2. EvaluationResult model enforces the strict status set.
3. Failed evaluations keep score=None (no silent 0.0 or default).
4. Each status string is preserved end-to-end through the model.
5. The feedback validator rejects placeholder strings.
6. Timeout exceptions are classified correctly.
7. API / credential errors are classified as "unavailable".
8. JSON parse errors are classified as "invalid_output".
9. Generic RuntimeError is classified as "failed".
10. classify_exception(None) returns "success".
11. EvaluationResult with status=success and score=None is allowed.
12. EvaluationResult rejects an invalid status string.
"""

import json
import asyncio
import concurrent.futures
import pytest
from pydantic import ValidationError

from evaluation_pipeline.data.models import EvaluationResult
from evaluation_pipeline.utils.error_handler import classify_exception


# ---------------------------------------------------------------------------
# Helper — build a minimal valid EvaluationResult
# ---------------------------------------------------------------------------

def _make_result(**overrides):
    defaults = dict(
        evaluator_name="test_evaluator",
        conversation_id="TEST-001",
        score=10.0,
        max_score=20.0,
        status="success",
        sub_scores={},
        feedback="This is a real, generated explanation of the score.",
        flagged=False,
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


# ===================================================================
# 1. classify_exception — timeout variants
# ===================================================================

class TestClassifyTimeout:
    def test_builtin_timeout_error(self):
        assert classify_exception(TimeoutError("operation timed out")) == "timeout"

    def test_concurrent_timeout(self):
        exc = concurrent.futures.TimeoutError()
        assert classify_exception(exc) == "timeout"

    def test_asyncio_timeout(self):
        exc = asyncio.TimeoutError()
        assert classify_exception(exc) == "timeout"

    def test_message_contains_timed_out(self):
        exc = RuntimeError("The request timed out after 30s")
        assert classify_exception(exc) == "timeout"


# ===================================================================
# 2. classify_exception — invalid_output variants
# ===================================================================

class TestClassifyInvalidOutput:
    def test_json_decode_error(self):
        exc = json.JSONDecodeError("Expecting value", "", 0)
        assert classify_exception(exc) == "invalid_output"

    def test_value_error(self):
        exc = ValueError("could not parse score from LLM output")
        assert classify_exception(exc) == "invalid_output"

    def test_type_error(self):
        exc = TypeError("expected int got str")
        assert classify_exception(exc) == "invalid_output"

    def test_validation_in_name(self):
        class ValidationError(Exception):
            pass
        assert classify_exception(ValidationError("bad data")) == "invalid_output"


# ===================================================================
# 3. classify_exception — unavailable
# ===================================================================

class TestClassifyUnavailable:
    def test_api_key_message(self):
        exc = RuntimeError("invalid api key provided")
        assert classify_exception(exc) == "unavailable"

    def test_rate_limit_message(self):
        exc = RuntimeError("429 rate limit exceeded")
        assert classify_exception(exc) == "unavailable"

    def test_connection_error(self):
        exc = ConnectionError("could not connect to service")
        assert classify_exception(exc) == "unavailable"

    def test_http_error(self):
        class HttpError(Exception):
            pass
        assert classify_exception(HttpError("503 Service Unavailable")) == "unavailable"


# ===================================================================
# 4. classify_exception — fallback to "failed"
# ===================================================================

class TestClassifyFailed:
    def test_generic_runtime_error(self):
        exc = RuntimeError("something completely unexpected")
        assert classify_exception(exc) == "failed"

    def test_attribute_error(self):
        exc = AttributeError("'NoneType' object has no attribute 'score'")
        assert classify_exception(exc) == "failed"


# ===================================================================
# 5. classify_exception(None) → "success"
# ===================================================================

class TestClassifyNone:
    def test_none_returns_success(self):
        assert classify_exception(None) == "success"


# ===================================================================
# 6. EvaluationResult — strict status validation
# ===================================================================

class TestModelStatusValidation:
    @pytest.mark.parametrize("status", [
        "success", "evaluated", "failed", "timeout",
        "invalid_output", "unavailable", "not_applicable",
    ])
    def test_all_valid_statuses_accepted(self, status):
        result = _make_result(status=status)
        assert result.status == status

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            _make_result(status="partial")

    def test_empty_status_rejected(self):
        with pytest.raises(ValidationError):
            _make_result(status="")

    def test_none_status_rejected(self):
        with pytest.raises(ValidationError):
            _make_result(status=None)


# ===================================================================
# 7. Score integrity — failed statuses keep score=None
# ===================================================================

class TestScoreIntegrity:
    @pytest.mark.parametrize("status", ["failed", "timeout", "invalid_output", "unavailable"])
    def test_failed_status_allows_none_score(self, status):
        result = _make_result(score=None, status=status)
        assert result.score is None

    def test_success_with_none_score_is_allowed(self):
        """Model allows score=None even with success — the evaluator layer
        should prevent this, but the model itself must not crash."""
        result = _make_result(score=None, status="success")
        assert result.score is None

    def test_success_with_real_score(self):
        result = _make_result(score=15.5, status="success")
        assert result.score == 15.5

    def test_negative_score_rejected(self):
        with pytest.raises(ValidationError):
            _make_result(score=-1.0)


# ===================================================================
# 8. Feedback validation — placeholder strings rejected
# ===================================================================

class TestFeedbackValidation:
    def test_placeholder_rejected(self):
        with pytest.raises(ValidationError):
            _make_result(feedback="N/A")

    def test_todo_rejected(self):
        with pytest.raises(ValidationError):
            _make_result(feedback="TODO")

    def test_real_feedback_accepted(self):
        result = _make_result(feedback="The response was accurate and well-structured.")
        assert "accurate" in result.feedback

    def test_too_short_feedback_rejected(self):
        with pytest.raises(ValidationError):
            _make_result(feedback="ok")


# ===================================================================
# 9. End-to-end: classify → model roundtrip
# ===================================================================

class TestClassifyToModel:
    def test_timeout_roundtrip(self):
        status = classify_exception(TimeoutError("timed out"))
        result = _make_result(score=None, status=status)
        assert result.status == "timeout"
        assert result.score is None

    def test_json_error_roundtrip(self):
        status = classify_exception(json.JSONDecodeError("err", "", 0))
        result = _make_result(score=None, status=status)
        assert result.status == "invalid_output"
        assert result.score is None

    def test_generic_failure_roundtrip(self):
        status = classify_exception(RuntimeError("kaboom"))
        result = _make_result(score=None, status=status)
        assert result.status == "failed"
        assert result.score is None
