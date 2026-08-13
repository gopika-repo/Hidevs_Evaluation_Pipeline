import pytest
import time
import os
from datetime import datetime
from evaluation_pipeline.data.models import EvaluationInput, ConversationType
from evaluation_pipeline.utils.retry_utils import execute_with_retry
from evaluation_pipeline.utils.llm_client import LLMJudge
from evaluation_pipeline.evaluators.groundedness_evaluator import _run_trulens_groundedness, _run_deepeval_faithfulness
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

class TestPhase8Timeouts:
    def test_deadline_calculation_and_budget_skips_retry(self):
        """
        Verify that execute_with_retry does not retry if the remaining budget
        is less than the sleep delay.
        """
        call_count = 0
        def dummy_failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Transient network drop")

        # Set a short deadline (current time + 1.5 seconds)
        # First execution will fail, retry delay is 2.0s.
        # Since 1.0 (start) + 2.0 (delay) > 2.5 (deadline), it should raise the exception immediately.
        deadline = time.time() + 1.5

        with pytest.raises(ValueError, match="Transient network drop"):
            execute_with_retry(
                dummy_failing_function,
                evaluator="test",
                framework="TestRetry",
                conversation_id="test_convo",
                max_retries=3,
                initial_delay=2.0,
                deadline=deadline
            )
        
        # Should have executed exactly once because the deadline prevented the retry attempt.
        assert call_count == 1

    def test_evaluate_endpoint_timeout_classification(self):
        """
        Verify that /evaluate endpoint sets status="timeout" if the request deadline is exceeded.
        """
        # Set request timeout environment to a very low value (e.g. 0.1s)
        os.environ["EVALUATION_REQUEST_TIMEOUT"] = "0.1"

        payload = {
            "conversation_id": "test_timeout_convo",
            "user_query": "Is this a timeout test?",
            "dave_response": "Yes, it is.",
            "retrieved_context": "Sample context that requires LLM processing.",
            "chat_history": "",
            "timestamp": "2026-08-13T12:00:00Z"
        }

        response = client.post("/evaluate", json=payload)
        assert response.status_code == 200
        data = response.json()

        # The request budget of 0.1s is guaranteed to expire, mapping evaluators to "timeout"
        convo = data["conversations"][0]
        eval_statuses = [res["status"] for res in convo["evaluations"].values()]
        assert "timeout" in eval_statuses

        # Overall health should reflect missing metrics
        assert convo["overall_health_score"] is None or convo["overall_health_score"] < 100.0

        # Reset environment
        del os.environ["EVALUATION_REQUEST_TIMEOUT"]
