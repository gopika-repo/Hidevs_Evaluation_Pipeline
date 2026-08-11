import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app
from evaluation_pipeline.data.models import EvaluationResult


class TestAPIValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("evaluation_pipeline.evaluators.response_quality_evaluator.ResponseQualityEvaluator.evaluate")
    def test_valid_request(self, mock_rq) -> None:
        """HTTP 200 returned on valid request payload."""
        mock_rq.return_value = EvaluationResult(
            evaluator_name="response_quality",
            conversation_id="VAL-001",
            score=20.0,
            max_score=20.0,
            status="success",
            feedback="Good response quality",
        )
        payload = {
            "conversation_id": "VAL-001",
            "user_query": "What is the capital of Japan?",
            "dave_response": "The capital of Japan is Tokyo.",
            "retrieved_context": "",
            "timestamp": "2026-08-11T12:00:00Z",
            "expected_intent": "technical",
        }
        # We patch LLMJudge call_with_json for other evaluators to avoid external API calls
        with patch("evaluation_pipeline.utils.llm_client.LLMJudge.call_with_json") as mock_call:
            mock_call.return_value = ({}, "{}")
            response = self.client.post("/evaluate", json=payload)
            self.assertEqual(response.status_code, 200)

    def test_missing_field(self) -> None:
        """HTTP 422 returned when required fields (like conversation_id) are missing."""
        payload = {
            "user_query": "Hello",
            "dave_response": "World",
            "timestamp": "2026-08-11T12:00:00Z",
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_blank_query(self) -> None:
        """HTTP 422 returned when user_query is blank / whitespace-only."""
        payload = {
            "conversation_id": "VAL-003",
            "user_query": "   ",
            "dave_response": "World",
            "timestamp": "2026-08-11T12:00:00Z",
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_blank_response(self) -> None:
        """HTTP 422 returned when dave_response is blank / whitespace-only."""
        payload = {
            "conversation_id": "VAL-004",
            "user_query": "Hello",
            "dave_response": "\n\t ",
            "timestamp": "2026-08-11T12:00:00Z",
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_invalid_timestamp(self) -> None:
        """HTTP 422 returned when timestamp is not parseable or invalid."""
        payload = {
            "conversation_id": "VAL-005",
            "user_query": "Hello",
            "dave_response": "World",
            "timestamp": "not-a-valid-datetime",
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_invalid_expected_intent(self) -> None:
        """HTTP 422 returned when expected_intent is not one of the allowed categories."""
        payload = {
            "conversation_id": "VAL-006",
            "user_query": "Hello",
            "dave_response": "World",
            "timestamp": "2026-08-11T12:00:00Z",
            "expected_intent": "invalid_category",
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 422)

    @patch("evaluation_pipeline.evaluators.response_quality_evaluator.ResponseQualityEvaluator.evaluate")
    def test_evaluator_failure_http_200(self, mock_rq) -> None:
        """HTTP 200 returned even if an individual evaluator fails, reporting status='failed'."""
        # Simulated exception inside Response Quality evaluator
        mock_rq.side_effect = Exception("Simulated connection failure during execution")
        
        payload = {
            "conversation_id": "VAL-007",
            "user_query": "Hello",
            "dave_response": "World",
            "timestamp": "2026-08-11T12:00:00Z",
        }

        with patch("evaluation_pipeline.utils.llm_client.LLMJudge.call_with_json") as mock_call:
            mock_call.return_value = ({}, "{}")
            response = self.client.post("/evaluate", json=payload)
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            convo = data["conversations"][0]
            rq_eval = convo["evaluations"]["response_quality"]
            
            # Should have score=None and status="failed"
            self.assertIsNone(rq_eval["score"])
            self.assertEqual(rq_eval["status"], "failed")
            self.assertTrue(convo["evaluation_failed"])
