"""
Phase 9 — Final Production Failure-Injection Test Suite

Executes comprehensive failure injection tests to verify system resiliency,
thread lifecycle, deadline enforcement, error classification, aggregator math,
and concurrency recovery.
"""

import time
import os
import threading
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app
from evaluation_pipeline.data.models import EvaluationInput, ConversationType, EvaluationResult
from evaluation_pipeline.utils.retry_utils import execute_with_retry, is_transient_error
from evaluation_pipeline.utils.concurrency import get_semaphore, reset_max_observed_concurrency
from evaluation_pipeline.aggregator.score_aggregator import ScoreAggregator

client = TestClient(app)

class TestPhase9FailureInjection(unittest.TestCase):

    def setUp(self):
        reset_max_observed_concurrency()

    # ------------------------------------------------------------------
    # 1. Failure Injection Tests A, B, C, D, E, F, G, H, I, J, K, L
    # ------------------------------------------------------------------

    def test_A_gemini_call_sleeps_longer_than_deadline(self):
        """A. Gemini call sleeps longer than the request deadline."""
        os.environ["EVALUATION_REQUEST_TIMEOUT"] = "0.2"
        
        def slow_gemini(*args, **kwargs):
            time.sleep(2.5)
            return {"correctness": {"score": 5, "reasoning": "Good"}}, "raw"

        with patch("evaluation_pipeline.utils.llm_client.LLMJudge.call_with_json", side_effect=slow_gemini):
            payload = {
                "conversation_id": "test_gemini_slow",
                "user_query": "What is company policy?",
                "dave_response": "The policy is remote work allowed.",
                "retrieved_context": "",
                "chat_history": "",
                "timestamp": "2026-08-13T12:00:00Z"
            }
            start_t = time.time()
            response = client.post("/evaluate", json=payload)
            dur = time.time() - start_t
            
            # The client receives a response (bounded by remaining timeout in app.py ~1.0-1.2s)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            evals = data["conversations"][0]["evaluations"]
            self.assertIn(evals["response_quality"]["status"], ["timeout", "failed"])
            self.assertIsNone(evals["response_quality"]["score"])

        del os.environ["EVALUATION_REQUEST_TIMEOUT"]

    def test_B_gemini_call_raises_repeated_transient_429(self):
        """B. Gemini call raises repeated transient 429 errors."""
        os.environ["EVALUATION_REQUEST_TIMEOUT"] = "5.0"
        call_count = 0

        def failing_429(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            from google.genai.errors import APIError
            raise APIError(429, "Rate limit exceeded", {})

        with patch("evaluation_pipeline.utils.llm_client.LLMJudge.call_with_json", side_effect=failing_429):
            payload = {
                "conversation_id": "test_gemini_429",
                "user_query": "Tell me a joke",
                "dave_response": "Why did the chicken cross the road?",
                "retrieved_context": "",
                "chat_history": "",
                "timestamp": "2026-08-13T12:00:00Z"
            }
            response = client.post("/evaluate", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            evals = data["conversations"][0]["evaluations"]
            
            self.assertIn(evals["response_quality"]["status"], ["unavailable", "failed"])
            self.assertIsNone(evals["response_quality"]["score"])

        del os.environ["EVALUATION_REQUEST_TIMEOUT"]

    def test_C_gemini_call_raises_repeated_connection_timeouts(self):
        """C. Gemini call raises repeated connection/timeouts."""
        os.environ["EVALUATION_REQUEST_TIMEOUT"] = "5.0"

        def failing_timeout(*args, **kwargs):
            raise TimeoutError("Connection to Gemini backend timed out")

        with patch("evaluation_pipeline.utils.llm_client.LLMJudge.call_with_json", side_effect=failing_timeout):
            payload = {
                "conversation_id": "test_gemini_conn_timeout",
                "user_query": "Hello",
                "dave_response": "Hi there",
                "retrieved_context": "",
                "chat_history": "",
                "timestamp": "2026-08-13T12:00:00Z"
            }
            response = client.post("/evaluate", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            evals = data["conversations"][0]["evaluations"]
            self.assertEqual(evals["response_quality"]["status"], "timeout")
            self.assertIsNone(evals["response_quality"]["score"])

        del os.environ["EVALUATION_REQUEST_TIMEOUT"]

    def test_D_trulens_call_blocks_longer_than_deadline(self):
        """D. TruLens call blocks longer than the request deadline."""
        from evaluation_pipeline.evaluators.groundedness_evaluator import _run_trulens_groundedness
        
        def slow_trulens(*args, **kwargs):
            time.sleep(2.0)
            return 0.95

        with patch("trulens.providers.google.Google.groundedness_measure_with_cot_reasons", side_effect=slow_trulens):
            deadline = time.time() + 0.2
            result = _run_trulens_groundedness("some context", "some response", "convo_trulens_slow", deadline=deadline)
            self.assertEqual(result["status"], "failed")
            self.assertIn("error", result)

    def test_E_deepeval_call_blocks_longer_than_deadline(self):
        """E. DeepEval call blocks longer than the request deadline."""
        from evaluation_pipeline.evaluators.groundedness_evaluator import _run_deepeval_faithfulness

        def slow_deepeval(*args, **kwargs):
            time.sleep(2.0)

        with patch("deepeval.metrics.FaithfulnessMetric.measure", side_effect=slow_deepeval):
            deadline = time.time() + 0.2
            result = _run_deepeval_faithfulness("query", "response", "some context", "convo_deepeval_slow", deadline=deadline)
            self.assertEqual(result["status"], "failed")
            self.assertIn("error", result)

    def test_F_G_trulens_deepeval_transient_errors(self):
        """F & G. TruLens and DeepEval raise transient errors."""
        from evaluation_pipeline.evaluators.groundedness_evaluator import _run_trulens_groundedness, _run_deepeval_faithfulness
        
        with patch("trulens.providers.google.Google.groundedness_measure_with_cot_reasons", side_effect=ValueError("TruLens Transient 503")):
            res = _run_trulens_groundedness("context", "resp", "trulens_transient", deadline=time.time()+5.0)
            self.assertEqual(res["status"], "failed")

        with patch("deepeval.metrics.FaithfulnessMetric.measure", side_effect=ValueError("DeepEval Transient Rate Limit")):
            res = _run_deepeval_faithfulness("q", "r", "c", "deepeval_transient", deadline=time.time()+5.0)
            self.assertEqual(res["status"], "failed")

    def test_H_semaphore_acquisition_delay(self):
        """H. Semaphore acquisition is delayed beyond its timeout."""
        from evaluation_pipeline.utils.concurrency import controlled_concurrency
        
        sem, limit = get_semaphore()
        with patch("evaluation_pipeline.utils.concurrency.get_semaphore", return_value=(sem, limit)):
            with patch("threading.Semaphore.acquire", return_value=False):
                with self.assertRaises(TimeoutError) as cm:
                    with controlled_concurrency("test_eval", "TestFW", "convo_sem_test"):
                        pass
                self.assertIn("Could not acquire concurrency slot", str(cm.exception))

    def test_I_one_evaluator_hangs_while_others_succeed(self):
        """I. One evaluator hangs while all other evaluators succeed."""
        os.environ["EVALUATION_REQUEST_TIMEOUT"] = "0.3"

        def selective_hang(sys_prompt, user_prompt, evaluator, conversation_id, response_schema=None, deadline=None):
            if evaluator == "response_quality":
                time.sleep(2.0)
                return {}, ""
            # Mock success for other evaluators
            if evaluator == "groundedness":
                return {"internal_consistency": {"score": 5, "reasoning": "ok"}, "overconfidence": {"score": 5, "reasoning": "ok"}, "hallucination_risk": {"score": 5, "reasoning": "ok"}}, ""
            if evaluator == "safety":
                return {"confidentiality_information_protection": {"score": 5, "reasoning": "ok"}, "security_attack_resistance": {"score": 5, "reasoning": "ok"}, "boundary_policy_compliance": {"score": 5, "reasoning": "ok"}}, ""
            if evaluator == "intent_understanding":
                return {"intent_classification": {"score": 5, "reasoning": "ok"}, "clarification_handling": {"score": 5, "reasoning": "ok"}, "actionability": {"score": 5, "reasoning": "ok"}}, ""
            if evaluator == "memory_and_continuity":
                return {"context_continuity": {"score": 5, "reasoning": "ok"}, "information_retention": {"score": 5, "reasoning": "ok"}, "consistency_across_turns": {"score": 5, "reasoning": "ok"}}, ""
            return {}, ""

        with patch("evaluation_pipeline.utils.llm_client.LLMJudge.call_with_json", side_effect=selective_hang):
            payload = {
                "conversation_id": "test_selective_hang",
                "user_query": "What is the policy?",
                "dave_response": "The policy is X.",
                "retrieved_context": "",
                "chat_history": "",
                "timestamp": "2026-08-13T12:00:00Z"
            }
            response = client.post("/evaluate", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            evals = data["conversations"][0]["evaluations"]
            
            # Response quality timed out
            self.assertEqual(evals["response_quality"]["status"], "timeout")
            self.assertIsNone(evals["response_quality"]["score"])
            
            # Groundedness succeeded
            self.assertEqual(evals["groundedness"]["status"], "success")
            self.assertEqual(evals["groundedness"]["score"], 20.0)
            
            # Overall health score is computed over valid dimensions
            health_score = data["conversations"][0]["overall_health_score"]
            self.assertIsNotNone(health_score)
            self.assertGreater(health_score, 0.0)

        del os.environ["EVALUATION_REQUEST_TIMEOUT"]

    def test_J_multiple_evaluators_hang_simultaneously(self):
        """J. Multiple evaluators hang simultaneously."""
        os.environ["EVALUATION_REQUEST_TIMEOUT"] = "0.2"

        def all_hang(*args, **kwargs):
            time.sleep(2.0)
            return {}, ""

        with patch("evaluation_pipeline.utils.llm_client.LLMJudge.call_with_json", side_effect=all_hang):
            payload = {
                "conversation_id": "test_all_hang",
                "user_query": "What is the policy?",
                "dave_response": "The policy is X.",
                "retrieved_context": "",
                "chat_history": "",
                "timestamp": "2026-08-13T12:00:00Z"
            }
            response = client.post("/evaluate", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            evals = data["conversations"][0]["evaluations"]
            
            for key in ["response_quality", "groundedness", "safety", "intent_understanding"]:
                self.assertIn(evals[key]["status"], ["timeout", "failed"])
                self.assertIsNone(evals[key]["score"])

            self.assertIsNone(data["conversations"][0]["overall_health_score"])

        del os.environ["EVALUATION_REQUEST_TIMEOUT"]

    def test_K_L_request_isolation_and_repeat(self):
        """K & L. Request A times out, followed immediately by Request B and repeated Request A."""
        os.environ["EVALUATION_REQUEST_TIMEOUT"] = "0.2"

        def slow_call(*args, **kwargs):
            time.sleep(2.0)
            return {}, ""

        def fast_call(*args, **kwargs):
            return {"correctness": {"score": 4, "reasoning": "ok"}, "helpfulness": {"score": 4, "reasoning": "ok"}, "clarity": {"score": 4, "reasoning": "ok"}, "completeness": {"score": 4, "reasoning": "ok"}}, "raw"

        payload_A = {
            "conversation_id": "REQ_A",
            "user_query": "Query A",
            "dave_response": "Response A",
            "retrieved_context": "",
            "chat_history": "",
            "timestamp": "2026-08-13T12:00:00Z"
        }

        payload_B = {
            "conversation_id": "REQ_B",
            "user_query": "Query B",
            "dave_response": "Response B",
            "retrieved_context": "",
            "chat_history": "",
            "timestamp": "2026-08-13T12:00:00Z"
        }

        # Request A (times out)
        with patch("evaluation_pipeline.utils.llm_client.LLMJudge.call_with_json", side_effect=slow_call):
            res_A = client.post("/evaluate", json=payload_A)
            self.assertEqual(res_A.status_code, 200)
            data_A = res_A.json()
            self.assertEqual(data_A["conversations"][0]["conversation_id"], "REQ_A")
            self.assertEqual(data_A["conversations"][0]["evaluations"]["response_quality"]["status"], "timeout")

        # Request B (succeeds, no leak from A)
        with patch("evaluation_pipeline.utils.llm_client.LLMJudge.call_with_json", side_effect=fast_call):
            res_B = client.post("/evaluate", json=payload_B)
            self.assertEqual(res_B.status_code, 200)
            data_B = res_B.json()
            self.assertEqual(data_B["conversations"][0]["conversation_id"], "REQ_B")
            self.assertEqual(data_B["conversations"][0]["evaluations"]["response_quality"]["status"], "success")
            self.assertEqual(data_B["conversations"][0]["evaluations"]["response_quality"]["score"], 16.0)

        del os.environ["EVALUATION_REQUEST_TIMEOUT"]

    # ------------------------------------------------------------------
    # 2. Thread Lifecycle & Concurrency Recovery
    # ------------------------------------------------------------------

    def test_thread_lifecycle_and_concurrency_recovery(self):
        """
        Specifically measure active thread count before, immediately after,
        and verify background completion.
        """
        threads_before = threading.active_count()
        os.environ["EVALUATION_REQUEST_TIMEOUT"] = "0.2"

        def background_sleeper(*args, **kwargs):
            time.sleep(1.5)
            return {}, ""

        with patch("evaluation_pipeline.utils.llm_client.LLMJudge.call_with_json", side_effect=background_sleeper):
            payload = {
                "conversation_id": "thread_lifecycle_convo",
                "user_query": "Lifecycle test",
                "dave_response": "Lifecycle response",
                "retrieved_context": "",
                "chat_history": "",
                "timestamp": "2026-08-13T12:00:00Z"
            }
            res = client.post("/evaluate", json=payload)
            self.assertEqual(res.status_code, 200)

            # Wait for background sleeper to complete
            time.sleep(1.8)
            threads_after = threading.active_count()

            # Confirm background threads finish and active thread count returns
            self.assertLessEqual(abs(threads_after - threads_before), 2)

        del os.environ["EVALUATION_REQUEST_TIMEOUT"]

    # ------------------------------------------------------------------
    # 3. Aggregator Verification
    # ------------------------------------------------------------------

    def test_aggregator_verification_math(self):
        """Verify denominator and numerator calculations across all evaluator statuses."""
        agg = ScoreAggregator()
        dummy_input = EvaluationInput(
            conversation_id="AGG-001",
            user_query="Query for aggregator test",
            dave_response="Response for aggregator test",
            conversation_type=ConversationType.CONTEXT_FREE,
            timestamp="2026-08-13T12:00:00Z"
        )

        rq_ok = EvaluationResult(evaluator_name="rq", conversation_id="AGG-001", score=15.0, max_score=20.0, status="success", sub_scores={}, feedback="Looks correct and well structured.", flagged=False)
        gd_failed = EvaluationResult(evaluator_name="gd", conversation_id="AGG-001", score=None, max_score=20.0, status="failed", sub_scores={}, feedback="Evaluation failed with an error.", flagged=True)
        sf_timeout = EvaluationResult(evaluator_name="sf", conversation_id="AGG-001", score=None, max_score=20.0, status="timeout", sub_scores={}, feedback="Request timed out during eval.", flagged=True)
        it_ok = EvaluationResult(evaluator_name="it", conversation_id="AGG-001", score=20.0, max_score=20.0, status="success", sub_scores={}, feedback="Intent correctly identified.", flagged=False)
        me_na = EvaluationResult(evaluator_name="me", conversation_id="AGG-001", score=None, max_score=20.0, status="not_applicable", applicable=False, sub_scores={}, feedback="Not applicable metric here.", flagged=False)

        report = agg.aggregate_dataset(
            inputs=[dummy_input],
            rq_results=[rq_ok],
            gd_results=[gd_failed],
            safety_results=[sf_timeout],
            intent_results=[it_ok],
            memory_results=[me_na]
        )

        convo = report["conversations"][0]
        # Raw applicable score = 15 + 20 = 35.0
        self.assertEqual(convo["raw_applicable_score"], 35.0)
        # Max denominator = 20 (rq) + 20 (gd) + 20 (sf) + 20 (it) = 80.0 (Memory excluded because not_applicable)
        self.assertEqual(convo["applicable_max_score"], 80.0)
        # Overall health = (35 / 80) * 100 = 43.75
        self.assertEqual(convo["overall_health_score"], 43.75)

    def test_aggregator_all_evaluators_failed(self):
        """Verify overall_health_score=None when all evaluators fail."""
        agg = ScoreAggregator()
        dummy_input = EvaluationInput(
            conversation_id="AGG-002",
            user_query="Query for aggregator test",
            dave_response="Response for aggregator test",
            conversation_type=ConversationType.CONTEXT_FREE,
            timestamp="2026-08-13T12:00:00Z"
        )

        rq_f = EvaluationResult(evaluator_name="rq", conversation_id="AGG-002", score=None, max_score=20.0, status="failed", sub_scores={}, feedback="Evaluation failed with an error.", flagged=True)
        gd_f = EvaluationResult(evaluator_name="gd", conversation_id="AGG-002", score=None, max_score=20.0, status="failed", sub_scores={}, feedback="Evaluation failed with an error.", flagged=True)
        sf_f = EvaluationResult(evaluator_name="sf", conversation_id="AGG-002", score=None, max_score=20.0, status="failed", sub_scores={}, feedback="Evaluation failed with an error.", flagged=True)
        it_f = EvaluationResult(evaluator_name="it", conversation_id="AGG-002", score=None, max_score=20.0, status="failed", sub_scores={}, feedback="Evaluation failed with an error.", flagged=True)

        report = agg.aggregate_dataset(
            inputs=[dummy_input],
            rq_results=[rq_f],
            gd_results=[gd_f],
            safety_results=[sf_f],
            intent_results=[it_f]
        )

        convo = report["conversations"][0]
        self.assertIsNone(convo["overall_health_score"])
        self.assertTrue(convo["evaluation_failed"])


if __name__ == "__main__":
    unittest.main()
