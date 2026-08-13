import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
from evaluation_pipeline.data.models import EvaluationInput, ConversationType
from evaluation_pipeline.evaluators.groundedness_evaluator import GroundednessEvaluator


class TestGroundednessFrameworks(unittest.TestCase):
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "mock-api-key"})
    def setUp(self) -> None:
        self.evaluator = GroundednessEvaluator()

    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_trulens_groundedness")
    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_deepeval_faithfulness")
    @patch.object(GroundednessEvaluator, "_run_custom_context_backed_judge")
    def test_both_frameworks_success(self, mock_custom, mock_deepeval, mock_trulens) -> None:
        """Verify output when both external frameworks succeed."""
        mock_custom.return_value = (
            {
                "internal_consistency": {"score": 5, "reasoning": "consistent"},
                "overconfidence": {"score": 5, "reasoning": "confident"},
                "hallucination_risk": {"score": 5, "reasoning": "no risk"},
                "explanation": "perfect response"
            },
            "{}"
        )
        mock_trulens.return_value = {"status": "success", "score": 0.95}
        mock_deepeval.return_value = {"status": "success", "score": 0.88}

        eval_input = EvaluationInput(
            conversation_id="conv_success",
            conversation_type=ConversationType.CONTEXT_BACKED,
            user_query="Hello",
            dave_response="World",
            retrieved_context="Context",
            timestamp=datetime.now(),
        )

        res = self.evaluator.evaluate(eval_input)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.score, 20.0)  # Max score is 20.0
        self.assertEqual(res.sub_scores["trulens_status"], "success")
        self.assertEqual(res.sub_scores["trulens_score"], 0.95)
        self.assertEqual(res.sub_scores["deepeval_status"], "success")
        self.assertEqual(res.sub_scores["deepeval_score"], 0.88)

    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_trulens_groundedness")
    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_deepeval_faithfulness")
    @patch.object(GroundednessEvaluator, "_run_custom_context_backed_judge")
    def test_trulens_failure_only(self, mock_custom, mock_deepeval, mock_trulens) -> None:
        """Verify overall score is retained when TruLens fails but custom judge and DeepEval succeed."""
        mock_custom.return_value = (
            {
                "internal_consistency": {"score": 5, "reasoning": "consistent"},
                "overconfidence": {"score": 5, "reasoning": "confident"},
                "hallucination_risk": {"score": 5, "reasoning": "no risk"},
                "explanation": "perfect response"
            },
            "{}"
        )
        mock_trulens.return_value = {"status": "failed", "error": "TruLens API Timeout"}
        mock_deepeval.return_value = {"status": "success", "score": 0.9}

        eval_input = EvaluationInput(
            conversation_id="conv_trulens_fail",
            conversation_type=ConversationType.CONTEXT_BACKED,
            user_query="Hello",
            dave_response="World",
            retrieved_context="Context",
            timestamp=datetime.now(),
        )

        res = self.evaluator.evaluate(eval_input)
        self.assertEqual(res.status, "success")  # The evaluation is still success because custom judge succeeded
        self.assertEqual(res.score, 20.0)
        self.assertEqual(res.sub_scores["trulens_status"], "failed")
        self.assertEqual(res.sub_scores["trulens_error"], "TruLens API Timeout")
        self.assertEqual(res.sub_scores["deepeval_status"], "success")

    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_trulens_groundedness")
    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_deepeval_faithfulness")
    @patch.object(GroundednessEvaluator, "_run_custom_context_backed_judge")
    def test_deepeval_failure_only(self, mock_custom, mock_deepeval, mock_trulens) -> None:
        """Verify overall score is retained when DeepEval fails but custom judge and TruLens succeed."""
        mock_custom.return_value = (
            {
                "internal_consistency": {"score": 5, "reasoning": "consistent"},
                "overconfidence": {"score": 5, "reasoning": "confident"},
                "hallucination_risk": {"score": 5, "reasoning": "no risk"},
                "explanation": "perfect response"
            },
            "{}"
        )
        mock_trulens.return_value = {"status": "success", "score": 0.92}
        mock_deepeval.return_value = {"status": "failed", "error": "DeepEval connection lost"}

        eval_input = EvaluationInput(
            conversation_id="conv_deepeval_fail",
            conversation_type=ConversationType.CONTEXT_BACKED,
            user_query="Hello",
            dave_response="World",
            retrieved_context="Context",
            timestamp=datetime.now(),
        )

        res = self.evaluator.evaluate(eval_input)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.score, 20.0)
        self.assertEqual(res.sub_scores["trulens_status"], "success")
        self.assertEqual(res.sub_scores["deepeval_status"], "failed")
        self.assertEqual(res.sub_scores["deepeval_error"], "DeepEval connection lost")

    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_trulens_groundedness")
    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_deepeval_faithfulness")
    @patch.object(GroundednessEvaluator, "_run_custom_context_backed_judge")
    def test_both_frameworks_failed(self, mock_custom, mock_deepeval, mock_trulens) -> None:
        """Verify overall score is retained even if both frameworks fail, provided custom judge succeeds."""
        mock_custom.return_value = (
            {
                "internal_consistency": {"score": 5, "reasoning": "consistent"},
                "overconfidence": {"score": 5, "reasoning": "confident"},
                "hallucination_risk": {"score": 5, "reasoning": "no risk"},
                "explanation": "perfect response"
            },
            "{}"
        )
        mock_trulens.return_value = {"status": "failed", "error": "TruLens Error"}
        mock_deepeval.return_value = {"status": "failed", "error": "DeepEval Error"}

        eval_input = EvaluationInput(
            conversation_id="conv_both_fail",
            conversation_type=ConversationType.CONTEXT_BACKED,
            user_query="Hello",
            dave_response="World",
            retrieved_context="Context",
            timestamp=datetime.now(),
        )

        res = self.evaluator.evaluate(eval_input)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.score, 20.0)
        self.assertEqual(res.sub_scores["trulens_status"], "failed")
        self.assertEqual(res.sub_scores["deepeval_status"], "failed")

    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_trulens_groundedness")
    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_deepeval_faithfulness")
    @patch.object(GroundednessEvaluator, "_run_custom_context_backed_judge")
    def test_no_context(self, mock_custom, mock_deepeval, mock_trulens) -> None:
        """Verify TruLens and DeepEval return status 'not_applicable' when retrieved_context is missing."""
        mock_custom.return_value = (
            {
                "internal_consistency": {"score": 5, "reasoning": "consistent"},
                "overconfidence": {"score": 5, "reasoning": "confident"},
                "hallucination_risk": {"score": 5, "reasoning": "no risk"},
                "explanation": "perfect response"
            },
            "{}"
        )
        # Mocking the functions to execute the actual fallback check
        mock_trulens.side_effect = lambda context, response, convo_id, *args, **kwargs: {
            "status": "not_applicable",
            "reason": "No retrieved context available"
        } if not context else {"status": "success", "score": 1.0}
        
        mock_deepeval.side_effect = lambda user_query, response, context, convo_id, *args, **kwargs: {
            "status": "not_applicable",
            "reason": "No retrieved context available"
        } if not context else {"status": "success", "score": 1.0}

        eval_input = EvaluationInput(
            conversation_id="conv_no_context",
            conversation_type=ConversationType.CONTEXT_BACKED,
            user_query="Hello",
            dave_response="World",
            retrieved_context="",  # empty
            timestamp=datetime.now(),
        )

        res = self.evaluator.evaluate(eval_input)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.score, 20.0)
        self.assertEqual(res.sub_scores["trulens_status"], "not_applicable")
        self.assertEqual(res.sub_scores["deepeval_status"], "not_applicable")

    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_trulens_groundedness")
    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_deepeval_faithfulness")
    @patch.object(GroundednessEvaluator, "_run_custom_context_backed_judge")
    def test_trulens_success_zero_score_preserved(self, mock_custom, mock_deepeval, mock_trulens) -> None:
        """Requirement 1: TruLens success with score 0.0 -> preserve 0.0."""
        mock_custom.return_value = (
            {
                "internal_consistency": {"score": 5, "reasoning": "ok"},
                "overconfidence": {"score": 5, "reasoning": "ok"},
                "hallucination_risk": {"score": 5, "reasoning": "ok"},
                "explanation": "ok"
            },
            "{}"
        )
        mock_trulens.return_value = {"status": "success", "score": 0.0}
        mock_deepeval.return_value = {"status": "success", "score": 1.0}

        eval_input = EvaluationInput(
            conversation_id="conv_trulens_zero",
            conversation_type=ConversationType.CONTEXT_BACKED,
            user_query="Q", dave_response="A", retrieved_context="C", timestamp=datetime.now()
        )
        res = self.evaluator.evaluate(eval_input)
        self.assertEqual(res.sub_scores["trulens_status"], "success")
        self.assertIn("trulens_score", res.sub_scores)
        self.assertEqual(res.sub_scores["trulens_score"], 0.0)

    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_trulens_groundedness")
    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_deepeval_faithfulness")
    @patch.object(GroundednessEvaluator, "_run_custom_context_backed_judge")
    def test_trulens_failed_no_numeric_score(self, mock_custom, mock_deepeval, mock_trulens) -> None:
        """Requirement 2: TruLens failed -> no numeric score."""
        mock_custom.return_value = (
            {
                "internal_consistency": {"score": 5, "reasoning": "ok"},
                "overconfidence": {"score": 5, "reasoning": "ok"},
                "hallucination_risk": {"score": 5, "reasoning": "ok"},
                "explanation": "ok"
            },
            "{}"
        )
        mock_trulens.return_value = {"status": "failed", "error": "Internal Error"}
        mock_deepeval.return_value = {"status": "success", "score": 1.0}

        eval_input = EvaluationInput(
            conversation_id="conv_trulens_failed",
            conversation_type=ConversationType.CONTEXT_BACKED,
            user_query="Q", dave_response="A", retrieved_context="C", timestamp=datetime.now()
        )
        res = self.evaluator.evaluate(eval_input)
        self.assertEqual(res.sub_scores["trulens_status"], "failed")
        self.assertNotIn("trulens_score", res.sub_scores)
        self.assertEqual(res.sub_scores["trulens_error"], "Internal Error")

    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_trulens_groundedness")
    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_deepeval_faithfulness")
    @patch.object(GroundednessEvaluator, "_run_custom_context_backed_judge")
    def test_trulens_not_applicable_no_numeric_score(self, mock_custom, mock_deepeval, mock_trulens) -> None:
        """Requirement 3: TruLens not_applicable -> no numeric score."""
        mock_custom.return_value = (
            {
                "internal_consistency": {"score": 5, "reasoning": "ok"},
                "overconfidence": {"score": 5, "reasoning": "ok"},
                "hallucination_risk": {"score": 5, "reasoning": "ok"},
                "explanation": "ok"
            },
            "{}"
        )
        mock_trulens.return_value = {"status": "not_applicable", "reason": "No context"}
        mock_deepeval.return_value = {"status": "success", "score": 1.0}

        eval_input = EvaluationInput(
            conversation_id="conv_trulens_na",
            conversation_type=ConversationType.CONTEXT_BACKED,
            user_query="Q", dave_response="A", retrieved_context="C", timestamp=datetime.now()
        )
        res = self.evaluator.evaluate(eval_input)
        self.assertEqual(res.sub_scores["trulens_status"], "not_applicable")
        self.assertNotIn("trulens_score", res.sub_scores)
        self.assertEqual(res.sub_scores["trulens_reason"], "No context")

    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_trulens_groundedness")
    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_deepeval_faithfulness")
    @patch.object(GroundednessEvaluator, "_run_custom_context_backed_judge")
    def test_deepeval_success_zero_score_preserved(self, mock_custom, mock_deepeval, mock_trulens) -> None:
        """Requirement 4: DeepEval success with score 0.0 -> preserve 0.0."""
        mock_custom.return_value = (
            {
                "internal_consistency": {"score": 5, "reasoning": "ok"},
                "overconfidence": {"score": 5, "reasoning": "ok"},
                "hallucination_risk": {"score": 5, "reasoning": "ok"},
                "explanation": "ok"
            },
            "{}"
        )
        mock_trulens.return_value = {"status": "success", "score": 1.0}
        mock_deepeval.return_value = {"status": "success", "score": 0.0}

        eval_input = EvaluationInput(
            conversation_id="conv_deepeval_zero",
            conversation_type=ConversationType.CONTEXT_BACKED,
            user_query="Q", dave_response="A", retrieved_context="C", timestamp=datetime.now()
        )
        res = self.evaluator.evaluate(eval_input)
        self.assertEqual(res.sub_scores["deepeval_status"], "success")
        self.assertIn("deepeval_score", res.sub_scores)
        self.assertEqual(res.sub_scores["deepeval_score"], 0.0)

    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_trulens_groundedness")
    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_deepeval_faithfulness")
    @patch.object(GroundednessEvaluator, "_run_custom_context_backed_judge")
    def test_deepeval_failed_no_numeric_score(self, mock_custom, mock_deepeval, mock_trulens) -> None:
        """Requirement 5: DeepEval failed -> no numeric score."""
        mock_custom.return_value = (
            {
                "internal_consistency": {"score": 5, "reasoning": "ok"},
                "overconfidence": {"score": 5, "reasoning": "ok"},
                "hallucination_risk": {"score": 5, "reasoning": "ok"},
                "explanation": "ok"
            },
            "{}"
        )
        mock_trulens.return_value = {"status": "success", "score": 1.0}
        mock_deepeval.return_value = {"status": "failed", "error": "API Error"}

        eval_input = EvaluationInput(
            conversation_id="conv_deepeval_failed",
            conversation_type=ConversationType.CONTEXT_BACKED,
            user_query="Q", dave_response="A", retrieved_context="C", timestamp=datetime.now()
        )
        res = self.evaluator.evaluate(eval_input)
        self.assertEqual(res.sub_scores["deepeval_status"], "failed")
        self.assertNotIn("deepeval_score", res.sub_scores)
        self.assertEqual(res.sub_scores["deepeval_error"], "API Error")

    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_trulens_groundedness")
    @patch("evaluation_pipeline.evaluators.groundedness_evaluator._run_deepeval_faithfulness")
    @patch.object(GroundednessEvaluator, "_run_custom_context_backed_judge")
    def test_deepeval_not_applicable_no_numeric_score(self, mock_custom, mock_deepeval, mock_trulens) -> None:
        """Requirement 6: DeepEval not_applicable -> no numeric score."""
        mock_custom.return_value = (
            {
                "internal_consistency": {"score": 5, "reasoning": "ok"},
                "overconfidence": {"score": 5, "reasoning": "ok"},
                "hallucination_risk": {"score": 5, "reasoning": "ok"},
                "explanation": "ok"
            },
            "{}"
        )
        mock_trulens.return_value = {"status": "success", "score": 1.0}
        mock_deepeval.return_value = {"status": "not_applicable", "reason": "No context"}

        eval_input = EvaluationInput(
            conversation_id="conv_deepeval_na",
            conversation_type=ConversationType.CONTEXT_BACKED,
            user_query="Q", dave_response="A", retrieved_context="C", timestamp=datetime.now()
        )
        res = self.evaluator.evaluate(eval_input)
        self.assertEqual(res.sub_scores["deepeval_status"], "not_applicable")
        self.assertNotIn("deepeval_score", res.sub_scores)
        self.assertEqual(res.sub_scores["deepeval_reason"], "No context")

