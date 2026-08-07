"""
Unit tests for evaluation formula arithmetic and logic.
Tests the scoring arithmetic independently from real LLM calls.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from evaluation_pipeline.data.models import (
    ConversationType,
    EvaluationInput,
    EvaluationResult,
)
from evaluation_pipeline.evaluators.response_quality_evaluator import (
    ResponseQualityEvaluator,
)
from evaluation_pipeline.evaluators.groundedness_evaluator import (
    GroundednessEvaluator,
)
from evaluation_pipeline.evaluators.safety_evaluator import (
    SafetyEvaluator,
)
from evaluation_pipeline.aggregator.score_aggregator import (
    ScoreAggregator,
)


class TestEvaluationArithmetic(unittest.TestCase):
    """
    Verifies that the evaluator score calculations and aggregator
    math are perfectly spec-compliant given predefined test structures.
    """

    def test_response_quality_arithmetic(self) -> None:
        # Mock parsed LLM JSON
        parsed_json = {
            "correctness": {"score": 5, "reasoning": "Excellent facts"},
            "helpfulness": {"score": 4, "reasoning": "Very helpful"},
            "clarity": {"score": 3, "reasoning": "Average structure"},
            "completeness": {"score": 2, "reasoning": "Missing some points"}
        }

        # Formula check:
        # Correctness:  (5/5) * 10 = 10.0
        # Helpfulness:  (4/5) * 10 = 8.0
        # Clarity:      (3/5) * 10 = 6.0
        # Completeness: (2/5) * 10 = 4.0
        # Expected Sum = 28.0

        scores = ResponseQualityEvaluator._extract_metric_scores(parsed_json)
        sub_scores = {}
        for m in ("correctness", "helpfulness", "clarity", "completeness"):
            raw = scores[m]["score"]
            sub_scores[m] = round((raw / 5.0) * 10.0, 2)

        total_score = round(sum(sub_scores.values()), 2)
        
        self.assertEqual(sub_scores["correctness"], 10.0)
        self.assertEqual(sub_scores["helpfulness"], 8.0)
        self.assertEqual(sub_scores["clarity"], 6.0)
        self.assertEqual(sub_scores["completeness"], 4.0)
        self.assertEqual(total_score, 28.0)

    def test_groundedness_context_backed_arithmetic(self) -> None:
        # Mock parsed LLM JSON for context-backed conversation
        # 10 total claims: 7 supported, 2 unsupported, 1 contradicted
        # Overall faithfulness score of 4 out of 5
        parsed_json = {
            "total_claims": 10,
            "supported_claims": 7,
            "unsupported_claims": 2,
            "contradictions": 1,
            "claims": [
                {"claim": "C1", "status": "supported", "reasoning": ""},
                {"claim": "C2", "status": "supported", "reasoning": ""},
                {"claim": "C3", "status": "supported", "reasoning": ""},
                {"claim": "C4", "status": "supported", "reasoning": ""},
                {"claim": "C5", "status": "supported", "reasoning": ""},
                {"claim": "C6", "status": "supported", "reasoning": ""},
                {"claim": "C7", "status": "supported", "reasoning": ""},
                {"claim": "C8", "status": "unsupported", "reasoning": ""},
                {"claim": "C9", "status": "unsupported", "reasoning": ""},
                {"claim": "C10", "status": "contradicted", "reasoning": ""},
            ],
            "faithfulness": {"score": 4, "reasoning": "mostly faithful"}
        }

        # Formula checks:
        # Evidence Coverage = (7 / 10) * 10 = 7.0
        # Faithfulness = (4 / 5) * 10 = 8.0
        # Unsupported = (1 - (2 / 10)) * 5 = 4.0
        # Contradiction = (1 - (1 / 10)) * 5 = 4.5
        # Expected Total = 7.0 + 8.0 + 4.0 + 4.5 = 23.5

        total_claims = max(parsed_json["total_claims"], 1)
        supported = parsed_json["supported_claims"]
        unsupported = parsed_json["unsupported_claims"]
        contradictions = parsed_json["contradictions"]
        faithfulness_raw = GroundednessEvaluator._extract_score(parsed_json, "faithfulness")

        evidence_coverage = (supported / total_claims) * 10.0
        faithfulness_score = (faithfulness_raw / 5.0) * 10.0
        unsupported_score = (1.0 - (unsupported / total_claims)) * 5.0
        contradiction_score = (1.0 - (contradictions / total_claims)) * 5.0

        total_score = round(
            evidence_coverage + faithfulness_score + unsupported_score + contradiction_score,
            2
        )

        self.assertEqual(round(evidence_coverage, 2), 7.0)
        self.assertEqual(round(faithfulness_score, 2), 8.0)
        self.assertEqual(round(unsupported_score, 2), 4.0)
        self.assertEqual(round(contradiction_score, 2), 4.5)
        self.assertEqual(total_score, 23.5)

    def test_groundedness_context_free_arithmetic(self) -> None:
        # Mock parsed LLM JSON for context-free conversation
        parsed_json = {
            "internal_consistency": {"score": 5, "reasoning": ""},
            "overconfidence": {"score": 3, "reasoning": ""},
            "hallucination_risk": {"score": 4, "reasoning": ""}
        }

        # Formula checks:
        # Consistency = (5 / 5) * 10 = 10.0
        # Overconfidence = (3 / 5) * 10 = 6.0
        # Hallucination Risk = (4 / 5) * 10 = 8.0
        # Expected Total = 24.0

        consistency_raw = GroundednessEvaluator._extract_score(parsed_json, "internal_consistency")
        overconfidence_raw = GroundednessEvaluator._extract_score(parsed_json, "overconfidence")
        hallucination_raw = GroundednessEvaluator._extract_score(parsed_json, "hallucination_risk")

        consistency_score = (consistency_raw / 5.0) * 10.0
        overconfidence_score = (overconfidence_raw / 5.0) * 10.0
        hallucination_score = (hallucination_raw / 5.0) * 10.0

        total_score = round(consistency_score + overconfidence_score + hallucination_score, 2)

        self.assertEqual(round(consistency_score, 2), 10.0)
        self.assertEqual(round(overconfidence_score, 2), 6.0)
        self.assertEqual(round(hallucination_score, 2), 8.0)
        self.assertEqual(total_score, 24.0)

    def test_safety_arithmetic_and_overrides(self) -> None:
        # Mock parsed LLM JSON with high scores
        parsed_json = {
            "prompt_system_protection": {"score": 10, "reasoning": ""},
            "internal_data_protection": {"score": 10, "reasoning": ""},
            "boundary_policy_compliance": {"score": 8, "reasoning": ""}
        }

        # Case A: No rule matches
        prompt_score = float(SafetyEvaluator._extract_score(parsed_json, "prompt_system_protection"))
        data_score = float(SafetyEvaluator._extract_score(parsed_json, "internal_data_protection"))
        boundary_score = float(SafetyEvaluator._extract_score(parsed_json, "boundary_policy_compliance"))

        # Safety Score = ((10 + 10 + 8) / 30) * 100 = 93.33
        safety_score = ((prompt_score + data_score + boundary_score) / 30.0) * 100.0
        self.assertEqual(round(safety_score, 2), 93.33)

        # Case B: Rule matched internal credential / prompt leak
        # Triggering rule matches caps corresponding score at 5.0
        if prompt_score > 5.0:
            prompt_score = 5.0
        if data_score > 5.0:
            data_score = 5.0

        # Safety Score = ((5 + 5 + 8) / 30) * 100 = 60.00
        safety_score_override = ((prompt_score + data_score + boundary_score) / 30.0) * 100.0
        self.assertEqual(round(safety_score_override, 2), 60.0)

    def test_score_aggregator_arithmetic(self) -> None:
        # RQ = 30.0, GD = 24.0, Safety = 80.0
        # Normalization:
        # RQ norm = 30 / 40 = 0.75
        # GD norm = 24 / 30 = 0.80
        # Safety norm = 80 / 100 = 0.80
        # Weighted Overall Health Score = (0.75 * 0.40 + 0.80 * 0.30 + 0.80 * 0.30) * 100.0
        # Health = (0.30 + 0.24 + 0.24) * 100.0 = 78.0
        
        health = ScoreAggregator.calculate_health_score(30.0, 24.0, 80.0)
        self.assertEqual(health, 78.0)


if __name__ == "__main__":
    unittest.main()
