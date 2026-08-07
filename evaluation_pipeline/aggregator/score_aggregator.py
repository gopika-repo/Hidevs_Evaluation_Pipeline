"""
Score Aggregator — Phase 0C

Combines scores from Response Quality, Groundedness, and Safety evaluators
into a unified Overall Health Score. Computes aggregate dataset statistics.
"""

from __future__ import annotations

import logging
from typing import Any

from evaluation_pipeline.data.models import ConversationType, EvaluationResult

logger = logging.getLogger(__name__)


class ScoreAggregator:
    """
    Aggregates evaluation results and calculates statistical summaries.
    """

    @staticmethod
    def calculate_health_score(
        rq_score: float,
        gd_score: float,
        safety_score: float,
    ) -> float:
        """
        Calculate Overall Health Score on a 0-100 scale.

        Formula:
          Health = Response Quality (max 40) + Groundedness (max 30) + Safety (max 30)
        """
        health = rq_score + gd_score + safety_score
        return round(health, 2)

    def aggregate_dataset(
        self,
        inputs: list[Any],
        rq_results: list[EvaluationResult],
        gd_results: list[EvaluationResult],
        safety_results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """
        Aggregate results across the dataset and compute top-level statistics.
        """
        # Create lookups by conversation ID
        rq_map = {r.conversation_id: r for r in rq_results}
        gd_map = {r.conversation_id: r for r in gd_results}
        safety_map = {r.conversation_id: r for r in safety_results}

        convo_records = []
        flagged_count = 0

        # Groundedness category breakdown lists
        gd_cb_scores = []
        gd_cf_scores = []

        total_rq = 0.0
        total_gd = 0.0
        total_safety = 0.0
        total_health = 0.0

        for inp in inputs:
            conv_id = inp.conversation_id
            rq = rq_map.get(conv_id)
            gd = gd_map.get(conv_id)
            safety = safety_map.get(conv_id)

            if not rq or not gd or not safety:
                logger.warning("Incomplete evaluation results for conversation '%s'", conv_id)
                continue

            health_score = self.calculate_health_score(rq.score, gd.score, safety.score)

            is_flagged = rq.flagged or gd.flagged or safety.flagged
            if is_flagged:
                flagged_count += 1

            # Stats aggregation
            total_rq += rq.score
            total_gd += gd.score
            total_safety += safety.score
            total_health += health_score

            if inp.conversation_type == ConversationType.CONTEXT_BACKED:
                gd_cb_scores.append(gd.score)
            else:
                gd_cf_scores.append(gd.score)

            convo_records.append({
                "conversation_id": conv_id,
                "conversation_type": inp.conversation_type.value,
                "overall_health_score": health_score,
                "flagged": is_flagged,
                "evaluations": {
                    "response_quality": {
                        "score": rq.score,
                        "max_score": rq.max_score,
                        "sub_scores": rq.sub_scores,
                        "feedback": rq.feedback,
                        "flagged": rq.flagged,
                    },
                    "groundedness": {
                        "score": gd.score,
                        "max_score": gd.max_score,
                        "sub_scores": gd.sub_scores,
                        "feedback": gd.feedback,
                        "flagged": gd.flagged,
                    },
                    "safety": {
                        "score": safety.score,
                        "max_score": safety.max_score,
                        "percentage": safety.percentage,
                        "sub_scores": safety.sub_scores,
                        "feedback": safety.feedback,
                        "flagged": safety.flagged,
                    }
                }
            })

        count = len(convo_records) or 1
        summary_stats = {
            "total_conversations": len(convo_records),
            "flagged_conversations": flagged_count,
            "averages": {
                "response_quality": round(total_rq / count, 2),
                "groundedness": round(total_gd / count, 2),
                "safety": round(total_safety / count, 2),
                "overall_health": round(total_health / count, 2),
            },
            "groundedness_breakdown": {
                "context_backed_average": round(sum(gd_cb_scores) / (len(gd_cb_scores) or 1), 2),
                "context_backed_count": len(gd_cb_scores),
                "context_free_average": round(sum(gd_cf_scores) / (len(gd_cf_scores) or 1), 2),
                "context_free_count": len(gd_cf_scores),
            }
        }

        return {
            "summary_stats": summary_stats,
            "conversations": convo_records,
        }
