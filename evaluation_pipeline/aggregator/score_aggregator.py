"""
Score Aggregator — Phase 1

Combines scores from Response Quality, Groundedness, Safety, Intent Understanding,
and Memory & Context Continuity evaluators into a unified Overall Health Score.
Computes aggregate dataset statistics.
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
        intent_score: float = 0.0,
        memory_score: float | None = None,
    ) -> float:
        """
        Calculate raw applicable score.
        """
        health = rq_score + gd_score + safety_score + intent_score
        if memory_score is not None:
            health += memory_score
        return round(health, 2)

    def aggregate_dataset(
        self,
        inputs: list[Any],
        rq_results: list[EvaluationResult],
        gd_results: list[EvaluationResult],
        safety_results: list[EvaluationResult],
        intent_results: list[EvaluationResult] | None = None,
        memory_results: list[EvaluationResult] | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate results across the dataset and compute top-level statistics.
        """
        # Create lookups by conversation ID
        rq_map = {r.conversation_id: r for r in rq_results}
        gd_map = {r.conversation_id: r for r in gd_results}
        safety_map = {r.conversation_id: r for r in safety_results}
        intent_map = {r.conversation_id: r for r in intent_results} if intent_results else {}
        memory_map = {r.conversation_id: r for r in memory_results} if memory_results else {}

        convo_records = []
        flagged_count = 0

        # Groundedness category breakdown lists
        gd_cb_scores = []
        gd_cf_scores = []

        total_rq = 0.0
        total_gd = 0.0
        total_safety = 0.0
        total_intent = 0.0
        total_memory = 0.0
        memory_applicable_count = 0
        total_normalized_health = 0.0

        for inp in inputs:
            conv_id = inp.conversation_id
            rq = rq_map.get(conv_id)
            gd = gd_map.get(conv_id)
            safety = safety_map.get(conv_id)
            intent = intent_map.get(conv_id)
            memory = memory_map.get(conv_id)

            if not rq or not gd or not safety:
                logger.warning("Incomplete evaluation results for conversation '%s'", conv_id)
                continue

            intent_score = intent.score if intent else 0.0
            
            memory_val = None
            max_health_convo = rq.max_score + gd.max_score + safety.max_score
            if intent:
                max_health_convo += intent.max_score

            if memory and memory.applicable:
                memory_val = memory.score
                max_health_convo += memory.max_score
                total_memory += memory.score
                memory_applicable_count += 1

            raw_app_score = self.calculate_health_score(rq.score, gd.score, safety.score, intent_score, memory_val)
            overall_health_score = round((raw_app_score / max_health_convo) * 100.0, 2)

            is_flagged = (
                rq.flagged 
                or gd.flagged 
                or safety.flagged 
                or (intent.flagged if intent else False)
                or (memory.flagged if (memory and memory.applicable) else False)
            )
            if is_flagged:
                flagged_count += 1

            # Stats aggregation
            total_rq += rq.score
            total_gd += gd.score
            total_safety += safety.score
            if intent:
                total_intent += intent.score
            total_normalized_health += overall_health_score

            if inp.conversation_type == ConversationType.CONTEXT_BACKED:
                gd_cb_scores.append(gd.score)
            else:
                gd_cf_scores.append(gd.score)

            evals = {
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
                    "critical_violation": getattr(safety, "critical_violation", False),
                }
            }

            if intent:
                evals["intent_understanding"] = {
                    "score": intent.score,
                    "max_score": intent.max_score,
                    "sub_scores": intent.sub_scores,
                    "feedback": intent.feedback,
                    "flagged": intent.flagged,
                }

            if memory:
                evals["memory_and_continuity"] = {
                    "score": memory.score,
                    "max_score": memory.max_score,
                    "applicable": memory.applicable,
                    "percentage": memory.percentage,
                    "sub_scores": memory.sub_scores,
                    "feedback": memory.feedback,
                    "flagged": memory.flagged,
                }

            convo_records.append({
                "conversation_id": conv_id,
                "conversation_type": inp.conversation_type.value,
                "raw_applicable_score": raw_app_score,
                "applicable_max_score": max_health_convo,
                "overall_health_score": overall_health_score,
                "flagged": is_flagged,
                "evaluations": evals
            })

        count = len(convo_records) or 1
        averages = {
            "response_quality": round(total_rq / count, 2),
            "groundedness": round(total_gd / count, 2),
            "safety": round(total_safety / count, 2),
            "overall_health": round(total_normalized_health / count, 2),
        }
        if intent_results:
            averages["intent_understanding"] = round(total_intent / count, 2)
        if memory_applicable_count > 0:
            averages["memory_and_continuity"] = round(total_memory / memory_applicable_count, 2)

        summary_stats = {
            "total_conversations": len(convo_records),
            "flagged_conversations": flagged_count,
            "averages": averages,
            "groundedness_breakdown": {
                "context_backed_average": round(sum(gd_cb_scores) / (len(gd_cb_scores) or 1), 2),
                "context_backed_count": len(gd_cb_scores),
                "context_free_average": round(sum(gd_cf_scores) / (len(gd_cf_scores) or 1), 2),
                "context_free_count": len(gd_cf_scores),
            }
        }

        return {
            "pipeline_phase": "phase_1",
            "summary_stats": summary_stats,
            "conversations": convo_records,
        }
