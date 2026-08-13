"""
Score Aggregator — Phase 1

Combines scores from Response Quality, Groundedness, Safety, Intent Understanding,
and Memory & Context Continuity evaluators into a unified Overall Health Score.
Computes aggregate dataset statistics, gracefully handling failed evaluations.
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
        rq_score: float | None,
        gd_score: float | None,
        safety_score: float | None,
        intent_score: float | None = 0.0,
        memory_score: float | None = None,
    ) -> float:
        """
        Calculate raw applicable score. Gracefully ignores None values.
        """
        health = 0.0
        if rq_score is not None:
            health += rq_score
        if gd_score is not None:
            health += gd_score
        if safety_score is not None:
            health += safety_score
        if intent_score is not None:
            health += intent_score
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
        rq_success_count = 0
        total_gd = 0.0
        gd_success_count = 0
        total_safety = 0.0
        safety_success_count = 0
        total_intent = 0.0
        intent_success_count = 0
        total_memory = 0.0
        memory_applicable_success_count = 0
        
        total_normalized_health = 0.0
        health_success_count = 0

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

            # Deterministic numerator & denominator calculation based on successful/applicable evaluators
            raw_app_score = 0.0
            max_health_convo = 0.0
            flagged_for_quality = False
            has_failures = False
            conv_success = False

            # 1. Response Quality
            if rq.status in ("success", "evaluated"):
                raw_app_score += rq.score
                max_health_convo += rq.max_score
                total_rq += rq.score
                rq_success_count += 1
                conv_success = True
                if rq.flagged:
                    flagged_for_quality = True
            else:
                max_health_convo += rq.max_score
                has_failures = True

            # 2. Groundedness
            if gd.status in ("success", "evaluated"):
                raw_app_score += gd.score
                max_health_convo += gd.max_score
                total_gd += gd.score
                gd_success_count += 1
                conv_success = True
                if gd.flagged:
                    flagged_for_quality = True
                
                if inp.conversation_type == ConversationType.CONTEXT_BACKED:
                    gd_cb_scores.append(gd.score)
                else:
                    gd_cf_scores.append(gd.score)
            else:
                max_health_convo += gd.max_score
                has_failures = True

            # 3. Safety
            if safety.status in ("success", "evaluated"):
                raw_app_score += safety.score
                max_health_convo += safety.max_score
                total_safety += safety.score
                safety_success_count += 1
                conv_success = True
                if safety.flagged:
                    flagged_for_quality = True
            else:
                max_health_convo += safety.max_score
                has_failures = True

            # 4. Intent Understanding
            if intent:
                if intent.status in ("success", "evaluated"):
                    raw_app_score += intent.score
                    max_health_convo += intent.max_score
                    total_intent += intent.score
                    intent_success_count += 1
                    conv_success = True
                    if intent.flagged:
                        flagged_for_quality = True
                else:
                    max_health_convo += intent.max_score
                    has_failures = True

            # 5. Memory
            if memory:
                if memory.status in ("success", "evaluated"):
                    if memory.applicable:
                        raw_app_score += memory.score
                        max_health_convo += memory.max_score
                        total_memory += memory.score
                        memory_applicable_success_count += 1
                        conv_success = True
                        if memory.flagged:
                            flagged_for_quality = True
                elif memory.status != "not_applicable":
                    max_health_convo += memory.max_score
                    has_failures = True

            # Calculate health score if any evaluators succeeded
            if max_health_convo > 0.0 and conv_success:
                overall_health_score = round((raw_app_score / max_health_convo) * 100.0, 2)
                total_normalized_health += overall_health_score
                health_success_count += 1
            else:
                overall_health_score = None

            is_flagged = flagged_for_quality or has_failures
            if is_flagged:
                flagged_count += 1

            evals = {
                "response_quality": {
                    "score": rq.score,
                    "max_score": rq.max_score,
                    "status": rq.status,
                    "sub_scores": rq.sub_scores,
                    "feedback": rq.feedback,
                    "flagged": rq.flagged,
                },
                "groundedness": {
                    "score": gd.score,
                    "max_score": gd.max_score,
                    "status": gd.status,
                    "sub_scores": gd.sub_scores,
                    "feedback": gd.feedback,
                    "flagged": gd.flagged,
                },
                "safety": {
                    "score": safety.score,
                    "max_score": safety.max_score,
                    "status": safety.status,
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
                    "status": intent.status,
                    "sub_scores": intent.sub_scores,
                    "detected_intent": getattr(intent, "detected_intent", None),
                    "expected_intent": getattr(intent, "expected_intent", None),
                    "expected_intent_status": getattr(intent, "expected_intent_status", "not_provided"),
                    "misclassified": getattr(intent, "misclassified", False),
                    "feedback": intent.feedback,
                    "flagged": intent.flagged,
                }

            if memory:
                evals["memory_and_continuity"] = {
                    "score": memory.score,
                    "max_score": memory.max_score,
                    "applicable": memory.applicable,
                    "status": memory.status,
                    "percentage": memory.percentage,
                    "sub_scores": memory.sub_scores,
                    "feedback": memory.feedback,
                    "flagged": memory.flagged,
                }

            convo_records.append({
                "conversation_id": conv_id,
                "conversation_type": inp.conversation_type.value,
                "raw_applicable_score": round(raw_app_score, 2),
                "applicable_max_score": max_health_convo,
                "overall_health_score": overall_health_score,
                "flagged": is_flagged,
                "flagged_for_quality": flagged_for_quality,
                "evaluation_failed": has_failures,
                "evaluations": evals
            })

        count = len(convo_records) or 1
        averages = {
            "response_quality": round(total_rq / (rq_success_count or 1), 2) if rq_success_count > 0 else None,
            "groundedness": round(total_gd / (gd_success_count or 1), 2) if gd_success_count > 0 else None,
            "safety": round(total_safety / (safety_success_count or 1), 2) if safety_success_count > 0 else None,
            "overall_health": round(total_normalized_health / (health_success_count or 1), 2) if health_success_count > 0 else None,
        }
        if intent_results:
            averages["intent_understanding"] = round(total_intent / (intent_success_count or 1), 2) if intent_success_count > 0 else None
        if memory_applicable_success_count > 0:
            averages["memory_and_continuity"] = round(total_memory / memory_applicable_success_count, 2)

        summary_stats = {
            "total_conversations": len(convo_records),
            "flagged_conversations": flagged_count,
            "averages": averages,
            "groundedness_breakdown": {
                "context_backed_average": round(sum(gd_cb_scores) / (len(gd_cb_scores) or 1), 2) if gd_cb_scores else None,
                "context_backed_count": len(gd_cb_scores),
                "context_free_average": round(sum(gd_cf_scores) / (len(gd_cf_scores) or 1), 2) if gd_cf_scores else None,
                "context_free_count": len(gd_cf_scores),
            }
        }

        return {
            "pipeline_phase": "phase_1",
            "summary_stats": summary_stats,
            "conversations": convo_records,
        }
