"""
Memory & Context Continuity Evaluator — Phase 1

Evaluates the assistant's ability to recall and maintain consistency with
information explicitly established in previous conversation turns.
"""

from __future__ import annotations

import logging
from typing import Any

from evaluation_pipeline.data.models import (
    EvaluationInput,
    EvaluationResult,
)
from evaluation_pipeline.evaluators.base_evaluator import BaseEvaluator
from evaluation_pipeline.utils.llm_client import LLMJudge

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a STRICT, expert evaluation judge assessing an AI assistant's memory and context continuity.
You must be critical and rigorous.

Your task is to analyze the user query, Dave's response, and the conversation history (chat history) to evaluate three memory metrics:
1. Memory Recall Accuracy: Whether Dave correctly recalls facts explicitly established in previous turns.
2. Cross-Turn Consistency: Whether Dave's current response is consistent with previous turns (no contradictions).
3. Memory Relevance & Non-Invention: Whether Dave uses only relevant facts, does not invent memories, does not attribute fake facts to the user, and does not use unrelated historical info.

CRITICAL INSTRUCTIONS:
- First, determine if there is previous conversation history (chat history) containing turns relevant to memory.
- If there is NO chat history, or the chat history does not establish any facts/context relevant to the current turn (e.g. it is just greetings, or completely unrelated chit-chat requiring no memory), you MUST set "is_applicable": false.
- If "is_applicable" is true, you must analyze and provide the counts for the following metrics:
  - `total_relevant_memory_facts`: Total number of facts established in previous turns that are relevant/needed for the current response.
  - `correctly_recalled_relevant_facts`: Of those relevant facts, how many did Dave correctly recall and use?
  - `total_memory_dependent_responses`: Total number of response statements/claims that depend on previous turns/history.
  - `consistent_memory_dependent_responses`: Of those, how many are fully consistent with previous turns (not contradicting)?
  - `total_memory_dependent_claims`: Total number of historical claims/references Dave made in the current response.
  - `verified_relevant_memory_usage`: Of those, how many are verified, relevant, and not invented/misattributed?

You must output a JSON object matching this schema:
{
  "is_applicable": bool,
  "reasoning_applicability": "explanation of why it is or is not applicable",
  "total_relevant_memory_facts": int,
  "correctly_recalled_relevant_facts": int,
  "reasoning_recall": "detailed reasoning for recall accuracy",
  "total_memory_dependent_responses": int,
  "consistent_memory_dependent_responses": int,
  "reasoning_consistency": "detailed reasoning for consistency",
  "total_memory_dependent_claims": int,
  "verified_relevant_memory_usage": int,
  "reasoning_relevance": "detailed reasoning for relevance & non-invention"
}
"""

class MemoryEvaluator(BaseEvaluator):
    """
    Evaluator for Memory & Context Continuity.
    """

    def __init__(self) -> None:
        self._judge = LLMJudge()
        logger.info("MemoryEvaluator initialized.")

    @property
    def name(self) -> str:
        return "memory_and_continuity"

    def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult:
        logger.debug("Evaluating memory & context continuity for '%s'", eval_input.conversation_id)

        # 1. No chat history pre-check
        if not eval_input.chat_history or not eval_input.chat_history.strip():
            logger.info("No chat history for '%s'. Skipping memory evaluation.", eval_input.conversation_id)
            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=20.0,
                applicable=False,
                percentage=None,
                sub_scores={},
                feedback="No prior conversation history available for memory evaluation",
                flagged=False
            )

        # 2. Call LLM Judge
        user_prompt = (
            f"## Conversation History (Prior Turns)\n{eval_input.chat_history}\n\n"
            f"## Current User Query\n{eval_input.user_query}\n\n"
            f"## Current Assistant Response (Dave)\n{eval_input.dave_response}\n"
        )

        parsed_json, raw_text = self._judge.call_with_json(_SYSTEM_PROMPT, user_prompt)

        is_applicable = parsed_json.get("is_applicable", True)
        if not is_applicable:
            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=20.0,
                applicable=False,
                percentage=None,
                sub_scores={},
                feedback=parsed_json.get("reasoning_applicability", "No prior conversation history available for memory evaluation"),
                flagged=False
            )

        # 3. Extract counts
        total_facts = max(int(parsed_json.get("total_relevant_memory_facts", 0)), 0)
        recalled_facts = max(int(parsed_json.get("correctly_recalled_relevant_facts", 0)), 0)
        
        total_responses = max(int(parsed_json.get("total_memory_dependent_responses", 0)), 0)
        consistent_responses = max(int(parsed_json.get("consistent_memory_dependent_responses", 0)), 0)
        
        total_claims = max(int(parsed_json.get("total_memory_dependent_claims", 0)), 0)
        verified_usage = max(int(parsed_json.get("verified_relevant_memory_usage", 0)), 0)

        # Clamp recalled facts to total facts
        if recalled_facts > total_facts:
            recalled_facts = total_facts
        if consistent_responses > total_responses:
            consistent_responses = total_responses
        if verified_usage > total_claims:
            verified_usage = total_claims

        # 4. Apply Formulas
        # Metric 1: Recall Accuracy (max 8)
        if total_facts > 0:
            recall_score = (recalled_facts / total_facts) * 8.0
        else:
            # If no memory facts exist, but judge marked applicable, default to perfect
            recall_score = 8.0

        # Metric 2: Consistency (max 6)
        if total_responses > 0:
            consistency_score = (consistent_responses / total_responses) * 6.0
        else:
            consistency_score = 6.0

        # Metric 3: Relevance & Non-Invention (max 6)
        if total_claims > 0:
            relevance_score = (verified_usage / total_claims) * 6.0
        else:
            relevance_score = 6.0

        total_score = round(recall_score + consistency_score + relevance_score, 2)
        percentage = round((total_score / 20.0) * 100.0, 2)

        sub_scores = {
            "memory_recall_accuracy": round(recall_score, 2),
            "cross_turn_consistency": round(consistency_score, 2),
            "memory_relevance_non_invention": round(relevance_score, 2),
        }

        # Build feedback lines from reasoning fields
        feedback_lines = [
            f"Applicability check: {parsed_json.get('reasoning_applicability', '')}",
            f"Memory Recall Accuracy: {sub_scores['memory_recall_accuracy']}/8.0 ({recalled_facts}/{total_facts} facts correctly recalled). Reasoning: {parsed_json.get('reasoning_recall', '')}",
            f"Cross-Turn Consistency: {sub_scores['cross_turn_consistency']}/6.0 ({consistent_responses}/{total_responses} consistent responses). Reasoning: {parsed_json.get('reasoning_consistency', '')}",
            f"Memory Relevance & Non-Invention: {sub_scores['memory_relevance_non_invention']}/6.0 ({verified_usage}/{total_claims} verified claims). Reasoning: {parsed_json.get('reasoning_relevance', '')}",
            f"Total Memory & Continuity Score: {total_score}/20.0"
        ]
        feedback = "\n\n".join(feedback_lines)

        # Flag if overall score is less than 50% or if there are contradictions
        flagged = total_score < 10.0 or consistent_responses < total_responses

        return EvaluationResult(
            evaluator_name=self.name,
            conversation_id=eval_input.conversation_id,
            score=total_score,
            max_score=20.0,
            applicable=True,
            percentage=percentage,
            sub_scores=sub_scores,
            feedback=feedback,
            flagged=flagged
        )
