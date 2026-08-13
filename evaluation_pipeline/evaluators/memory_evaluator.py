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
from evaluation_pipeline.utils.schemas import MemorySchema
from evaluation_pipeline.utils.error_handler import classify_exception

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a STRICT, expert evaluation judge assessing an AI assistant's memory and context continuity.
You must be critical and rigorous.

Your task is to analyze the user query, Dave's response, and the conversation history (chat history) to evaluate three memory metrics:

1. **Context Continuity** (1-5):
   Evaluate whether Dave correctly uses relevant information from previous conversation turns when it is necessary to answer the current query.
   - 5 = Correctly uses established preferences/context.
   - 1 = Ignores required context.
   
2. **Information Retention** (1-5):
   Evaluate whether Dave correctly remembers relevant facts/preferences/information explicitly provided earlier in the conversation.
   Does he invent memories, remember things incorrectly, or unnecessarily ignore established information?
   - 5 = Remembers all facts correctly.
   - 1 = Invents or misremembers critical facts.

3. **Consistency Across Turns** (1-5):
   Evaluate whether Dave remains consistent with information and commitments established earlier in the conversation.
   Does Dave contradict previous answers, change established facts without explanation, or act inconsistently with commitments?
   - 5 = Completely consistent across all turns.
   - 1 = Severe contradictions.

CRITICAL INSTRUCTIONS:
- First, determine if there is previous conversation history (chat history) containing turns relevant to memory.
- If there is NO chat history, or the chat history does not establish any facts/context relevant to the current turn, you MUST set "is_applicable": false.
- Otherwise, set "is_applicable": true.

Return ONLY valid JSON matching this schema:
{
  "is_applicable": bool,
  "reasoning_applicability": "explanation of why it is or is not applicable",
  "context_continuity": {
    "score": <1-5>,
    "reasoning": "detailed reasoning for context continuity score"
  },
  "information_retention": {
    "score": <1-5>,
    "reasoning": "detailed reasoning for information retention score"
  },
  "consistency_across_turns": {
    "score": <1-5>,
    "reasoning": "detailed reasoning for consistency across turns score"
  },
  "overall_reasoning": "summary reasoning"
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
                status="not_applicable",
                percentage=None,
                sub_scores={
                    "context_continuity": None,
                    "information_retention": None,
                    "consistency_across_turns": None
                },
                feedback="No prior conversation history available for memory evaluation",
                flagged=False
            )

        # 2. Call LLM Judge
        try:
            user_prompt = (
                f"## Conversation History (Prior Turns)\n{eval_input.chat_history}\n\n"
                f"## Current User Query\n{eval_input.user_query}\n\n"
                f"## Current Assistant Response (Dave)\n{eval_input.dave_response}\n"
            )

            parsed_json, raw_text = self._judge.call_with_json(
                _SYSTEM_PROMPT,
                user_prompt,
                evaluator=self.name,
                conversation_id=eval_input.conversation_id,
                response_schema=MemorySchema,
            )

            if not parsed_json:
                raise ValueError("Empty or invalid JSON response from memory judge.")

            is_applicable = parsed_json.get("is_applicable", True)
            if not is_applicable:
                return EvaluationResult(
                    evaluator_name=self.name,
                    conversation_id=eval_input.conversation_id,
                    score=None,
                    max_score=20.0,
                    applicable=False,
                    status="not_applicable",
                    percentage=None,
                    sub_scores={
                        "context_continuity": None,
                        "information_retention": None,
                        "consistency_across_turns": None
                    },
                    feedback=parsed_json.get("reasoning_applicability", "No prior conversation history available for memory evaluation"),
                    flagged=False
                )

            # 3. Extract scores
            context_raw = self._extract_score(parsed_json, "context_continuity")
            retention_raw = self._extract_score(parsed_json, "information_retention")
            consistency_raw = self._extract_score(parsed_json, "consistency_across_turns")

            # 4. Apply Formulas
            # Metric 1: Context Continuity (max 8)
            recall_score = (context_raw / 5.0) * 8.0
            # Metric 2: Information Retention (max 6)
            relevance_score = (retention_raw / 5.0) * 6.0
            # Metric 3: Consistency Across Turns (max 6)
            consistency_score = (consistency_raw / 5.0) * 6.0

            total_score = round(recall_score + relevance_score + consistency_score, 2)
            percentage = round((total_score / 20.0) * 100.0, 2)

            sub_scores = {
                "context_continuity": round(recall_score, 2),
                "information_retention": round(relevance_score, 2),
                "consistency_across_turns": round(consistency_score, 2),
            }

            # Build feedback lines from reasoning fields
            feedback_lines = [
                f"Applicability check: {parsed_json.get('reasoning_applicability', '')}",
                f"Context Continuity: {sub_scores['context_continuity']}/8.0 ({context_raw}/5). Reasoning: {parsed_json.get('context_continuity', {}).get('reasoning', '') if isinstance(parsed_json.get('context_continuity'), dict) else ''}",
                f"Information Retention: {sub_scores['information_retention']}/6.0 ({retention_raw}/5). Reasoning: {parsed_json.get('information_retention', {}).get('reasoning', '') if isinstance(parsed_json.get('information_retention'), dict) else ''}",
                f"Consistency Across Turns: {sub_scores['consistency_across_turns']}/6.0 ({consistency_raw}/5). Reasoning: {parsed_json.get('consistency_across_turns', {}).get('reasoning', '') if isinstance(parsed_json.get('consistency_across_turns'), dict) else ''}",
                f"Total Memory & Continuity Score: {total_score}/20.0"
            ]
            feedback = "\n\n".join(feedback_lines)

            # Flag if overall score is less than 50%
            flagged = total_score < 10.0

            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=total_score,
                max_score=20.0,
                applicable=True,
                status="evaluated",
                percentage=percentage,
                sub_scores=sub_scores,
                feedback=feedback,
                flagged=flagged
            )
        except Exception as exc:
            logger.error("MemoryEvaluator failed for %s: %s", eval_input.conversation_id, exc)
            error_status = classify_exception(exc)
            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=20.0,
                applicable=False,
                status=error_status,
                percentage=None,
                sub_scores={
                    "context_continuity": None,
                    "information_retention": None,
                    "consistency_across_turns": None
                },
                feedback=f"Memory evaluation failed with error: {exc}",
                flagged=True,
            )

    @staticmethod
    def _extract_score(parsed: dict[str, Any], key: str) -> int:
        """Safely extract a 1–5 integer score from a parsed JSON entry. Raises ValueError if missing."""
        entry = parsed.get(key)
        if entry is None:
            raise ValueError(f"Missing key '{key}' in LLM memory response.")
            
        if isinstance(entry, dict):
            raw = entry.get("score")
            if raw is None:
                raise ValueError(f"Missing 'score' field inside key '{key}' in LLM memory response.")
        else:
            raw = entry
            
        try:
            score = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Non-integer score for '{key}': {raw}")
            
        if not 1 <= score <= 5:
            raise ValueError(f"Out-of-range score for '{key}': {score}")
            
        return score
