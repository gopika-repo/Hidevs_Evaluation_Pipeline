"""
Memory & Context Continuity Evaluator — Phase 1

Evaluates Dave's ability to maintain context, reference prior turns, and keep consistency
across a multi-turn conversation.

Scoring (Max 20.0):
  - Context Carryover (10.0): (Score [1-5] / 5.0) * 10.0
  - Context Consistency (10.0): (Score [1-5] / 5.0) * 10.0
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

# Constants
MAX_SCORE = 20.0

_SYSTEM_PROMPT = """\
You are a STRICT, expert evaluation judge assessing an AI assistant's memory and context continuity.
You must be critical and rigorous.

Your task is to:
1. Analyze the conversation history, the new user query, and the assistant's response.
2. Evaluate the assistant's response on:
   - **context_carryover** (1-5): Does the assistant remember and properly use relevant context, facts, preferences, or details established in the prior conversation turns?
     - 5 = Excellent, completely carries over and references prior context seamlessly.
     - 3 = Average, partially tracks details but has minor gaps or asks redundant questions.
     - 1 = Poor, ignores prior conversation context entirely.
   - **context_consistency** (1-5): Is the response consistent with facts or statements made in prior turns?
     - 5 = Fully consistent; no contradictions.
     - 3 = Minor contradiction or confusing change in details.
     - 1 = Major contradiction; directly conflicts with facts stated in prior turns.

Return your evaluation as a JSON object with EXACTLY this structure (no extra keys):
{
  "context_carryover": {"score": <1-5>, "reasoning": "<specific explanation citing prior turns and response>"},
  "context_consistency": {"score": <1-5>, "reasoning": "<specific explanation citing prior turns and response>"},
  "explanation": "<overall explanation of memory and continuity quality>"
}
"""

def _build_user_prompt(eval_input: EvaluationInput) -> str:
    parts = [
        "Evaluate the memory and context continuity of the assistant's response below.\n",
        f"## Conversation History\n{eval_input.chat_history or 'No prior turns.'}\n",
        f"## Current User Query\n{eval_input.user_query}\n",
        f"## Assistant Response\n{eval_input.dave_response}\n"
    ]
    return "".join(parts)


class MemoryEvaluator(BaseEvaluator):
    """
    Evaluates an assistant's response for memory and context continuity.
    """

    name = "memory_and_continuity"

    def __init__(self) -> None:
        self._judge = LLMJudge()

    def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult:
        logger.debug("Evaluating memory/continuity for '%s'", eval_input.conversation_id)

        # If no chat history is present, memory is not applicable (default to full score)
        if not eval_input.chat_history or not eval_input.chat_history.strip():
            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=MAX_SCORE,
                max_score=MAX_SCORE,
                applicable=False,
                sub_scores={
                    "context_carryover": 10.0,
                    "context_consistency": 10.0
                },
                feedback="Not applicable — single-turn conversation without chat history.",
                flagged=False
            )

        user_prompt = _build_user_prompt(eval_input)
        parsed_json, raw_text = self._judge.call_with_json(
            _SYSTEM_PROMPT, user_prompt
        )

        carryover_raw = self._extract_score(parsed_json, "context_carryover")
        consistency_raw = self._extract_score(parsed_json, "context_consistency")

        carryover_score = (carryover_raw / 5.0) * 10.0
        consistency_score = (consistency_raw / 5.0) * 10.0

        sub_scores = {
            "context_carryover": round(carryover_score, 2),
            "context_consistency": round(consistency_score, 2)
        }

        total_score = round(carryover_score + consistency_score, 2)

        feedback_parts = [
            f"Context Carryover: {carryover_raw}/5 — {parsed_json.get('context_carryover', {}).get('reasoning', '')}",
            f"Context Consistency: {consistency_raw}/5 — {parsed_json.get('context_consistency', {}).get('reasoning', '')}",
            f"Explanation: {parsed_json.get('explanation', '')}"
        ]
        feedback = "\n\n".join(feedback_parts)

        # Flag if score is below 60% or if consistency is very poor (score <= 2)
        flagged = total_score < (MAX_SCORE * 0.6) or consistency_raw <= 2

        return EvaluationResult(
            evaluator_name=self.name,
            conversation_id=eval_input.conversation_id,
            score=total_score,
            max_score=MAX_SCORE,
            sub_scores=sub_scores,
            feedback=feedback,
            flagged=flagged
        )

    @staticmethod
    def _extract_score(parsed: dict[str, Any], key: str, default: int = 3) -> int:
        entry = parsed.get(key)
        if isinstance(entry, dict):
            raw = entry.get("score", default)
        else:
            raw = entry if entry is not None else default

        try:
            score = int(raw)
        except (TypeError, ValueError):
            return default

        return max(1, min(score, 5))
