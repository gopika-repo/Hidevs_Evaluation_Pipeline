"""
Intent Understanding Evaluator — Phase 1B

Evaluates whether Dave correctly understood the category of the user's request.
Intent Categories:
  - personal: progress/performance of the user
  - technical: how-to/conceptual query requiring knowledge base
  - platform: questions about platform features/tools
  - out_of_scope: unrelated to Dave's role
  - ambiguous: genuinely unclear query

Scoring (Max 15):
  - Intent Accuracy: (Accuracy Score [1-5] / 5) * 6
  - Clarification Handling: (Score [1-5] / 5) * 5
  - Misclassification Penalty: (1 - was_misclassified) * 4
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
INTENT_ACCURACY_WEIGHT = 8.0
CLARIFICATION_HANDLING_WEIGHT = 20.0 / 3.0
MISCLASSIFICATION_WEIGHT = 16.0 / 3.0
MAX_SCORE = 20.0

_SYSTEM_PROMPT = """\
You are a STRICT, expert evaluation judge assessing an AI assistant's intent understanding.
You must be critical and rigorous.

Your task is to:
1. Classify the user query and chat history into one of the following 5 TRUE intents:
   - "personal" — questions about the user's own progress/performance
   - "technical" — how-to / conceptual questions requiring the knowledge base
   - "platform" — questions about platform features/tools
   - "out_of_scope" — unrelated to Dave's role entirely
   - "ambiguous" — genuinely unclear what's being asked

2. Evaluate the assistant's response on:
   - **intent_accuracy** (1-5): Does the assistant's response reflect a correct understanding of the true intent?
     - 5 = Excellent, completely understood and directly addressed the correct intent category.
     - 3 = Average, understood but addressed it with minor category confusion or unnecessary details.
     - 1 = Poor, did not address the true intent at all.
   - **clarification_handling** (1-5):
     - If the intent was "ambiguous": Did the assistant ask for clarification? (Yes = 5, No/guessed = 1)
     - If the intent was clear (not ambiguous): Did the assistant avoid unnecessary clarification questions? (Yes/directly answered = 5, No/unnecessarily asked questions = 1)
   - **was_misclassified** (true/false): Did the assistant respond as if this were a completely different category of request? (e.g. treated a "personal" query as a "technical" query).

Return your evaluation as a JSON object with EXACTLY this structure (no extra keys):
{
  "detected_true_intent": "personal|technical|platform|out_of_scope|ambiguous",
  "intent_accuracy": {"score": <1-5>, "reasoning": "<specific explanation citing response content>"},
  "clarification_handling": {"score": <1-5>, "reasoning": "<specific explanation citing response content>"},
  "was_misclassified": <true/false>,
  "explanation": "<overall explanation of intent classification and match>"
}
"""

def _build_user_prompt(eval_input: EvaluationInput) -> str:
    """Construct the user evaluation prompt."""
    parts = [
        "Evaluate the assistant's intent understanding.\n",
        f"## User Query\n{eval_input.user_query}\n",
        f"## AI Assistant Response\n{eval_input.dave_response}\n",
    ]

    if eval_input.chat_history:
        parts.append(
            f"## Conversation History\n{eval_input.chat_history}\n"
        )

    parts.append(
        "\nDetermine the true intent and score intent accuracy, clarification handling, and misclassification. "
        "Return ONLY valid JSON."
    )
    return "\n".join(parts)


class IntentEvaluator(BaseEvaluator):
    """
    LLM-as-judge evaluator for user intent understanding.
    Scores: Accuracy (max 6) + Clarification (max 5) + Misclassification (max 4) = 15.0 max.
    """

    name: str = "intent_understanding"

    def __init__(self) -> None:
        self._judge = LLMJudge()
        logger.info("IntentEvaluator initialized.")

    def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult:
        """Run the LLM judge to evaluate intent understanding."""
        logger.debug(
            "Evaluating intent understanding for '%s'", eval_input.conversation_id
        )

        user_prompt = _build_user_prompt(eval_input)
        parsed_json, raw_text = self._judge.call_with_json(
            _SYSTEM_PROMPT, user_prompt
        )

        # Extract and validate values
        detected_true_intent = parsed_json.get("detected_true_intent", "ambiguous").strip().lower()
        allowed_intents = {"personal", "technical", "platform", "out_of_scope", "ambiguous"}
        if detected_true_intent not in allowed_intents:
            detected_true_intent = "ambiguous"

        accuracy_raw = self._extract_score(parsed_json, "intent_accuracy")
        clarification_raw = self._extract_score(parsed_json, "clarification_handling")
        was_misclassified = bool(parsed_json.get("was_misclassified", False))

        # Compute scoring
        accuracy_score = (accuracy_raw / 5.0) * INTENT_ACCURACY_WEIGHT
        clarification_score = (clarification_raw / 5.0) * CLARIFICATION_HANDLING_WEIGHT
        misclassification_score = (0.0 if was_misclassified else 1.0) * MISCLASSIFICATION_WEIGHT

        sub_scores = {
            "intent_accuracy": round(accuracy_score, 2),
            "clarification_handling": round(clarification_score, 2),
            "misclassification_penalty": round(misclassification_score, 2),
        }

        # Calculate ground truth validation check if expected_intent is present
        if eval_input.expected_intent:
            expected = eval_input.expected_intent.strip().lower()
            intent_match = (detected_true_intent == expected)
            sub_scores["intent_match"] = 1.0 if intent_match else 0.0

        total_score = round(accuracy_score + clarification_score + misclassification_score, 2)

        # Generate feedback
        feedback_parts = [
            f"Detected True Intent: {detected_true_intent}",
            f"Expected Intent: {eval_input.expected_intent or 'Not provided'}",
            f"Intent Accuracy: {accuracy_raw}/5 — {parsed_json.get('intent_accuracy', {}).get('reasoning', '')}",
            f"Clarification Handling: {clarification_raw}/5 — {parsed_json.get('clarification_handling', {}).get('reasoning', '')}",
            f"Was Misclassified: {'Yes' if was_misclassified else 'No'}",
            f"Explanation: {parsed_json.get('explanation', '')}"
        ]
        feedback = "\n\n".join(feedback_parts)

        # Flag if score is below 50% or if it was misclassified
        flagged = total_score < (MAX_SCORE * 0.5) or was_misclassified

        return EvaluationResult(
            evaluator_name=self.name,
            conversation_id=eval_input.conversation_id,
            score=total_score,
            max_score=MAX_SCORE,
            sub_scores=sub_scores,
            feedback=feedback,
            flagged=flagged,
        )

    @staticmethod
    def _extract_score(parsed: dict[str, Any], key: str, default: int = 3) -> int:
        """Safely extract a 1-5 integer score from key dict structure."""
        entry = parsed.get(key)
        if isinstance(entry, dict):
            raw = entry.get("score", default)
        else:
            raw = entry if entry is not None else default

        try:
            score = int(raw)
        except (TypeError, ValueError):
            return default

        # Clamp to [1, 5]
        return max(1, min(score, 5))
