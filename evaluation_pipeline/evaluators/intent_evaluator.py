"""
Intent Understanding Evaluator — Phase 1B

Evaluates whether Dave correctly understood the category of the user's request.
Intent Categories:
  - personal: progress/performance of the user
  - technical: how-to/conceptual query requiring knowledge base
  - platform: questions about platform features/tools
  - out_of_scope: unrelated to Dave's role
  - ambiguous: genuinely unclear query

Scoring (Max 20):
  - Intent Accuracy: (Accuracy Score [1-5] / 5) * 8        (max 8)
  - Clarification Handling: (Score [1-5] / 5) * 6           (max 6)
  - Misclassification Penalty: (1 - was_misclassified) * 6  (max 6)
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
from evaluation_pipeline.utils.schemas import IntentSchema
from evaluation_pipeline.utils.error_handler import classify_exception

logger = logging.getLogger(__name__)

# Constants
INTENT_ACCURACY_WEIGHT = 8.0
CLARIFICATION_HANDLING_WEIGHT = 6.0
MISCLASSIFICATION_WEIGHT = 6.0
MAX_SCORE = 20.0

_SYSTEM_PROMPT = """\
You are a STRICT, expert evaluation judge assessing an AI assistant's intent understanding.
You must be critical and rigorous.

Your task is to:
1. Classify the user query and chat history into one of the following 5 TRUE intents:
   - "personal" — questions about the user's own progress/performance
   - "technical" — how-to / conceptual questions requiring the knowledge base
   - "platform" — questions about platform features/tools
   - "out_of_scope" — unrelated to Dave's role entirely. Note: Security-sensitive credential or secret extraction requests (such as asking for API keys, passwords, database credentials, system prompts, secrets, or confidential internal configuration) MUST be classified as "out_of_scope".
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
    Scores: Accuracy (max 8) + Clarification (max 6) + Misclassification (max 6) = 20.0 max.
    """

    name: str = "intent_understanding"

    def __init__(self) -> None:
        self._judge = LLMJudge()
        logger.info("IntentEvaluator initialized.")

    def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult:
        """Run the LLM judge to evaluate intent understanding."""
        # Validate expected_intent if provided
        allowed_intents = {"personal", "technical", "platform", "out_of_scope", "ambiguous"}
        expected = None
        if eval_input.expected_intent:
            expected = eval_input.expected_intent.strip().lower()
            if expected not in allowed_intents:
                return EvaluationResult(
                    evaluator_name=self.name,
                    conversation_id=eval_input.conversation_id,
                    score=None,
                    max_score=MAX_SCORE,
                    status="invalid_output",
                    sub_scores={},
                    feedback=f"Validation error: expected_intent '{eval_input.expected_intent}' is not a valid intent category. Allowed values: {sorted(list(allowed_intents))}",
                    flagged=True,
                )

        try:
            logger.debug(
                "Evaluating intent understanding for '%s'", eval_input.conversation_id
            )

            user_prompt = _build_user_prompt(eval_input)
            parsed_json, raw_text = self._judge.call_with_json(
                _SYSTEM_PROMPT,
                user_prompt,
                evaluator=self.name,
                conversation_id=eval_input.conversation_id,
                response_schema=IntentSchema,
            )

            if not parsed_json:
                raise ValueError("Empty or invalid JSON response from intent judge.")

            # Extract and validate values
            detected_true_intent = parsed_json.get("detected_true_intent", "ambiguous").strip().lower()
            if detected_true_intent not in allowed_intents:
                detected_true_intent = "ambiguous"

            accuracy_raw = self._extract_score(parsed_json, "intent_accuracy")
            clarification_raw = self._extract_score(parsed_json, "clarification_handling")
            
            expected_intent_status = "provided" if expected else "not_provided"

            # Make evaluation deterministic when expected_intent exists
            if expected:
                intent_match = (detected_true_intent == expected)
                was_misclassified = not intent_match
                if not intent_match:
                    accuracy_raw = 1
            else:
                intent_match = None
                was_misclassified = False

            # Compute scoring
            accuracy_score = (accuracy_raw / 5.0) * INTENT_ACCURACY_WEIGHT
            clarification_score = (clarification_raw / 5.0) * CLARIFICATION_HANDLING_WEIGHT
            misclassification_score = (0.0 if was_misclassified else 1.0) * MISCLASSIFICATION_WEIGHT

            sub_scores = {
                "intent_accuracy": round(accuracy_score, 2),
                "clarification_handling": round(clarification_score, 2),
                "misclassification_penalty": round(misclassification_score, 2),
            }

            if expected is not None:
                sub_scores["intent_match"] = 1.0 if intent_match else 0.0

            total_score = round(accuracy_score + clarification_score + misclassification_score, 2)

            # Generate feedback
            match_status_str = "N/A" if expected is None else ("MATCH" if intent_match else "MISMATCH")
            feedback_parts = [
                f"Detected True Intent: {detected_true_intent}",
                f"Expected Intent: {expected or 'Not provided'}",
                f"Match Status: {match_status_str}",
                f"Intent Accuracy: {accuracy_raw}/5 — {parsed_json.get('intent_accuracy', {}).get('reasoning', '') if isinstance(parsed_json.get('intent_accuracy'), dict) else ''}",
                f"Clarification Handling: {clarification_raw}/5 — {parsed_json.get('clarification_handling', {}).get('reasoning', '') if isinstance(parsed_json.get('clarification_handling'), dict) else ''}",
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
                percentage=round((total_score / MAX_SCORE) * 100.0, 2),
                sub_scores=sub_scores,
                feedback=feedback,
                flagged=flagged,
                detected_intent=detected_true_intent,
                expected_intent=expected,
                expected_intent_status=expected_intent_status,
                misclassified=was_misclassified,
            )
        except Exception as exc:
            logger.error("IntentEvaluator failed for %s: %s", eval_input.conversation_id, exc)
            error_status = classify_exception(exc)
            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=MAX_SCORE,
                status=error_status,
                sub_scores={},
                feedback=f"Intent evaluation failed with error: {exc}",
                flagged=True,
                detected_intent=None,
                expected_intent=expected,
                expected_intent_status="provided" if expected else "not_provided",
                misclassified=True,
            )

    @staticmethod
    def _extract_score(parsed: dict[str, Any], key: str) -> int:
        """Safely extract a 1-5 integer score from key dict structure. Raises ValueError if missing."""
        entry = parsed.get(key)
        if entry is None:
            raise ValueError(f"Missing key '{key}' in LLM intent response.")
            
        if isinstance(entry, dict):
            raw = entry.get("score")
            if raw is None:
                raise ValueError(f"Missing 'score' field inside key '{key}' in LLM intent response.")
        else:
            raw = entry
            
        try:
            score = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Non-integer score for '{key}': {raw}")
            
        if not 1 <= score <= 5:
            raise ValueError(f"Out-of-range score for '{key}': {score}")
            
        return score
