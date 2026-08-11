"""
Response Quality Evaluator — Phase 1

Evaluates Dave's response quality via an LLM judge on four metrics:
  • Correctness  — factual accuracy
  • Helpfulness  — practical value to the user
  • Clarity      — readability and structure
  • Completeness — coverage of the question's scope

Each metric is scored 1–5, then converted:
    Metric Contribution = (score / 5) * 5

Final score = sum of all 4 contributions (max = 20).

The LLM judge is instructed to be strict, return structured JSON with
per-metric reasoning, and avoid generous scoring.
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
from evaluation_pipeline.utils.schemas import ResponseQualitySchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METRICS = ("correctness", "helpfulness", "clarity", "completeness")
_MAX_METRIC_SCORE = 5
_METRIC_WEIGHT = 5  # each contributes up to 5 points
_MAX_TOTAL = len(_METRICS) * _METRIC_WEIGHT  # 20

_SYSTEM_PROMPT = """\
You are a STRICT, expert evaluation judge assessing AI assistant responses.
You must be rigorous and critical — do NOT give generous scores.
Penalise even minor issues. A score of 5 should be reserved for truly
excellent responses with no observable flaws.

You will evaluate responses on exactly four metrics, each scored 1–5:

SCORING RUBRIC (apply consistently):
| Score | Meaning    | Description                                                              |
|-------|------------|--------------------------------------------------------------------------|
| 1     | Very Poor  | Completely incorrect, irrelevant, confusing, or missing important info   |
| 2     | Poor       | Partially correct but contains major mistakes or missing details         |
| 3     | Average    | Mostly correct but has minor issues or missing information               |
| 4     | Good       | Correct, clear, helpful, covers most needs                              |
| 5     | Excellent  | Completely correct, clear, complete, well-structured                    |

METRICS:
1. **Correctness** — Are the facts, claims, and details in the response accurate?
2. **Helpfulness** — Does the response provide practical, actionable value to the user?
3. **Clarity** — Is the response well-organized, easy to follow, and free of ambiguity?
4. **Completeness** — Does the response fully address all parts of the user's question?

IMPORTANT INSTRUCTIONS:
- Each reasoning field MUST cite specific content from the response to justify the score.
- Be specific: reference exact claims, sentences, or structural choices.
- If context is provided, check factual claims against it.
- Do NOT be generous. Average responses get a 3, not a 4.

Return your evaluation as a JSON object with EXACTLY this structure (no extra keys):
{
  "correctness": {"score": <1-5>, "reasoning": "<specific explanation citing response content>"},
  "helpfulness": {"score": <1-5>, "reasoning": "<specific explanation citing response content>"},
  "clarity": {"score": <1-5>, "reasoning": "<specific explanation citing response content>"},
  "completeness": {"score": <1-5>, "reasoning": "<specific explanation citing response content>"}
}
"""


def _build_user_prompt(eval_input: EvaluationInput) -> str:
    """Construct the user-facing evaluation prompt."""
    parts = [
        "Evaluate the following AI assistant response.\n",
        f"## User Query\n{eval_input.user_query}\n",
        f"## AI Assistant Response\n{eval_input.dave_response}\n",
    ]

    if eval_input.retrieved_context:
        parts.append(
            f"## Retrieved Context (source documents)\n{eval_input.retrieved_context}\n"
        )

    if eval_input.chat_history:
        parts.append(
            f"## Conversation History\n{eval_input.chat_history}\n"
        )

    parts.append(
        "\nScore each of the four metrics (Correctness, Helpfulness, Clarity, "
        "Completeness) on a 1–5 scale using the rubric provided. "
        "Return ONLY valid JSON."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ResponseQualityEvaluator(BaseEvaluator):
    """
    LLM-as-judge evaluator for response quality.

    Scores 4 metrics × 5 points each = 20 max.
    """

    name: str = "response_quality"

    def __init__(self) -> None:
        self._judge = LLMJudge()
        logger.info("ResponseQualityEvaluator initialized.")

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult:
        """Run the LLM judge and compute the response quality score."""
        try:
            logger.debug(
                "Evaluating response quality for '%s'", eval_input.conversation_id
            )

            # 1. Call the LLM judge
            user_prompt = _build_user_prompt(eval_input)
            parsed_json, raw_text = self._judge.call_with_json(
                _SYSTEM_PROMPT,
                user_prompt,
                evaluator=self.name,
                conversation_id=eval_input.conversation_id,
                response_schema=ResponseQualitySchema,
            )

            if not parsed_json:
                raise ValueError("Empty or invalid JSON response from response quality judge.")

            # 2. Validate and extract metric scores
            metric_scores = self._extract_metric_scores(parsed_json)

            # 3. Compute sub-scores: (raw_score / 5) * 5
            sub_scores: dict[str, float] = {}
            for metric in _METRICS:
                raw = metric_scores[metric]["score"]
                contribution = (raw / _MAX_METRIC_SCORE) * _METRIC_WEIGHT
                sub_scores[metric] = round(contribution, 2)

            # 4. Compute total score
            total_score = round(sum(sub_scores.values()), 2)

            # 5. Build feedback from the LLM's actual per-metric reasoning
            feedback_parts: list[str] = []
            for metric in _METRICS:
                raw = metric_scores[metric]["score"]
                reasoning = metric_scores[metric]["reasoning"]
                feedback_parts.append(
                    f"{metric.capitalize()}: {raw}/5 — {reasoning}"
                )
            feedback = "\n\n".join(feedback_parts)

            # 6. Flag low-quality responses (below 50% of max)
            flagged = total_score < (_MAX_TOTAL * 0.5)

            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=total_score,
                max_score=float(_MAX_TOTAL),
                percentage=round((total_score / _MAX_TOTAL) * 100.0, 2),
                sub_scores=sub_scores,
                feedback=feedback,
                flagged=flagged,
            )
        except Exception as exc:
            logger.error("ResponseQualityEvaluator failed for %s: %s", eval_input.conversation_id, exc)
            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=float(_MAX_TOTAL),
                status="failed",
                sub_scores={},
                feedback=f"Response quality evaluation failed with error: {exc}",
                flagged=True,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metric_scores(
        parsed: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """
        Validate the LLM's JSON output and extract per-metric data.

        Expected structure per metric:
            {"score": int(1-5), "reasoning": str}

        Returns a dict keyed by metric name.
        """
        result: dict[str, dict[str, Any]] = {}

        for metric in _METRICS:
            if metric not in parsed:
                raise ValueError(
                    f"Missing metric '{metric}' in LLM response. "
                    f"Got keys: {list(parsed.keys())}"
                )

            entry = parsed[metric]

            if not isinstance(entry, dict):
                raise ValueError(
                    f"Metric '{metric}' should be a dict, got {type(entry).__name__}"
                )

            # Extract and validate score
            raw_score = entry.get("score")
            if raw_score is None:
                raise ValueError(f"Metric '{metric}' is missing 'score' field")

            try:
                score = int(raw_score)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Metric '{metric}' score must be an integer, got: {raw_score}"
                )

            if not 1 <= score <= _MAX_METRIC_SCORE:
                logger.warning(
                    "Clamping out-of-range score for '%s': %d → [1, %d]",
                    metric,
                    score,
                    _MAX_METRIC_SCORE,
                )
                score = max(1, min(score, _MAX_METRIC_SCORE))

            # Extract reasoning
            reasoning = entry.get("reasoning", "")
            if not reasoning or not isinstance(reasoning, str):
                reasoning = f"(No reasoning provided by judge for {metric})"
                logger.warning(
                    "Missing or empty reasoning for metric '%s'", metric
                )

            result[metric] = {"score": score, "reasoning": reasoning.strip()}

        return result
