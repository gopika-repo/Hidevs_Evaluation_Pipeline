"""
Groundedness / Hallucination Evaluator — Phase 1

Evaluates whether Dave's response is grounded in evidence and free of
fabricated claims. Branches logic based on conversation type:

**Context-Backed** (max = 15):
  • Evidence Coverage       — (supported / total claims) × 5
  • Faithfulness to Context — (faithfulness_score / 5) × 5
  • Unsupported Claims      — (1 − unsupported / total) × 2.5
  • Contradiction Detection — (1 − contradictions / total) × 2.5
  + TruLens groundedness score (stored for comparison, not averaged in)
  + DeepEval faithfulness score (stored for comparison, not averaged in)

**Context-Free** (max = 15):
  • Internal Consistency  — (consistency_score / 5) × 5
  • Overconfidence        — (overconfidence_score / 5) × 5
  • Hallucination Risk    — (hallucination_score / 5) × 5
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from evaluation_pipeline.data.models import (
    ConversationType,
    EvaluationInput,
    EvaluationResult,
)
from evaluation_pipeline.evaluators.base_evaluator import BaseEvaluator
from evaluation_pipeline.utils.llm_client import LLMJudge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SCORE_CONTEXT_BACKED = 15.0  # Phase 1: halved from 30
_MAX_SCORE_CONTEXT_FREE = 15.0    # Phase 1: halved from 30

# ---------------------------------------------------------------------------
# Prompts — Context-Backed
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_CONTEXT_BACKED = """\
You are a STRICT, expert evaluation judge specialising in groundedness
and hallucination detection. You must be rigorous — penalise any
unsupported or fabricated claim, no matter how minor.

TASK:
Given an AI assistant's response and the retrieved source context,
you must:

1. **Extract every distinct factual claim** made in the response.
   A "claim" is any statement asserting a fact, number, process, URL,
   name, or specific detail. List each claim separately.

2. For each claim, classify it as:
   - "supported" — clearly backed by the source context
   - "unsupported" — not present in the context (neither confirmed nor denied)
   - "contradicted" — directly contradicts something in the context

3. **Score Faithfulness** (1–5): Overall, how faithful is the response
   to the source context?
   | Score | Meaning |
   |-------|---------|
   | 1 | Mostly fabricated; majority of claims unsupported |
   | 2 | Several significant unsupported claims |
   | 3 | Some unsupported claims but core is grounded |
   | 4 | Nearly all claims supported; minor gaps only |
   | 5 | Every claim is fully supported by the context |

IMPORTANT:
- Be exhaustive when extracting claims — do NOT skip minor details.
- A claim is "unsupported" even if it is likely true but not in the context.
- Check numbers, dates, URLs, and proper nouns with extra care.
- Include internal metadata, config variables, or system details that
  appear in the response — these are often leaks and should be flagged.

Return ONLY valid JSON with this exact structure:
{
  "claims": [
    {"claim": "<exact claim text>", "status": "supported|unsupported|contradicted", "reasoning": "<why>"}
  ],
  "total_claims": <int>,
  "supported_claims": <int>,
  "unsupported_claims": <int>,
  "contradictions": <int>,
  "faithfulness": {"score": <1-5>, "reasoning": "<overall assessment>"},
  "overall_reasoning": "<summary of groundedness assessment>"
}
"""

_USER_PROMPT_CONTEXT_BACKED = """\
Evaluate the groundedness of the following AI assistant response against
the provided source context.

## User Query
{user_query}

## AI Assistant Response
{dave_response}

## Retrieved Source Context
{retrieved_context}

{chat_history_section}

Extract ALL factual claims from the response and classify each against
the source context. Return ONLY valid JSON.
"""

# ---------------------------------------------------------------------------
# Prompts — Context-Free
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_CONTEXT_FREE = """\
You are a STRICT, expert evaluation judge specialising in detecting
hallucination and overconfidence in AI responses where no source context
is available.

TASK:
Assess the AI assistant's response on three dimensions, each scored 1–5.
A score of 5 means GOOD (no issues), and 1 means BAD (severe issues).

1. **Internal Consistency** (1–5):
   Does the response contradict itself? Are all statements logically
   compatible with each other and with the conversation history?
   | 5 = Fully consistent | 1 = Contains clear self-contradictions |

2. **Overconfidence** (1–5):
   Does the response present uncertain or unverifiable information as if
   it were confirmed fact? Does it appropriately hedge when making claims
   that cannot be independently verified?
   | 5 = Appropriately hedged | 1 = Presents speculation as fact |

3. **Hallucination Risk** (1–5):
   Assess whether this response presents uncertain or unverifiable
   information as if it were confirmed fact, and whether it contradicts
   anything stated earlier in the conversation history provided. Does it
   fabricate specific details (names, numbers, URLs, dates) that seem
   invented rather than recalled from reliable knowledge?
   | 5 = No hallucination indicators | 1 = Clearly fabricated content |

IMPORTANT:
- Be strict. General knowledge responses can still be overconfident.
- Check for invented statistics, fake references, or made-up specifics.
- If conversation history is provided, check for contradictions with it.

Return ONLY valid JSON with this exact structure:
{
  "internal_consistency": {"score": <1-5>, "reasoning": "<specific assessment>"},
  "overconfidence": {"score": <1-5>, "reasoning": "<specific assessment>"},
  "hallucination_risk": {"score": <1-5>, "reasoning": "<specific assessment>"},
  "overall_reasoning": "<summary>"
}
"""

_USER_PROMPT_CONTEXT_FREE = """\
Evaluate the following AI assistant response for internal consistency,
overconfidence, and hallucination risk. No source context is available
for this conversation — assess based on the response's own coherence
and epistemic calibration.

## User Query
{user_query}

## AI Assistant Response
{dave_response}

{chat_history_section}

Score each dimension 1–5 (5 = good, no issues). Return ONLY valid JSON.
"""


# ---------------------------------------------------------------------------
# TruLens integration (optional — graceful degradation)
# ---------------------------------------------------------------------------

def _run_trulens_groundedness(
    context: str, response: str
) -> float | None:
    """
    Run TruLens groundedness evaluation.

    Returns a float score [0, 1] or None if TruLens is unavailable.
    """
    try:
        from trulens.providers.litellm import LiteLLM
        import os

        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        provider = LiteLLM(model_engine=f"gemini/{model_name}")

        score = provider.groundedness_measure_with_cot_reasons(
            source=context,
            statement=response,
        )
        # TruLens returns (score, dict_of_reasons) or just a score
        if isinstance(score, tuple):
            return float(score[0])
        return float(score)

    except ImportError:
        logger.warning(
            "TruLens not available (import failed). "
            "Skipping TruLens groundedness comparison."
        )
        return None
    except Exception as exc:
        logger.warning(
            "TruLens groundedness evaluation failed: %s. "
            "Continuing with custom evaluation only.",
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# DeepEval integration (optional — graceful degradation)
# ---------------------------------------------------------------------------

def _run_deepeval_faithfulness(
    user_query: str, response: str, context: str
) -> float | None:
    """
    Run DeepEval faithfulness evaluation.

    Returns a float score [0, 1] or None if DeepEval is unavailable.
    """
    try:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
        import os

        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

        test_case = LLMTestCase(
            input=user_query,
            actual_output=response,
            retrieval_context=[context],
        )

        metric = FaithfulnessMetric(
            model=f"gemini/{model_name}",
            threshold=0.7,
        )
        metric.measure(test_case)
        return float(metric.score)

    except ImportError:
        logger.warning(
            "DeepEval not available (import failed). "
            "Skipping DeepEval faithfulness comparison."
        )
        return None
    except Exception as exc:
        logger.warning(
            "DeepEval faithfulness evaluation failed: %s. "
            "Continuing with custom evaluation only.",
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class GroundednessEvaluator(BaseEvaluator):
    """
    Groundedness / hallucination evaluator with branching logic:
      - Context-backed → claim extraction + faithfulness (max 15)
      - Context-free   → consistency + overconfidence + hallucination risk (max 15)

    Also runs TruLens and DeepEval in parallel for comparison (context-backed only).
    """

    name: str = "groundedness"

    def __init__(self) -> None:
        self._judge = LLMJudge()
        logger.info("GroundednessEvaluator initialized.")

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult:
        """Branch to context-backed or context-free evaluation."""

        if eval_input.conversation_type == ConversationType.CONTEXT_BACKED:
            return self._evaluate_context_backed(eval_input)
        else:
            return self._evaluate_context_free(eval_input)

    # ------------------------------------------------------------------
    # Context-Backed evaluation
    # ------------------------------------------------------------------

    def _evaluate_context_backed(
        self, eval_input: EvaluationInput
    ) -> EvaluationResult:
        """
        Evaluate groundedness for a context-backed conversation.

        Runs custom LLM judge + TruLens + DeepEval in parallel.
        """
        logger.debug(
            "Evaluating groundedness (context-backed) for '%s'",
            eval_input.conversation_id,
        )

        context = eval_input.retrieved_context or ""
        response = eval_input.dave_response

        # --- Run custom judge + TruLens + DeepEval concurrently ----------
        trulens_score: float | None = None
        deepeval_score: float | None = None
        parsed_json: dict[str, Any] = {}
        raw_text: str = ""

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all three evaluations
            future_custom = executor.submit(
                self._run_custom_context_backed_judge, eval_input
            )
            future_trulens = executor.submit(
                _run_trulens_groundedness, context, response
            )
            future_deepeval = executor.submit(
                _run_deepeval_faithfulness,
                eval_input.user_query,
                response,
                context,
            )

            # Collect results
            parsed_json, raw_text = future_custom.result()
            trulens_score = future_trulens.result()
            deepeval_score = future_deepeval.result()

        # --- Compute scores from custom judge ----
        total_claims = max(parsed_json.get("total_claims", 1), 1)  # avoid div/0
        supported = parsed_json.get("supported_claims", 0)
        unsupported = parsed_json.get("unsupported_claims", 0)
        contradictions = parsed_json.get("contradictions", 0)

        faithfulness_raw = self._extract_score(
            parsed_json, "faithfulness", default=3
        )

        # Apply the exact formulas (Phase 1: halved multipliers)
        evidence_coverage = (supported / total_claims) * 5
        faithfulness_score = (faithfulness_raw / 5) * 5
        unsupported_score = (1 - (unsupported / total_claims)) * 2.5
        contradiction_score = (1 - (contradictions / total_claims)) * 2.5

        sub_scores: dict[str, float] = {
            "evidence_coverage": round(evidence_coverage, 2),
            "faithfulness_to_context": round(faithfulness_score, 2),
            "unsupported_claims": round(unsupported_score, 2),
            "contradiction_detection": round(contradiction_score, 2),
        }

        # Store comparison scores from external frameworks
        if trulens_score is not None:
            sub_scores["trulens_groundedness"] = round(trulens_score, 4)
        if deepeval_score is not None:
            sub_scores["deepeval_faithfulness"] = round(deepeval_score, 4)

        total_score = round(
            evidence_coverage
            + faithfulness_score
            + unsupported_score
            + contradiction_score,
            2,
        )

        # Build feedback from LLM's actual output
        feedback = self._build_context_backed_feedback(
            parsed_json, sub_scores, trulens_score, deepeval_score
        )

        # Flag if score < 50% or any contradictions found
        flagged = total_score < (_MAX_SCORE_CONTEXT_BACKED * 0.5) or contradictions > 0

        return EvaluationResult(
            evaluator_name=self.name,
            conversation_id=eval_input.conversation_id,
            score=total_score,
            max_score=_MAX_SCORE_CONTEXT_BACKED,
            sub_scores=sub_scores,
            feedback=feedback,
            flagged=flagged,
        )

    def _run_custom_context_backed_judge(
        self, eval_input: EvaluationInput
    ) -> tuple[dict[str, Any], str]:
        """Run the custom LLM judge for context-backed conversations."""
        chat_section = ""
        if eval_input.chat_history:
            chat_section = (
                f"## Conversation History\n{eval_input.chat_history}"
            )

        user_prompt = _USER_PROMPT_CONTEXT_BACKED.format(
            user_query=eval_input.user_query,
            dave_response=eval_input.dave_response,
            retrieved_context=eval_input.retrieved_context or "",
            chat_history_section=chat_section,
        )

        return self._judge.call_with_json(
            _SYSTEM_PROMPT_CONTEXT_BACKED, user_prompt
        )

    @staticmethod
    def _build_context_backed_feedback(
        parsed: dict[str, Any],
        sub_scores: dict[str, float],
        trulens_score: float | None,
        deepeval_score: float | None,
    ) -> str:
        """Build human-readable feedback from the judge's analysis."""
        parts: list[str] = []

        # Overall reasoning from LLM
        overall = parsed.get("overall_reasoning", "")
        if overall:
            parts.append(f"Overall Assessment: {overall}")

        # Faithfulness reasoning
        faith = parsed.get("faithfulness", {})
        if isinstance(faith, dict) and faith.get("reasoning"):
            parts.append(
                f"Faithfulness ({faith.get('score', '?')}/5): "
                f"{faith['reasoning']}"
            )

        # Claim-by-claim details (show unsupported/contradicted)
        claims = parsed.get("claims", [])
        problem_claims = [
            c for c in claims
            if isinstance(c, dict) and c.get("status") in ("unsupported", "contradicted")
        ]
        if problem_claims:
            parts.append("Problematic Claims:")
            for c in problem_claims:
                parts.append(
                    f"  [{c.get('status', '?').upper()}] "
                    f"\"{c.get('claim', '?')}\" — {c.get('reasoning', '')}"
                )

        # Sub-score summary
        parts.append(
            f"Sub-scores: Evidence Coverage={sub_scores.get('evidence_coverage', 0)}/5, "
            f"Faithfulness={sub_scores.get('faithfulness_to_context', 0)}/5, "
            f"Unsupported={sub_scores.get('unsupported_claims', 0)}/2.5, "
            f"Contradictions={sub_scores.get('contradiction_detection', 0)}/2.5"
        )

        # External framework comparison
        if trulens_score is not None:
            parts.append(f"TruLens Groundedness (comparison): {trulens_score:.4f}")
        if deepeval_score is not None:
            parts.append(f"DeepEval Faithfulness (comparison): {deepeval_score:.4f}")

        return "\n\n".join(parts) if parts else "No feedback generated."

    # ------------------------------------------------------------------
    # Context-Free evaluation
    # ------------------------------------------------------------------

    def _evaluate_context_free(
        self, eval_input: EvaluationInput
    ) -> EvaluationResult:
        """Evaluate groundedness for a context-free conversation."""

        logger.debug(
            "Evaluating groundedness (context-free) for '%s'",
            eval_input.conversation_id,
        )

        chat_section = ""
        if eval_input.chat_history:
            chat_section = (
                f"## Conversation History\n{eval_input.chat_history}"
            )

        user_prompt = _USER_PROMPT_CONTEXT_FREE.format(
            user_query=eval_input.user_query,
            dave_response=eval_input.dave_response,
            chat_history_section=chat_section,
        )

        parsed_json, raw_text = self._judge.call_with_json(
            _SYSTEM_PROMPT_CONTEXT_FREE, user_prompt
        )

        # Extract scores
        consistency_raw = self._extract_score(
            parsed_json, "internal_consistency", default=3
        )
        overconfidence_raw = self._extract_score(
            parsed_json, "overconfidence", default=3
        )
        hallucination_raw = self._extract_score(
            parsed_json, "hallucination_risk", default=3
        )

        # Apply formulas: (score / 5) × 5  (Phase 1: halved from ×10)
        consistency_score = (consistency_raw / 5) * 5
        overconfidence_score = (overconfidence_raw / 5) * 5
        hallucination_score = (hallucination_raw / 5) * 5

        sub_scores: dict[str, float] = {
            "internal_consistency": round(consistency_score, 2),
            "overconfidence": round(overconfidence_score, 2),
            "hallucination_risk": round(hallucination_score, 2),
        }

        total_score = round(
            consistency_score + overconfidence_score + hallucination_score, 2
        )

        # Build feedback from LLM's actual reasoning
        feedback = self._build_context_free_feedback(parsed_json, sub_scores)

        # Flag if score < 50%
        flagged = total_score < (_MAX_SCORE_CONTEXT_FREE * 0.5)

        return EvaluationResult(
            evaluator_name=self.name,
            conversation_id=eval_input.conversation_id,
            score=total_score,
            max_score=_MAX_SCORE_CONTEXT_FREE,
            sub_scores=sub_scores,
            feedback=feedback,
            flagged=flagged,
        )

    @staticmethod
    def _build_context_free_feedback(
        parsed: dict[str, Any],
        sub_scores: dict[str, float],
    ) -> str:
        """Build human-readable feedback for context-free evaluation."""
        parts: list[str] = []

        # Per-metric reasoning from LLM
        for metric_key, label in [
            ("internal_consistency", "Internal Consistency"),
            ("overconfidence", "Overconfidence"),
            ("hallucination_risk", "Hallucination Risk"),
        ]:
            entry = parsed.get(metric_key, {})
            if isinstance(entry, dict) and entry.get("reasoning"):
                parts.append(
                    f"{label} ({entry.get('score', '?')}/5): "
                    f"{entry['reasoning']}"
                )

        # Overall reasoning
        overall = parsed.get("overall_reasoning", "")
        if overall:
            parts.append(f"Overall: {overall}")

        # Sub-score summary
        parts.append(
            f"Sub-scores: Consistency={sub_scores.get('internal_consistency', 0)}/5, "
            f"Overconfidence={sub_scores.get('overconfidence', 0)}/5, "
            f"Hallucination Risk={sub_scores.get('hallucination_risk', 0)}/5"
        )

        return "\n\n".join(parts) if parts else "No feedback generated."

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_score(
        parsed: dict[str, Any],
        key: str,
        default: int = 3,
    ) -> int:
        """
        Safely extract a 1–5 integer score from a parsed JSON entry.

        The entry can be either:
          - {"score": 4, "reasoning": "..."}   (nested dict)
          - 4                                    (bare int)
        """
        entry = parsed.get(key)

        if entry is None:
            logger.warning("Missing key '%s' in LLM response, using default %d", key, default)
            return default

        if isinstance(entry, dict):
            raw = entry.get("score", default)
        else:
            raw = entry

        try:
            score = int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Non-integer score for '%s': %s, using default %d",
                key,
                raw,
                default,
            )
            return default

        # Clamp to [1, 5]
        if not 1 <= score <= 5:
            logger.warning(
                "Out-of-range score for '%s': %d, clamping to [1, 5]", key, score
            )
            score = max(1, min(score, 5))

        return score
