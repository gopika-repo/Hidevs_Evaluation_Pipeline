"""
Groundedness / Hallucination Evaluator — Phase 1

Evaluates whether Dave's response is grounded in evidence and free of
fabricated claims. Uses the same three custom LLM metrics for both paths:

**Context-Backed** (max = 20):
  Custom LLM judge scores:
  • Internal Consistency  — (score / 5) × 6   (max 6)
  • Overconfidence        — (score / 5) × 6   (max 6)
  • Hallucination Risk    — (score / 5) × 8   (max 8)
  + TruLens groundedness score (stored for comparison, not added to score)
  + DeepEval faithfulness score (stored for comparison, not added to score)

**Context-Free** (max = 20):
  Custom LLM judge scores:
  • Internal Consistency  — (score / 5) × 6   (max 6)
  • Overconfidence        — (score / 5) × 6   (max 6)
  • Hallucination Risk    — (score / 5) × 8   (max 8)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from evaluation_pipeline.data.models import (
    ConversationType,
    EvaluationInput,
    EvaluationResult,
)
from evaluation_pipeline.evaluators.base_evaluator import BaseEvaluator
from evaluation_pipeline.utils.llm_client import LLMJudge
from evaluation_pipeline.utils.schemas import GroundednessSchema
from evaluation_pipeline.utils.error_handler import classify_exception

logger = logging.getLogger(__name__)

_shared_executor = ThreadPoolExecutor(max_workers=32)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SCORE_CONTEXT_BACKED = 20.0
_MAX_SCORE_CONTEXT_FREE = 20.0

# ---------------------------------------------------------------------------
# Prompts — Context-Backed
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_CONTEXT_BACKED = """\
You are a STRICT, expert evaluation judge specialising in groundedness
and hallucination detection. You must be rigorous — penalise any
unsupported or fabricated claim, no matter how minor.

You will evaluate the response on three metrics, each scored 1–5 (5 = Excellent, 1 = Very Poor):

1. **Internal Consistency** (1-5):
   Does the response contradict itself? Are all statements logically
   compatible with each other and with the conversation history?
   | 5 = Fully consistent | 1 = Contains clear self-contradictions |

2. **Overconfidence** (1-5):
   Does the response present uncertain or unverifiable information as if
   it were confirmed fact? Does it appropriately hedge when making claims
   that cannot be independently verified?
   | 5 = Appropriately hedged | 1 = Presents speculation as fact |

3. **Hallucination Risk** (1-5):
   Assess whether this response presents claims that are NOT grounded in the retrieved source context,
   or if it contradicts anything in the retrieved source context. Does it
   fabricate specific details (names, numbers, URLs, dates) that seem
   invented or unsupported by the context?
   | 5 = Fully grounded in context | 1 = Clearly fabricated or contains contradicted content |

Return your evaluation as a JSON object with EXACTLY this structure:
{
  "internal_consistency": {"score": <1-5>, "reasoning": "<specific assessment>"},
  "overconfidence": {"score": <1-5>, "reasoning": "<specific assessment>"},
  "hallucination_risk": {"score": <1-5>, "reasoning": "<specific assessment>"},
  "overall_reasoning": "<summary>"
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

Score each dimension 1–5 (5 = good, no issues). Return ONLY valid JSON.
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
    context: str, response: str, conversation_id: str = "unknown", deadline: float | None = None
) -> dict[str, Any]:
    """
    Run TruLens groundedness evaluation using native Google provider with controlled concurrency, retries, and timeout.
    """
    if not context or not context.strip():
        return {
            "status": "not_applicable",
            "reason": "No retrieved context available"
        }
    
    from evaluation_pipeline.utils.concurrency import controlled_concurrency
    from evaluation_pipeline.utils.retry_utils import execute_with_retry
    import os
    import concurrent.futures

    trulens_timeout = float(os.getenv("TRULENS_TIMEOUT", "45.0"))
    if deadline is not None:
        remaining = deadline - time.time()
        trulens_timeout = max(1.0, min(trulens_timeout, remaining))

    def _call_trulens():
        from trulens.providers.google import Google
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
        api_key = os.getenv("GOOGLE_API_KEY")
        provider = Google(model_engine=model_name, api_key=api_key)
        
        def _invoke():
            with controlled_concurrency("groundedness", "TruLens", conversation_id):
                score = provider.groundedness_measure_with_cot_reasons(
                    source=context,
                    statement=response,
                )
                if isinstance(score, tuple):
                    return float(score[0])
                return float(score)

        fut = _shared_executor.submit(_invoke)
        return fut.result(timeout=trulens_timeout)

    try:
        score_val = execute_with_retry(
            _call_trulens,
            evaluator="groundedness",
            framework="TruLens",
            conversation_id=conversation_id,
            max_retries=3,
            initial_delay=2.0,
            deadline=deadline,
        )
        return {
            "status": "success",
            "score": score_val
        }
    except Exception as exc:
        logger.warning("TruLens groundedness evaluation failed: %s", exc)
        return {
            "status": "failed",
            "error": str(exc)
        }


def _run_deepeval_faithfulness(
    user_query: str, response: str, context: str, conversation_id: str = "unknown", deadline: float | None = None
) -> dict[str, Any]:
    """
    Run DeepEval faithfulness evaluation using GeminiModel with controlled concurrency, retries, and timeout.
    """
    if not context or not context.strip():
        return {
            "status": "not_applicable",
            "reason": "No retrieved context available"
        }
    
    from evaluation_pipeline.utils.concurrency import controlled_concurrency
    from evaluation_pipeline.utils.retry_utils import execute_with_retry
    import os
    import concurrent.futures

    deepeval_timeout = float(os.getenv("DEEPEVAL_TIMEOUT", "45.0"))
    if deadline is not None:
        remaining = deadline - time.time()
        deepeval_timeout = max(1.0, min(deepeval_timeout, remaining))

    def _call_deepeval():
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
        from deepeval.models import GeminiModel
        
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
        api_key = os.getenv("GOOGLE_API_KEY")

        model = GeminiModel(
            model=model_name,
            api_key=api_key,
            temperature=0.0
        )
        test_case = LLMTestCase(
            input=user_query,
            actual_output=response,
            retrieval_context=[context],
        )
        metric = FaithfulnessMetric(
            model=model,
            threshold=0.7,
        )

        def _invoke():
            with controlled_concurrency("groundedness", "DeepEval", conversation_id):
                metric.measure(test_case)
                return float(metric.score)

        fut = _shared_executor.submit(_invoke)
        return fut.result(timeout=deepeval_timeout)

    try:
        score_val = execute_with_retry(
            _call_deepeval,
            evaluator="groundedness",
            framework="DeepEval",
            conversation_id=conversation_id,
            max_retries=3,
            initial_delay=2.0,
            deadline=deadline,
        )
        return {
            "status": "success",
            "score": score_val
        }
    except Exception as exc:
        logger.warning("DeepEval faithfulness evaluation failed: %s", exc)
        return {
            "status": "failed",
            "error": str(exc)
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class GroundednessEvaluator(BaseEvaluator):
    """
    Groundedness / hallucination evaluator with branching logic:
      - Context-backed → consistency + overconfidence + hallucination risk (max 20)
                         + TruLens and DeepEval for comparison
      - Context-free   → consistency + overconfidence + hallucination risk (max 20)
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
        try:
            if eval_input.conversation_type == ConversationType.CONTEXT_BACKED:
                return self._evaluate_context_backed(eval_input)
            else:
                return self._evaluate_context_free(eval_input)
        except Exception as exc:
            logger.error("GroundednessEvaluator failed for %s: %s", eval_input.conversation_id, exc)
            error_status = classify_exception(exc)
            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=20.0,
                status=error_status,
                sub_scores={},
                feedback=f"Groundedness evaluation failed with error: {exc}",
                flagged=True,
            )

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
        trulens_res: dict[str, Any] = {}
        deepeval_res: dict[str, Any] = {}
        parsed_json: dict[str, Any] = {}
        raw_text: str = ""

        deadline = eval_input.deadline

        # Submit all three evaluations to the shared executor, passing the request deadline
        future_custom = _shared_executor.submit(
            self._run_custom_context_backed_judge, eval_input
        )
        future_trulens = _shared_executor.submit(
            _run_trulens_groundedness, context, response, eval_input.conversation_id, deadline
        )
        future_deepeval = _shared_executor.submit(
            _run_deepeval_faithfulness,
            eval_input.user_query,
            response,
            context,
            eval_input.conversation_id,
            deadline
        )

        custom_exc = None
        # Collect results safely with remaining time budget limits
        remaining = None
        if deadline is not None:
            remaining = max(1.0, deadline - time.time())
        try:
            parsed_json, raw_text = future_custom.result(timeout=remaining)
        except Exception as exc:
            logger.error("Groundedness custom judge failed for %s: %s", eval_input.conversation_id, exc)
            parsed_json = {}
            raw_text = ""
            custom_exc = exc
            
        if deadline is not None:
            remaining = max(1.0, deadline - time.time())
        try:
            trulens_res = future_trulens.result(timeout=remaining)
        except Exception as exc:
            logger.error("Groundedness TruLens failed for %s: %s", eval_input.conversation_id, exc)
            trulens_res = {"status": "failed", "error": str(exc)}
            
        if deadline is not None:
            remaining = max(1.0, deadline - time.time())
        try:
            deepeval_res = future_deepeval.result(timeout=remaining)
        except Exception as exc:
            logger.error("Groundedness DeepEval failed for %s: %s", eval_input.conversation_id, exc)
            deepeval_res = {"status": "failed", "error": str(exc)}

        # --- Check if custom judge failed ----
        if not parsed_json:
            error_status = "failed"
            feedback_msg = "Groundedness custom judge call failed."
            if custom_exc:
                error_status = classify_exception(custom_exc)
                feedback_msg = f"Groundedness custom judge call failed with error: {custom_exc}"
            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=_MAX_SCORE_CONTEXT_BACKED,
                status=error_status,
                sub_scores={},
                feedback=feedback_msg,
                flagged=True,
            )

        # --- Compute scores from custom judge ----
        consistency_raw = self._extract_score(parsed_json, "internal_consistency")
        overconfidence_raw = self._extract_score(parsed_json, "overconfidence")
        hallucination_raw = self._extract_score(parsed_json, "hallucination_risk")

        consistency_score = (consistency_raw / 5.0) * 6.0
        overconfidence_score = (overconfidence_raw / 5.0) * 6.0
        hallucination_score = (hallucination_raw / 5.0) * 8.0

        sub_scores: dict[str, float] = {
            "internal_consistency": round(consistency_score, 2),
            "overconfidence": round(overconfidence_score, 2),
            "hallucination_risk": round(hallucination_score, 2),
        }

        # Store comparison details from external frameworks
        trulens_status = trulens_res.get("status")
        if trulens_status == "success":
            tr_score = trulens_res.get("score")
            if tr_score is not None and isinstance(tr_score, (int, float)):
                sub_scores["trulens_status"] = "success"
                sub_scores["trulens_score"] = round(float(tr_score), 4)
            else:
                sub_scores["trulens_status"] = "failed"
                sub_scores["trulens_error"] = "TruLens reported success but returned missing or invalid score."
        elif trulens_status == "failed":
            sub_scores["trulens_status"] = "failed"
            sub_scores["trulens_error"] = trulens_res.get("error", "Unknown TruLens error")
        else:
            sub_scores["trulens_status"] = "not_applicable"
            sub_scores["trulens_reason"] = trulens_res.get("reason", "No retrieved context available")

        deepeval_status = deepeval_res.get("status")
        if deepeval_status == "success":
            de_score = deepeval_res.get("score")
            if de_score is not None and isinstance(de_score, (int, float)):
                sub_scores["deepeval_status"] = "success"
                sub_scores["deepeval_score"] = round(float(de_score), 4)
            else:
                sub_scores["deepeval_status"] = "failed"
                sub_scores["deepeval_error"] = "DeepEval reported success but returned missing or invalid score."
        elif deepeval_status == "failed":
            sub_scores["deepeval_status"] = "failed"
            sub_scores["deepeval_error"] = deepeval_res.get("error", "Unknown DeepEval error")
        else:
            sub_scores["deepeval_status"] = "not_applicable"
            sub_scores["deepeval_reason"] = deepeval_res.get("reason", "No retrieved context available")

        total_score = round(
            consistency_score
            + overconfidence_score
            + hallucination_score,
            2,
        )

        # Build feedback from LLM's actual output
        feedback = self._build_context_backed_feedback(
            parsed_json, sub_scores
        )

        # Flag if score < 50%
        flagged = total_score < (_MAX_SCORE_CONTEXT_BACKED * 0.5)

        return EvaluationResult(
            evaluator_name=self.name,
            conversation_id=eval_input.conversation_id,
            score=total_score,
            max_score=_MAX_SCORE_CONTEXT_BACKED,
            percentage=round((total_score / _MAX_SCORE_CONTEXT_BACKED) * 100.0, 2),
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
            _SYSTEM_PROMPT_CONTEXT_BACKED,
            user_prompt,
            evaluator=self.name,
            conversation_id=eval_input.conversation_id,
            response_schema=GroundednessSchema,
            deadline=eval_input.deadline,
        )

    @staticmethod
    def _build_context_backed_feedback(
        parsed: dict[str, Any],
        sub_scores: dict[str, float]
    ) -> str:
        """Build human-readable feedback from the judge's analysis."""
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

        # Overall reasoning from LLM
        overall = parsed.get("overall_reasoning", "")
        if overall:
            parts.append(f"Overall Assessment: {overall}")

        # Sub-score summary
        parts.append(
            f"Sub-scores: Consistency={sub_scores.get('internal_consistency', 0)}/6.0, "
            f"Overconfidence={sub_scores.get('overconfidence', 0)}/6.0, "
            f"Hallucination Risk={sub_scores.get('hallucination_risk', 0)}/8.0"
        )

        # External framework comparison
        trulens_status = sub_scores.get("trulens_status", "unknown")
        if trulens_status == "success" and "trulens_score" in sub_scores:
            parts.append(f"TruLens Groundedness (comparison): {float(sub_scores['trulens_score']):.4f}")
        elif trulens_status == "failed":
            parts.append(f"TruLens Groundedness (comparison): FAILED. Error: {sub_scores.get('trulens_error', '')}")
        else:
            parts.append(f"TruLens Groundedness (comparison): NOT APPLICABLE. Reason: {sub_scores.get('trulens_reason', '')}")

        deepeval_status = sub_scores.get("deepeval_status", "unknown")
        if deepeval_status == "success" and "deepeval_score" in sub_scores:
            parts.append(f"DeepEval Faithfulness (comparison): {float(sub_scores['deepeval_score']):.4f}")
        elif deepeval_status == "failed":
            parts.append(f"DeepEval Faithfulness (comparison): FAILED. Error: {sub_scores.get('deepeval_error', '')}")
        else:
            parts.append(f"DeepEval Faithfulness (comparison): NOT APPLICABLE. Reason: {sub_scores.get('deepeval_reason', '')}")

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
            _SYSTEM_PROMPT_CONTEXT_FREE,
            user_prompt,
            evaluator=self.name,
            conversation_id=eval_input.conversation_id,
            response_schema=GroundednessSchema,
            deadline=eval_input.deadline,
        )

        if not parsed_json:
            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=_MAX_SCORE_CONTEXT_FREE,
                status="failed",
                sub_scores={},
                feedback="Groundedness custom judge call failed.",
                flagged=True,
            )

        # Extract scores
        consistency_raw = self._extract_score(parsed_json, "internal_consistency")
        overconfidence_raw = self._extract_score(parsed_json, "overconfidence")
        hallucination_raw = self._extract_score(parsed_json, "hallucination_risk")

        # Apply formulas: (score / 5) × category_max (6, 6, 8)
        consistency_score = (consistency_raw / 5.0) * 6.0
        overconfidence_score = (overconfidence_raw / 5.0) * 6.0
        hallucination_score = (hallucination_raw / 5.0) * 8.0

        sub_scores: dict[str, float] = {
            "internal_consistency": round(consistency_score, 2),
            "overconfidence": round(overconfidence_score, 2),
            "hallucination_risk": round(hallucination_score, 2),
            "trulens_status": "not_applicable",
            "trulens_reason": "No retrieved context available",
            "deepeval_status": "not_applicable",
            "deepeval_reason": "No retrieved context available",
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
            percentage=round((total_score / _MAX_SCORE_CONTEXT_FREE) * 100.0, 2),
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
            f"Sub-scores: Consistency={sub_scores.get('internal_consistency', 0)}/6.0, "
            f"Overconfidence={sub_scores.get('overconfidence', 0)}/6.0, "
            f"Hallucination Risk={sub_scores.get('hallucination_risk', 0)}/8.0"
        )

        # External framework comparison
        parts.append("TruLens Groundedness (comparison): NOT APPLICABLE. Reason: No retrieved context available")
        parts.append("DeepEval Faithfulness (comparison): NOT APPLICABLE. Reason: No retrieved context available")

        return "\n\n".join(parts) if parts else "No feedback generated."

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_score(
        parsed: dict[str, Any],
        key: str,
    ) -> int:
        """
        Safely extract a 1–5 integer score from a parsed JSON entry.
        Raises ValueError if missing or invalid.
        """
        entry = parsed.get(key)

        if entry is None:
            raise ValueError(f"Missing key '{key}' in LLM response.")

        if isinstance(entry, dict):
            raw = entry.get("score")
            if raw is None:
                raise ValueError(f"Missing 'score' field inside key '{key}' in LLM response.")
        else:
            raw = entry

        try:
            score = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Non-integer score for '{key}': {raw}")

        # Clamp to [1, 5]
        if not 1 <= score <= 5:
            raise ValueError(f"Out-of-range score for '{key}': {score}")

        return score
