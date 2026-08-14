"""
Groundedness / Hallucination Evaluator — Phase 1

Context-Backed:
  Custom LLM judge:
    - Internal Consistency  -> 6 points
    - Overconfidence        -> 6 points
    - Hallucination Risk    -> 8 points
    Total                    -> 20 points

  TruLens and DeepEval are comparison metrics only.
  They do NOT contribute to the official groundedness score.

Context-Free:
  Custom LLM judge:
    - Internal Consistency  -> 6 points
    - Overconfidence        -> 6 points
    - Hallucination Risk    -> 8 points
    Total                    -> 20 points
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from evaluation_pipeline.data.models import (
    ConversationType,
    EvaluationInput,
    EvaluationResult,
)
from evaluation_pipeline.evaluators.base_evaluator import BaseEvaluator
from evaluation_pipeline.utils.error_handler import classify_exception
from evaluation_pipeline.utils.llm_client import LLMJudge
from evaluation_pipeline.utils.schemas import GroundednessSchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------

# IMPORTANT:
# Existing groundedness tests patch:
#
#   evaluation_pipeline.evaluators.groundedness_evaluator._shared_executor.submit
#
# Therefore TruLens and DeepEval MUST use _shared_executor.
#
# This executor is reserved for actual framework calls.
_shared_executor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="groundedness-framework",
)


# Separate executor for orchestration of:
#   - custom judge
#   - TruLens wrapper
#   - DeepEval wrapper
#
# This prevents the outer orchestration layer from consuming the same
# worker threads needed by the framework futures.
_orchestration_executor = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="groundedness-orchestration",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SCORE_CONTEXT_BACKED = 20.0
_MAX_SCORE_CONTEXT_FREE = 20.0

_MIN_FRAMEWORK_TIMEOUT = 0.05


# ---------------------------------------------------------------------------
# Prompts — Context-Backed
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_CONTEXT_BACKED = """\
You are a STRICT, expert evaluation judge specialising in groundedness
and hallucination detection.

You must be rigorous. Penalise unsupported, fabricated, or contradicted
claims.

Evaluate the AI assistant response on three dimensions.

1. Internal Consistency (1-5)

Does the response contradict itself?

Are all statements logically compatible with:
- the user query,
- the conversation history,
- and the response itself?

5 = Fully consistent
1 = Contains clear contradictions

2. Overconfidence (1-5)

Does the response present uncertain or unverifiable information as confirmed
fact?

Does it appropriately hedge claims that cannot be verified from the provided
context?

5 = Appropriately calibrated
1 = Presents speculation as fact

3. Hallucination Risk (1-5)

Are the claims supported by the retrieved source context?

Penalise:
- fabricated facts
- unsupported names
- unsupported numbers
- unsupported dates
- unsupported URLs
- unsupported locations
- unsupported statistics
- claims contradicting the retrieved context

5 = Fully grounded
1 = Clearly fabricated or contradicted

Return ONLY valid JSON with exactly this structure:

{
  "internal_consistency": {
    "score": <1-5>,
    "reasoning": "<specific assessment>"
  },
  "overconfidence": {
    "score": <1-5>,
    "reasoning": "<specific assessment>"
  },
  "hallucination_risk": {
    "score": <1-5>,
    "reasoning": "<specific assessment>"
  },
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

Score each dimension from 1–5.

5 = good / no issue
1 = severe issue

Return ONLY valid JSON.
"""


# ---------------------------------------------------------------------------
# Prompts — Context-Free
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_CONTEXT_FREE = """\
You are a STRICT, expert evaluation judge specialising in hallucination
and overconfidence detection when no retrieved source context is available.

Evaluate the response on three dimensions.

1. Internal Consistency (1-5)

Does the response contradict itself or previous conversation information?

5 = Fully consistent
1 = Clear contradictions

2. Overconfidence (1-5)

Does the response present uncertain or unverifiable information as fact?

5 = Appropriately calibrated
1 = Presents speculation as fact

3. Hallucination Risk (1-5)

Does the response contain fabricated, invented, or unsupported specifics?

Look for:
- invented statistics
- fake references
- made-up URLs
- fabricated dates
- fabricated names
- unsupported claims

5 = No hallucination indicators
1 = Clearly fabricated content

Return ONLY valid JSON with exactly this structure:

{
  "internal_consistency": {
    "score": <1-5>,
    "reasoning": "<specific assessment>"
  },
  "overconfidence": {
    "score": <1-5>,
    "reasoning": "<specific assessment>"
  },
  "hallucination_risk": {
    "score": <1-5>,
    "reasoning": "<specific assessment>"
  },
  "overall_reasoning": "<summary>"
}
"""


_USER_PROMPT_CONTEXT_FREE = """\
Evaluate the following AI assistant response for:

- internal consistency
- overconfidence
- hallucination risk

No source context is available.

## User Query
{user_query}

## AI Assistant Response
{dave_response}

{chat_history_section}

Score each dimension from 1–5.

5 = good / no issue
1 = severe issue

Return ONLY valid JSON.
"""


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

def _calculate_framework_timeout(
    env_name: str,
    default: float,
    deadline: float | None,
) -> float:
    """
    Calculate a bounded timeout for TruLens / DeepEval.

    A very small remaining deadline is floored to 0.05 seconds because
    the test suite explicitly verifies this behaviour.
    """
    try:
        timeout = float(
            os.getenv(
                env_name,
                str(default),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {env_name} value."
        ) from exc

    if timeout <= 0:
        raise ValueError(
            f"{env_name} must be greater than 0."
        )

    if deadline is None:
        return timeout

    remaining = deadline - time.time()

    return min(
        timeout,
        max(
            _MIN_FRAMEWORK_TIMEOUT,
            remaining,
        ),
    )


# ---------------------------------------------------------------------------
# TruLens integration
# ---------------------------------------------------------------------------

def _run_trulens_groundedness(
    context: str,
    response: str,
    conversation_id: str = "unknown",
    deadline: float | None = None,
) -> dict[str, Any]:
    """
    Run TruLens groundedness evaluation.

    TruLens is a comparison metric only.
    It does NOT contribute to the official groundedness score.
    """

    if not context or not context.strip():
        return {
            "status": "not_applicable",
            "reason": "No retrieved context available",
        }

    from evaluation_pipeline.utils.concurrency import (
        controlled_concurrency,
    )
    from evaluation_pipeline.utils.retry_utils import (
        execute_with_retry,
    )

    trulens_timeout = _calculate_framework_timeout(
        env_name="TRULENS_TIMEOUT",
        default=45.0,
        deadline=deadline,
    )

    def framework_call() -> float:
        """
        Actual TruLens execution.

        This function executes INSIDE _shared_executor.
        """
        from trulens.providers.google import Google

        model_name = os.getenv(
            "GEMINI_MODEL_NAME",
            "gemini-3.5-flash",
        )

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not configured."
            )

        provider = Google(
            model_engine=model_name,
            api_key=api_key,
        )

        with controlled_concurrency(
            "groundedness",
            "TruLens",
            conversation_id,
        ):
            score = provider.groundedness_measure_with_cot_reasons(
                source=context,
                statement=response,
            )

        if isinstance(score, tuple):
            score = score[0]

        score = float(score)

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"TruLens returned out-of-range score: {score}"
            )

        return score

    def invoke_with_timeout() -> float:
        """
        Submit the framework call to the shared framework executor and
        enforce the framework-specific timeout.

        Existing tests intentionally patch _shared_executor.submit().
        """
        future = _shared_executor.submit(
            framework_call
        )

        return float(
            future.result(
                timeout=trulens_timeout
            )
        )

    try:
        score = execute_with_retry(
            invoke_with_timeout,
            evaluator="groundedness",
            framework="TruLens",
            conversation_id=conversation_id,
            max_retries=3,
            initial_delay=2.0,
            deadline=deadline,
        )

        return {
            "status": "success",
            "score": float(score),
        }

    except Exception as exc:
        logger.warning(
            "TruLens groundedness evaluation failed for %s: %s",
            conversation_id,
            exc,
        )

        return {
            "status": "failed",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# DeepEval integration
# ---------------------------------------------------------------------------

def _run_deepeval_faithfulness(
    user_query: str,
    response: str,
    context: str,
    conversation_id: str = "unknown",
    deadline: float | None = None,
) -> dict[str, Any]:
    """
    Run DeepEval Faithfulness evaluation.

    DeepEval is a comparison metric only.
    It does NOT contribute to the official groundedness score.
    """

    if not context or not context.strip():
        return {
            "status": "not_applicable",
            "reason": "No retrieved context available",
        }

    from evaluation_pipeline.utils.concurrency import (
        controlled_concurrency,
    )
    from evaluation_pipeline.utils.retry_utils import (
        execute_with_retry,
    )

    deepeval_timeout = _calculate_framework_timeout(
        env_name="DEEPEVAL_TIMEOUT",
        default=45.0,
        deadline=deadline,
    )

    def framework_call() -> float:
        """
        Actual DeepEval execution.

        This function executes INSIDE _shared_executor.
        """
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.models import GeminiModel
        from deepeval.test_case import LLMTestCase

        model_name = os.getenv(
            "GEMINI_MODEL_NAME",
            "gemini-3.5-flash",
        )

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not configured."
            )

        model = GeminiModel(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
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

        with controlled_concurrency(
            "groundedness",
            "DeepEval",
            conversation_id,
        ):
            metric.measure(test_case)

        score = metric.score

        if score is None:
            raise ValueError(
                "DeepEval returned no score."
            )

        if not isinstance(
            score,
            (int, float),
        ):
            raise ValueError(
                "DeepEval returned invalid score type: "
                f"{type(score).__name__}"
            )

        score = float(score)

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"DeepEval returned out-of-range score: {score}"
            )

        return score

    def invoke_with_timeout() -> float:
        """
        Submit DeepEval to _shared_executor and enforce timeout.

        Existing tests patch _shared_executor.submit().
        """
        future = _shared_executor.submit(
            framework_call
        )

        return float(
            future.result(
                timeout=deepeval_timeout
            )
        )

    try:
        score = execute_with_retry(
            invoke_with_timeout,
            evaluator="groundedness",
            framework="DeepEval",
            conversation_id=conversation_id,
            max_retries=3,
            initial_delay=2.0,
            deadline=deadline,
        )

        return {
            "status": "success",
            "score": float(score),
        }

    except Exception as exc:
        logger.warning(
            "DeepEval faithfulness evaluation failed for %s: %s",
            conversation_id,
            exc,
        )

        return {
            "status": "failed",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Groundedness Evaluator
# ---------------------------------------------------------------------------

class GroundednessEvaluator(BaseEvaluator):
    """
    Groundedness / hallucination evaluator.

    Context-backed:
        - Custom LLM judge
        - TruLens comparison
        - DeepEval comparison

    Context-free:
        - Custom LLM judge
        - TruLens not applicable
        - DeepEval not applicable
    """

    name: str = "groundedness"

    def __init__(self) -> None:
        self._judge = LLMJudge()

        logger.info(
            "GroundednessEvaluator initialized."
        )

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        eval_input: EvaluationInput,
    ) -> EvaluationResult:
        try:
            if (
                eval_input.conversation_type
                == ConversationType.CONTEXT_BACKED
            ):
                return self._evaluate_context_backed(
                    eval_input
                )

            return self._evaluate_context_free(
                eval_input
            )

        except Exception as exc:
            logger.error(
                "GroundednessEvaluator failed for %s: %s",
                eval_input.conversation_id,
                exc,
            )

            error_status = classify_exception(
                exc
            )

            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=20.0,
                status=error_status,
                sub_scores={},
                feedback=(
                    "Groundedness evaluation failed "
                    f"with error: {exc}"
                ),
                flagged=True,
            )

    # ------------------------------------------------------------------
    # Context-backed
    # ------------------------------------------------------------------

    def _evaluate_context_backed(
        self,
        eval_input: EvaluationInput,
    ) -> EvaluationResult:
        logger.debug(
            "Evaluating groundedness (context-backed) for '%s'",
            eval_input.conversation_id,
        )

        context = (
            eval_input.retrieved_context
            or ""
        )

        response = eval_input.dave_response
        deadline = eval_input.deadline

        # Outer orchestration futures use the separate orchestration pool.
        future_custom = _orchestration_executor.submit(
            self._run_custom_context_backed_judge,
            eval_input,
        )

        future_trulens = _orchestration_executor.submit(
            _run_trulens_groundedness,
            context,
            response,
            eval_input.conversation_id,
            deadline,
        )

        future_deepeval = _orchestration_executor.submit(
            _run_deepeval_faithfulness,
            eval_input.user_query,
            response,
            context,
            eval_input.conversation_id,
            deadline,
        )

        parsed_json: dict[str, Any] = {}
        raw_text = ""

        trulens_res: dict[str, Any] = {}
        deepeval_res: dict[str, Any] = {}

        custom_exc: Exception | None = None

        # --------------------------------------------------------------
        # Custom judge
        # --------------------------------------------------------------

        try:
            remaining = self._remaining_timeout(
                deadline
            )

            parsed_json, raw_text = (
                future_custom.result(
                    timeout=remaining
                )
            )

        except Exception as exc:
            custom_exc = exc

            logger.error(
                "Groundedness custom judge failed for %s: %s",
                eval_input.conversation_id,
                exc,
            )

        # --------------------------------------------------------------
        # TruLens
        # --------------------------------------------------------------

        try:
            remaining = self._remaining_timeout(
                deadline
            )

            trulens_res = future_trulens.result(
                timeout=remaining
            )

        except Exception as exc:
            logger.error(
                "Groundedness TruLens failed for %s: %s",
                eval_input.conversation_id,
                exc,
            )

            trulens_res = {
                "status": "failed",
                "error": str(exc),
            }

        # --------------------------------------------------------------
        # DeepEval
        # --------------------------------------------------------------

        try:
            remaining = self._remaining_timeout(
                deadline
            )

            deepeval_res = future_deepeval.result(
                timeout=remaining
            )

        except Exception as exc:
            logger.error(
                "Groundedness DeepEval failed for %s: %s",
                eval_input.conversation_id,
                exc,
            )

            deepeval_res = {
                "status": "failed",
                "error": str(exc),
            }

        # --------------------------------------------------------------
        # Custom judge failure
        # --------------------------------------------------------------

        if not parsed_json:
            error_status = "failed"

            feedback = (
                "Groundedness custom judge call failed."
            )

            if custom_exc is not None:
                error_status = classify_exception(
                    custom_exc
                )

                feedback = (
                    "Groundedness custom judge call "
                    f"failed with error: {custom_exc}"
                )

            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=_MAX_SCORE_CONTEXT_BACKED,
                status=error_status,
                sub_scores={},
                feedback=feedback,
                flagged=True,
            )

        # --------------------------------------------------------------
        # Extract custom scores
        # --------------------------------------------------------------

        consistency_raw = self._extract_score(
            parsed_json,
            "internal_consistency",
        )

        overconfidence_raw = self._extract_score(
            parsed_json,
            "overconfidence",
        )

        hallucination_raw = self._extract_score(
            parsed_json,
            "hallucination_risk",
        )

        consistency_score = (
            consistency_raw / 5.0
        ) * 6.0

        overconfidence_score = (
            overconfidence_raw / 5.0
        ) * 6.0

        hallucination_score = (
            hallucination_raw / 5.0
        ) * 8.0

        sub_scores: dict[str, Any] = {
            "internal_consistency": round(
                consistency_score,
                2,
            ),
            "overconfidence": round(
                overconfidence_score,
                2,
            ),
            "hallucination_risk": round(
                hallucination_score,
                2,
            ),
        }

        # --------------------------------------------------------------
        # External framework comparison results
        # --------------------------------------------------------------

        self._add_trulens_result(
            sub_scores,
            trulens_res,
        )

        self._add_deepeval_result(
            sub_scores,
            deepeval_res,
        )

        # --------------------------------------------------------------
        # Official score
        # --------------------------------------------------------------

        total_score = round(
            consistency_score
            + overconfidence_score
            + hallucination_score,
            2,
        )

        feedback = (
            self._build_context_backed_feedback(
                parsed_json,
                sub_scores,
            )
        )

        flagged = (
            total_score
            < (_MAX_SCORE_CONTEXT_BACKED * 0.5)
        )

        return EvaluationResult(
            evaluator_name=self.name,
            conversation_id=eval_input.conversation_id,
            score=total_score,
            max_score=_MAX_SCORE_CONTEXT_BACKED,
            percentage=round(
                (
                    total_score
                    / _MAX_SCORE_CONTEXT_BACKED
                )
                * 100.0,
                2,
            ),
            sub_scores=sub_scores,
            feedback=feedback,
            flagged=flagged,
        )

    # ------------------------------------------------------------------
    # Custom context-backed judge
    # ------------------------------------------------------------------

    def _run_custom_context_backed_judge(
        self,
        eval_input: EvaluationInput,
    ) -> tuple[dict[str, Any], str]:

        chat_section = ""

        if eval_input.chat_history:
            chat_section = (
                "## Conversation History\n"
                f"{eval_input.chat_history}"
            )

        user_prompt = (
            _USER_PROMPT_CONTEXT_BACKED.format(
                user_query=eval_input.user_query,
                dave_response=eval_input.dave_response,
                retrieved_context=(
                    eval_input.retrieved_context
                    or ""
                ),
                chat_history_section=chat_section,
            )
        )

        return self._judge.call_with_json(
            _SYSTEM_PROMPT_CONTEXT_BACKED,
            user_prompt,
            evaluator=self.name,
            conversation_id=(
                eval_input.conversation_id
            ),
            response_schema=GroundednessSchema,
            deadline=eval_input.deadline,
        )

    # ------------------------------------------------------------------
    # Context-backed feedback
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_backed_feedback(
        parsed: dict[str, Any],
        sub_scores: dict[str, Any],
    ) -> str:

        parts: list[str] = []

        for metric_key, label in [
            (
                "internal_consistency",
                "Internal Consistency",
            ),
            (
                "overconfidence",
                "Overconfidence",
            ),
            (
                "hallucination_risk",
                "Hallucination Risk",
            ),
        ]:
            entry = parsed.get(
                metric_key,
                {},
            )

            if (
                isinstance(entry, dict)
                and entry.get("reasoning")
            ):
                parts.append(
                    f"{label} "
                    f"({entry.get('score', '?')}/5): "
                    f"{entry['reasoning']}"
                )

        overall = parsed.get(
            "overall_reasoning",
            "",
        )

        if overall:
            parts.append(
                f"Overall Assessment: {overall}"
            )

        parts.append(
            "Sub-scores: "
            f"Consistency="
            f"{sub_scores.get('internal_consistency', 0)}/6.0, "
            f"Overconfidence="
            f"{sub_scores.get('overconfidence', 0)}/6.0, "
            f"Hallucination Risk="
            f"{sub_scores.get('hallucination_risk', 0)}/8.0"
        )

        trulens_status = sub_scores.get(
            "trulens_status",
            "unknown",
        )

        if (
            trulens_status == "success"
            and "trulens_score" in sub_scores
        ):
            parts.append(
                "TruLens Groundedness "
                "(comparison): "
                f"{float(sub_scores['trulens_score']):.4f}"
            )

        elif trulens_status == "failed":
            parts.append(
                "TruLens Groundedness "
                "(comparison): FAILED. "
                f"Error: "
                f"{sub_scores.get('trulens_error', '')}"
            )

        else:
            parts.append(
                "TruLens Groundedness "
                "(comparison): NOT APPLICABLE. "
                f"Reason: "
                f"{sub_scores.get('trulens_reason', '')}"
            )

        deepeval_status = sub_scores.get(
            "deepeval_status",
            "unknown",
        )

        if (
            deepeval_status == "success"
            and "deepeval_score" in sub_scores
        ):
            parts.append(
                "DeepEval Faithfulness "
                "(comparison): "
                f"{float(sub_scores['deepeval_score']):.4f}"
            )

        elif deepeval_status == "failed":
            parts.append(
                "DeepEval Faithfulness "
                "(comparison): FAILED. "
                f"Error: "
                f"{sub_scores.get('deepeval_error', '')}"
            )

        else:
            parts.append(
                "DeepEval Faithfulness "
                "(comparison): NOT APPLICABLE. "
                f"Reason: "
                f"{sub_scores.get('deepeval_reason', '')}"
            )

        return (
            "\n\n".join(parts)
            if parts
            else "No feedback generated."
        )

    # ------------------------------------------------------------------
    # Context-free evaluation
    # ------------------------------------------------------------------

    def _evaluate_context_free(
        self,
        eval_input: EvaluationInput,
    ) -> EvaluationResult:

        logger.debug(
            "Evaluating groundedness (context-free) "
            "for '%s'",
            eval_input.conversation_id,
        )

        chat_section = ""

        if eval_input.chat_history:
            chat_section = (
                "## Conversation History\n"
                f"{eval_input.chat_history}"
            )

        user_prompt = (
            _USER_PROMPT_CONTEXT_FREE.format(
                user_query=eval_input.user_query,
                dave_response=eval_input.dave_response,
                chat_history_section=chat_section,
            )
        )

        parsed_json, raw_text = (
            self._judge.call_with_json(
                _SYSTEM_PROMPT_CONTEXT_FREE,
                user_prompt,
                evaluator=self.name,
                conversation_id=(
                    eval_input.conversation_id
                ),
                response_schema=GroundednessSchema,
                deadline=eval_input.deadline,
            )
        )

        if not parsed_json:
            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=_MAX_SCORE_CONTEXT_FREE,
                status="failed",
                sub_scores={},
                feedback=(
                    "Groundedness custom judge "
                    "call failed."
                ),
                flagged=True,
            )

        consistency_raw = self._extract_score(
            parsed_json,
            "internal_consistency",
        )

        overconfidence_raw = self._extract_score(
            parsed_json,
            "overconfidence",
        )

        hallucination_raw = self._extract_score(
            parsed_json,
            "hallucination_risk",
        )

        consistency_score = (
            consistency_raw / 5.0
        ) * 6.0

        overconfidence_score = (
            overconfidence_raw / 5.0
        ) * 6.0

        hallucination_score = (
            hallucination_raw / 5.0
        ) * 8.0

        sub_scores: dict[str, Any] = {
            "internal_consistency": round(
                consistency_score,
                2,
            ),
            "overconfidence": round(
                overconfidence_score,
                2,
            ),
            "hallucination_risk": round(
                hallucination_score,
                2,
            ),
            "trulens_status": "not_applicable",
            "trulens_reason": (
                "No retrieved context available"
            ),
            "deepeval_status": "not_applicable",
            "deepeval_reason": (
                "No retrieved context available"
            ),
        }

        total_score = round(
            consistency_score
            + overconfidence_score
            + hallucination_score,
            2,
        )

        feedback = (
            self._build_context_free_feedback(
                parsed_json,
                sub_scores,
            )
        )

        flagged = (
            total_score
            < (_MAX_SCORE_CONTEXT_FREE * 0.5)
        )

        return EvaluationResult(
            evaluator_name=self.name,
            conversation_id=eval_input.conversation_id,
            score=total_score,
            max_score=_MAX_SCORE_CONTEXT_FREE,
            percentage=round(
                (
                    total_score
                    / _MAX_SCORE_CONTEXT_FREE
                )
                * 100.0,
                2,
            ),
            sub_scores=sub_scores,
            feedback=feedback,
            flagged=flagged,
        )

    # ------------------------------------------------------------------
    # Context-free feedback
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_free_feedback(
        parsed: dict[str, Any],
        sub_scores: dict[str, Any],
    ) -> str:

        parts: list[str] = []

        for metric_key, label in [
            (
                "internal_consistency",
                "Internal Consistency",
            ),
            (
                "overconfidence",
                "Overconfidence",
            ),
            (
                "hallucination_risk",
                "Hallucination Risk",
            ),
        ]:
            entry = parsed.get(
                metric_key,
                {},
            )

            if (
                isinstance(entry, dict)
                and entry.get("reasoning")
            ):
                parts.append(
                    f"{label} "
                    f"({entry.get('score', '?')}/5): "
                    f"{entry['reasoning']}"
                )

        overall = parsed.get(
            "overall_reasoning",
            "",
        )

        if overall:
            parts.append(
                f"Overall: {overall}"
            )

        parts.append(
            "Sub-scores: "
            f"Consistency="
            f"{sub_scores.get('internal_consistency', 0)}/6.0, "
            f"Overconfidence="
            f"{sub_scores.get('overconfidence', 0)}/6.0, "
            f"Hallucination Risk="
            f"{sub_scores.get('hallucination_risk', 0)}/8.0"
        )

        parts.append(
            "TruLens Groundedness "
            "(comparison): NOT APPLICABLE. "
            "Reason: No retrieved context available"
        )

        parts.append(
            "DeepEval Faithfulness "
            "(comparison): NOT APPLICABLE. "
            "Reason: No retrieved context available"
        )

        return (
            "\n\n".join(parts)
            if parts
            else "No feedback generated."
        )

    # ------------------------------------------------------------------
    # External framework result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_trulens_result(
        sub_scores: dict[str, Any],
        result: dict[str, Any],
    ) -> None:

        status = result.get(
            "status"
        )

        if status == "success":
            score = result.get(
                "score"
            )

            if isinstance(
                score,
                (int, float),
            ):
                sub_scores[
                    "trulens_status"
                ] = "success"

                sub_scores[
                    "trulens_score"
                ] = round(
                    float(score),
                    4,
                )

            else:
                sub_scores[
                    "trulens_status"
                ] = "failed"

                sub_scores[
                    "trulens_error"
                ] = (
                    "TruLens reported success "
                    "but returned an invalid score."
                )

        elif status == "failed":
            sub_scores[
                "trulens_status"
            ] = "failed"

            sub_scores[
                "trulens_error"
            ] = result.get(
                "error",
                "Unknown TruLens error",
            )

        else:
            sub_scores[
                "trulens_status"
            ] = "not_applicable"

            sub_scores[
                "trulens_reason"
            ] = result.get(
                "reason",
                "No retrieved context available",
            )

    @staticmethod
    def _add_deepeval_result(
        sub_scores: dict[str, Any],
        result: dict[str, Any],
    ) -> None:

        status = result.get(
            "status"
        )

        if status == "success":
            score = result.get(
                "score"
            )

            if isinstance(
                score,
                (int, float),
            ):
                sub_scores[
                    "deepeval_status"
                ] = "success"

                sub_scores[
                    "deepeval_score"
                ] = round(
                    float(score),
                    4,
                )

            else:
                sub_scores[
                    "deepeval_status"
                ] = "failed"

                sub_scores[
                    "deepeval_error"
                ] = (
                    "DeepEval reported success "
                    "but returned an invalid score."
                )

        elif status == "failed":
            sub_scores[
                "deepeval_status"
            ] = "failed"

            sub_scores[
                "deepeval_error"
            ] = result.get(
                "error",
                "Unknown DeepEval error",
            )

        else:
            sub_scores[
                "deepeval_status"
            ] = "not_applicable"

            sub_scores[
                "deepeval_reason"
            ] = result.get(
                "reason",
                "No retrieved context available",
            )

    # ------------------------------------------------------------------
    # Remaining timeout
    # ------------------------------------------------------------------

    @staticmethod
    def _remaining_timeout(
        deadline: float | None,
    ) -> float | None:

        if deadline is None:
            return None

        return max(
            _MIN_FRAMEWORK_TIMEOUT,
            deadline - time.time(),
        )

    # ------------------------------------------------------------------
    # Score extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_score(
        parsed: dict[str, Any],
        key: str,
    ) -> int:

        entry = parsed.get(
            key
        )

        if entry is None:
            raise ValueError(
                f"Missing key '{key}' "
                "in LLM response."
            )

        if isinstance(
            entry,
            dict,
        ):
            raw = entry.get(
                "score"
            )

            if raw is None:
                raise ValueError(
                    f"Missing 'score' field "
                    f"inside key '{key}' "
                    "in LLM response."
                )

        else:
            raw = entry

        try:
            score = int(
                raw
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Non-integer score for "
                f"'{key}': {raw}"
            ) from exc

        if not 1 <= score <= 5:
            raise ValueError(
                f"Out-of-range score for "
                f"'{key}': {score}"
            )

        return score