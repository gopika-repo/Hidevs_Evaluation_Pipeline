"""
Groundedness / Hallucination Evaluator — Phase 1

Context-Backed
--------------
Official score:
    Internal Consistency -> 6
    Overconfidence       -> 6
    Hallucination Risk   -> 8

Comparison metrics:
    TruLens Groundedness
    DeepEval Faithfulness

TruLens and DeepEval do NOT contribute to the official score.

Context-Free
------------
Official score:
    Internal Consistency -> 6
    Overconfidence       -> 6
    Hallucination Risk   -> 8

TruLens and DeepEval are not applicable.

TruLens / Google GenAI compatibility
-------------------------------------
The installed environment uses:

    trulens-core              2.12.0
    trulens-feedback          2.12.0
    trulens-providers-google  2.12.0
    google-genai              2.18.1

The installed TruLens Google provider passes a Pydantic response model
through Google's typed `response_schema` path.

That path has produced errors such as:

    Unknown name "additional_properties"

and:

    response_schema.required[0]: property is not defined

The compatibility provider below keeps TruLens' groundedness logic but
overrides only the Gemini completion call.

When TruLens requests structured output:

    1. Obtain the real Pydantic JSON schema.
    2. Resolve local $ref / $defs references.
    3. Validate that required fields actually exist in properties.
    4. Send the resulting schema through Gemini's raw JSON-schema field:
           response_json_schema
       rather than:
           response_schema

This avoids the incompatible typed-schema conversion layer.

No mock data is used.
No API key is hardcoded.
No evaluation score is hardcoded.
The Gemini model is read from GEMINI_MODEL_NAME.
The API key is read from GOOGLE_API_KEY.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Sequence

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


# ============================================================================
# EXECUTOR
# ============================================================================

try:
    _SHARED_EXECUTOR_WORKERS = max(
        8,
        int(
            os.getenv(
                "GROUNDEDNESS_EXECUTOR_WORKERS",
                "64",
            )
        ),
    )
except (TypeError, ValueError):
    _SHARED_EXECUTOR_WORKERS = 64


_shared_executor = ThreadPoolExecutor(
    max_workers=_SHARED_EXECUTOR_WORKERS,
    thread_name_prefix="groundedness",
)


# ============================================================================
# CONSTANTS
# ============================================================================

_MAX_SCORE_CONTEXT_BACKED = 20.0
_MAX_SCORE_CONTEXT_FREE = 20.0

_MIN_FRAMEWORK_TIMEOUT = 0.05

_DEFAULT_TRULENS_TIMEOUT = 45.0
_DEFAULT_DEEPEVAL_TIMEOUT = 45.0

_FRAMEWORK_MAX_RETRIES = 3
_FRAMEWORK_RETRY_DELAY = 2.0


# ============================================================================
# PROMPTS — CONTEXT BACKED
# ============================================================================

_SYSTEM_PROMPT_CONTEXT_BACKED = """\
You are a STRICT expert judge specializing in groundedness and hallucination
detection.

Evaluate the AI assistant response against the retrieved source context.

You must score exactly three dimensions from 1 to 5.

1. INTERNAL CONSISTENCY
Does the response contradict itself or the supplied conversation context?

5 = Fully consistent
4 = Minor issue
3 = Some inconsistency
2 = Significant inconsistency
1 = Clear contradiction

2. OVERCONFIDENCE
Does the response present uncertain or unverifiable information as established
fact?

5 = Fully calibrated
4 = Mostly calibrated
3 = Some unnecessary certainty
2 = Frequently overconfident
1 = Strongly presents unsupported claims as facts

3. HALLUCINATION RISK
Are the response's claims actually supported by the retrieved source context?

Penalize:
- fabricated facts
- unsupported names
- unsupported numbers
- unsupported dates
- unsupported URLs
- unsupported locations
- unsupported statistics
- direct contradictions of the retrieved context

5 = Fully grounded
4 = Mostly grounded
3 = Some unsupported details
2 = Significant unsupported content
1 = Clearly fabricated or contradicted

Return ONLY JSON matching this exact structure:

{
  "internal_consistency": {
    "score": 1,
    "reasoning": "specific assessment"
  },
  "overconfidence": {
    "score": 1,
    "reasoning": "specific assessment"
  },
  "hallucination_risk": {
    "score": 1,
    "reasoning": "specific assessment"
  },
  "overall_reasoning": "summary"
}

Do not include markdown.
Do not include additional fields.
"""


_USER_PROMPT_CONTEXT_BACKED = """\
Evaluate the AI assistant response against the retrieved source context.

## USER QUERY
{user_query}

## AI ASSISTANT RESPONSE
{dave_response}

## RETRIEVED SOURCE CONTEXT
{retrieved_context}

{chat_history_section}

Score all three dimensions from 1 to 5.

Return ONLY valid JSON.
"""


# ============================================================================
# PROMPTS — CONTEXT FREE
# ============================================================================

_SYSTEM_PROMPT_CONTEXT_FREE = """\
You are a STRICT expert judge specializing in hallucination and
overconfidence detection.

No retrieved source context is available.

Evaluate exactly three dimensions from 1 to 5.

1. INTERNAL CONSISTENCY
Does the response contradict itself or information explicitly present in
conversation history?

5 = Fully consistent
1 = Clear contradiction

2. OVERCONFIDENCE
Does the response present uncertain or unverifiable information as fact?

5 = Fully calibrated
1 = Strongly overconfident

3. HALLUCINATION RISK
Does the response contain fabricated or suspiciously invented specifics?

Look for:
- invented statistics
- fabricated names
- fabricated dates
- fake URLs
- made-up references
- unsupported precise claims

5 = No hallucination indicators
1 = Clearly fabricated content

Return ONLY JSON matching this exact structure:

{
  "internal_consistency": {
    "score": 1,
    "reasoning": "specific assessment"
  },
  "overconfidence": {
    "score": 1,
    "reasoning": "specific assessment"
  },
  "hallucination_risk": {
    "score": 1,
    "reasoning": "specific assessment"
  },
  "overall_reasoning": "summary"
}

Do not include markdown.
Do not include additional fields.
"""


_USER_PROMPT_CONTEXT_FREE = """\
Evaluate the following AI assistant response.

No retrieved source context is available.

## USER QUERY
{user_query}

## AI ASSISTANT RESPONSE
{dave_response}

{chat_history_section}

Score all three dimensions from 1 to 5.

Return ONLY valid JSON.
"""


# ============================================================================
# TIMEOUT HELPERS
# ============================================================================

def _calculate_framework_timeout(
    env_name: str,
    default: float,
    deadline: float | None,
) -> float:
    """
    Return a bounded timeout for an external framework.
    """

    try:
        configured = float(
            os.getenv(
                env_name,
                str(default),
            )
        )
    except (TypeError, ValueError):
        configured = default

    if configured <= 0:
        configured = default

    if deadline is None:
        return configured

    remaining = deadline - time.time()

    return min(
        configured,
        max(
            _MIN_FRAMEWORK_TIMEOUT,
            remaining,
        ),
    )


def _remaining_deadline(
    deadline: float | None,
) -> float | None:
    """
    Return remaining evaluation time.

    None means there is no request deadline.
    """

    if deadline is None:
        return None

    return max(
        _MIN_FRAMEWORK_TIMEOUT,
        deadline - time.time(),
    )


# ============================================================================
# TRULENS / GEMINI JSON-SCHEMA COMPATIBILITY
# ============================================================================

def _resolve_json_schema(
    schema: Any,
) -> Any:
    """
    Resolve Pydantic JSON-schema $ref/$defs structures into a standalone
    schema suitable for direct Gemini JSON-schema submission.

    This function uses the actual schema supplied by TruLens/Pydantic.

    It does not invent fields, values, evaluation scores, or test data.
    """

    if not isinstance(schema, dict):
        return schema

    definitions = schema.get(
        "$defs",
        {},
    )

    if not isinstance(
        definitions,
        dict,
    ):
        definitions = {}

    resolving: set[str] = set()

    def resolve(
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            list,
        ):
            return [
                resolve(item)
                for item in value
            ]

        if not isinstance(
            value,
            dict,
        ):
            return value

        # ------------------------------------------------------------
        # Resolve local references.
        # ------------------------------------------------------------

        ref = value.get(
            "$ref"
        )

        if (
            isinstance(ref, str)
            and ref.startswith("#/$defs/")
        ):
            ref_name = ref[
                len("#/$defs/"):
            ]

            if ref_name in resolving:
                # Prevent recursive-reference loops.
                return {}

            referenced_schema = definitions.get(
                ref_name
            )

            if isinstance(
                referenced_schema,
                dict,
            ):

                resolving.add(
                    ref_name
                )

                resolved_target = resolve(
                    referenced_schema
                )

                resolving.remove(
                    ref_name
                )

                # Merge local metadata from the ref node.
                if isinstance(
                    resolved_target,
                    dict,
                ):

                    merged = dict(
                        resolved_target
                    )

                    for key, item in value.items():
                        if key != "$ref":
                            merged[key] = item

                    return resolve(
                        merged
                    )

                return resolved_target

        # ------------------------------------------------------------
        # Normal recursive processing.
        # ------------------------------------------------------------

        result: dict[str, Any] = {}

        for key, item in value.items():

            # $defs is no longer required after inlining references.
            if key in {
                "$defs",
                "$schema",
                "$id",
            }:
                continue

            result[
                key
            ] = resolve(
                item
            )

        # ------------------------------------------------------------
        # Normalize object schemas.
        # ------------------------------------------------------------

        schema_type = result.get(
            "type"
        )

        if isinstance(
            schema_type,
            str,
        ):
            normalized_type = schema_type.lower()
        else:
            normalized_type = schema_type

        if normalized_type == "object":

            properties = result.get(
                "properties"
            )

            if not isinstance(
                properties,
                dict,
            ):
                properties = {}

            cleaned_properties: dict[str, Any] = {}

            for property_name, property_schema in properties.items():
                cleaned_properties[
                    str(property_name)
                ] = resolve(
                    property_schema
                )

            result[
                "properties"
            ] = cleaned_properties

            # --------------------------------------------------------
            # Gemini requires every name in required to be present in
            # properties.
            # --------------------------------------------------------

            required = result.get(
                "required"
            )

            if isinstance(
                required,
                list,
            ):

                valid_required = [
                    str(name)
                    for name in required
                    if str(name)
                    in cleaned_properties
                ]

                if valid_required:
                    result[
                        "required"
                    ] = valid_required
                else:
                    result.pop(
                        "required",
                        None,
                    )

            # Preserve or derive property ordering from actual
            # properties. This is supported by Gemini.
            if cleaned_properties:
                result[
                    "propertyOrdering"
                ] = list(
                    cleaned_properties.keys()
                )

        return result

    resolved_schema = resolve(
        schema
    )

    if not isinstance(
        resolved_schema,
        dict,
    ):
        raise ValueError(
            "Resolved TruLens JSON schema is not an object."
        )

    return resolved_schema


def _schema_from_response_format(
    response_format: Any,
) -> dict[str, Any] | None:
    """
    Extract the real JSON schema from the response model passed by TruLens.

    Supports Pydantic v2 model_json_schema() and compatible schema() APIs.
    """

    if response_format is None:
        return None

    # Pydantic v2.
    model_json_schema = getattr(
        response_format,
        "model_json_schema",
        None,
    )

    if callable(
        model_json_schema
    ):

        schema = model_json_schema()

        if isinstance(
            schema,
            dict,
        ):
            return _resolve_json_schema(
                schema
            )

    # Compatibility fallback.
    schema_method = getattr(
        response_format,
        "schema",
        None,
    )

    if callable(
        schema_method
    ):

        schema = schema_method()

        if isinstance(
            schema,
            dict,
        ):
            return _resolve_json_schema(
                schema
            )

    raise ValueError(
        "TruLens supplied a response format that does not expose "
        "a usable JSON schema."
    )


# ============================================================================
# COMPATIBLE TRULENS GOOGLE PROVIDER
# ============================================================================

class _CompatibleTruLensGoogleMixin:
    """
    Override only TruLens' Gemini completion implementation.

    TruLens continues to own:

        - groundedness sentence splitting
        - trivial statement filtering
        - groundedness prompting
        - score normalization
        - reason construction

    Only the low-level Google GenAI structured-output request is replaced.
    """

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[
            Sequence[dict[str, Any]]
        ] = None,
        response_format: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[str]:

        endpoint = getattr(
            self,
            "endpoint",
            None,
        )

        if endpoint is None:
            raise RuntimeError(
                "TruLens Google endpoint is not initialized."
            )

        client = getattr(
            endpoint,
            "client",
            None,
        )

        if client is None:
            raise RuntimeError(
                "TruLens Google endpoint does not expose "
                "a configured GenAI client."
            )

        # ------------------------------------------------------------
        # Build the Gemini request content from TruLens messages.
        # ------------------------------------------------------------

        contents: list[str] = []
        system_instruction = ""

        if messages is not None:

            for message in messages:

                if not isinstance(
                    message,
                    dict,
                ):
                    continue

                role = str(
                    message.get(
                        "role",
                        "",
                    )
                )

                content = message.get(
                    "content",
                    "",
                )

                if role == "system":

                    system_instruction = str(
                        content
                    )

                elif role in {
                    "user",
                    "assistant",
                    "model",
                }:

                    contents.append(
                        str(content)
                    )

        elif prompt is not None:

            contents.append(
                str(prompt)
            )

        if not contents:
            contents = [""]

        # ------------------------------------------------------------
        # Gemini GenerateContentConfig.
        # ------------------------------------------------------------

        from google.genai import types

        config_kwargs: dict[str, Any] = {}

        temperature = kwargs.get(
            "temperature"
        )

        if temperature is not None:
            config_kwargs[
                "temperature"
            ] = float(
                temperature
            )

        if system_instruction:
            config_kwargs[
                "system_instruction"
            ] = system_instruction

        raw_schema = (
            _schema_from_response_format(
                response_format
            )
            if response_format is not None
            else None
        )

        if raw_schema is not None:

            # ========================================================
            # CRITICAL FIX
            #
            # Do NOT use:
            #
            #     response_schema=raw_schema
            #
            # because google-genai treats that as its typed Schema
            # conversion path.
            #
            # Send the actual raw JSON schema using:
            #
            #     response_json_schema
            #
            # This bypasses the Pydantic/Schema translation layer.
            # ========================================================

            config_kwargs[
                "response_mime_type"
            ] = "application/json"

            config_kwargs[
                "response_json_schema"
            ] = raw_schema

        config = types.GenerateContentConfig(
            **config_kwargs
        )

        model_name = getattr(
            self,
            "model_engine",
            None,
        )

        if not model_name:
            raise RuntimeError(
                "TruLens model_engine is not configured."
            )

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        text = getattr(
            response,
            "text",
            None,
        )

        if text is None:
            raise RuntimeError(
                "Gemini returned no response text."
            )

        return str(
            text
        )


def _create_compatible_trulens_provider(
    client: Any,
    model_engine: str,
) -> Any:
    """
    Create a real TruLens Google provider subclass with only its Gemini
    completion method overridden.

    The installed TruLens Google provider remains the parent implementation.
    """

    from trulens.providers.google import Google

    class CompatibleTruLensGoogle(
        _CompatibleTruLensGoogleMixin,
        Google,
    ):
        pass

    provider = CompatibleTruLensGoogle(
        client=client,
        model_engine=model_engine,
    )

    if not isinstance(
        provider,
        Google,
    ):
        raise RuntimeError(
            "Compatible TruLens provider is not a TruLens Google provider."
        )

    return provider


# ============================================================================
# TRULENS
# ============================================================================

def _run_trulens_groundedness(
    context: str,
    response: str,
    conversation_id: str = "unknown",
    deadline: float | None = None,
) -> dict[str, Any]:
    """
    Run TruLens groundedness as a comparison metric.

    TruLens does NOT contribute to the official groundedness score.
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

    timeout = _calculate_framework_timeout(
        env_name="TRULENS_TIMEOUT",
        default=_DEFAULT_TRULENS_TIMEOUT,
        deadline=deadline,
    )

    def _invoke_trulens() -> float:

        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                f"Unable to import Google GenAI client: {exc}"
            ) from exc

        try:
            from trulens.providers.google import Google
        except ImportError as exc:
            raise RuntimeError(
                f"Unable to import TruLens Google provider: {exc}"
            ) from exc

        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not configured."
            )

        model_name = os.getenv(
            "GEMINI_MODEL_NAME"
        )

        if not model_name:
            raise ValueError(
                "GEMINI_MODEL_NAME is not configured."
            )

        client = genai.Client(
            api_key=api_key
        )

        provider = _create_compatible_trulens_provider(
            client=client,
            model_engine=model_name,
        )

        if not isinstance(
            provider,
            Google,
        ):
            raise RuntimeError(
                "Failed to initialize compatible TruLens Google provider."
            )

        with controlled_concurrency(
            "groundedness",
            "TruLens",
            conversation_id,
        ):

            result = (
                provider.groundedness_measure_with_cot_reasons(
                    source=context,
                    statement=response,
                )
            )

        if isinstance(
            result,
            tuple,
        ):
            score = result[0]
        else:
            score = result

        if score is None:
            raise ValueError(
                "TruLens returned no score."
            )

        try:

            score_float = float(
                score
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "TruLens returned an invalid score: "
                f"{score!r}"
            ) from exc

        if not 0.0 <= score_float <= 1.0:

            raise ValueError(
                "TruLens returned out-of-range score: "
                f"{score_float}"
            )

        return score_float

    def _submit_and_wait() -> float:

        future = _shared_executor.submit(
            _invoke_trulens
        )

        return float(
            future.result(
                timeout=timeout
            )
        )

    try:

        score = execute_with_retry(
            _submit_and_wait,
            evaluator="groundedness",
            framework="TruLens",
            conversation_id=conversation_id,
            max_retries=_FRAMEWORK_MAX_RETRIES,
            initial_delay=_FRAMEWORK_RETRY_DELAY,
            deadline=deadline,
        )

        return {
            "status": "success",
            "score": float(
                score
            ),
        }

    except Exception as exc:

        logger.warning(
            "TruLens groundedness evaluation failed "
            "for %s: %s",
            conversation_id,
            exc,
        )

        return {
            "status": "failed",
            "error": str(
                exc
            ),
        }


# ============================================================================
# DEEPEVAL
# ============================================================================

def _run_deepeval_faithfulness(
    user_query: str,
    response: str,
    context: str,
    conversation_id: str = "unknown",
    deadline: float | None = None,
) -> dict[str, Any]:
    """
    Run DeepEval faithfulness as a comparison metric.

    DeepEval does NOT contribute to the official groundedness score.
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

    timeout = _calculate_framework_timeout(
        env_name="DEEPEVAL_TIMEOUT",
        default=_DEFAULT_DEEPEVAL_TIMEOUT,
        deadline=deadline,
    )

    def _invoke_deepeval() -> float:

        try:

            from deepeval.metrics import (
                FaithfulnessMetric,
            )

            from deepeval.models import (
                GeminiModel,
            )

            from deepeval.test_case import (
                LLMTestCase,
            )

        except ImportError as exc:

            raise RuntimeError(
                f"Unable to import DeepEval components: {exc}"
            ) from exc

        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not configured."
            )

        model_name = os.getenv(
            "GEMINI_MODEL_NAME"
        )

        if not model_name:
            raise ValueError(
                "GEMINI_MODEL_NAME is not configured."
            )

        model = GeminiModel(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
        )

        test_case = LLMTestCase(
            input=user_query,
            actual_output=response,
            retrieval_context=[
                context
            ],
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

            metric.measure(
                test_case
            )

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

        score_float = float(
            score
        )

        if not 0.0 <= score_float <= 1.0:

            raise ValueError(
                "DeepEval returned out-of-range score: "
                f"{score_float}"
            )

        return score_float

    def _submit_and_wait() -> float:

        future = _shared_executor.submit(
            _invoke_deepeval
        )

        return float(
            future.result(
                timeout=timeout
            )
        )

    try:

        score = execute_with_retry(
            _submit_and_wait,
            evaluator="groundedness",
            framework="DeepEval",
            conversation_id=conversation_id,
            max_retries=_FRAMEWORK_MAX_RETRIES,
            initial_delay=_FRAMEWORK_RETRY_DELAY,
            deadline=deadline,
        )

        return {
            "status": "success",
            "score": float(
                score
            ),
        }

    except Exception as exc:

        logger.warning(
            "DeepEval faithfulness evaluation failed "
            "for %s: %s",
            conversation_id,
            exc,
        )

        return {
            "status": "failed",
            "error": str(
                exc
            ),
        }


# ============================================================================
# GROUNDEDNESS EVALUATOR
# ============================================================================

class GroundednessEvaluator(BaseEvaluator):
    """
    Groundedness / hallucination evaluator.
    """

    name: str = "groundedness"

    def __init__(self) -> None:

        self._judge = LLMJudge()

        logger.info(
            "GroundednessEvaluator initialized."
        )

    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================

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
                exc_info=True,
            )

            status = classify_exception(
                exc
            )

            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=(
                    eval_input.conversation_id
                ),
                score=None,
                max_score=20.0,
                status=status,
                sub_scores={},
                feedback=(
                    "Groundedness evaluation failed "
                    f"with error: {exc}"
                ),
                flagged=True,
            )

    # ========================================================================
    # CONTEXT-BACKED
    # ========================================================================

    def _evaluate_context_backed(
        self,
        eval_input: EvaluationInput,
    ) -> EvaluationResult:

        logger.debug(
            "Evaluating groundedness (context-backed) "
            "for '%s'",
            eval_input.conversation_id,
        )

        context = (
            eval_input.retrieved_context
            or ""
        )

        response = (
            eval_input.dave_response
        )

        deadline = (
            eval_input.deadline
        )

        future_custom = _shared_executor.submit(
            self._run_custom_context_backed_judge,
            eval_input,
        )

        future_trulens = _shared_executor.submit(
            _run_trulens_groundedness,
            context,
            response,
            eval_input.conversation_id,
            deadline,
        )

        future_deepeval = _shared_executor.submit(
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

        # --------------------------------------------------------------------
        # Custom judge
        # --------------------------------------------------------------------

        try:

            parsed_json, raw_text = (
                future_custom.result(
                    timeout=_remaining_deadline(
                        deadline
                    )
                )
            )

        except Exception as exc:

            custom_exc = exc

            logger.error(
                "Groundedness custom judge failed "
                "for %s: %s",
                eval_input.conversation_id,
                exc,
            )

        # --------------------------------------------------------------------
        # TruLens
        # --------------------------------------------------------------------

        try:

            trulens_res = (
                future_trulens.result(
                    timeout=_remaining_deadline(
                        deadline
                    )
                )
            )

        except Exception as exc:

            logger.error(
                "Groundedness TruLens failed "
                "for %s: %s",
                eval_input.conversation_id,
                exc,
            )

            trulens_res = {
                "status": "failed",
                "error": str(
                    exc
                ),
            }

        # --------------------------------------------------------------------
        # DeepEval
        # --------------------------------------------------------------------

        try:

            deepeval_res = (
                future_deepeval.result(
                    timeout=_remaining_deadline(
                        deadline
                    )
                )
            )

        except Exception as exc:

            logger.error(
                "Groundedness DeepEval failed "
                "for %s: %s",
                eval_input.conversation_id,
                exc,
            )

            deepeval_res = {
                "status": "failed",
                "error": str(
                    exc
                ),
            }

        # --------------------------------------------------------------------
        # Custom judge failure
        # --------------------------------------------------------------------

        if not parsed_json:

            status = "failed"

            feedback = (
                "Groundedness custom judge call failed."
            )

            if custom_exc is not None:

                status = classify_exception(
                    custom_exc
                )

                feedback = (
                    "Groundedness custom judge call "
                    f"failed with error: {custom_exc}"
                )

            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=(
                    eval_input.conversation_id
                ),
                score=None,
                max_score=(
                    _MAX_SCORE_CONTEXT_BACKED
                ),
                status=status,
                sub_scores={},
                feedback=feedback,
                flagged=True,
            )

        # --------------------------------------------------------------------
        # Extract scores
        # --------------------------------------------------------------------

        try:

            consistency_raw = (
                self._extract_score(
                    parsed_json,
                    "internal_consistency",
                )
            )

            overconfidence_raw = (
                self._extract_score(
                    parsed_json,
                    "overconfidence",
                )
            )

            hallucination_raw = (
                self._extract_score(
                    parsed_json,
                    "hallucination_risk",
                )
            )

        except Exception as exc:

            logger.error(
                "Invalid groundedness judge output "
                "for %s: %s",
                eval_input.conversation_id,
                exc,
                exc_info=True,
            )

            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=(
                    eval_input.conversation_id
                ),
                score=None,
                max_score=(
                    _MAX_SCORE_CONTEXT_BACKED
                ),
                status="failed",
                sub_scores={},
                feedback=(
                    "Groundedness judge returned invalid "
                    f"structured output: {exc}"
                ),
                flagged=True,
            )

        # --------------------------------------------------------------------
        # Official score
        # --------------------------------------------------------------------

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

        self._add_trulens_result(
            sub_scores,
            trulens_res,
        )

        self._add_deepeval_result(
            sub_scores,
            deepeval_res,
        )

        total_score = round(
            consistency_score
            + overconfidence_score
            + hallucination_score,
            2,
        )

        percentage = round(
            (
                total_score
                / _MAX_SCORE_CONTEXT_BACKED
            ) * 100.0,
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
            < (
                _MAX_SCORE_CONTEXT_BACKED
                * 0.5
            )
        )

        return EvaluationResult(
            evaluator_name=self.name,
            conversation_id=(
                eval_input.conversation_id
            ),
            score=total_score,
            max_score=(
                _MAX_SCORE_CONTEXT_BACKED
            ),
            percentage=percentage,
            sub_scores=sub_scores,
            feedback=feedback,
            flagged=flagged,
        )

    # ========================================================================
    # CUSTOM CONTEXT-BACKED JUDGE
    # ========================================================================

    def _run_custom_context_backed_judge(
        self,
        eval_input: EvaluationInput,
    ) -> tuple[dict[str, Any], str]:

        chat_history_section = ""

        if eval_input.chat_history:

            chat_history_section = (
                "## Conversation History\n"
                f"{eval_input.chat_history}"
            )

        user_prompt = (
            _USER_PROMPT_CONTEXT_BACKED.format(
                user_query=(
                    eval_input.user_query
                ),
                dave_response=(
                    eval_input.dave_response
                ),
                retrieved_context=(
                    eval_input.retrieved_context
                    or ""
                ),
                chat_history_section=(
                    chat_history_section
                ),
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
            deadline=(
                eval_input.deadline
            ),
        )

    # ========================================================================
    # CONTEXT-BACKED FEEDBACK
    # ========================================================================

    @staticmethod
    def _build_context_backed_feedback(
        parsed: dict[str, Any],
        sub_scores: dict[str, Any],
    ) -> str:

        parts: list[str] = []

        for metric_key, label in (
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
        ):

            entry = parsed.get(
                metric_key,
                {},
            )

            if (
                isinstance(
                    entry,
                    dict,
                )
                and entry.get(
                    "reasoning"
                )
            ):

                parts.append(
                    f"{label} "
                    f"({entry.get('score', '?')}/5): "
                    f"{entry['reasoning']}"
                )

        overall_reasoning = (
            parsed.get(
                "overall_reasoning"
            )
            or parsed.get(
                "explanation"
            )
            or ""
        )

        if overall_reasoning:

            parts.append(
                "Overall Assessment: "
                f"{overall_reasoning}"
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

        # --------------------------------------------------------------------
        # TruLens
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # DeepEval
        # --------------------------------------------------------------------

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

    # ========================================================================
    # CONTEXT-FREE
    # ========================================================================

    def _evaluate_context_free(
        self,
        eval_input: EvaluationInput,
    ) -> EvaluationResult:

        logger.debug(
            "Evaluating groundedness (context-free) "
            "for '%s'",
            eval_input.conversation_id,
        )

        chat_history_section = ""

        if eval_input.chat_history:

            chat_history_section = (
                "## Conversation History\n"
                f"{eval_input.chat_history}"
            )

        user_prompt = (
            _USER_PROMPT_CONTEXT_FREE.format(
                user_query=(
                    eval_input.user_query
                ),
                dave_response=(
                    eval_input.dave_response
                ),
                chat_history_section=(
                    chat_history_section
                ),
            )
        )

        parsed_json, _raw_text = (
            self._judge.call_with_json(
                _SYSTEM_PROMPT_CONTEXT_FREE,
                user_prompt,
                evaluator=self.name,
                conversation_id=(
                    eval_input.conversation_id
                ),
                response_schema=GroundednessSchema,
                deadline=(
                    eval_input.deadline
                ),
            )
        )

        if not parsed_json:

            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=(
                    eval_input.conversation_id
                ),
                score=None,
                max_score=(
                    _MAX_SCORE_CONTEXT_FREE
                ),
                status="failed",
                sub_scores={},
                feedback=(
                    "Groundedness custom judge "
                    "call failed."
                ),
                flagged=True,
            )

        try:

            consistency_raw = (
                self._extract_score(
                    parsed_json,
                    "internal_consistency",
                )
            )

            overconfidence_raw = (
                self._extract_score(
                    parsed_json,
                    "overconfidence",
                )
            )

            hallucination_raw = (
                self._extract_score(
                    parsed_json,
                    "hallucination_risk",
                )
            )

        except Exception as exc:

            logger.error(
                "Invalid context-free groundedness output "
                "for %s: %s",
                eval_input.conversation_id,
                exc,
                exc_info=True,
            )

            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=(
                    eval_input.conversation_id
                ),
                score=None,
                max_score=(
                    _MAX_SCORE_CONTEXT_FREE
                ),
                status="failed",
                sub_scores={},
                feedback=(
                    "Groundedness judge returned invalid "
                    f"structured output: {exc}"
                ),
                flagged=True,
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

        total_score = round(
            consistency_score
            + overconfidence_score
            + hallucination_score,
            2,
        )

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

        percentage = round(
            (
                total_score
                / _MAX_SCORE_CONTEXT_FREE
            ) * 100.0,
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
            < (
                _MAX_SCORE_CONTEXT_FREE
                * 0.5
            )
        )

        return EvaluationResult(
            evaluator_name=self.name,
            conversation_id=(
                eval_input.conversation_id
            ),
            score=total_score,
            max_score=(
                _MAX_SCORE_CONTEXT_FREE
            ),
            percentage=percentage,
            sub_scores=sub_scores,
            feedback=feedback,
            flagged=flagged,
        )

    # ========================================================================
    # CONTEXT-FREE FEEDBACK
    # ========================================================================

    @staticmethod
    def _build_context_free_feedback(
        parsed: dict[str, Any],
        sub_scores: dict[str, Any],
    ) -> str:

        parts: list[str] = []

        for metric_key, label in (
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
        ):

            entry = parsed.get(
                metric_key,
                {},
            )

            if (
                isinstance(
                    entry,
                    dict,
                )
                and entry.get(
                    "reasoning"
                )
            ):

                parts.append(
                    f"{label} "
                    f"({entry.get('score', '?')}/5): "
                    f"{entry['reasoning']}"
                )

        overall_reasoning = (
            parsed.get(
                "overall_reasoning"
            )
            or parsed.get(
                "explanation"
            )
            or ""
        )

        if overall_reasoning:

            parts.append(
                "Overall Assessment: "
                f"{overall_reasoning}"
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

    # ========================================================================
    # TRULENS RESULT
    # ========================================================================

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
                    float(
                        score
                    ),
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

    # ========================================================================
    # DEEPEVAL RESULT
    # ========================================================================

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
                    float(
                        score
                    ),
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

    # ========================================================================
    # SCORE EXTRACTION
    # ========================================================================

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