from concurrent.futures import ThreadPoolExecutor, as_completed, Future, TimeoutError
import logging
import os
import time

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from evaluation_pipeline.aggregator.score_aggregator import ScoreAggregator
from evaluation_pipeline.data.dataset_builder import DatasetBuilder
from evaluation_pipeline.data.models import ConversationRecord, EvaluationResult
from evaluation_pipeline.evaluators.groundedness_evaluator import GroundednessEvaluator
from evaluation_pipeline.evaluators.intent_evaluator import IntentEvaluator
from evaluation_pipeline.evaluators.memory_evaluator import MemoryEvaluator
from evaluation_pipeline.evaluators.response_quality_evaluator import ResponseQualityEvaluator
from evaluation_pipeline.evaluators.safety_evaluator import SafetyEvaluator
from evaluation_pipeline.utils.cors_config import get_allowed_origins
from evaluation_pipeline.utils.error_handler import classify_exception

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Evaluation Pipeline API",
)

# ---------------------------------------------------------------------------
# Persistent evaluator executor
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# Do NOT create a ThreadPoolExecutor with "with ..." inside /evaluate.
# The context manager performs shutdown(wait=True), which can block the
# HTTP response while slow evaluator workers are still running.
#
# This executor remains alive for the process lifetime.
# ---------------------------------------------------------------------------

_EVAL_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(
        os.getenv("EVALUATION_EXECUTOR_WORKERS", "10")
    )
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

allowed_origins = get_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(
        "--> Incoming request: %s %s",
        request.method,
        request.url,
    )

    try:
        response = await call_next(request)

        logger.info(
            "<-- Request finished: %s %s - Status: %s",
            request.method,
            request.url,
            response.status_code,
        )

        return response

    except Exception as exc:
        logger.exception(
            "<-- Request failed: %s %s - Error: %s",
            request.method,
            request.url,
            exc,
        )
        raise


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "message": (
            "Welcome to the Dave Evaluation Pipeline API. "
            "Use /evaluate to evaluate a conversation."
        )
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "dave-evaluation-pipeline",
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _remaining_time(deadline: float) -> float:
    """
    Return remaining request time.

    Never return zero/negative time to Future.result().
    A tiny floor prevents invalid timeout values while still allowing the
    caller to recognize that the request deadline is effectively exhausted.
    """
    return max(
        0.05,
        deadline - time.time(),
    )


def _build_failed_result(
    *,
    evaluator_name: str,
    conversation_id: str,
    status: str,
    error: str,
    applicable: bool = True,
) -> EvaluationResult:
    """
    Build a consistent evaluator failure/timeout result.
    """
    kwargs = {
        "evaluator_name": evaluator_name,
        "conversation_id": conversation_id,
        "score": None,
        "max_score": 20.0,
        "status": status,
        "sub_scores": {},
        "feedback": error,
        "flagged": True,
    }

    if not applicable:
        kwargs["applicable"] = False

    return EvaluationResult(**kwargs)


def _collect_evaluator_results(
    *,
    futures: dict[Future, tuple[str, object]],
    conversation_id: str,
    deadline: float,
) -> dict[str, EvaluationResult]:
    """
    Collect ALL evaluator futures against ONE shared request deadline.

    The critical difference from the previous implementation:

        BAD:
            wait RQ -> then wait GD -> then wait Safety -> ...

        GOOD:
            submit everything immediately
            wait for all completions concurrently
            once the global deadline expires, mark only unfinished evaluators
            as timed out

    Therefore a fast evaluator cannot consume the entire remaining deadline
    before groundedness gets a chance to return.
    """

    results: dict[str, EvaluationResult] = {}

    # Reverse lookup:
    # Future -> evaluator name / evaluator instance
    future_metadata = dict(futures)

    if not future_metadata:
        return results

    try:
        remaining = _remaining_time(deadline)

        for future in as_completed(
            future_metadata,
            timeout=remaining,
        ):
            evaluator_name, evaluator_instance = future_metadata[future]

            try:
                results[evaluator_name] = future.result()

            except Exception as exc:
                logger.exception(
                    "%s failed for conversation_id=%s",
                    evaluator_name,
                    conversation_id,
                )

                results[evaluator_name] = _build_failed_result(
                    evaluator_name=evaluator_instance.name,
                    conversation_id=conversation_id,
                    status=classify_exception(exc),
                    error=(
                        f"{evaluator_name} evaluation failed: "
                        f"{exc}"
                    ),
                    applicable=(
                        evaluator_name != "memory_and_continuity"
                    ),
                )

    except TimeoutError:
        logger.warning(
            "Evaluation request deadline reached for conversation_id=%s",
            conversation_id,
        )

    # Any future that did not finish before the request deadline is now
    # represented explicitly as a timeout.
    for future, (
        evaluator_name,
        evaluator_instance,
    ) in future_metadata.items():

        if evaluator_name in results:
            continue

        if future.done():
            try:
                results[evaluator_name] = future.result()
                continue

            except Exception as exc:
                logger.exception(
                    "%s completed with an exception for conversation_id=%s",
                    evaluator_name,
                    conversation_id,
                )

                results[evaluator_name] = _build_failed_result(
                    evaluator_name=evaluator_instance.name,
                    conversation_id=conversation_id,
                    status=classify_exception(exc),
                    error=(
                        f"{evaluator_name} evaluation failed: "
                        f"{exc}"
                    ),
                    applicable=(
                        evaluator_name != "memory_and_continuity"
                    ),
                )

                continue

        # Future is still running / pending.
        logger.error(
            "Evaluator timed out: evaluator=%s conversation_id=%s",
            evaluator_name,
            conversation_id,
        )

        results[evaluator_name] = _build_failed_result(
            evaluator_name=evaluator_instance.name,
            conversation_id=conversation_id,
            status="timeout",
            error=(
                f"{evaluator_name} evaluation timed out "
                f"before the request deadline."
            ),
            applicable=(
                evaluator_name != "memory_and_continuity"
            ),
        )

    return results


# ---------------------------------------------------------------------------
# Main evaluation endpoint
# ---------------------------------------------------------------------------

@app.post("/evaluate")
def run_evaluation(record: ConversationRecord):
    """
    Evaluate one conversation record and return the complete evaluation report.
    """

    # -----------------------------------------------------------------------
    # 1. Build EvaluationInput
    # -----------------------------------------------------------------------

    builder = DatasetBuilder()

    try:
        inputs = builder.build([record])
    except Exception as exc:
        logger.exception(
            "Failed to build evaluation input for conversation_id=%s",
            getattr(record, "conversation_id", "unknown"),
        )

        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse input record: {exc}",
        ) from exc

    if not inputs:
        raise HTTPException(
            status_code=400,
            detail="Failed to parse input record",
        )

    evaluation_input = inputs[0]

    # -----------------------------------------------------------------------
    # 2. Establish ONE request deadline
    # -----------------------------------------------------------------------

    try:
        request_timeout = float(
            os.getenv(
                "EVALUATION_REQUEST_TIMEOUT",
                "45.0",
            )
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Invalid EVALUATION_REQUEST_TIMEOUT configuration.",
        ) from exc

    if request_timeout <= 0:
        raise HTTPException(
            status_code=500,
            detail="EVALUATION_REQUEST_TIMEOUT must be greater than zero.",
        )

    deadline = time.time() + request_timeout

    evaluation_input.deadline = deadline

    conversation_id = evaluation_input.conversation_id

    logger.info(
        "Starting evaluation request: conversation_id=%s timeout=%.2fs",
        conversation_id,
        request_timeout,
    )

    # -----------------------------------------------------------------------
    # 3. Create evaluator instances
    # -----------------------------------------------------------------------

    rq_evaluator = ResponseQualityEvaluator()
    gd_evaluator = GroundednessEvaluator()
    sf_evaluator = SafetyEvaluator()
    it_evaluator = IntentEvaluator()
    me_evaluator = MemoryEvaluator()

    # -----------------------------------------------------------------------
    # 4. Submit ALL evaluators immediately
    # -----------------------------------------------------------------------
    #
    # This is the critical reliability fix.
    #
    # All five evaluators start at approximately the same time.
    # We do NOT:
    #
    #   result RQ
    #   then result GD
    #   then result Safety
    #
    # because that allows one slow evaluator to consume the request budget
    # before the other evaluators get their result.
    # -----------------------------------------------------------------------

    future_rq = _EVAL_EXECUTOR.submit(
        rq_evaluator.evaluate,
        evaluation_input,
    )

    future_gd = _EVAL_EXECUTOR.submit(
        gd_evaluator.evaluate,
        evaluation_input,
    )

    future_sf = _EVAL_EXECUTOR.submit(
        sf_evaluator.evaluate,
        evaluation_input,
    )

    future_it = _EVAL_EXECUTOR.submit(
        it_evaluator.evaluate,
        evaluation_input,
    )

    future_me = _EVAL_EXECUTOR.submit(
        me_evaluator.evaluate,
        evaluation_input,
    )

    futures = {
        future_rq: (
            "response_quality",
            rq_evaluator,
        ),
        future_gd: (
            "groundedness",
            gd_evaluator,
        ),
        future_sf: (
            "safety",
            sf_evaluator,
        ),
        future_it: (
            "intent_understanding",
            it_evaluator,
        ),
        future_me: (
            "memory_and_continuity",
            me_evaluator,
        ),
    }

    # -----------------------------------------------------------------------
    # 5. Collect results concurrently against the shared deadline
    # -----------------------------------------------------------------------

    try:
        results = _collect_evaluator_results(
            futures=futures,
            conversation_id=conversation_id,
            deadline=deadline,
        )
    except Exception as exc:
        logger.exception(
            "Unexpected evaluator collection failure for conversation_id=%s",
            conversation_id,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Evaluation collection failed: {exc}",
        ) from exc

    # -----------------------------------------------------------------------
    # 6. Extract evaluator results
    # -----------------------------------------------------------------------

    rq_res = results.get(
        "response_quality"
    )

    if rq_res is None:
        rq_res = _build_failed_result(
            evaluator_name=rq_evaluator.name,
            conversation_id=conversation_id,
            status="timeout",
            error="Response Quality result was not produced.",
        )

    gd_res = results.get(
        "groundedness"
    )

    if gd_res is None:
        gd_res = _build_failed_result(
            evaluator_name=gd_evaluator.name,
            conversation_id=conversation_id,
            status="timeout",
            error="Groundedness result was not produced.",
        )

    sf_res = results.get(
        "safety"
    )

    if sf_res is None:
        sf_res = _build_failed_result(
            evaluator_name=sf_evaluator.name,
            conversation_id=conversation_id,
            status="timeout",
            error="Safety result was not produced.",
        )

    it_res = results.get(
        "intent_understanding"
    )

    if it_res is None:
        it_res = _build_failed_result(
            evaluator_name=it_evaluator.name,
            conversation_id=conversation_id,
            status="timeout",
            error="Intent result was not produced.",
        )

    me_res = results.get(
        "memory_and_continuity"
    )

    if me_res is None:
        me_res = _build_failed_result(
            evaluator_name=me_evaluator.name,
            conversation_id=conversation_id,
            status="timeout",
            error="Memory result was not produced.",
            applicable=False,
        )

    # -----------------------------------------------------------------------
    # 7. Aggregate final report
    # -----------------------------------------------------------------------

    try:
        aggregator = ScoreAggregator()

        report = aggregator.aggregate_dataset(
            inputs=[evaluation_input],
            rq_results=[rq_res],
            gd_results=[gd_res],
            safety_results=[sf_res],
            intent_results=[it_res],
            memory_results=[me_res],
        )

    except Exception as exc:
        logger.exception(
            "Score aggregation failed for conversation_id=%s",
            conversation_id,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Evaluation aggregation failed: {exc}",
        ) from exc

    logger.info(
        "Evaluation request completed: conversation_id=%s",
        conversation_id,
    )

    return report


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )