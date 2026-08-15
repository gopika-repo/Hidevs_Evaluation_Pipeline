from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    Future,
    TimeoutError,
)
import logging
import os
import time

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from evaluation_pipeline.aggregator.score_aggregator import ScoreAggregator
from evaluation_pipeline.data.dataset_builder import DatasetBuilder
from evaluation_pipeline.data.models import (
    ConversationRecord,
    EvaluationResult,
)
from evaluation_pipeline.evaluators.groundedness_evaluator import (
    GroundednessEvaluator,
)
from evaluation_pipeline.evaluators.intent_evaluator import IntentEvaluator
from evaluation_pipeline.evaluators.memory_evaluator import MemoryEvaluator
from evaluation_pipeline.evaluators.response_quality_evaluator import (
    ResponseQualityEvaluator,
)
from evaluation_pipeline.evaluators.safety_evaluator import SafetyEvaluator
from evaluation_pipeline.utils.cors_config import get_allowed_origins
from evaluation_pipeline.utils.error_handler import classify_exception


load_dotenv()


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Evaluation Pipeline API",
)


# ============================================================================
# PERSISTENT EVALUATOR EXECUTOR
# ============================================================================

try:
    _EVALUATION_EXECUTOR_WORKERS = max(
        5,
        int(
            os.getenv(
                "EVALUATION_EXECUTOR_WORKERS",
                "10",
            )
        ),
    )
except (TypeError, ValueError):
    _EVALUATION_EXECUTOR_WORKERS = 10


_EVAL_EXECUTOR = ThreadPoolExecutor(
    max_workers=_EVALUATION_EXECUTOR_WORKERS,
    thread_name_prefix="evaluation",
)


# ============================================================================
# CORS
# ============================================================================

allowed_origins = get_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()

    logger.info(
        "--> Incoming request: %s %s",
        request.method,
        request.url,
    )

    try:
        response = await call_next(request)

        duration = time.time() - start

        logger.info(
            "<-- Request finished: %s %s - Status: %s - Duration: %.3fs",
            request.method,
            request.url,
            response.status_code,
            duration,
        )

        return response

    except Exception as exc:
        duration = time.time() - start

        logger.exception(
            "<-- Request failed: %s %s - Duration: %.3fs - Error: %s",
            request.method,
            request.url,
            duration,
            exc,
        )

        raise


# ============================================================================
# HEALTH ENDPOINTS
# ============================================================================

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


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def _remaining_time(
    deadline: float,
) -> float:
    """
    Return the remaining request time.

    A small positive floor avoids passing zero/negative values to
    Future.result(), while the caller still treats an effectively
    exhausted deadline as expired.
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
    Build a consistent evaluator failure result.
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

    return EvaluationResult(
        **kwargs
    )


def _collect_evaluator_results(
    *,
    futures: dict[
        Future,
        tuple[str, object],
    ],
    conversation_id: str,
    deadline: float,
) -> dict[str, EvaluationResult]:
    """
    Collect evaluator futures against one shared request deadline.

    All evaluators are started concurrently.

    Once the HTTP request deadline expires:
        - completed evaluators are returned normally;
        - unfinished evaluators receive a timeout result;
        - already-running evaluator threads are NOT forcibly cancelled.

    This prevents the API request lifecycle from interfering with background
    evaluator cleanup.
    """

    results: dict[str, EvaluationResult] = {}

    if not futures:
        return results

    future_metadata = dict(
        futures
    )

    try:
        remaining = _remaining_time(
            deadline
        )

        if remaining <= 0.05:
            raise TimeoutError()

        for future in as_completed(
            future_metadata,
            timeout=remaining,
        ):
            evaluator_name, evaluator_instance = future_metadata[
                future
            ]

            try:
                result = future.result()

                results[
                    evaluator_name
                ] = result

                logger.info(
                    "Evaluator completed: "
                    "conversation_id=%s evaluator=%s",
                    conversation_id,
                    evaluator_name,
                )

            except Exception as exc:
                logger.exception(
                    "Evaluator failed: "
                    "conversation_id=%s evaluator=%s error=%s",
                    conversation_id,
                    evaluator_name,
                    exc,
                )

                results[
                    evaluator_name
                ] = _build_failed_result(
                    evaluator_name=evaluator_instance.name,
                    conversation_id=conversation_id,
                    status=classify_exception(
                        exc
                    ),
                    error=(
                        f"{evaluator_name} evaluation failed: "
                        f"{exc}"
                    ),
                    applicable=(
                        evaluator_name
                        != "memory_and_continuity"
                    ),
                )

    except TimeoutError:
        logger.warning(
            "Evaluation request deadline reached: "
            "conversation_id=%s",
            conversation_id,
        )

    # ------------------------------------------------------------------------
    # Handle anything that did not finish before the request deadline.
    # ------------------------------------------------------------------------

    for future, (
        evaluator_name,
        evaluator_instance,
    ) in future_metadata.items():

        if evaluator_name in results:
            continue

        # A future may have completed right after as_completed() timed out.
        if future.done():

            try:
                result = future.result()

                results[
                    evaluator_name
                ] = result

                logger.info(
                    "Evaluator completed after collection timeout: "
                    "conversation_id=%s evaluator=%s",
                    conversation_id,
                    evaluator_name,
                )

                continue

            except Exception as exc:
                logger.exception(
                    "Evaluator completed with exception: "
                    "conversation_id=%s evaluator=%s error=%s",
                    conversation_id,
                    evaluator_name,
                    exc,
                )

                results[
                    evaluator_name
                ] = _build_failed_result(
                    evaluator_name=evaluator_instance.name,
                    conversation_id=conversation_id,
                    status=classify_exception(
                        exc
                    ),
                    error=(
                        f"{evaluator_name} evaluation failed: "
                        f"{exc}"
                    ),
                    applicable=(
                        evaluator_name
                        != "memory_and_continuity"
                    ),
                )

                continue

        # --------------------------------------------------------------------
        # Evaluator is still running.
        #
        # DO NOT cancel it here.
        #
        # The evaluator may already have an active Gemini/TruLens/DeepEval
        # request. Let that work finish independently.
        # --------------------------------------------------------------------

        logger.warning(
            "Evaluator timed out for current HTTP request: "
            "conversation_id=%s evaluator=%s",
            conversation_id,
            evaluator_name,
        )

        results[
            evaluator_name
        ] = _build_failed_result(
            evaluator_name=evaluator_instance.name,
            conversation_id=conversation_id,
            status="timeout",
            error=(
                f"{evaluator_name} evaluation exceeded the "
                f"request deadline."
            ),
            applicable=(
                evaluator_name
                != "memory_and_continuity"
            ),
        )

    return results


# ============================================================================
# MAIN EVALUATION ENDPOINT
# ============================================================================

@app.post("/evaluate")
def run_evaluation(
    record: ConversationRecord,
):
    """
    Evaluate one conversation record and return the complete evaluation report.
    """

    request_start = time.time()

    # =========================================================================
    # 1. BUILD EVALUATION INPUT
    # =========================================================================

    builder = DatasetBuilder()

    try:
        inputs = builder.build(
            [record]
        )

    except Exception as exc:
        logger.exception(
            "Failed to build evaluation input for conversation_id=%s",
            getattr(
                record,
                "conversation_id",
                "unknown",
            ),
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Failed to parse input record: {exc}"
            ),
        ) from exc

    if not inputs:
        raise HTTPException(
            status_code=400,
            detail="Failed to parse input record",
        )

    evaluation_input = inputs[0]

    conversation_id = evaluation_input.conversation_id

    # =========================================================================
    # 2. REQUEST TIMEOUT
    # =========================================================================
    #
    # Groundedness is the most expensive evaluator because it can involve:
    #
    #     custom Gemini judge
    #     TruLens
    #     DeepEval
    #
    # Therefore the old 45-second default is too aggressive for a cold
    # production request.
    #
    # It remains configurable through:
    #
    #     EVALUATION_REQUEST_TIMEOUT
    #
    # =========================================================================

    try:
        request_timeout = float(
            os.getenv(
                "EVALUATION_REQUEST_TIMEOUT",
                "90.0",
            )
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid EVALUATION_REQUEST_TIMEOUT "
                "configuration."
            ),
        ) from exc

    if request_timeout <= 0:

        raise HTTPException(
            status_code=500,
            detail=(
                "EVALUATION_REQUEST_TIMEOUT must be "
                "greater than zero."
            ),
        )

    deadline = (
        time.time()
        + request_timeout
    )

    evaluation_input.deadline = deadline

    logger.info(
        "Starting evaluation: "
        "conversation_id=%s request_timeout=%.2fs",
        conversation_id,
        request_timeout,
    )

    # =========================================================================
    # 3. CREATE EVALUATORS
    # =========================================================================

    construction_start = time.time()

    rq_evaluator = ResponseQualityEvaluator()
    gd_evaluator = GroundednessEvaluator()
    sf_evaluator = SafetyEvaluator()
    it_evaluator = IntentEvaluator()
    me_evaluator = MemoryEvaluator()

    logger.info(
        "Evaluator initialization completed: "
        "conversation_id=%s duration=%.3fs",
        conversation_id,
        time.time() - construction_start,
    )

    # =========================================================================
    # 4. SUBMIT ALL EVALUATORS IMMEDIATELY
    # =========================================================================

    submission_start = time.time()

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

    logger.info(
        "All evaluators submitted: "
        "conversation_id=%s duration=%.3fs",
        conversation_id,
        time.time() - submission_start,
    )

    # =========================================================================
    # 5. COLLECT RESULTS
    # =========================================================================

    try:
        results = _collect_evaluator_results(
            futures=futures,
            conversation_id=conversation_id,
            deadline=deadline,
        )

    except Exception as exc:
        logger.exception(
            "Unexpected evaluator collection failure: "
            "conversation_id=%s error=%s",
            conversation_id,
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Evaluation collection failed: {exc}"
            ),
        ) from exc

    # =========================================================================
    # 6. GUARANTEE RESULT OBJECTS EXIST
    # =========================================================================

    rq_res = results.get(
        "response_quality"
    )

    if rq_res is None:

        rq_res = _build_failed_result(
            evaluator_name=rq_evaluator.name,
            conversation_id=conversation_id,
            status="timeout",
            error=(
                "Response Quality result "
                "was not produced."
            ),
        )

    gd_res = results.get(
        "groundedness"
    )

    if gd_res is None:

        gd_res = _build_failed_result(
            evaluator_name=gd_evaluator.name,
            conversation_id=conversation_id,
            status="timeout",
            error=(
                "Groundedness result "
                "was not produced."
            ),
        )

    sf_res = results.get(
        "safety"
    )

    if sf_res is None:

        sf_res = _build_failed_result(
            evaluator_name=sf_evaluator.name,
            conversation_id=conversation_id,
            status="timeout",
            error=(
                "Safety result "
                "was not produced."
            ),
        )

    it_res = results.get(
        "intent_understanding"
    )

    if it_res is None:

        it_res = _build_failed_result(
            evaluator_name=it_evaluator.name,
            conversation_id=conversation_id,
            status="timeout",
            error=(
                "Intent result "
                "was not produced."
            ),
        )

    me_res = results.get(
        "memory_and_continuity"
    )

    if me_res is None:

        me_res = _build_failed_result(
            evaluator_name=me_evaluator.name,
            conversation_id=conversation_id,
            status="timeout",
            error=(
                "Memory result "
                "was not produced."
            ),
            applicable=False,
        )

    # =========================================================================
    # 7. AGGREGATE FINAL REPORT
    # =========================================================================

    try:

        aggregation_start = time.time()

        aggregator = ScoreAggregator()

        report = aggregator.aggregate_dataset(
            inputs=[
                evaluation_input
            ],
            rq_results=[
                rq_res
            ],
            gd_results=[
                gd_res
            ],
            safety_results=[
                sf_res
            ],
            intent_results=[
                it_res
            ],
            memory_results=[
                me_res
            ],
        )

        logger.info(
            "Score aggregation completed: "
            "conversation_id=%s duration=%.3fs",
            conversation_id,
            time.time() - aggregation_start,
        )

    except Exception as exc:
        logger.exception(
            "Score aggregation failed: "
            "conversation_id=%s error=%s",
            conversation_id,
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Evaluation aggregation failed: {exc}"
            ),
        ) from exc

    # =========================================================================
    # 8. FINAL LOGGING
    # =========================================================================

    total_duration = (
        time.time()
        - request_start
    )

    logger.info(
        "Evaluation request completed: "
        "conversation_id=%s total_duration=%.3fs",
        conversation_id,
        total_duration,
    )

    return report


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )