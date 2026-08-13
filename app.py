from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from evaluation_pipeline.utils.cors_config import get_allowed_origins
import uvicorn
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pydantic import BaseModel
from datetime import datetime
from evaluation_pipeline.data.models import ConversationRecord, EvaluationResult
from evaluation_pipeline.data.dataset_builder import DatasetBuilder
from evaluation_pipeline.evaluators.response_quality_evaluator import ResponseQualityEvaluator
from evaluation_pipeline.evaluators.groundedness_evaluator import GroundednessEvaluator
from evaluation_pipeline.evaluators.safety_evaluator import SafetyEvaluator
from evaluation_pipeline.evaluators.intent_evaluator import IntentEvaluator
from evaluation_pipeline.evaluators.memory_evaluator import MemoryEvaluator
from evaluation_pipeline.aggregator.score_aggregator import ScoreAggregator
from evaluation_pipeline.utils.error_handler import classify_exception

app = FastAPI(title="Evaluation Pipeline API")

# Module-level persistent executor — shared across all requests.
# CRITICAL: We NEVER use `with ThreadPoolExecutor() as executor:` inside a
# request handler because the context-manager __exit__ calls
# `executor.shutdown(wait=True)`, which blocks until ALL submitted worker
# threads finish. That means a slow/hung evaluator worker can delay the
# HTTP response far beyond the configured EVALUATION_REQUEST_TIMEOUT.
#
# Solution: use a persistently alive executor. Request handlers submit
# futures and call future.result(timeout=...) per-future, then return.
# Slow background workers continue independently, release their semaphore
# slot via the controlled_concurrency finally-block, then terminate.
_EVAL_EXECUTOR = ThreadPoolExecutor(max_workers=10)

# CORS — environment-driven origin policy (see evaluation_pipeline/utils/cors_config.py)
allowed_origins = get_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"--> Incoming request: {request.method} {request.url}", flush=True)
    try:
        response = await call_next(request)
        print(f"<-- Request finished: {request.method} {request.url} - Status: {response.status_code}", flush=True)
        return response
    except Exception as e:
        print(f"<-- Request failed: {request.method} {request.url} - Error: {str(e)}", flush=True)
        raise

@app.get("/")
def read_root():
    return {"message": "Welcome to the Dave Evaluation Pipeline API. Use /evaluate to evaluate a conversation."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "dave-evaluation-pipeline"
    }

@app.post("/evaluate")
def run_evaluation(record: ConversationRecord):
    """
    Evaluates a single conversation record on-the-fly and returns the JSON report.
    """
    # 1. Build dataset
    builder = DatasetBuilder()
    inputs = builder.build([record])
    if not inputs:
        raise HTTPException(status_code=400, detail="Failed to parse input record")
    
    evaluation_input = inputs[0]
    
    request_timeout = float(os.getenv("EVALUATION_REQUEST_TIMEOUT", "45.0"))
    deadline = time.time() + request_timeout
    evaluation_input.deadline = deadline

    try:
        # 2. Run evaluators concurrently with timeouts.
        # NOTE: We use the module-level _EVAL_EXECUTOR (NOT a `with` block).
        # A `with ThreadPoolExecutor() as executor:` context manager calls
        # shutdown(wait=True) on __exit__, which blocks the HTTP response
        # until ALL worker threads finish — even if they have already timed out.
        # Using the persistent _EVAL_EXECUTOR avoids this entirely.
        rq_evaluator = ResponseQualityEvaluator()
        gd_evaluator = GroundednessEvaluator()
        sf_evaluator = SafetyEvaluator()
        it_evaluator = IntentEvaluator()
        me_evaluator = MemoryEvaluator()

        # Submit all evaluators immediately so they run in parallel
        future_rq = _EVAL_EXECUTOR.submit(rq_evaluator.evaluate, evaluation_input)
        future_gd = _EVAL_EXECUTOR.submit(gd_evaluator.evaluate, evaluation_input)
        future_sf = _EVAL_EXECUTOR.submit(sf_evaluator.evaluate, evaluation_input)
        future_it = _EVAL_EXECUTOR.submit(it_evaluator.evaluate, evaluation_input)
        future_me = _EVAL_EXECUTOR.submit(me_evaluator.evaluate, evaluation_input)

        # Collect results sequentially, each with a shrinking remaining budget.
        # After each future.result(timeout=...) call, we either have the result
        # or an exception — the request handler does NOT wait for the underlying
        # worker thread to finish. Background workers continue independently,
        # release their semaphore slot in the controlled_concurrency finally-block,
        # and terminate when their own downstream timeout (GEMINI_TIMEOUT etc.) fires.

        # Response Quality
        try:
            remaining = max(0.05, deadline - time.time())
            rq_res = future_rq.result(timeout=remaining)
        except Exception as exc:
            logger.error("ResponseQualityEvaluator failed: %s", exc, exc_info=True)
            error_status = classify_exception(exc)
            rq_res = EvaluationResult(
                evaluator_name=rq_evaluator.name,
                conversation_id=evaluation_input.conversation_id,
                score=None,
                max_score=20.0,
                status=error_status,
                sub_scores={},
                feedback=f"Evaluation failed with error: {exc}",
                flagged=True,
            )

        # Groundedness
        try:
            remaining = max(0.05, deadline - time.time())
            gd_res = future_gd.result(timeout=remaining)
        except Exception as exc:
            logger.error("GroundednessEvaluator failed: %s", exc, exc_info=True)
            error_status = classify_exception(exc)
            gd_res = EvaluationResult(
                evaluator_name=gd_evaluator.name,
                conversation_id=evaluation_input.conversation_id,
                score=None,
                max_score=20.0,
                status=error_status,
                sub_scores={},
                feedback=f"Evaluation failed with error: {exc}",
                flagged=True,
            )

        # Safety
        try:
            remaining = max(0.05, deadline - time.time())
            sf_res = future_sf.result(timeout=remaining)
        except Exception as exc:
            logger.error("SafetyEvaluator failed: %s", exc, exc_info=True)
            error_status = classify_exception(exc)
            sf_res = EvaluationResult(
                evaluator_name=sf_evaluator.name,
                conversation_id=evaluation_input.conversation_id,
                score=None,
                max_score=20.0,
                status=error_status,
                sub_scores={},
                feedback=f"Evaluation failed with error: {exc}",
                flagged=True,
            )

        # Intent
        try:
            remaining = max(0.05, deadline - time.time())
            it_res = future_it.result(timeout=remaining)
        except Exception as exc:
            logger.error("IntentEvaluator failed: %s", exc, exc_info=True)
            error_status = classify_exception(exc)
            it_res = EvaluationResult(
                evaluator_name=it_evaluator.name,
                conversation_id=evaluation_input.conversation_id,
                score=None,
                max_score=20.0,
                status=error_status,
                sub_scores={},
                feedback=f"Evaluation failed with error: {exc}",
                flagged=True,
            )

        # Memory
        try:
            remaining = max(0.05, deadline - time.time())
            me_res = future_me.result(timeout=remaining)
        except Exception as exc:
            logger.error("MemoryEvaluator failed: %s", exc, exc_info=True)
            error_status = classify_exception(exc)
            me_res = EvaluationResult(
                evaluator_name=me_evaluator.name,
                conversation_id=evaluation_input.conversation_id,
                score=None,
                max_score=20.0,
                applicable=False,
                status=error_status,
                sub_scores={},
                feedback=f"Evaluation failed with error: {exc}",
                flagged=True,
            )
        
        # 3. Aggregate
        aggregator = ScoreAggregator()
        report = aggregator.aggregate_dataset(
            inputs=[evaluation_input],
            rq_results=[rq_res],
            gd_results=[gd_res],
            safety_results=[sf_res],
            intent_results=[it_res],
            memory_results=[me_res]
        )
        
        return report
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
