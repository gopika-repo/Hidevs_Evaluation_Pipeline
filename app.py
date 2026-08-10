from fastapi import FastAPI, BackgroundTasks, HTTPException
import uvicorn
import os
import logging
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

app = FastAPI(title="Evaluation Pipeline API")

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

@app.post("/evaluate")
def run_evaluation(record: ConversationRecord):
    """
    Evaluates a single conversation record on-the-fly and returns the JSON report.
    """
    # 1. Build dataset
    builder = DatasetBuilder()
    inputs = builder.build([record])
    if not inputs:
        return {"error": "Failed to parse input record"}
    
    evaluation_input = inputs[0]
    
    try:
        # 2. Run evaluators concurrently with timeouts
        rq_evaluator = ResponseQualityEvaluator()
        gd_evaluator = GroundednessEvaluator()
        sf_evaluator = SafetyEvaluator()
        it_evaluator = IntentEvaluator()
        me_evaluator = MemoryEvaluator()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_rq = executor.submit(rq_evaluator.evaluate, evaluation_input)
            future_gd = executor.submit(gd_evaluator.evaluate, evaluation_input)
            future_sf = executor.submit(sf_evaluator.evaluate, evaluation_input)
            future_it = executor.submit(it_evaluator.evaluate, evaluation_input)
            future_me = executor.submit(me_evaluator.evaluate, evaluation_input)
            
            # Response Quality
            try:
                rq_res = future_rq.result(timeout=45.0)
            except Exception as exc:
                logger.error("ResponseQualityEvaluator failed: %s", exc, exc_info=True)
                rq_res = EvaluationResult(
                    evaluator_name=rq_evaluator.name,
                    conversation_id=evaluation_input.conversation_id,
                    score=0.0,
                    max_score=20.0,
                    status="failed",
                    sub_scores={},
                    feedback=f"Evaluation failed with error: {exc}",
                    flagged=True,
                )

            # Groundedness
            try:
                gd_res = future_gd.result(timeout=45.0)
            except Exception as exc:
                logger.error("GroundednessEvaluator failed: %s", exc, exc_info=True)
                gd_res = EvaluationResult(
                    evaluator_name=gd_evaluator.name,
                    conversation_id=evaluation_input.conversation_id,
                    score=0.0,
                    max_score=20.0,
                    status="failed",
                    sub_scores={},
                    feedback=f"Evaluation failed with error: {exc}",
                    flagged=True,
                )

            # Safety
            try:
                sf_res = future_sf.result(timeout=45.0)
            except Exception as exc:
                logger.error("SafetyEvaluator failed: %s", exc, exc_info=True)
                sf_res = EvaluationResult(
                    evaluator_name=sf_evaluator.name,
                    conversation_id=evaluation_input.conversation_id,
                    score=0.0,
                    max_score=20.0,
                    status="failed",
                    sub_scores={},
                    feedback=f"Evaluation failed with error: {exc}",
                    flagged=True,
                )

            # Intent
            try:
                it_res = future_it.result(timeout=45.0)
            except Exception as exc:
                logger.error("IntentEvaluator failed: %s", exc, exc_info=True)
                it_res = EvaluationResult(
                    evaluator_name=it_evaluator.name,
                    conversation_id=evaluation_input.conversation_id,
                    score=0.0,
                    max_score=20.0,
                    status="failed",
                    sub_scores={},
                    feedback=f"Evaluation failed with error: {exc}",
                    flagged=True,
                )

            # Memory
            try:
                me_res = future_me.result(timeout=45.0)
            except Exception as exc:
                logger.error("MemoryEvaluator failed: %s", exc, exc_info=True)
                me_res = EvaluationResult(
                    evaluator_name=me_evaluator.name,
                    conversation_id=evaluation_input.conversation_id,
                    score=None,
                    max_score=20.0,
                    applicable=False,
                    status="failed",
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
