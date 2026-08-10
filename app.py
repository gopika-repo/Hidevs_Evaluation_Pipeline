from fastapi import FastAPI, BackgroundTasks, HTTPException
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel
from datetime import datetime
from evaluation_pipeline.data.models import ConversationRecord
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
        # 2. Run evaluators
        rq_evaluator = ResponseQualityEvaluator()
        gd_evaluator = GroundednessEvaluator()
        sf_evaluator = SafetyEvaluator()
        it_evaluator = IntentEvaluator()
        me_evaluator = MemoryEvaluator()
        
        rq_res = rq_evaluator.evaluate(evaluation_input)
        gd_res = gd_evaluator.evaluate(evaluation_input)
        sf_res = sf_evaluator.evaluate(evaluation_input)
        it_res = it_evaluator.evaluate(evaluation_input)
        me_res = me_evaluator.evaluate(evaluation_input)
        
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
