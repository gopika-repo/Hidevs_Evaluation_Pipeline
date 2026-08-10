from fastapi import FastAPI, BackgroundTasks, HTTPException
import uvicorn
import subprocess
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
from evaluation_pipeline.evaluators.retrieval_evaluator import RetrievalEvaluator
from evaluation_pipeline.aggregator.score_aggregator import ScoreAggregator

app = FastAPI(title="Evaluation Pipeline API")

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
        rt_evaluator = RetrievalEvaluator()
        
        rq_res = rq_evaluator.evaluate(evaluation_input)
        gd_res = gd_evaluator.evaluate(evaluation_input)
        sf_res = sf_evaluator.evaluate(evaluation_input)
        it_res = it_evaluator.evaluate(evaluation_input)
        rt_res = rt_evaluator.evaluate(evaluation_input)
        
        # 3. Aggregate
        aggregator = ScoreAggregator()
        report = aggregator.aggregate_dataset(
            inputs=[evaluation_input],
            rq_results=[rq_res],
            gd_results=[gd_res],
            safety_results=[sf_res],
            intent_results=[it_res],
            retrieval_results=[rt_res]
        )
        
        return report
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
