from fastapi import FastAPI, BackgroundTasks
import uvicorn
import subprocess
import os

from pydantic import BaseModel
from datetime import datetime
from evaluation_pipeline.data.models import ConversationRecord
from evaluation_pipeline.data.dataset_builder import DatasetBuilder
from evaluation_pipeline.evaluators.response_quality_evaluator import ResponseQualityEvaluator
from evaluation_pipeline.evaluators.groundedness_evaluator import GroundednessEvaluator
from evaluation_pipeline.evaluators.safety_evaluator import SafetyEvaluator
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
    
    # 2. Run evaluators
    rq_evaluator = ResponseQualityEvaluator()
    gd_evaluator = GroundednessEvaluator()
    sf_evaluator = SafetyEvaluator()
    
    rq_res = rq_evaluator.evaluate(evaluation_input)
    gd_res = gd_evaluator.evaluate(evaluation_input)
    sf_res = sf_evaluator.evaluate(evaluation_input)
    
    # 3. Aggregate
    aggregator = ScoreAggregator()
    report = aggregator.aggregate_dataset(
        inputs=[evaluation_input],
        rq_results=[rq_res],
        gd_results=[gd_res],
        safety_results=[sf_res]
    )
    
    return report

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
