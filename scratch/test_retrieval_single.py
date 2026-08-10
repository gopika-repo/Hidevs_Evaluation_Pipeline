import time
import os
import sys
sys.path.append(os.path.abspath("."))

# Ensure we import the compatibility patch
import evaluation_pipeline.utils.ragas_compat_patch

load_dotenv = lambda: None
from dotenv import load_dotenv
load_dotenv()

from evaluation_pipeline.data.mock_conversations import get_mock_conversations
from evaluation_pipeline.data.dataset_builder import DatasetBuilder
from evaluation_pipeline.evaluators.retrieval_evaluator import RetrievalEvaluator

def main():
    print("Initializing RetrievalEvaluator...")
    start_init = time.time()
    evaluator = RetrievalEvaluator()
    print(f"Evaluator initialized in {time.time() - start_init:.2f}s")
    
    raw_convs = get_mock_conversations()
    # Find a context_backed conversation like CB-001
    cb_conv = next(c for c in raw_convs if c.conversation_id == "CB-001")
    
    builder = DatasetBuilder()
    inputs = builder.build([cb_conv])
    eval_input = inputs[0]
    
    print(f"Running evaluation for {eval_input.conversation_id}...")
    start_eval = time.time()
    result = evaluator.evaluate(eval_input)
    elapsed = time.time() - start_eval
    
    print("\n" + "="*50)
    print("EVALUATION RESULT:")
    print(f"Conversation ID: {result.conversation_id}")
    print(f"Score:           {result.score}/{result.max_score}")
    print(f"Sub-scores:      {result.sub_scores}")
    print(f"Applicable:      {result.applicable}")
    print(f"Flagged:         {result.flagged}")
    if result.feedback:
        print(f"Feedback preview:\n{result.feedback[:500]}...")
    print(f"Evaluation took: {elapsed:.2f}s")
    print("="*50)

if __name__ == "__main__":
    main()
