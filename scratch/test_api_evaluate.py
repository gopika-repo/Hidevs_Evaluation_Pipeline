import os
import sys
import json
sys.path.append(os.path.abspath("."))

from fastapi.testclient import TestClient
from app import app
from evaluation_pipeline.data.mock_conversations import get_mock_conversations

client = TestClient(app)

def test_api():
    raw_convs = get_mock_conversations()
    
    # 1. Context-backed
    cb_record = next(c for c in raw_convs if c.conversation_id == "CB-001")
    # Convert ConversationRecord Pydantic model to dict
    cb_data = json.loads(cb_record.model_dump_json())
    
    print("Sending CB-001 (context_backed) to /evaluate...")
    res_cb = client.post("/evaluate", json=cb_data)
    assert res_cb.status_code == 200, f"Failed: {res_cb.text}"
    report_cb = res_cb.json()
    print("CB-001 Response Keys:", report_cb.keys())
    convo_cb = report_cb["conversations"][0]
    print("Overall Health Score:", convo_cb["overall_health_score"])
    print("-"*50)
    
    # 2. Context-free
    cf_record = next(c for c in raw_convs if c.conversation_id == "CF-001")
    cf_data = json.loads(cf_record.model_dump_json())
    
    print("Sending CF-001 (context_free) to /evaluate...")
    res_cf = client.post("/evaluate", json=cf_data)
    assert res_cf.status_code == 200, f"Failed: {res_cf.text}"
    report_cf = res_cf.json()
    convo_cf = report_cf["conversations"][0]
    print("Overall Health Score:", convo_cf["overall_health_score"])
    print("="*50)

if __name__ == "__main__":
    test_api()
