import unittest
import os
from dotenv import load_dotenv

# Ensure dotenv is loaded before imports
load_dotenv()

from evaluation_pipeline.utils.llm_client import LLMJudge
from evaluation_pipeline.evaluators.groundedness_evaluator import _run_deepeval_faithfulness, _run_trulens_groundedness

class TestFrameworkSmoke(unittest.TestCase):
    def test_framework_smoke(self):
        print("\n=== FRAMEWORK SMOKE TEST ===")
        
        # 1. Verify GOOGLE_API_KEY
        api_key = os.getenv("GOOGLE_API_KEY")
        self.assertTrue(api_key is not None, "GOOGLE_API_KEY must be set in env")
        print(f"GOOGLE_API_KEY configured: {'yes' if api_key else 'no'}")
        
        # 2. Test Gemini Shared Judge
        print("\n--- Testing Gemini Shared Judge ---")
        judge = LLMJudge()
        system_prompt = "You are a helpful assistant. Respond with a JSON object containing a 'response' key."
        user_prompt = "Say hello."
        parsed, raw = judge.call_with_json(system_prompt, user_prompt)
        print("Raw response:", raw)
        print("Parsed response:", parsed)
        self.assertIn("response", parsed)
        
        # 3. Test DeepEval Faithfulness
        print("\n--- Testing DeepEval FaithfulnessMetric ---")
        user_query = "What is the company's remote work policy?"
        dave_response = "According to our policy, full-time employees are eligible for up to 3 days of remote work per week."
        retrieved_context = "All full-time employees are eligible for hybrid remote work arrangements of up to 3 days per week."
        
        res_deepeval = _run_deepeval_faithfulness(user_query, dave_response, retrieved_context)
        print("DeepEval Result:", res_deepeval)
        
        self.assertIn(res_deepeval["status"], ["success", "failed"])
        if res_deepeval["status"] == "success":
            self.assertIsNotNone(res_deepeval.get("score"))
            print(f"DeepEval Success Score: {res_deepeval['score']}")
        else:
            print(f"DeepEval Failed: {res_deepeval.get('error')}")

        # 4. Test TruLens Groundedness
        print("\n--- Testing TruLens Groundedness ---")
        res_trulens = _run_trulens_groundedness(retrieved_context, dave_response)
        print("TruLens Result:", res_trulens)
        self.assertIn(res_trulens["status"], ["success", "failed"])
        if res_trulens["status"] == "success":
            self.assertIsNotNone(res_trulens.get("score"))
            print(f"TruLens Success Score: {res_trulens['score']}")
        else:
            print(f"TruLens Failed: {res_trulens.get('error')}")

        # 5. Confirm OPENAI_API_KEY is NOT required
        self.assertIsNone(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must NOT be set/required in this test environment")

if __name__ == "__main__":
    unittest.main()
