import unittest
import concurrent.futures
import time
import os

# Set concurrency limit to 2
os.environ["EVALUATION_MAX_CONCURRENCY"] = "2"
from evaluation_pipeline.utils.concurrency import controlled_concurrency, get_max_observed_concurrency, reset_max_observed_concurrency

class TestConcurrencyBounds(unittest.TestCase):
    def test_concurrency_never_exceeds_limit(self):
        reset_max_observed_concurrency()
        
        def mock_external_call(i):
            with controlled_concurrency("test", "MockFramework", f"conv_{i}"):
                time.sleep(0.3)
                
        # Run 5 concurrent calls in a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(mock_external_call, i) for i in range(5)]
            concurrent.futures.wait(futures)
            
        max_observed = get_max_observed_concurrency()
        print(f"Max observed concurrency: {max_observed}")
        self.assertLessEqual(max_observed, 2)
