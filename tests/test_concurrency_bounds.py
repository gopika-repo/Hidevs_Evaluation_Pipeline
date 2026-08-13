import unittest
import concurrent.futures
import time
import os

from evaluation_pipeline.utils.concurrency import controlled_concurrency, get_max_observed_concurrency, reset_max_observed_concurrency
from evaluation_pipeline.utils import concurrency as conc_mod

class TestConcurrencyBounds(unittest.TestCase):
    def setUp(self):
        # Force concurrency limit to 2 and rebuild semaphore
        os.environ["EVALUATION_MAX_CONCURRENCY"] = "2"
        conc_mod._semaphore = None
        conc_mod._current_limit = None
        reset_max_observed_concurrency()

    def tearDown(self):
        # Clean up
        os.environ.pop("EVALUATION_MAX_CONCURRENCY", None)
        conc_mod._semaphore = None
        conc_mod._current_limit = None

    def test_concurrency_never_exceeds_limit(self):
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

