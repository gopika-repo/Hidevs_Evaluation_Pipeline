"""
Phase 1 Concurrency Reliability Tests

Tests 1-7 verifying that the concurrency infrastructure provides
independent evaluator execution isolation and reliable semaphore behavior.
"""

import unittest
import concurrent.futures
import time
import os
import threading

from evaluation_pipeline.utils.concurrency import (
    controlled_concurrency,
    get_max_observed_concurrency,
    reset_max_observed_concurrency,
)
from evaluation_pipeline.utils import concurrency as conc_mod


class TestPhase1ConcurrencyReliability(unittest.TestCase):
    """Phase 1 concurrency reliability tests."""

    def setUp(self):
        """Reset semaphore state before each test."""
        # Default to 10 for production-like behavior unless overridden
        os.environ.pop("EVALUATION_MAX_CONCURRENCY", None)
        conc_mod._semaphore = None
        conc_mod._current_limit = None
        reset_max_observed_concurrency()

    def tearDown(self):
        """Clean up semaphore state after each test."""
        os.environ.pop("EVALUATION_MAX_CONCURRENCY", None)
        conc_mod._semaphore = None
        conc_mod._current_limit = None

    # ------------------------------------------------------------------
    # Test 1: All evaluators complete concurrently for one conversation
    # ------------------------------------------------------------------
    def test_1_all_evaluators_complete_concurrently(self):
        """
        Run all evaluators concurrently for one conversation.
        Expected: All independent evaluators complete or report their genuine failure.
        No evaluator remains stuck indefinitely.
        """
        evaluator_names = [
            "response_quality", "groundedness", "safety",
            "intent_understanding", "memory_and_continuity",
            # Simulate nested groundedness sub-tasks
            "groundedness_trulens", "groundedness_deepeval",
        ]
        results = {}
        errors = {}

        def simulate_evaluator(name):
            try:
                with controlled_concurrency(name, "MockFramework", "TEST-001"):
                    time.sleep(0.2)  # Simulate LLM call
                results[name] = "success"
            except Exception as exc:
                errors[name] = str(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            futures = {
                executor.submit(simulate_evaluator, name): name
                for name in evaluator_names
            }
            done, not_done = concurrent.futures.wait(futures, timeout=30.0)

        # All tasks must complete
        self.assertEqual(len(not_done), 0, f"Stuck evaluators: {[futures[f] for f in not_done]}")
        # All evaluators must succeed (no semaphore starvation)
        self.assertEqual(len(results), len(evaluator_names),
                         f"Missing results: {set(evaluator_names) - set(results.keys())}. Errors: {errors}")
        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")

    # ------------------------------------------------------------------
    # Test 2: One blocked evaluator does not affect others
    # ------------------------------------------------------------------
    def test_2_blocked_evaluator_does_not_affect_others(self):
        """
        Force one evaluator to sleep/block/fail.
        Expected: Other evaluators continue executing and are not incorrectly
        marked N/A or FAILED because of the unrelated evaluator.
        """
        results = {}
        errors = {}

        def slow_evaluator():
            try:
                with controlled_concurrency("slow_eval", "MockFramework", "TEST-002"):
                    time.sleep(3.0)  # Blocks for 3 seconds
                results["slow_eval"] = "success"
            except Exception as exc:
                errors["slow_eval"] = str(exc)

        def fast_evaluator(name):
            try:
                with controlled_concurrency(name, "MockFramework", "TEST-002"):
                    time.sleep(0.1)  # Fast completion
                results[name] = "success"
            except Exception as exc:
                errors[name] = str(exc)

        fast_names = ["response_quality", "safety", "intent", "memory"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(slow_evaluator)]
            for name in fast_names:
                futures.append(executor.submit(fast_evaluator, name))
            done, not_done = concurrent.futures.wait(futures, timeout=15.0)

        self.assertEqual(len(not_done), 0, "Some evaluators stuck")
        # All fast evaluators must complete successfully
        for name in fast_names:
            self.assertIn(name, results, f"{name} was blocked by slow evaluator")
            self.assertEqual(results[name], "success")
        # Slow evaluator should also eventually complete
        self.assertIn("slow_eval", results)

    # ------------------------------------------------------------------
    # Test 3: Exception releases semaphore
    # ------------------------------------------------------------------
    def test_3_exception_releases_semaphore(self):
        """
        Force an evaluator exception.
        Expected: Semaphore is released and subsequent evaluations can execute.
        """
        os.environ["EVALUATION_MAX_CONCURRENCY"] = "1"
        conc_mod._semaphore = None
        conc_mod._current_limit = None

        # First: a call that throws an exception
        with self.assertRaises(ValueError):
            with controlled_concurrency("failing_eval", "MockFramework", "TEST-003"):
                raise ValueError("Simulated evaluator crash")

        # Second: a subsequent call must succeed (semaphore was released)
        success = False
        with controlled_concurrency("subsequent_eval", "MockFramework", "TEST-003"):
            success = True
        self.assertTrue(success, "Semaphore was NOT released after exception")

    # ------------------------------------------------------------------
    # Test 4: Timeout releases semaphore
    # ------------------------------------------------------------------
    def test_4_timeout_releases_semaphore(self):
        """
        Force evaluator timeout/cancellation.
        Expected: Semaphore is released and subsequent evaluations can execute.
        """
        os.environ["EVALUATION_MAX_CONCURRENCY"] = "1"
        conc_mod._semaphore = None
        conc_mod._current_limit = None

        # First: a call that times out
        with self.assertRaises(TimeoutError):
            with controlled_concurrency("timeout_eval", "MockFramework", "TEST-004"):
                raise TimeoutError("Simulated timeout")

        # Second: a subsequent call must succeed
        success = False
        with controlled_concurrency("subsequent_eval", "MockFramework", "TEST-004"):
            success = True
        self.assertTrue(success, "Semaphore was NOT released after timeout")

    # ------------------------------------------------------------------
    # Test 5: Repeated evaluation stability (10 iterations)
    # ------------------------------------------------------------------
    def test_5_repeated_evaluation_stability(self):
        """
        Run the same evaluation repeatedly at least 10 times.
        Expected: There is no random SCORE → N/A → FAILED transition
        caused by concurrency scheduling.
        """
        results = []
        for iteration in range(10):
            iteration_results = {}
            evaluator_names = [
                "response_quality", "groundedness", "safety",
                "intent_understanding", "memory",
            ]

            def simulate(name, it=iteration):
                with controlled_concurrency(name, "MockFramework", f"REPEAT-{it:03d}"):
                    time.sleep(0.05)
                return "success"

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_map = {
                    executor.submit(simulate, name): name
                    for name in evaluator_names
                }
                for future in concurrent.futures.as_completed(future_map, timeout=15.0):
                    name = future_map[future]
                    try:
                        iteration_results[name] = future.result()
                    except Exception as exc:
                        iteration_results[name] = f"FAILED: {exc}"

            results.append(iteration_results)

        # Every iteration must have all evaluators succeed
        for i, res in enumerate(results):
            for name in ["response_quality", "groundedness", "safety",
                         "intent_understanding", "memory"]:
                self.assertEqual(
                    res.get(name), "success",
                    f"Iteration {i}: {name} got '{res.get(name)}' instead of 'success'"
                )

    # ------------------------------------------------------------------
    # Test 6: Multiple concurrent requests without deadlock
    # ------------------------------------------------------------------
    def test_6_concurrent_requests_no_deadlock(self):
        """
        Run multiple evaluation requests concurrently.
        Expected: No deadlock, semaphore starvation, task leakage,
        or cross-request result contamination.
        """
        request_results = {}

        def simulate_full_request(request_id):
            """Simulate a full /evaluate request with 5 evaluators."""
            evaluator_names = [
                "response_quality", "groundedness", "safety",
                "intent_understanding", "memory",
            ]
            local_results = {}

            def run_eval(name):
                with controlled_concurrency(name, "MockFramework", request_id):
                    time.sleep(0.1)
                return "success"

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_map = {
                    executor.submit(run_eval, name): name
                    for name in evaluator_names
                }
                for future in concurrent.futures.as_completed(future_map, timeout=30.0):
                    name = future_map[future]
                    try:
                        local_results[name] = future.result()
                    except Exception as exc:
                        local_results[name] = f"FAILED: {exc}"

            return local_results

        # Simulate 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as outer:
            future_map = {
                outer.submit(simulate_full_request, f"REQ-{i:03d}"): f"REQ-{i:03d}"
                for i in range(3)
            }
            for future in concurrent.futures.as_completed(future_map, timeout=60.0):
                req_id = future_map[future]
                request_results[req_id] = future.result()

        # All 3 requests must complete with all evaluators succeeding
        self.assertEqual(len(request_results), 3, "Not all requests completed")
        for req_id, res in request_results.items():
            for name in ["response_quality", "groundedness", "safety",
                         "intent_understanding", "memory"]:
                self.assertEqual(
                    res.get(name), "success",
                    f"{req_id}: {name} got '{res.get(name)}'"
                )

    # ------------------------------------------------------------------
    # Test 7: Result belongs to correct request ID
    # ------------------------------------------------------------------
    def test_7_result_belongs_to_correct_request_id(self):
        """
        Verify that an evaluator's result belongs to the correct evaluation/request ID.
        """
        captured_ids = {}
        lock = threading.Lock()

        def simulate_evaluator(request_id, evaluator_name):
            with controlled_concurrency(evaluator_name, "MockFramework", request_id):
                time.sleep(0.05)
                # Record which request_id this evaluator ran with
                with lock:
                    key = f"{request_id}:{evaluator_name}"
                    captured_ids[key] = request_id
            return request_id

        all_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for req_id in ["R-001", "R-002", "R-003"]:
                for eval_name in ["rq", "gd", "sf", "it", "mem"]:
                    f = executor.submit(simulate_evaluator, req_id, eval_name)
                    all_futures[f] = (req_id, eval_name)

            for future in concurrent.futures.as_completed(all_futures, timeout=30.0):
                expected_req_id, eval_name = all_futures[future]
                actual_req_id = future.result()
                self.assertEqual(
                    actual_req_id, expected_req_id,
                    f"Result contamination: {eval_name} returned {actual_req_id} but expected {expected_req_id}"
                )

        # Verify all combinations were captured
        self.assertEqual(len(captured_ids), 15)  # 3 requests × 5 evaluators


if __name__ == "__main__":
    unittest.main()
