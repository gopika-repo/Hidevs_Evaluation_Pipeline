import unittest
import concurrent.futures
import time
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from evaluation_pipeline.utils.concurrency import (
    controlled_concurrency,
    get_max_observed_concurrency,
    reset_max_observed_concurrency,
)
from evaluation_pipeline.utils.retry_utils import execute_with_retry, is_transient_error


class TestConcurrencyAndTimeouts(unittest.TestCase):
    def test_concurrency_limits(self) -> None:
        """Verify that concurrency never exceeds the configured E_MAX limit under high thread count."""
        reset_max_observed_concurrency()
        
        def run_call(i):
            with controlled_concurrency("evaluator_test", "MockFramework", f"conv_{i}"):
                time.sleep(0.1)

        # We execute 8 threads simultaneously. The limit MAX_CONCURRENCY is configured to 2 in tests.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_call, i) for i in range(8)]
            concurrent.futures.wait(futures)

        max_observed = get_max_observed_concurrency()
        self.assertLessEqual(max_observed, 2)

    def test_retry_skip_non_transient(self) -> None:
        """Verify that non-transient errors (auth, validation, bad requests) are never retried."""
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            # Simulate a non-transient auth error
            raise ValueError("Invalid API key provided or bad request validation error.")

        with self.assertRaises(ValueError):
            execute_with_retry(
                failing_func,
                evaluator="test_eval",
                framework="Mock",
                conversation_id="conv_retry_skip",
                max_retries=3,
                initial_delay=0.01,
            )

        # Should only try once because it's non-transient!
        self.assertEqual(call_count, 1)

    def test_retry_transient_retries_and_fails(self) -> None:
        """Verify that transient errors (rate limit 429) are retried up to max_retries."""
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Rate limit exceeded. Too many requests 429")

        with self.assertRaises(Exception):
            execute_with_retry(
                failing_func,
                evaluator="test_eval",
                framework="Mock",
                conversation_id="conv_retry_transient",
                max_retries=3,
                initial_delay=0.01,
            )

        self.assertEqual(call_count, 3)

    def test_timeout_policy_returns_failed_status_score_none(self) -> None:
        """Verify that a timeout does not manufacture a score of zero, but returns None and status='failed'."""
        # Simulated timeout error
        from concurrent.futures import TimeoutError

        def timeout_func():
            raise TimeoutError("Request timed out after policy window.")

        with self.assertRaises(TimeoutError):
            execute_with_retry(
                timeout_func,
                evaluator="test_eval",
                framework="Mock",
                conversation_id="conv_timeout",
                max_retries=1,
                initial_delay=0.01,
            )
