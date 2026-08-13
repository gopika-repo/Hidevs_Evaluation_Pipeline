import os
import threading
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_semaphore = None
_current_limit = None
_semaphore_lock = threading.Lock()
_active_calls = 0
_active_calls_lock = threading.Lock()
_max_observed_concurrency = 0

def get_semaphore():
    global _semaphore, _current_limit
    limit = int(os.getenv("EVALUATION_MAX_CONCURRENCY", "10"))
    with _semaphore_lock:
        if _semaphore is None or _current_limit != limit:
            _semaphore = threading.Semaphore(limit)
            _current_limit = limit
    return _semaphore, limit

def get_max_observed_concurrency() -> int:
    global _max_observed_concurrency
    with _active_calls_lock:
        return _max_observed_concurrency

def reset_max_observed_concurrency():
    global _max_observed_concurrency
    with _active_calls_lock:
        _max_observed_concurrency = 0

@contextmanager
def controlled_concurrency(evaluator: str, framework: str, conversation_id: str):
    global _active_calls, _max_observed_concurrency
    
    semaphore, limit = get_semaphore()
    start_wait = time.time()
    logger.info(
        "Evaluation execution state: START | conversation_id=%s | evaluator=%s | framework=%s | max_limit=%d | state=waiting_for_semaphore",
        conversation_id, evaluator, framework, limit
    )
    
    # Use 30.0s timeout to acquire the semaphore to prevent deadlocks and ensure cleanup
    # of orphaned threads if parent requests time out.
    acquired = semaphore.acquire(timeout=30.0)
    
    if not acquired:
        duration = time.time() - start_wait
        logger.error(
            "Evaluation execution state: FAILED | conversation_id=%s | evaluator=%s | framework=%s | duration=%.3fs | state=semaphore_timeout | error=Could not acquire concurrency slot",
            conversation_id, evaluator, framework, duration
        )
        raise TimeoutError(f"Could not acquire concurrency slot for {evaluator} ({framework}) within 30 seconds.")
    
    wait_duration = time.time() - start_wait
    if wait_duration > 0.1:
        logger.info(
            "Evaluation execution state: WAITED | conversation_id=%s | evaluator=%s | framework=%s | wait_duration=%.3fs",
            conversation_id, evaluator, framework, wait_duration
        )
        
    with _active_calls_lock:
        _active_calls += 1
        if _active_calls > _max_observed_concurrency:
            _max_observed_concurrency = _active_calls
        current = _active_calls
    
    logger.info(
        "Evaluation execution state: ACQUIRED | conversation_id=%s | evaluator=%s | framework=%s | active_slots=%d/%d",
        conversation_id, evaluator, framework, current, limit
    )
    
    start_execution = time.time()
    try:
        yield
        execution_duration = time.time() - start_execution
        logger.info(
            "Evaluation execution state: SUCCESS | conversation_id=%s | evaluator=%s | framework=%s | duration=%.3fs | state=completed",
            conversation_id, evaluator, framework, execution_duration
        )
    except Exception as exc:
        execution_duration = time.time() - start_execution
        logger.error(
            "Evaluation execution state: FAILED | conversation_id=%s | evaluator=%s | framework=%s | duration=%.3fs | state=exception | error=%s",
            conversation_id, evaluator, framework, execution_duration, str(exc)
        )
        raise exc
    finally:
        with _active_calls_lock:
            _active_calls -= 1
            current = _active_calls
        semaphore.release()
        logger.info(
            "Evaluation execution state: RELEASED | conversation_id=%s | evaluator=%s | framework=%s | active_slots=%d/%d",
            conversation_id, evaluator, framework, current, limit
        )
