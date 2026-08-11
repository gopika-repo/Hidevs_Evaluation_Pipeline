import os
import threading
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Configurable concurrency limit via environment variable
MAX_CONCURRENCY = int(os.getenv("EVALUATION_MAX_CONCURRENCY", "2"))

_semaphore = threading.Semaphore(MAX_CONCURRENCY)
_active_calls = 0
_active_calls_lock = threading.Lock()
_max_observed_concurrency = 0

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
    
    logger.debug(
        "[%s][%s][%s] Waiting for concurrency slot. Max Limit: %d",
        conversation_id, evaluator, framework, MAX_CONCURRENCY
    )
    
    with _semaphore:
        with _active_calls_lock:
            _active_calls += 1
            if _active_calls > _max_observed_concurrency:
                _max_observed_concurrency = _active_calls
            current = _active_calls
        
        logger.debug(
            "[%s][%s][%s] Acquired concurrency slot. Active external calls: %d/%d",
            conversation_id, evaluator, framework, current, MAX_CONCURRENCY
        )
        try:
            yield
        finally:
            with _active_calls_lock:
                _active_calls -= 1
                current = _active_calls
            logger.debug(
                "[%s][%s][%s] Released concurrency slot. Active external calls: %d/%d",
                conversation_id, evaluator, framework, current, MAX_CONCURRENCY
            )
