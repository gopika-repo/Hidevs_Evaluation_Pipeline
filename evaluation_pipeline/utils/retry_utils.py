import time
import logging
from google.genai.errors import APIError

logger = logging.getLogger(__name__)

def is_transient_error(exc: Exception) -> bool:
    """
    Detects if the exception is a transient failure that should be retried.
    """
    exc_name = exc.__class__.__name__
    if "ValidationError" in exc_name or "PydanticUserError" in exc_name:
        return False

    # 1. APIError from google-genai
    if isinstance(exc, APIError):
        code = getattr(exc, "code", None)
        if code is not None:
            if code == 429 or (500 <= code < 600):
                return True
            return False
        message = str(exc).lower()
        if "429" in message or "quota" in message or "rate limit" in message or "overloaded" in message:
            return True
        return False
    
    # 2. General check based on exception status code attributes
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        try:
            code_int = int(status_code)
            if code_int == 429 or (500 <= code_int < 600):
                return True
            return False
        except ValueError:
            pass

    # 3. Check error message / type for connection / timeout / transient strings
    exc_name_lower = exc_name.lower()
    if "timeout" in exc_name_lower or "timeouterror" in exc_name_lower or "connectionerror" in exc_name_lower or "connecterror" in exc_name_lower:
        return True
        
    message = str(exc).lower()
    
    # Explicitly do NOT retry if it's an API key or auth issue or invalid model name
    non_retry_keywords = ["api key", "apikey", "unauthorized", "auth", "credential", "invalid key", "not found", "not exist", "does not exist", "bad request", "invalid_argument", "validation"]
    if any(kw in message for kw in non_retry_keywords):
        return False
        
    transient_keywords = ["429", "quota", "rate limit", "rate_limit", "overloaded", "503", "502", "504", "500", "temporary", "service unavailable", "gateway", "timeout", "time out", "try again"]
    if any(kw in message for kw in transient_keywords):
        return True
        
    return False

def execute_with_retry(
    func,
    evaluator: str,
    framework: str,
    conversation_id: str,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    *args,
    **kwargs
):
    """
    Executes a function with exponential backoff for transient errors, and logs attempts in a structured format.
    """
    attempt = 1
    delay = initial_delay
    
    while True:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            # Structured logging: conversation_id, evaluator, framework, attempt, duration, status, transient
            logger.info(
                "Evaluation attempt SUCCESS: conversation_id=%s | evaluator=%s | framework=%s | attempt=%d | duration=%.3fs | status=success | transient=None",
                conversation_id, evaluator, framework, attempt, duration
            )
            return result
        except Exception as exc:
            duration = time.time() - start_time
            transient = is_transient_error(exc)
            
            logger.warning(
                "Evaluation attempt FAILED: conversation_id=%s | evaluator=%s | framework=%s | attempt=%d | duration=%.3fs | status=failed | transient=%s | error=%s",
                conversation_id, evaluator, framework, attempt, duration, str(transient), str(exc)
            )
            
            if not transient or attempt >= max_retries:
                raise exc
            
            logger.info("Retrying in %.1fs due to transient error...", delay)
            time.sleep(delay)
            attempt += 1
            delay *= backoff_factor
