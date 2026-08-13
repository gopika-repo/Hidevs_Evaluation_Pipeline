import json
import concurrent.futures
import asyncio
from google.genai.errors import APIError

def classify_exception(exc: Exception) -> str:
    """
    Classifies a raised exception into one of the strict status values:
    - timeout
    - invalid_output
    - unavailable
    - failed
    """
    if exc is None:
        return "success"
        
    exc_name = exc.__class__.__name__
    exc_msg = str(exc).lower()

    # 1. Timeout Check
    if (isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError, asyncio.TimeoutError))
            or "timeout" in exc_name.lower()
            or "timed out" in exc_msg):
        return "timeout"

    # 2. Invalid Output / Format / Parse Check
    if (isinstance(exc, (json.JSONDecodeError, ValueError, TypeError))
            or "validation" in exc_name.lower()
            or "pydantic" in exc_name.lower()
            or "json" in exc_msg
            or "format" in exc_msg
            or "extract" in exc_msg):
        return "invalid_output"

    # 3. Unavailable Check (Service, auth, credentials, network, quota, rate limit)
    non_retry_keywords = ["api key", "apikey", "unauthorized", "auth", "credential", "invalid key", "not found", "does not exist", "bad request", "invalid_argument", "validation"]
    transient_keywords = ["429", "quota", "rate limit", "rate_limit", "overloaded", "503", "502", "504", "500", "temporary", "service unavailable", "gateway", "connection", "http", "try again"]
    
    if (isinstance(exc, APIError)
            or any(kw in exc_msg for kw in non_retry_keywords)
            or any(kw in exc_msg for kw in transient_keywords)
            or "connection" in exc_name.lower()
            or "http" in exc_name.lower()):
        return "unavailable"

    # 4. Fallback General Failure
    return "failed"
