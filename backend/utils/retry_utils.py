import asyncio
import logging
import random
from typing import Callable, Any, Optional, Type, Tuple, List, Dict

logger = logging.getLogger("RetryUtils")

class RetryManager:
    """
    Generic utility for retrying asynchronous operations with exponential backoff.
    """
    
    @staticmethod
    async def retry_async(
        func: Callable,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retry_on_exceptions: Optional[List[Type[Exception]]] = None,
        retry_on_status_codes: Optional[List[int]] = None,
        operation_name: str = "Operation"
    ) -> Any:
        """
        Retry an async function with exponential backoff.
        """
        args = args or ()
        kwargs = kwargs or {}
        retry_on_exceptions = retry_on_exceptions or [Exception]
        
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                
                # Check status code if result has one (e.g. httpx Response)
                if retry_on_status_codes and hasattr(result, 'status_code'):
                    if result.status_code in retry_on_status_codes:
                        if attempt < max_retries:
                            wait_time = delay * (random.uniform(0.5, 1.5) if jitter else 1.0)
                            logger.warning(f"⚠️ {operation_name} returned status {result.status_code}. Attempt {attempt + 1}/{max_retries}. Retrying in {wait_time:.2f}s...")
                            await asyncio.sleep(wait_time)
                            delay *= backoff_factor
                            continue
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry on this exception type
                should_retry = any(isinstance(e, ex_type) for ex_type in retry_on_exceptions)
                
                if should_retry and attempt < max_retries:
                    wait_time = delay * (random.uniform(0.5, 1.5) if jitter else 1.0)
                    logger.warning(f"❌ {operation_name} failed: {str(e)}. Attempt {attempt + 1}/{max_retries}. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                    delay *= backoff_factor
                else:
                    logger.error(f"🚫 {operation_name} failed after {attempt + 1} attempts: {str(e)}")
                    raise e
                    
        if last_exception:
            raise last_exception
