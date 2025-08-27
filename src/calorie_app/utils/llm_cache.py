# utils/llm_cache.py
import hashlib
import json
import pickle
import time
import asyncio
import inspect
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Callable, Dict, Union
import logging

from .config import SystemLimits

logger = logging.getLogger(__name__)


class LLMCache:
    """Cache for LLM and AI agent responses to avoid hitting API limits during development."""

    def __init__(
        self,
        cache_dir: str = ".llm_cache",
        max_size: int = SystemLimits.LLM_CACHE_SIZE,
        ttl_hours: int = SystemLimits.CACHE_TTL_HOURS,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.memory_cache = {}

    def _get_request_hash(
        self,
        messages: Union[str, list],
        model: str,
        temperature: float = None,
        max_tokens: int = None,
        extra_params: Dict = None,
    ) -> str:
        """Generate hash for LLM request parameters."""
        try:
            # Create a consistent string representation
            if isinstance(messages, str):
                content = messages
            elif isinstance(messages, list):
                # Handle list of messages (ChatOpenAI format)
                content = json.dumps(
                    [
                        {
                            "type": getattr(msg, "__class__", {}).get(
                                "__name__", "unknown"
                            ),
                            "content": getattr(msg, "content", str(msg)),
                        }
                        for msg in messages
                    ],
                    sort_keys=True,
                )
            else:
                content = str(messages)

            # Include all parameters that affect the response
            request_data = {
                "messages": content,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_params": extra_params or {},
            }

            request_string = json.dumps(request_data, sort_keys=True)
            return hashlib.sha256(request_string.encode()).hexdigest()[
                : SystemLimits.HASH_LENGTH
            ]

        except Exception as e:
            logger.warning(f"Failed to hash LLM request: {e}")
            # Fallback hash
            return hashlib.sha256(str(messages).encode()).hexdigest()[
                : SystemLimits.HASH_LENGTH
            ]

    def get(
        self,
        messages: Union[str, list],
        model: str,
        temperature: float = None,
        max_tokens: int = None,
        extra_params: Dict = None,
    ) -> Optional[Any]:
        """Get cached LLM response."""
        cache_key = self._get_request_hash(
            messages, model, temperature, max_tokens, extra_params
        )

        # Check memory cache first
        if cache_key in self.memory_cache:
            cached_item = self.memory_cache[cache_key]
            if self._is_cache_valid(cached_item["timestamp"]):
                logger.debug(f"LLM cache hit (memory): {cache_key}")
                return cached_item["response"]
            else:
                # Remove expired item
                del self.memory_cache[cache_key]

        # Check disk cache
        cache_file = self.cache_dir / f"llm_{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_item = pickle.load(f)

                    if self._is_cache_valid(cached_item["timestamp"]):
                        # Store in memory for faster access
                        self.memory_cache[cache_key] = cached_item
                        logger.debug(f"LLM cache hit (disk): {cache_key}")
                        return cached_item["response"]
                    else:
                        # Remove expired file
                        cache_file.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(f"Failed to load LLM cache file {cache_file}: {e}")
                cache_file.unlink(missing_ok=True)

        return None

    def _is_cacheable(self, response: Any) -> bool:
        """Check if response can be safely cached (no coroutines or other non-serializable objects)."""
        try:
            # Check for coroutine objects
            if asyncio.iscoroutine(response):
                return False

            # Check for generator objects
            if inspect.isgenerator(response):
                return False

            # Try a test pickle to see if it's serializable
            pickle.dumps(response)
            return True
        except (TypeError, pickle.PicklingError):
            return False

    def set(
        self,
        messages: Union[str, list],
        model: str,
        response: Any,
        temperature: float = None,
        max_tokens: int = None,
        extra_params: Dict = None,
    ) -> None:
        """Store LLM response in cache."""
        # Check if response is cacheable
        if not self._is_cacheable(response):
            logger.debug(
                f"Skipping cache for non-serializable response (type: {type(response)})"
            )
            return

        cache_key = self._get_request_hash(
            messages, model, temperature, max_tokens, extra_params
        )
        current_time = time.time()

        cached_item = {
            "response": response,
            "timestamp": current_time,
            "model": model,
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_params": extra_params,
            },
        }

        # Store in memory cache
        self.memory_cache[cache_key] = cached_item

        # Store in disk cache
        cache_file = self.cache_dir / f"llm_{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cached_item, f)
            logger.debug(f"Cached LLM response: {cache_key} (model: {model})")
        except Exception as e:
            logger.warning(f"Failed to cache LLM response {cache_key}: {e}")

        # Clean up if cache is too large
        self._cleanup_if_needed()

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached item is still valid based on TTL."""
        return (time.time() - timestamp) < self.ttl_seconds

    def _cleanup_if_needed(self) -> None:
        """Clean up cache if it exceeds max size."""
        cache_files = list(self.cache_dir.glob("llm_*.pkl"))
        if len(cache_files) > self.max_size:
            # Sort by modification time and remove oldest
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            files_to_remove = cache_files[: -self.max_size]

            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    # Also remove from memory cache
                    cache_key = file_path.stem.replace("llm_", "")
                    self.memory_cache.pop(cache_key, None)
                except Exception as e:
                    logger.warning(f"Failed to remove LLM cache file {file_path}: {e}")

    def clear_expired(self) -> int:
        """Clear all expired cache entries and return count of removed items."""
        removed_count = 0

        # Clear expired memory cache items
        expired_keys = []
        for key, item in self.memory_cache.items():
            if not self._is_cache_valid(item["timestamp"]):
                expired_keys.append(key)

        for key in expired_keys:
            del self.memory_cache[key]
            removed_count += 1

        # Clear expired disk cache items
        for cache_file in self.cache_dir.glob("llm_*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    cached_item = pickle.load(f)
                    if not self._is_cache_valid(cached_item["timestamp"]):
                        cache_file.unlink()
                        removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to check cache file {cache_file}: {e}")
                # Remove corrupted file
                cache_file.unlink(missing_ok=True)
                removed_count += 1

        logger.info(f"Cleared {removed_count} expired LLM cache entries")
        return removed_count

    def clear_all(self) -> None:
        """Clear all cached data."""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("llm_*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove LLM cache file {cache_file}: {e}")
        logger.info("Cleared all LLM cache data")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_files = list(self.cache_dir.glob("llm_*.pkl"))

        return {
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": len(disk_files),
            "max_size": self.max_size,
            "ttl_hours": self.ttl_seconds / 3600,
            "cache_directory": str(self.cache_dir),
        }


class AgentCache(LLMCache):
    """Specialized cache for AI agent responses."""

    def __init__(
        self,
        cache_dir: str = ".agent_cache",
        max_size: int = SystemLimits.AGENT_CACHE_SIZE,
        ttl_hours: int = SystemLimits.CACHE_TTL_HOURS,
    ):
        super().__init__(cache_dir, max_size, ttl_hours)

    def _get_request_hash(
        self,
        messages: Union[str, list],
        model: str,
        temperature: float = None,
        max_tokens: int = None,
        extra_params: Dict = None,
    ) -> str:
        """Generate hash for agent request parameters with additional context."""
        try:
            # Include agent-specific parameters
            agent_data = {
                "messages": str(messages),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_params": extra_params or {},
                "request_type": "agent",
            }

            request_string = json.dumps(agent_data, sort_keys=True)
            return hashlib.sha256(request_string.encode()).hexdigest()[
                : SystemLimits.HASH_LENGTH
            ]

        except Exception as e:
            logger.warning(f"Failed to hash agent request: {e}")
            return hashlib.sha256(str(messages).encode()).hexdigest()[
                : SystemLimits.HASH_LENGTH
            ]

    def _cleanup_if_needed(self) -> None:
        """Clean up agent cache files."""
        cache_files = list(self.cache_dir.glob("llm_*.pkl"))
        if len(cache_files) > self.max_size:
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            files_to_remove = cache_files[: -self.max_size]

            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    cache_key = file_path.stem.replace("llm_", "")
                    self.memory_cache.pop(cache_key, None)
                except Exception as e:
                    logger.warning(
                        f"Failed to remove agent cache file {file_path}: {e}"
                    )


# Global cache instances
llm_cache = LLMCache()
agent_cache = AgentCache()


def cached_llm_response(cache_instance: LLMCache = llm_cache):
    """Decorator for caching LLM responses."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract cache-relevant parameters
            self_arg = args[0] if args else None
            messages = None
            model = None
            temperature = None
            max_tokens = None

            # Try to extract parameters from different LLM call patterns
            if hasattr(self_arg, "llm"):
                model = getattr(self_arg.llm, "model_name", "unknown")
                temperature = getattr(self_arg.llm, "temperature", None)
                max_tokens = getattr(self_arg.llm, "max_tokens", None)

            # Look for messages in args or kwargs
            if len(args) > 1:
                messages = args[1]  # Common pattern: self, messages
            elif "messages" in kwargs:
                messages = kwargs["messages"]
            elif "prompt" in kwargs:
                messages = kwargs["prompt"]
            elif len(args) > 0 and isinstance(args[-1], (str, list)):
                messages = args[-1]

            if messages and model:
                # Try to get from cache
                cached_result = cache_instance.get(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if cached_result is not None:
                    logger.info(f"Using cached LLM response for model {model}")
                    return cached_result

            # Call original function
            result = func(*args, **kwargs)

            # Cache the result if we have the required parameters
            if messages and model and result is not None:
                cache_instance.set(
                    messages=messages,
                    model=model,
                    response=result,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            return result

        return wrapper

    return decorator


def cached_agent_response(cache_instance: AgentCache = agent_cache):
    """Decorator for caching AI agent responses."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract agent-specific parameters
            agent_self = args[0] if args else None
            task_description = None
            model = None

            # Extract task/query from common agent patterns
            if len(args) > 1:
                task_description = str(args[1])
            elif "query" in kwargs:
                task_description = str(kwargs["query"])
            elif "task" in kwargs:
                task_description = str(kwargs["task"])
            elif "ingredient_name" in kwargs:
                task_description = f"ingredient_search:{kwargs['ingredient_name']}"

            if hasattr(agent_self, "llm") and hasattr(agent_self.llm, "model_name"):
                model = agent_self.llm.model_name
            elif hasattr(agent_self, "model"):
                model = agent_self.model
            else:
                model = "agent_unknown"

            if task_description and model:
                # Try to get from cache
                cached_result = cache_instance.get(
                    messages=task_description,
                    model=model,
                    extra_params={"function": func.__name__},
                )

                if cached_result is not None:
                    logger.info(f"Using cached agent response for {func.__name__}")
                    return cached_result

            # Call original function
            result = func(*args, **kwargs)

            # Cache the result
            if task_description and model and result is not None:
                cache_instance.set(
                    messages=task_description,
                    model=model,
                    response=result,
                    extra_params={"function": func.__name__},
                )

            return result

        return wrapper

    return decorator
