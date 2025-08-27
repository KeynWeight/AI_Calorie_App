# utils/cache.py
import hashlib
import json
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Callable
import logging

from .config import SystemLimits

logger = logging.getLogger(__name__)

class VLMCache:
    """Cache for VLM responses to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = ".cache", max_size: int = SystemLimits.MAX_CACHE_SIZE):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.memory_cache = {}
    
    def _get_image_hash(self, image_path: str) -> str:
        """Generate hash for image file content."""
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()[:SystemLimits.HASH_LENGTH]
        except Exception as e:
            logger.warning(f"Failed to hash image {image_path}: {e}")
            return hashlib.sha256(image_path.encode()).hexdigest()[:SystemLimits.HASH_LENGTH]
    
    def _get_cache_key(self, image_path: str, model_name: str) -> str:
        """Generate cache key for image and model combination."""
        image_hash = self._get_image_hash(image_path)
        model_hash = hashlib.sha256(model_name.encode()).hexdigest()[:8]
        return f"{image_hash}_{model_hash}"
    
    def get(self, image_path: str, model_name: str) -> Optional[Any]:
        """Get cached VLM response."""
        cache_key = self._get_cache_key(image_path, model_name)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            logger.debug(f"[CHE] Memory hit: {cache_key[:16]}...")
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    # Store in memory for faster access
                    self.memory_cache[cache_key] = data
                    logger.debug(f"[CHE] Disk hit: {cache_key[:16]}...")
                    return data
            except Exception as e:
                logger.warning(f"[CHE] Load failed: {e}")
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def set(self, image_path: str, model_name: str, data: Any) -> None:
        """Store VLM response in cache."""
        cache_key = self._get_cache_key(image_path, model_name)
        
        # Store in memory cache
        self.memory_cache[cache_key] = data
        
        # Store in disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"[CHE] Stored: {cache_key[:16]}...")
        except Exception as e:
            logger.warning(f"[CHE] Store failed: {e}")
        
        # Clean up if cache is too large
        self._cleanup_if_needed()
    
    def clear(self) -> None:
        """Clear all cached data."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache
        cache_files = list(self.cache_dir.glob("*.pkl"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"[CHE] Cache cleared: {len(cache_files)} files removed")
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if it exceeds max size."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        if len(cache_files) > self.max_size:
            # Remove oldest files
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            files_to_remove = cache_files[:-self.max_size]
            
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    # Also remove from memory cache
                    cache_key = file_path.stem
                    self.memory_cache.pop(cache_key, None)
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {file_path}: {e}")
    

# Global cache instance
vlm_cache = VLMCache()

def cached_vlm_analysis(cache_instance: VLMCache = vlm_cache):
    """Decorator for caching VLM analysis results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, image_path: str, *args, **kwargs):
            # Get model name from self
            model_name = getattr(self, 'model_name', 'unknown')
            
            # Try to get from cache
            cached_result = cache_instance.get(image_path, model_name)
            if cached_result is not None:
                logger.info(f"[CHE] Using cached analysis for {Path(image_path).name}")
                return cached_result
            
            # Call original function
            result = func(self, image_path, *args, **kwargs)
            
            # Cache the result if successful
            if result is not None:
                cache_instance.set(image_path, model_name, result)
            
            return result
        return wrapper
    return decorator