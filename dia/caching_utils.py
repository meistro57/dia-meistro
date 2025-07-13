"""Caching utilities for the Dia text-to-speech model.

This module provides caching mechanisms for model compilation,
component-level caching, and other performance optimizations.
"""

import hashlib
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Union
import warnings

import torch

from .runtime_config import get_runtime_config
from .exceptions import ResourceError


class CompilationCache:
    """Cache for torch.compile compiled functions."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize compilation cache.
        
        Args:
            cache_dir: Directory to store cache files (uses temp dir if None).
        """
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()
        
        self.cache_dir = Path(cache_dir) / "dia_compilation_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache = {}
        self._enabled = get_runtime_config().compilation_cache_enabled
    
    def get_cache_key(self, func: Callable, *args, **kwargs) -> str:
        """Generate cache key for a function and its arguments.
        
        Args:
            func: Function to cache.
            *args: Function arguments.
            **kwargs: Function keyword arguments.
            
        Returns:
            Cache key string.
        """
        # Create a hash from function name and serializable arguments
        key_data = {
            "func_name": func.__name__,
            "module": getattr(func, "__module__", ""),
            "args_hash": self._hash_args(args, kwargs)
        }
        
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _hash_args(self, args, kwargs) -> str:
        """Create hash from function arguments."""
        try:
            # Convert torch tensors to shape/dtype info for hashing
            processed_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    processed_args.append(f"tensor_{arg.shape}_{arg.dtype}")
                else:
                    processed_args.append(str(arg))
            
            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    processed_kwargs[k] = f"tensor_{v.shape}_{v.dtype}"
                else:
                    processed_kwargs[k] = str(v)
            
            combined = {"args": processed_args, "kwargs": processed_kwargs}
            return hashlib.md5(str(combined).encode()).hexdigest()
            
        except Exception:
            # Fallback to simple string representation
            return hashlib.md5(f"{args}_{kwargs}".encode()).hexdigest()
    
    def get_compiled_function(self, func: Callable, compile_kwargs: Dict[str, Any] = None) -> Callable:
        """Get compiled version of function with caching.
        
        Args:
            func: Function to compile.
            compile_kwargs: Arguments for torch.compile.
            
        Returns:
            Compiled function.
        """
        if not self._enabled:
            return func
        
        compile_kwargs = compile_kwargs or {}
        cache_key = f"{func.__name__}_{compile_kwargs}"
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Compile the function
        try:
            print(f"Compiling function {func.__name__}...")
            compiled_func = torch.compile(func, **compile_kwargs)
            
            # Cache in memory
            self._memory_cache[cache_key] = compiled_func
            
            print(f"Function {func.__name__} compiled and cached")
            return compiled_func
            
        except Exception as e:
            warnings.warn(f"Failed to compile {func.__name__}: {e}")
            return func
    
    def clear_cache(self) -> None:
        """Clear all cached compiled functions."""
        self._memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.compiled"):
            try:
                cache_file.unlink()
            except OSError:
                pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "memory_cached_functions": len(self._memory_cache),
            "cache_enabled": self._enabled,
            "cache_dir": str(self.cache_dir)
        }


class ComponentCache:
    """Cache for expensive model components and intermediate results."""
    
    def __init__(self, max_memory_mb: float = 512):
        """Initialize component cache.
        
        Args:
            max_memory_mb: Maximum memory to use for caching in MB.
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache = {}
        self._access_times = {}
        self._cache_sizes = {}
        self._total_size = 0
    
    def get(self, key: str) -> Any:
        """Get item from cache.
        
        Args:
            key: Cache key.
            
        Returns:
            Cached item or None if not found.
        """
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
    
    def put(self, key: str, value: Any, size_bytes: Optional[int] = None) -> bool:
        """Put item in cache with size tracking.
        
        Args:
            key: Cache key.
            value: Value to cache.
            size_bytes: Size in bytes (estimated if None).
            
        Returns:
            True if successfully cached, False otherwise.
        """
        if size_bytes is None:
            size_bytes = self._estimate_size(value)
        
        # Check if we need to evict items
        while self._total_size + size_bytes > self.max_memory_bytes and self._cache:
            self._evict_lru()
        
        # Add to cache if there's space
        if self._total_size + size_bytes <= self.max_memory_bytes:
            self._cache[key] = value
            self._access_times[key] = time.time()
            self._cache_sizes[key] = size_bytes
            self._total_size += size_bytes
            return True
        
        return False
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation
            return 1024  # 1KB default
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return
        
        # Find LRU item
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Remove from cache
        del self._cache[lru_key]
        del self._access_times[lru_key]
        self._total_size -= self._cache_sizes[lru_key]
        del self._cache_sizes[lru_key]
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._access_times.clear()
        self._cache_sizes.clear()
        self._total_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "num_items": len(self._cache),
            "total_size_mb": self._total_size / (1024 * 1024),
            "max_size_mb": self.max_memory_bytes / (1024 * 1024),
            "utilization_pct": (self._total_size / self.max_memory_bytes) * 100 if self.max_memory_bytes > 0 else 0
        }


class EncoderOutputCache(ComponentCache):
    """Specialized cache for encoder outputs."""
    
    def cache_encoder_output(self, text_hash: str, encoder_output: torch.Tensor) -> bool:
        """Cache encoder output for text.
        
        Args:
            text_hash: Hash of the input text.
            encoder_output: Encoder output tensor.
            
        Returns:
            True if cached successfully.
        """
        key = f"encoder_output_{text_hash}"
        return self.put(key, encoder_output.detach().cpu())
    
    def get_encoder_output(self, text_hash: str, device: torch.device) -> Optional[torch.Tensor]:
        """Get cached encoder output.
        
        Args:
            text_hash: Hash of the input text.
            device: Target device for the tensor.
            
        Returns:
            Encoder output tensor or None if not cached.
        """
        key = f"encoder_output_{text_hash}"
        output = self.get(key)
        if output is not None:
            return output.to(device)
        return None


class CrossAttentionCache(ComponentCache):
    """Specialized cache for cross-attention caches."""
    
    def cache_cross_attention(self, encoder_hash: str, cross_attn_cache: Any) -> bool:
        """Cache cross-attention data.
        
        Args:
            encoder_hash: Hash of the encoder output.
            cross_attn_cache: Cross-attention cache data.
            
        Returns:
            True if cached successfully.
        """
        key = f"cross_attn_{encoder_hash}"
        return self.put(key, cross_attn_cache)
    
    def get_cross_attention(self, encoder_hash: str) -> Any:
        """Get cached cross-attention data.
        
        Args:
            encoder_hash: Hash of the encoder output.
            
        Returns:
            Cross-attention cache or None if not cached.
        """
        key = f"cross_attn_{encoder_hash}"
        return self.get(key)


# Global cache instances
_compilation_cache: Optional[CompilationCache] = None
_encoder_cache: Optional[EncoderOutputCache] = None
_cross_attention_cache: Optional[CrossAttentionCache] = None


def get_compilation_cache() -> CompilationCache:
    """Get global compilation cache instance."""
    global _compilation_cache
    if _compilation_cache is None:
        _compilation_cache = CompilationCache()
    return _compilation_cache


def get_encoder_cache() -> EncoderOutputCache:
    """Get global encoder output cache instance."""
    global _encoder_cache
    if _encoder_cache is None:
        _encoder_cache = EncoderOutputCache(max_memory_mb=256)
    return _encoder_cache


def get_cross_attention_cache() -> CrossAttentionCache:
    """Get global cross-attention cache instance."""
    global _cross_attention_cache
    if _cross_attention_cache is None:
        _cross_attention_cache = CrossAttentionCache(max_memory_mb=256)
    return _cross_attention_cache


def clear_all_caches() -> None:
    """Clear all global caches."""
    if _compilation_cache:
        _compilation_cache.clear_cache()
    if _encoder_cache:
        _encoder_cache.clear()
    if _cross_attention_cache:
        _cross_attention_cache.clear()


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches."""
    stats = {}
    
    if _compilation_cache:
        stats["compilation"] = _compilation_cache.get_cache_stats()
    
    if _encoder_cache:
        stats["encoder"] = _encoder_cache.get_stats()
    
    if _cross_attention_cache:
        stats["cross_attention"] = _cross_attention_cache.get_stats()
    
    return stats


def create_text_hash(text: Union[str, list]) -> str:
    """Create hash for text input for caching purposes.
    
    Args:
        text: Text string or list of strings.
        
    Returns:
        Hash string.
    """
    if isinstance(text, list):
        text_str = "|".join(text)
    else:
        text_str = text
    
    return hashlib.md5(text_str.encode()).hexdigest()