"""Memory management utilities for the Dia text-to-speech model.

This module provides context managers and utilities for efficient memory
management, especially for GPU operations and large tensor handling.
"""

import contextlib
import gc
import time
from typing import Optional, Generator, Dict, Any
import warnings

import torch

from .exceptions import ResourceError
from .runtime_config import get_runtime_config


class MemoryTracker:
    """Track GPU memory usage for monitoring and debugging."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.snapshots = []
        self.start_memory = None
        
    def snapshot(self, name: str = "") -> Dict[str, Any]:
        """Take a memory snapshot."""
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return {"name": name, "timestamp": time.time(), "memory_mb": 0}
        
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2   # MB
        
        snapshot = {
            "name": name,
            "timestamp": time.time(),
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "device": str(self.device)
        }
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def start_tracking(self) -> None:
        """Start memory tracking."""
        self.start_memory = self.snapshot("start")
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage since tracking started (in MB)."""
        if not self.snapshots:
            return 0.0
        return max(s.get("allocated_mb", 0) for s in self.snapshots)
    
    def get_current_usage(self) -> float:
        """Get current memory usage (in MB)."""
        current = self.snapshot()
        return current.get("allocated_mb", 0)
    
    def get_memory_diff(self) -> float:
        """Get memory difference from start (in MB)."""
        if not self.start_memory:
            return 0.0
        current = self.get_current_usage()
        return current - self.start_memory.get("allocated_mb", 0)
    
    def print_summary(self) -> None:
        """Print memory usage summary."""
        if not self.snapshots:
            print("No memory snapshots available")
            return
        
        print("Memory Usage Summary:")
        print("-" * 40)
        for snapshot in self.snapshots:
            if "allocated_mb" in snapshot:
                print(f"{snapshot['name']}: {snapshot['allocated_mb']:.1f}MB allocated, "
                      f"{snapshot['reserved_mb']:.1f}MB reserved")
        
        if self.start_memory:
            diff = self.get_memory_diff()
            print(f"Total change: {diff:+.1f}MB")


@contextlib.contextmanager
def gpu_memory_cleanup(device: Optional[torch.device] = None, 
                      aggressive: bool = False) -> Generator[None, None, None]:
    """Context manager for automatic GPU memory cleanup.
    
    Args:
        device: The device to clean up (defaults to current device).
        aggressive: Whether to perform aggressive cleanup (slower but more thorough).
        
    Yields:
        None
        
    Example:
        with gpu_memory_cleanup():
            # GPU memory operations
            tensor = torch.randn(1000, 1000, device="cuda")
            # Memory automatically cleaned up on exit
    """
    config = get_runtime_config()
    if not config.memory_check_enabled:
        yield
        return
    
    initial_memory = None
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        elif isinstance(device, torch.device) and device.type == "cuda":
            device = device.index if device.index is not None else 0
        
        torch.cuda.synchronize(device)
        initial_memory = torch.cuda.memory_allocated(device)
    
    try:
        yield
    finally:
        if torch.cuda.is_available():
            # Clear cache and run garbage collection
            if aggressive:
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()  # Second pass for good measure
            else:
                torch.cuda.empty_cache()
            
            torch.cuda.synchronize(device)
            
            # Optional: warn about memory leaks
            if initial_memory is not None:
                final_memory = torch.cuda.memory_allocated(device)
                memory_diff = final_memory - initial_memory
                
                if memory_diff > 100 * 1024**2:  # 100MB threshold
                    warnings.warn(
                        f"Potential memory leak detected: {memory_diff / 1024**2:.1f}MB "
                        f"not freed after context exit",
                        ResourceWarning
                    )


@contextlib.contextmanager
def memory_efficient_generation(batch_size: int, 
                               device: Optional[torch.device] = None) -> Generator[MemoryTracker, None, None]:
    """Context manager for memory-efficient generation operations.
    
    Args:
        batch_size: Number of items in the batch.
        device: The device being used.
        
    Yields:
        MemoryTracker: Memory tracker instance for monitoring.
        
    Raises:
        ResourceError: If insufficient memory is available.
    """
    config = get_runtime_config()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Estimate memory requirements
    base_memory_gb = config.gpu_memory_required_gb
    estimated_memory_gb = base_memory_gb * batch_size
    
    # Check available memory
    if torch.cuda.is_available() and device.type == "cuda":
        device_props = torch.cuda.get_device_properties(device)
        total_memory = device_props.total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        available_memory = total_memory - allocated_memory
        available_gb = available_memory / (1024**3)
        
        if available_gb < estimated_memory_gb:
            raise ResourceError(
                f"Insufficient GPU memory for batch size {batch_size}: "
                f"{available_gb:.1f}GB available, {estimated_memory_gb:.1f}GB estimated required",
                "gpu_memory"
            )
    
    tracker = MemoryTracker(device)
    tracker.start_tracking()
    
    try:
        with gpu_memory_cleanup(device, aggressive=False):
            yield tracker
    finally:
        if config.memory_check_enabled:
            peak_usage = tracker.get_peak_usage()
            if peak_usage > estimated_memory_gb * 1024 * 1.5:  # 50% over estimate
                warnings.warn(
                    f"Memory usage ({peak_usage:.1f}MB) significantly exceeded "
                    f"estimate ({estimated_memory_gb * 1024:.1f}MB)",
                    ResourceWarning
                )


@contextlib.contextmanager
def temporary_tensor_cache() -> Generator[Dict[str, torch.Tensor], None, None]:
    """Context manager for temporary tensor caching within a scope.
    
    Automatically cleans up cached tensors when exiting the context.
    
    Yields:
        Dict[str, torch.Tensor]: Cache dictionary for storing tensors.
    """
    cache = {}
    try:
        yield cache
    finally:
        # Clear all cached tensors
        for key in list(cache.keys()):
            tensor = cache.pop(key)
            if isinstance(tensor, torch.Tensor):
                del tensor
        
        # Force cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def optimize_tensor_memory(tensor: torch.Tensor, 
                          target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Optimize tensor memory usage.
    
    Args:
        tensor: The tensor to optimize.
        target_dtype: Target data type (if different from current).
        
    Returns:
        torch.Tensor: The optimized tensor.
    """
    if tensor is None:
        return tensor
    
    # Convert to target dtype if specified
    if target_dtype is not None and tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)
    
    # Make contiguous if not already
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    return tensor


def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Get current memory statistics.
    
    Args:
        device: The device to check (defaults to current CUDA device).
        
    Returns:
        Dict[str, float]: Memory statistics in MB.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "available_mb": 0}
    
    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        if device.type != "cuda":
            return {"allocated_mb": 0, "reserved_mb": 0, "available_mb": 0}
        device = device.index if device.index is not None else 0
    
    props = torch.cuda.get_device_properties(device)
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    total = props.total_memory / 1024**2
    available = total - allocated
    
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "total_mb": total,
        "available_mb": available,
        "utilization_pct": (allocated / total) * 100 if total > 0 else 0
    }


def check_memory_fragmentation(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Check for GPU memory fragmentation.
    
    Args:
        device: The device to check.
        
    Returns:
        Dict[str, Any]: Fragmentation statistics.
    """
    if not torch.cuda.is_available():
        return {"fragmented": False, "reason": "CUDA not available"}
    
    stats = get_memory_stats(device)
    allocated = stats["allocated_mb"]
    reserved = stats["reserved_mb"]
    
    # Simple heuristic: if reserved >> allocated, likely fragmented
    fragmentation_ratio = reserved / allocated if allocated > 0 else 0
    fragmented = fragmentation_ratio > 2.0  # Reserved is more than 2x allocated
    
    return {
        "fragmented": fragmented,
        "fragmentation_ratio": fragmentation_ratio,
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "recommendation": "Consider calling torch.cuda.empty_cache()" if fragmented else "Memory OK"
    }