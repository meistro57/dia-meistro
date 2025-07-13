"""Model loading utilities for the Dia text-to-speech model.

This module provides shared model loading functionality to reduce
code duplication between CLI, Gradio app, and other interfaces.
"""

import os
from pathlib import Path
from typing import Optional, Union

import torch

from .model import Dia, ComputeDtype
from .config import DiaConfig
from .exceptions import ModelLoadError, ValidationError, DeviceError
from .validation import validate_device
from .runtime_config import get_runtime_config
from .memory_utils import gpu_memory_cleanup


class ModelLoader:
    """Centralized model loading with caching and optimization."""
    
    def __init__(self):
        self._loaded_models = {}
        self._config = get_runtime_config()
    
    def load_model(self,
                   model_source: Optional[str] = None,
                   config_path: Optional[str] = None,
                   checkpoint_path: Optional[str] = None,
                   compute_dtype: Union[str, ComputeDtype] = None,
                   device: Optional[torch.device] = None,
                   load_dac: bool = True,
                   force_reload: bool = False) -> Dia:
        """Load a Dia model with optimized loading and caching.
        
        Args:
            model_source: Hugging Face model ID or 'local' for local loading.
            config_path: Path to local config file (required if model_source='local').
            checkpoint_path: Path to local checkpoint (required if model_source='local').
            compute_dtype: Computation data type.
            device: Target device.
            load_dac: Whether to load the DAC model.
            force_reload: Whether to force reload even if cached.
            
        Returns:
            Loaded Dia model.
            
        Raises:
            ModelLoadError: If model loading fails.
            ValidationError: If parameters are invalid.
        """
        # Set defaults
        if model_source is None:
            model_source = self._config.default_model_repo
        if compute_dtype is None:
            compute_dtype = self._config.default_compute_dtype
        if device is None:
            device = self._get_default_device()
        
        # Validate device
        device = validate_device(device)
        
        # Create cache key
        cache_key = self._create_cache_key(
            model_source, config_path, checkpoint_path, 
            compute_dtype, device, load_dac
        )
        
        # Check cache if not forcing reload
        if not force_reload and cache_key in self._loaded_models:
            cached_model = self._loaded_models[cache_key]
            if self._validate_cached_model(cached_model, device):
                return cached_model
            else:
                # Remove invalid cached model
                del self._loaded_models[cache_key]
        
        # Load the model
        try:
            with gpu_memory_cleanup(device):
                if model_source == 'local':
                    model = self._load_local_model(
                        config_path, checkpoint_path, compute_dtype, device, load_dac
                    )
                else:
                    model = self._load_pretrained_model(
                        model_source, compute_dtype, device, load_dac
                    )
                
                # Cache the loaded model
                self._loaded_models[cache_key] = model
                
                return model
                
        except Exception as e:
            if isinstance(e, (ModelLoadError, ValidationError, DeviceError)):
                raise
            raise ModelLoadError(f"Unexpected error loading model: {e}")
    
    def _get_default_device(self) -> torch.device:
        """Get the default device for model loading."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _create_cache_key(self, model_source, config_path, checkpoint_path, 
                         compute_dtype, device, load_dac) -> str:
        """Create a cache key for the model."""
        key_parts = [
            str(model_source),
            str(config_path) if config_path else "None",
            str(checkpoint_path) if checkpoint_path else "None",
            str(compute_dtype),
            str(device),
            str(load_dac)
        ]
        return "|".join(key_parts)
    
    def _validate_cached_model(self, model: Dia, target_device: torch.device) -> bool:
        """Validate that a cached model is still usable."""
        try:
            # Check if model is on the correct device
            if model.device != target_device:
                return False
            
            # Check if model is still in eval mode
            if model.model.training:
                model.model.eval()
            
            return True
            
        except Exception:
            return False
    
    def _load_local_model(self, config_path: str, checkpoint_path: str,
                         compute_dtype: Union[str, ComputeDtype],
                         device: torch.device, load_dac: bool) -> Dia:
        """Load model from local files."""
        # Validate input paths
        if not config_path or not checkpoint_path:
            raise ValidationError("Both config_path and checkpoint_path are required for local loading")
        
        if not os.path.exists(config_path):
            raise ModelLoadError(f"Config file not found: {config_path}", config_path)
        
        if not os.path.exists(checkpoint_path):
            raise ModelLoadError(f"Checkpoint file not found: {checkpoint_path}", checkpoint_path)
        
        print(f"Loading model from local files:")
        print(f"  Config: {config_path}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Device: {device}")
        print(f"  Dtype: {compute_dtype}")
        
        try:
            return Dia.from_local(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                compute_dtype=compute_dtype,
                device=device,
                load_dac=load_dac
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load local model: {e}")
    
    def _load_pretrained_model(self, model_name: str,
                              compute_dtype: Union[str, ComputeDtype],
                              device: torch.device, load_dac: bool) -> Dia:
        """Load model from Hugging Face Hub."""
        print(f"Loading model from Hugging Face Hub:")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Dtype: {compute_dtype}")
        
        try:
            return Dia.from_pretrained(
                model_name=model_name,
                compute_dtype=compute_dtype,
                device=device,
                load_dac=load_dac
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load pretrained model '{model_name}': {e}")
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._loaded_models.clear()
    
    def get_cache_info(self) -> dict:
        """Get information about cached models."""
        return {
            "num_cached_models": len(self._loaded_models),
            "cached_keys": list(self._loaded_models.keys())
        }


# Global model loader instance
_global_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = ModelLoader()
    return _global_loader


def load_model_smart(model_source: Optional[str] = None,
                    config_path: Optional[str] = None,
                    checkpoint_path: Optional[str] = None,
                    compute_dtype: Optional[str] = None,
                    device: Optional[str] = None,
                    load_dac: bool = True,
                    force_reload: bool = False) -> Dia:
    """Smart model loading with automatic parameter detection.
    
    This is a convenience function that provides intelligent defaults
    and parameter detection for common use cases.
    
    Args:
        model_source: Model source ('local' or HF model ID).
        config_path: Path to config file (for local loading).
        checkpoint_path: Path to checkpoint file (for local loading).
        compute_dtype: Computation data type.
        device: Target device.
        load_dac: Whether to load DAC model.
        force_reload: Whether to force reload.
        
    Returns:
        Loaded Dia model.
    """
    loader = get_model_loader()
    
    # Auto-detect local loading
    if model_source is None:
        if config_path and checkpoint_path:
            model_source = 'local'
    
    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)
    
    return loader.load_model(
        model_source=model_source,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        compute_dtype=compute_dtype,
        device=device,
        load_dac=load_dac,
        force_reload=force_reload
    )


def validate_model_paths(config_path: Optional[str] = None,
                        checkpoint_path: Optional[str] = None) -> dict:
    """Validate model file paths and return information.
    
    Args:
        config_path: Path to config file.
        checkpoint_path: Path to checkpoint file.
        
    Returns:
        Dict with validation results and file information.
    """
    results = {
        "config_valid": False,
        "checkpoint_valid": False,
        "config_size": None,
        "checkpoint_size": None,
        "errors": []
    }
    
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            results["config_valid"] = True
            results["config_size"] = config_file.stat().st_size
        else:
            results["errors"].append(f"Config file not found: {config_path}")
    
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            results["checkpoint_valid"] = True
            results["checkpoint_size"] = checkpoint_file.stat().st_size
        else:
            results["errors"].append(f"Checkpoint file not found: {checkpoint_path}")
    
    return results


def detect_device_capabilities() -> dict:
    """Detect available device capabilities for model loading.
    
    Returns:
        Dict with device capability information.
    """
    capabilities = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": False,
        "cpu_available": True,
        "recommended_device": "cpu",
        "cuda_devices": 0,
        "memory_info": {}
    }
    
    # Check CUDA
    if torch.cuda.is_available():
        capabilities["cuda_devices"] = torch.cuda.device_count()
        capabilities["recommended_device"] = "cuda"
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            capabilities["memory_info"][f"cuda:{i}"] = {
                "total_memory_gb": props.total_memory / (1024**3),
                "name": props.name
            }
    
    # Check MPS (Metal Performance Shaders)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        capabilities["mps_available"] = True
        if not capabilities["cuda_available"]:
            capabilities["recommended_device"] = "mps"
    
    return capabilities