"""Input validation utilities for the Dia text-to-speech model.

This module provides validation functions for various inputs to ensure
data integrity and provide clear error messages for invalid inputs.
"""

import os
from pathlib import Path
from typing import Any

import torch

from .exceptions import ValidationError, AudioProcessingError, ResourceError
from .runtime_config import get_runtime_config


def validate_text_input(text: str | list[str]) -> None:
    """Validate text input for generation.
    
    Args:
        text: Text string or list of text strings to validate.
        
    Raises:
        ValidationError: If text input is invalid.
    """
    config = get_runtime_config()
    max_length = config.max_text_length_bytes
    
    if isinstance(text, str):
        if not text or text.isspace():
            raise ValidationError("Text input cannot be empty or whitespace only", "text")
        if len(text.encode('utf-8')) > max_length:
            raise ValidationError(f"Text input too long (max {max_length} bytes)", "text")
    elif isinstance(text, list):
        if not text:
            raise ValidationError("Text list cannot be empty", "text")
        for i, t in enumerate(text):
            if not isinstance(t, str):
                raise ValidationError(f"Text item {i} must be a string", "text")
            if not t or t.isspace():
                raise ValidationError(f"Text item {i} cannot be empty or whitespace only", "text")
            if len(t.encode('utf-8')) > max_length:
                raise ValidationError(f"Text item {i} too long (max {max_length} bytes)", "text")
    else:
        raise ValidationError("Text must be a string or list of strings", "text")


def validate_audio_file(file_path: str) -> None:
    """Validate audio file path and basic properties.
    
    Args:
        file_path: Path to the audio file.
        
    Raises:
        ValidationError: If file path is invalid.
        AudioProcessingError: If file cannot be read or is invalid format.
    """
    config = get_runtime_config()
    
    if not isinstance(file_path, str):
        raise ValidationError("Audio file path must be a string", "audio_file")
    
    if not file_path:
        raise ValidationError("Audio file path cannot be empty", "audio_file")
    
    # Check for path traversal attempts (if security checks are enabled)
    if config.path_traversal_protection:
        normalized_path = os.path.normpath(file_path)
        if ".." in normalized_path or normalized_path.startswith("/"):
            if not os.path.abspath(normalized_path).startswith(os.getcwd()):
                raise ValidationError("Audio file path not allowed (security restriction)", "audio_file")
    
    path = Path(file_path)
    if not path.exists():
        raise AudioProcessingError(f"Audio file not found: {file_path}", file_path)
    
    if not path.is_file():
        raise AudioProcessingError(f"Audio path is not a file: {file_path}", file_path)
    
    # Check file size
    max_size_bytes = config.max_audio_file_size_mb * 1024 * 1024
    file_size = path.stat().st_size
    if file_size > max_size_bytes:
        max_size_mb = config.max_audio_file_size_mb
        actual_size_mb = file_size / 1024 / 1024
        raise AudioProcessingError(f"Audio file too large: {actual_size_mb:.1f}MB (max {max_size_mb}MB)", file_path)
    
    # Check file extension
    if path.suffix.lower() not in config.supported_audio_extensions:
        supported_formats = ', '.join(config.supported_audio_extensions)
        raise AudioProcessingError(f"Unsupported audio format: {path.suffix} (supported: {supported_formats})", file_path)


def validate_generation_parameters(
    max_tokens: int | None = None,
    cfg_scale: float = 3.0,
    temperature: float = 1.2,
    top_p: float = 0.95,
    cfg_filter_top_k: int = 45,
) -> None:
    """Validate generation parameters.
    
    Args:
        max_tokens: Maximum number of tokens to generate.
        cfg_scale: Classifier-free guidance scale.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        cfg_filter_top_k: Top-k filtering for CFG.
        
    Raises:
        ValidationError: If any parameter is invalid.
    """
    config = get_runtime_config()
    
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValidationError("max_tokens must be a positive integer", "max_tokens")
        if max_tokens > config.max_generation_tokens:
            raise ValidationError(f"max_tokens too large (max {config.max_generation_tokens})", "max_tokens")
    
    if not isinstance(cfg_scale, (int, float)) or cfg_scale < config.min_cfg_scale:
        raise ValidationError(f"cfg_scale must be >= {config.min_cfg_scale}", "cfg_scale")
    if cfg_scale > config.max_cfg_scale:
        raise ValidationError(f"cfg_scale too large (max {config.max_cfg_scale})", "cfg_scale")
    
    if not isinstance(temperature, (int, float)) or temperature < config.min_temperature:
        raise ValidationError(f"temperature must be >= {config.min_temperature}", "temperature")
    if temperature > config.max_temperature:
        raise ValidationError(f"temperature too large (max {config.max_temperature})", "temperature")
    
    if not isinstance(top_p, (int, float)) or not (config.min_top_p <= top_p <= config.max_top_p):
        raise ValidationError(f"top_p must be between {config.min_top_p} and {config.max_top_p}", "top_p")
    
    if not isinstance(cfg_filter_top_k, int) or cfg_filter_top_k < config.min_top_k:
        raise ValidationError(f"cfg_filter_top_k must be >= {config.min_top_k}", "cfg_filter_top_k")
    if cfg_filter_top_k > config.max_top_k:
        raise ValidationError(f"cfg_filter_top_k too large (max {config.max_top_k})", "cfg_filter_top_k")


def validate_device(device: torch.device | str | None) -> torch.device:
    """Validate and normalize device specification.
    
    Args:
        device: Device specification.
        
    Returns:
        Validated torch.device object.
        
    Raises:
        ValidationError: If device specification is invalid.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(device, str):
        try:
            device = torch.device(device)
        except Exception as e:
            raise ValidationError(f"Invalid device specification: {device}") from e
    
    if not isinstance(device, torch.device):
        raise ValidationError("Device must be a torch.device, string, or None", "device")
    
    # Validate device availability
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValidationError("CUDA device specified but CUDA is not available", "device")
    
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValidationError("MPS device specified but MPS is not available", "device")
    
    return device


def validate_batch_size(batch_size: int, max_batch_size: int | None = None) -> None:
    """Validate batch size for generation.
    
    Args:
        batch_size: Number of items in the batch.
        max_batch_size: Maximum allowed batch size (uses runtime config if None).
        
    Raises:
        ValidationError: If batch size is invalid.
    """
    config = get_runtime_config()
    if max_batch_size is None:
        max_batch_size = config.max_batch_size
    
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValidationError("Batch size must be a positive integer", "batch_size")
    
    if batch_size > max_batch_size:
        raise ValidationError(f"Batch size too large (max {max_batch_size})", "batch_size")


def check_available_memory(required_gb: float | None = None) -> None:
    """Check if sufficient memory is available for generation.
    
    Args:
        required_gb: Required memory in GB (uses runtime config if None).
        
    Raises:
        ResourceError: If insufficient memory is available.
    """
    config = get_runtime_config()
    if not config.memory_check_enabled:
        return
    
    if required_gb is None:
        required_gb = config.gpu_memory_required_gb
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = total_memory - torch.cuda.memory_allocated(0)
        free_gb = free_memory / (1024**3)
        
        if free_gb < required_gb:
            raise ResourceError(
                f"Insufficient GPU memory: {free_gb:.1f}GB available, {required_gb:.1f}GB required",
                "gpu_memory"
            )
    else:
        # For CPU, we can't easily check available RAM, so we'll skip this check
        pass


def validate_output_path(output_path: str) -> None:
    """Validate output file path.
    
    Args:
        output_path: Path where output will be saved.
        
    Raises:
        ValidationError: If output path is invalid.
    """
    if not isinstance(output_path, str):
        raise ValidationError("Output path must be a string", "output_path")
    
    if not output_path:
        raise ValidationError("Output path cannot be empty", "output_path")
    
    # Check for path traversal attempts
    normalized_path = os.path.normpath(output_path)
    if ".." in normalized_path:
        abs_path = os.path.abspath(normalized_path)
        if not abs_path.startswith(os.getcwd()):
            raise ValidationError("Output path not allowed (security restriction)", "output_path")
    
    # Check if directory exists or can be created
    output_dir = os.path.dirname(output_path) or "."
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise ValidationError(f"Cannot create output directory: {e}", "output_path")
    
    # Check write permissions
    if not os.access(output_dir, os.W_OK):
        raise ValidationError(f"No write permission for output directory: {output_dir}", "output_path")