"""Custom exceptions for the Dia text-to-speech model.

This module defines specific exception classes to provide better error handling
and more informative error messages throughout the Dia codebase.
"""


class DiaError(Exception):
    """Base exception class for all Dia-related errors."""

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.error_code = error_code


class ModelLoadError(DiaError):
    """Raised when there's an error loading the Dia model or its components."""

    def __init__(self, message: str, model_path: str | None = None):
        super().__init__(message, "MODEL_LOAD_ERROR")
        self.model_path = model_path


class AudioProcessingError(DiaError):
    """Raised when there's an error processing audio files or data."""

    def __init__(self, message: str, audio_path: str | None = None):
        super().__init__(message, "AUDIO_PROCESSING_ERROR")
        self.audio_path = audio_path


class GenerationError(DiaError):
    """Raised when there's an error during audio generation."""

    def __init__(self, message: str, generation_step: int | None = None):
        super().__init__(message, "GENERATION_ERROR")
        self.generation_step = generation_step


class ValidationError(DiaError):
    """Raised when input validation fails."""

    def __init__(self, message: str, parameter_name: str | None = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.parameter_name = parameter_name


class ConfigurationError(DiaError):
    """Raised when there's an error with model configuration."""

    def __init__(self, message: str, config_field: str | None = None):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_field = config_field


class ResourceError(DiaError):
    """Raised when there's an error with system resources (memory, disk, etc.)."""

    def __init__(self, message: str, resource_type: str | None = None):
        super().__init__(message, "RESOURCE_ERROR")
        self.resource_type = resource_type


class DeviceError(DiaError):
    """Raised when there's an error with device (CUDA, MPS, etc.) operations."""

    def __init__(self, message: str, device_type: str | None = None):
        super().__init__(message, "DEVICE_ERROR")
        self.device_type = device_type