"""Runtime configuration management for the Dia model.

This module provides runtime configuration options that can be set via environment
variables or programmatically, separate from the model architecture configuration.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RuntimeConfig:
    """Runtime configuration for Dia model operations.
    
    These settings control runtime behavior and can be overridden via environment variables.
    """
    
    # Audio processing constants
    default_sample_rate: int = 44100
    sample_rate_ratio: int = 512
    max_audio_file_size_mb: int = 100
    max_text_length_bytes: int = 10000
    
    # Generation limits
    max_batch_size: int = 32
    max_generation_tokens: int = 10000
    default_max_tokens: int = 3072
    
    # Memory and performance
    memory_check_enabled: bool = True
    gpu_memory_required_gb: float = 2.0
    compilation_cache_enabled: bool = True
    
    # File I/O settings
    output_directory_permissions: int = 0o755
    temp_file_cleanup: bool = True
    
    # Validation settings
    strict_validation: bool = True
    security_checks_enabled: bool = True
    path_traversal_protection: bool = True
    
    # Model defaults
    default_model_repo: str = "nari-labs/Dia-1.6B-0626"
    default_compute_dtype: str = "float32"
    
    # Audio format support
    supported_audio_extensions: tuple = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    
    # Generation parameter bounds
    min_cfg_scale: float = 0.0
    max_cfg_scale: float = 10.0
    min_temperature: float = 0.1
    max_temperature: float = 5.0
    min_top_p: float = 0.01
    max_top_p: float = 1.0
    min_top_k: int = 1
    max_top_k: int = 1000
    
    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Create a RuntimeConfig instance from environment variables.
        
        Environment variables should be prefixed with 'DIA_' and use uppercase names.
        For example: DIA_DEFAULT_SAMPLE_RATE, DIA_MAX_BATCH_SIZE, etc.
        
        Returns:
            RuntimeConfig instance with values from environment variables.
        """
        def get_env_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(f"DIA_{name}", default))
            except ValueError:
                return default
        
        def get_env_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(f"DIA_{name}", default))
            except ValueError:
                return default
        
        def get_env_bool(name: str, default: bool) -> bool:
            value = os.getenv(f"DIA_{name}", "").lower()
            if value in ("true", "1", "yes", "on"):
                return True
            elif value in ("false", "0", "no", "off"):
                return False
            return default
        
        def get_env_str(name: str, default: str) -> str:
            return os.getenv(f"DIA_{name}", default)
        
        return cls(
            # Audio processing constants
            default_sample_rate=get_env_int("DEFAULT_SAMPLE_RATE", 44100),
            sample_rate_ratio=get_env_int("SAMPLE_RATE_RATIO", 512),
            max_audio_file_size_mb=get_env_int("MAX_AUDIO_FILE_SIZE_MB", 100),
            max_text_length_bytes=get_env_int("MAX_TEXT_LENGTH_BYTES", 10000),
            
            # Generation limits
            max_batch_size=get_env_int("MAX_BATCH_SIZE", 32),
            max_generation_tokens=get_env_int("MAX_GENERATION_TOKENS", 10000),
            default_max_tokens=get_env_int("DEFAULT_MAX_TOKENS", 3072),
            
            # Memory and performance
            memory_check_enabled=get_env_bool("MEMORY_CHECK_ENABLED", True),
            gpu_memory_required_gb=get_env_float("GPU_MEMORY_REQUIRED_GB", 2.0),
            compilation_cache_enabled=get_env_bool("COMPILATION_CACHE_ENABLED", True),
            
            # File I/O settings
            output_directory_permissions=int(get_env_str("OUTPUT_DIRECTORY_PERMISSIONS", "0o755"), 8),
            temp_file_cleanup=get_env_bool("TEMP_FILE_CLEANUP", True),
            
            # Validation settings
            strict_validation=get_env_bool("STRICT_VALIDATION", True),
            security_checks_enabled=get_env_bool("SECURITY_CHECKS_ENABLED", True),
            path_traversal_protection=get_env_bool("PATH_TRAVERSAL_PROTECTION", True),
            
            # Model defaults
            default_model_repo=get_env_str("DEFAULT_MODEL_REPO", "nari-labs/Dia-1.6B-0626"),
            default_compute_dtype=get_env_str("DEFAULT_COMPUTE_DTYPE", "float32"),
            
            # Audio format support
            supported_audio_extensions=tuple(
                get_env_str("SUPPORTED_AUDIO_EXTENSIONS", ".wav,.mp3,.flac,.ogg,.m4a").split(",")
            ),
            
            # Generation parameter bounds
            min_cfg_scale=get_env_float("MIN_CFG_SCALE", 0.0),
            max_cfg_scale=get_env_float("MAX_CFG_SCALE", 10.0),
            min_temperature=get_env_float("MIN_TEMPERATURE", 0.1),
            max_temperature=get_env_float("MAX_TEMPERATURE", 5.0),
            min_top_p=get_env_float("MIN_TOP_P", 0.01),
            max_top_p=get_env_float("MAX_TOP_P", 1.0),
            min_top_k=get_env_int("MIN_TOP_K", 1),
            max_top_k=get_env_int("MAX_TOP_K", 1000),
        )


# Global runtime configuration instance
_runtime_config: Optional[RuntimeConfig] = None


def get_runtime_config() -> RuntimeConfig:
    """Get the global runtime configuration instance.
    
    The configuration is loaded once from environment variables on first access.
    
    Returns:
        The global RuntimeConfig instance.
    """
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = RuntimeConfig.from_env()
    return _runtime_config


def set_runtime_config(config: RuntimeConfig) -> None:
    """Set the global runtime configuration instance.
    
    Args:
        config: The RuntimeConfig instance to use globally.
    """
    global _runtime_config
    _runtime_config = config


def reset_runtime_config() -> None:
    """Reset the global runtime configuration to reload from environment variables."""
    global _runtime_config
    _runtime_config = None