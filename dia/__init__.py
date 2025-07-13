from .model import Dia
from .exceptions import (
    DiaError,
    ModelLoadError,
    AudioProcessingError,
    GenerationError,
    ValidationError,
    ConfigurationError,
    ResourceError,
    DeviceError,
)
from .validation import (
    validate_text_input,
    validate_audio_file,
    validate_generation_parameters,
    validate_device,
    validate_batch_size,
    check_available_memory,
    validate_output_path,
)
from .runtime_config import (
    RuntimeConfig,
    get_runtime_config,
    set_runtime_config,
    reset_runtime_config,
)
from .memory_utils import (
    MemoryTracker,
    gpu_memory_cleanup,
    memory_efficient_generation,
    get_memory_stats,
)
from .audio_utils import (
    AudioProcessor,
    create_temp_audio_file,
    validate_audio_tensor,
)
from .model_loader import (
    ModelLoader,
    load_model_smart,
    get_model_loader,
)
from .caching_utils import (
    get_compilation_cache,
    get_encoder_cache,
    clear_all_caches,
)
from .gradio_backend import GradioGenerationService


__all__ = [
    # Core model
    "Dia",
    
    # Exceptions
    "DiaError",
    "ModelLoadError", 
    "AudioProcessingError",
    "GenerationError",
    "ValidationError",
    "ConfigurationError",
    "ResourceError",
    "DeviceError",
    
    # Validation functions
    "validate_text_input",
    "validate_audio_file", 
    "validate_generation_parameters",
    "validate_device",
    "validate_batch_size",
    "check_available_memory",
    "validate_output_path",
    
    # Runtime configuration
    "RuntimeConfig",
    "get_runtime_config",
    "set_runtime_config", 
    "reset_runtime_config",
    
    # Memory management
    "MemoryTracker",
    "gpu_memory_cleanup",
    "memory_efficient_generation",
    "get_memory_stats",
    
    # Audio utilities
    "AudioProcessor",
    "create_temp_audio_file",
    "validate_audio_tensor",
    
    # Model loading
    "ModelLoader",
    "load_model_smart",
    "get_model_loader",
    
    # Caching
    "get_compilation_cache",
    "get_encoder_cache", 
    "clear_all_caches",
    
    # Gradio backend
    "GradioGenerationService",
]
