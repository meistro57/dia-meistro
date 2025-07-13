"""Business logic backend for the Gradio interface.

This module separates the core generation logic from the UI components,
making the code more maintainable and testable.
"""

import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import soundfile as sf

from .model_loader import load_model_smart
from .exceptions import DiaError, ValidationError, GenerationError, AudioProcessingError
from .validation import validate_text_input, validate_generation_parameters
from .audio_utils import create_temp_audio_file, validate_audio_tensor
from .memory_utils import get_memory_stats


class GradioGenerationService:
    """Service class for handling generation requests from Gradio interface."""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """Initialize the generation service.
        
        Args:
            model_config: Configuration for model loading.
        """
        self.model = None
        self.model_config = model_config or {}
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model with error handling."""
        try:
            print("Loading Dia model...")
            self.model = load_model_smart(**self.model_config)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_audio(self,
                      text_input: str,
                      audio_prompt_text_input: str = "",
                      audio_prompt_input: Optional[Tuple[int, np.ndarray]] = None,
                      max_new_tokens: int = 3072,
                      cfg_scale: float = 3.0,
                      temperature: float = 1.8,
                      top_p: float = 0.95,
                      cfg_filter_top_k: int = 45,
                      speed_factor: float = 1.0,
                      seed: Optional[int] = None) -> Tuple[Tuple[int, np.ndarray], int, str]:
        """Generate audio from text input with comprehensive error handling.
        
        Args:
            text_input: The text to convert to speech.
            audio_prompt_text_input: Transcript of the audio prompt.
            audio_prompt_input: Audio prompt data (sample_rate, audio_array).
            max_new_tokens: Maximum tokens to generate.
            cfg_scale: Classifier-free guidance scale.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            cfg_filter_top_k: Top-k filtering parameter.
            speed_factor: Audio speed adjustment factor.
            seed: Random seed for reproducibility.
            
        Returns:
            Tuple of (audio_output, used_seed, console_output).
        """
        console_log = []
        
        try:
            # Validate inputs
            self._validate_generation_inputs(
                text_input, audio_prompt_text_input, audio_prompt_input,
                max_new_tokens, cfg_scale, temperature, top_p, cfg_filter_top_k
            )
            
            # Process text input
            final_text = self._process_text_input(text_input, audio_prompt_text_input, audio_prompt_input)
            console_log.append(f"Processing text: {final_text[:100]}...")
            
            # Process audio prompt
            audio_prompt_path = None
            if audio_prompt_input is not None:
                audio_prompt_path = self._process_audio_prompt(audio_prompt_input, console_log)
            
            # Set random seed
            used_seed = self._set_generation_seed(seed)
            console_log.append(f"Using seed: {used_seed}")
            
            # Monitor memory
            memory_before = get_memory_stats(self.model.device)
            console_log.append(f"GPU memory before generation: {memory_before.get('allocated_mb', 0):.1f}MB")
            
            # Generate audio
            console_log.append("Starting audio generation...")
            start_time = time.time()
            
            audio_np = self.model.generate(
                text=final_text,
                max_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                use_torch_compile=False,  # Keep False for Gradio stability
                audio_prompt=audio_prompt_path,
                verbose=True,
            )
            
            generation_time = time.time() - start_time
            console_log.append(f"Generation completed in {generation_time:.2f} seconds")
            
            # Post-process audio
            output_audio = self._post_process_audio(audio_np, speed_factor, console_log)
            
            # Memory after generation
            memory_after = get_memory_stats(self.model.device)
            console_log.append(f"GPU memory after generation: {memory_after.get('allocated_mb', 0):.1f}MB")
            
            return output_audio, used_seed, "\n".join(console_log)
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            console_log.append(error_msg)
            
            # Return default audio on error
            default_audio = (44100, np.zeros(1000, dtype=np.int16))
            return default_audio, seed or 0, "\n".join(console_log)
        
        finally:
            # Cleanup temporary files
            if audio_prompt_path and Path(audio_prompt_path).exists():
                try:
                    Path(audio_prompt_path).unlink()
                except OSError:
                    pass
    
    def _validate_generation_inputs(self, text_input: str, audio_prompt_text: str,
                                   audio_prompt_input: Optional[Tuple], max_new_tokens: int,
                                   cfg_scale: float, temperature: float, top_p: float,
                                   cfg_filter_top_k: int) -> None:
        """Validate all generation inputs."""
        # Validate text input
        if not text_input or text_input.isspace():
            raise ValidationError("Text input cannot be empty", "text_input")
        
        validate_text_input(text_input)
        
        # Validate audio prompt consistency
        if audio_prompt_input and (not audio_prompt_text or audio_prompt_text.isspace()):
            raise ValidationError("Audio prompt text is required when audio prompt is provided", "audio_prompt_text")
        
        # Validate generation parameters
        validate_generation_parameters(max_new_tokens, cfg_scale, temperature, top_p, cfg_filter_top_k)
    
    def _process_text_input(self, text_input: str, audio_prompt_text: str,
                           audio_prompt_input: Optional[Tuple]) -> str:
        """Process and combine text inputs."""
        if audio_prompt_input and audio_prompt_text and not audio_prompt_text.isspace():
            # Prepend transcript text if audio prompt is provided
            final_text = audio_prompt_text.strip() + "\n" + text_input.strip()
        else:
            final_text = text_input.strip()
        
        return final_text
    
    def _process_audio_prompt(self, audio_prompt_input: Tuple[int, np.ndarray],
                             console_log: list) -> Optional[str]:
        """Process audio prompt and return temporary file path."""
        sr, audio_data = audio_prompt_input
        
        # Validate audio data
        if audio_data is None or audio_data.size == 0:
            console_log.append("Warning: Audio prompt is empty, ignoring")
            return None
        
        if np.all(np.abs(audio_data) < 1e-6):
            console_log.append("Warning: Audio prompt appears to be silent, ignoring")
            return None
        
        try:
            # Validate the audio tensor
            validate_audio_tensor(audio_data, "audio_prompt")
            
            # Preprocess audio data
            processed_audio = self._preprocess_audio_data(audio_data)
            
            # Create temporary file
            temp_path = create_temp_audio_file(processed_audio, sr, suffix=".wav")
            console_log.append(f"Created temporary audio prompt: {temp_path}")
            
            return temp_path
            
        except Exception as e:
            console_log.append(f"Error processing audio prompt: {e}")
            return None
    
    def _preprocess_audio_data(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio data for consistency."""
        # Convert to float32 in [-1, 1] range if integer type
        if np.issubdtype(audio_data.dtype, np.integer):
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(np.float32) / max_val
        elif not np.issubdtype(audio_data.dtype, np.floating):
            # Attempt conversion for other types
            audio_data = audio_data.astype(np.float32)
        
        # Ensure mono (average channels if stereo)
        if audio_data.ndim > 1:
            if audio_data.shape[0] == 2:  # Assume (2, N)
                audio_data = np.mean(audio_data, axis=0)
            elif audio_data.shape[1] == 2:  # Assume (N, 2)
                audio_data = np.mean(audio_data, axis=1)
            else:
                # Take first channel for unexpected shapes
                audio_data = audio_data.flatten() if audio_data.size > 0 else audio_data[:, 0]
        
        # Ensure contiguous array
        return np.ascontiguousarray(audio_data)
    
    def _set_generation_seed(self, seed: Optional[int]) -> int:
        """Set random seed for generation."""
        import random
        import torch
        
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        
        # Set all random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        return seed
    
    def _post_process_audio(self, audio_np: np.ndarray, speed_factor: float,
                           console_log: list) -> Tuple[int, np.ndarray]:
        """Post-process generated audio."""
        if audio_np is None:
            console_log.append("Warning: No audio generated")
            return (44100, np.zeros(1000, dtype=np.int16))
        
        sample_rate = 44100  # Default sample rate
        
        # Apply speed adjustment
        if speed_factor != 1.0:
            audio_np = self._adjust_audio_speed(audio_np, speed_factor, console_log)
        
        # Convert to int16 for Gradio
        if audio_np.dtype in [np.float32, np.float64]:
            # Clip to [-1, 1] and convert to int16
            audio_clipped = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
        else:
            audio_int16 = audio_np.astype(np.int16)
        
        console_log.append(f"Audio shape: {audio_int16.shape}, Sample rate: {sample_rate}Hz")
        
        return (sample_rate, audio_int16)
    
    def _adjust_audio_speed(self, audio: np.ndarray, speed_factor: float,
                           console_log: list) -> np.ndarray:
        """Adjust audio playback speed."""
        speed_factor = max(0.1, min(speed_factor, 5.0))  # Clamp to reasonable range
        
        if speed_factor == 1.0:
            return audio
        
        original_len = len(audio)
        target_len = int(original_len / speed_factor)
        
        if target_len > 0:
            x_original = np.arange(original_len)
            x_resampled = np.linspace(0, original_len - 1, target_len)
            resampled_audio = np.interp(x_resampled, x_original, audio)
            console_log.append(f"Adjusted speed: {speed_factor:.2f}x ({original_len} -> {target_len} samples)")
            return resampled_audio.astype(audio.dtype)
        else:
            console_log.append("Warning: Speed adjustment failed, using original audio")
            return audio
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "device": str(self.model.device),
            "compute_dtype": str(self.model.compute_dtype),
            "dac_loaded": self.model._dac_loaded,
            "config": {
                "max_tokens": self.model.config.decoder_config.max_position_embeddings,
                "vocab_size": self.model.config.decoder_config.vocab_size,
                "num_channels": self.model.config.decoder_config.num_channels,
            }
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return get_memory_stats(self.model.device)
    
    def reload_model(self, **new_config) -> bool:
        """Reload the model with new configuration.
        
        Args:
            **new_config: New configuration parameters.
            
        Returns:
            True if reload was successful, False otherwise.
        """
        try:
            self.model_config.update(new_config)
            self._load_model()
            return True
        except Exception as e:
            print(f"Failed to reload model: {e}")
            return False