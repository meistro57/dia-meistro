"""Audio processing utilities for the Dia text-to-speech model.

This module provides shared audio preprocessing functions, optimized operations,
and utilities to reduce code duplication across the codebase.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union, List
import warnings

import torch
import torchaudio
import numpy as np

from .exceptions import AudioProcessingError, ValidationError
from .validation import validate_audio_file
from .runtime_config import get_runtime_config
from .memory_utils import optimize_tensor_memory


class AudioProcessor:
    """Centralized audio processing with caching and optimization."""
    
    def __init__(self, sample_rate: Optional[int] = None, device: Optional[torch.device] = None):
        """Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate (uses runtime config if None).
            device: Target device (auto-detected if None).
        """
        config = get_runtime_config()
        self.sample_rate = sample_rate or config.default_sample_rate
        self.device = device or self._get_default_device()
        self._cache = {}
        
    def _get_default_device(self) -> torch.device:
        """Get the default device for audio processing."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def load_and_preprocess(self, 
                           audio_path: str, 
                           target_sample_rate: Optional[int] = None,
                           normalize: bool = True,
                           to_mono: bool = True) -> torch.Tensor:
        """Load and preprocess an audio file with optimizations.
        
        Args:
            audio_path: Path to the audio file.
            target_sample_rate: Target sample rate (uses instance default if None).
            normalize: Whether to normalize the audio to [-1, 1].
            to_mono: Whether to convert to mono.
            
        Returns:
            torch.Tensor: Preprocessed audio tensor.
            
        Raises:
            AudioProcessingError: If audio processing fails.
        """
        # Validate the audio file first
        validate_audio_file(audio_path)
        
        target_sr = target_sample_rate or self.sample_rate
        
        try:
            # Load audio file
            audio, sr = torchaudio.load(audio_path, channels_first=True)
            
            # Validate audio content
            if audio.numel() == 0:
                raise AudioProcessingError(f"Audio file is empty: {audio_path}", audio_path)
            
            # Check for silence
            if torch.all(torch.abs(audio) < 1e-6):
                warnings.warn(f"Audio file appears to be silent: {audio_path}", UserWarning)
            
            # Normalize if requested
            if normalize:
                audio = self._normalize_audio(audio)
            
            # Convert to mono if requested
            if to_mono and audio.shape[0] > 1:
                audio = self._convert_to_mono(audio)
            
            # Resample if necessary
            if sr != target_sr:
                audio = self._resample_audio(audio, sr, target_sr)
            
            # Optimize memory usage
            audio = optimize_tensor_memory(audio, torch.float32)
            
            return audio.to(self.device)
            
        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            raise AudioProcessingError(f"Failed to load and preprocess audio: {e}", audio_path)
    
    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        if audio.dtype in [torch.int16, torch.int32]:
            # Convert integer types to float
            max_val = torch.iinfo(audio.dtype).max
            audio = audio.float() / max_val
        elif audio.dtype == torch.float64:
            audio = audio.float()
        
        # Ensure in [-1, 1] range
        max_val = torch.max(torch.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        
        return audio
    
    def _convert_to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel audio to mono."""
        if audio.shape[0] == 1:
            return audio
        
        # Simple average across channels
        return torch.mean(audio, dim=0, keepdim=True)
    
    def _resample_audio(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate."""
        try:
            return torchaudio.functional.resample(audio, orig_sr, target_sr)
        except Exception as e:
            raise AudioProcessingError(f"Failed to resample audio from {orig_sr}Hz to {target_sr}Hz: {e}")
    
    def save_audio(self, 
                   audio: Union[torch.Tensor, np.ndarray], 
                   path: str, 
                   sample_rate: Optional[int] = None,
                   format: Optional[str] = None) -> None:
        """Save audio to file with format detection and optimization.
        
        Args:
            audio: Audio tensor or numpy array.
            path: Output file path.
            sample_rate: Sample rate (uses instance default if None).
            format: Audio format (auto-detected from extension if None).
        """
        import soundfile as sf
        
        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
        
        # Ensure correct shape (samples,) or (samples, channels)
        if audio_np.ndim == 1:
            pass  # Already correct for mono
        elif audio_np.ndim == 2:
            if audio_np.shape[0] == 1:
                audio_np = audio_np.squeeze(0)  # Remove channel dimension for mono
        else:
            raise AudioProcessingError(f"Unsupported audio shape: {audio_np.shape}")
        
        # Use instance sample rate if not specified
        sr = sample_rate or self.sample_rate
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            
            # Save audio
            sf.write(path, audio_np, sr, format=format)
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to save audio to {path}: {e}")
    
    def encode_with_dac(self, audio: torch.Tensor, dac_model) -> torch.Tensor:
        """Encode audio using DAC model with error handling.
        
        Args:
            audio: Audio tensor to encode.
            dac_model: DAC model instance.
            
        Returns:
            torch.Tensor: Encoded audio codes.
        """
        if dac_model is None:
            raise AudioProcessingError("DAC model is required but not provided")
        
        try:
            # Ensure audio is in the right shape and on the right device
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)  # Add batch dimension
            
            # Preprocess and encode
            audio_data = dac_model.preprocess(audio, self.sample_rate)
            _, encoded_frame, _, _, _ = dac_model.encode(audio_data)
            
            # Return in expected format [T, C]
            return encoded_frame.squeeze(0).transpose(0, 1)
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to encode audio with DAC: {e}")
    
    def decode_with_dac(self, audio_codes: torch.Tensor, dac_model) -> torch.Tensor:
        """Decode audio codes using DAC model.
        
        Args:
            audio_codes: Audio codes to decode.
            dac_model: DAC model instance.
            
        Returns:
            torch.Tensor: Decoded audio waveform.
        """
        if dac_model is None:
            raise AudioProcessingError("DAC model is required but not provided")
        
        try:
            # Reshape for DAC: [1, C, T]
            audio_codes = audio_codes.unsqueeze(0).transpose(1, 2)
            
            # Decode
            audio_values, _, _ = dac_model.quantizer.from_codes(audio_codes)
            audio_values = dac_model.decode(audio_values)
            
            return audio_values.squeeze()
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to decode audio with DAC: {e}")
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()


def create_temp_audio_file(audio: Union[torch.Tensor, np.ndarray],
                          sample_rate: int,
                          suffix: str = ".wav",
                          cleanup: bool = True) -> str:
    """Create a temporary audio file.
    
    Args:
        audio: Audio data.
        sample_rate: Sample rate.
        suffix: File suffix.
        cleanup: Whether to register for automatic cleanup.
        
    Returns:
        str: Path to the temporary file.
    """
    import soundfile as sf
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        temp_path = f.name
    
    try:
        # Convert tensor to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
        
        # Write audio file
        sf.write(temp_path, audio_np, sample_rate)
        
        return temp_path
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise AudioProcessingError(f"Failed to create temporary audio file: {e}")


def batch_audio_preprocessing(audio_paths: List[str],
                             processor: Optional[AudioProcessor] = None,
                             parallel: bool = False) -> List[torch.Tensor]:
    """Preprocess multiple audio files efficiently.
    
    Args:
        audio_paths: List of audio file paths.
        processor: AudioProcessor instance (creates new one if None).
        parallel: Whether to use parallel processing (if available).
        
    Returns:
        List[torch.Tensor]: List of preprocessed audio tensors.
    """
    if processor is None:
        processor = AudioProcessor()
    
    processed_audio = []
    
    for path in audio_paths:
        try:
            audio = processor.load_and_preprocess(path)
            processed_audio.append(audio)
        except Exception as e:
            # Log error but continue with other files
            warnings.warn(f"Failed to process {path}: {e}", UserWarning)
            processed_audio.append(None)
    
    return processed_audio


def validate_audio_tensor(audio: torch.Tensor, 
                         name: str = "audio") -> None:
    """Validate an audio tensor for common issues.
    
    Args:
        audio: Audio tensor to validate.
        name: Name for error messages.
        
    Raises:
        ValidationError: If audio tensor is invalid.
    """
    if not isinstance(audio, torch.Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor", name)
    
    if audio.numel() == 0:
        raise ValidationError(f"{name} tensor is empty", name)
    
    if audio.dim() not in [1, 2, 3]:
        raise ValidationError(f"{name} tensor must be 1D, 2D, or 3D, got {audio.dim()}D", name)
    
    # Check for NaN or infinite values
    if torch.isnan(audio).any():
        raise ValidationError(f"{name} tensor contains NaN values", name)
    
    if torch.isinf(audio).any():
        raise ValidationError(f"{name} tensor contains infinite values", name)
    
    # Check value range for float tensors
    if audio.dtype.is_floating_point:
        max_val = torch.max(torch.abs(audio)).item()
        if max_val > 10.0:  # Reasonable threshold
            warnings.warn(f"{name} tensor has large values (max: {max_val:.2f})", UserWarning)


def estimate_audio_memory_usage(audio_length_seconds: float,
                               sample_rate: int = 44100,
                               channels: int = 1,
                               dtype: torch.dtype = torch.float32) -> float:
    """Estimate memory usage for audio tensor.
    
    Args:
        audio_length_seconds: Length of audio in seconds.
        sample_rate: Audio sample rate.
        channels: Number of audio channels.
        dtype: Tensor data type.
        
    Returns:
        float: Estimated memory usage in MB.
    """
    num_samples = int(audio_length_seconds * sample_rate)
    
    # Get size of dtype in bytes
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    
    total_bytes = num_samples * channels * dtype_size
    return total_bytes / (1024 ** 2)  # Convert to MB