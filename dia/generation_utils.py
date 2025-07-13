"""Generation utilities for the Dia text-to-speech model.

This module provides helper functions for the generation process,
breaking down the large generate method into smaller, manageable components.
"""

import time
from typing import List, Optional, Tuple, Union

import torch
import numpy as np

from .exceptions import GenerationError, ValidationError, AudioProcessingError
from .memory_utils import temporary_tensor_cache


def process_audio_prompts(
    audio_prompt: Union[List[Union[str, torch.Tensor, None]], str, torch.Tensor, None],
    batch_size: int,
    load_audio_fn,
) -> List[Optional[torch.Tensor]]:
    """Process and validate audio prompts for generation.
    
    Args:
        audio_prompt: Audio prompt(s) to process.
        batch_size: Expected batch size.
        load_audio_fn: Function to load audio from path.
        
    Returns:
        List of processed audio prompts.
        
    Raises:
        ValidationError: If audio prompts are invalid.
        AudioProcessingError: If audio processing fails.
    """
    try:
        if isinstance(audio_prompt, list):
            processed_prompts = []
            for i, p in enumerate(audio_prompt):
                if isinstance(p, str):
                    processed_prompts.append(load_audio_fn(p))
                elif p is None or isinstance(p, torch.Tensor):
                    processed_prompts.append(p)
                else:
                    raise ValidationError(f"Audio prompt {i} must be string, tensor, or None", "audio_prompt")
            audio_prompt = processed_prompts
        elif isinstance(audio_prompt, str):
            audio_prompt = [load_audio_fn(audio_prompt)]
        elif isinstance(audio_prompt, torch.Tensor):
            audio_prompt = [audio_prompt]
        elif audio_prompt is None:
            audio_prompt = [None] * batch_size
        else:
            raise ValidationError("audio_prompt must be string, tensor, list, or None", "audio_prompt")

        if len(audio_prompt) != batch_size:
            raise ValidationError(
                f"Number of audio prompts ({len(audio_prompt)}) must match batch size ({batch_size})", 
                "audio_prompt"
            )
        
        return audio_prompt
        
    except Exception as e:
        if isinstance(e, (ValidationError, AudioProcessingError)):
            raise
        raise AudioProcessingError(f"Failed to process audio prompts: {e}")


def encode_text_inputs(text: Union[str, List[str]], encode_text_fn) -> torch.Tensor:
    """Encode text inputs for generation.
    
    Args:
        text: Text input(s) to encode.
        encode_text_fn: Function to encode individual text strings.
        
    Returns:
        Padded text tensor.
        
    Raises:
        GenerationError: If text encoding fails.
    """
    try:
        if isinstance(text, list):
            encoded_texts = [encode_text_fn(t) for t in text]
        else:
            encoded_texts = [encode_text_fn(text)]
        return encoded_texts
    except Exception as e:
        raise GenerationError(f"Failed to encode text input: {e}")


def setup_generation_state(
    text: torch.Tensor,
    audio_prompts: List[Optional[torch.Tensor]],
    max_tokens: Optional[int],
    prepare_generation_fn,
) -> Tuple:
    """Set up the initial generation state.
    
    Args:
        text: Encoded text tensor.
        audio_prompts: Processed audio prompts.
        max_tokens: Maximum tokens to generate.
        prepare_generation_fn: Function to prepare generation state.
        
    Returns:
        Tuple of (decoder_state, decoder_output).
        
    Raises:
        GenerationError: If generation setup fails.
    """
    try:
        return prepare_generation_fn(text, audio_prompts, max_tokens=max_tokens)
    except Exception as e:
        raise GenerationError(f"Failed to prepare generation: {e}")


def run_generation_loop(
    dec_state,
    dec_output,
    max_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    decoder_step_fn,
    config,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the main generation loop.
    
    Args:
        dec_state: Decoder state object.
        dec_output: Decoder output object.
        max_tokens: Maximum tokens to generate.
        cfg_scale: Classifier-free guidance scale.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        cfg_filter_top_k: Top-k filtering parameter.
        decoder_step_fn: Decoder step function.
        config: Model configuration.
        device: Target device.
        verbose: Whether to print progress.
        
    Returns:
        Tuple of (eos_detected, eos_countdown, finished_step).
    """
    batch_size = len(dec_output.prefill_steps)
    audio_eos_value = config.eos_token_id
    audio_pad_value = config.pad_token_id
    delay_pattern = config.delay_pattern
    max_delay_pattern = max(delay_pattern)
    delay_pattern_Cx = torch.tensor(delay_pattern, device=device, dtype=torch.long)
    
    dec_step = min(dec_output.prefill_steps) - 1
    current_idx = torch.tensor([dec_step], device=device)
    
    eos_detected_Bx = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    eos_countdown_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    finished_step_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    
    bos_over = False
    
    if verbose:
        print("generate: starting generation loop")
        start_time = time.time()
    
    # Use temporary cache for intermediate tensors
    with temporary_tensor_cache() as cache:
        try:
            while dec_step < max_tokens:
                if (eos_countdown_Bx == 0).all():
                    break

                current_step_idx = dec_step + 1
                
                # Prepare decoder step
                dec_state.prepare_step(dec_step)
                tokens_Bx1xC = dec_output.get_tokens_at(dec_step).repeat_interleave(2, dim=0)
                
                # Run decoder step
                try:
                    pred_BxC = decoder_step_fn(
                        tokens_Bx1xC,
                        dec_state,
                        cfg_scale,
                        temperature,
                        top_p,
                        cfg_filter_top_k,
                        current_idx,
                    )
                except Exception as e:
                    raise GenerationError(f"Generation failed at step {dec_step}: {e}", dec_step)

                current_idx += 1

                # EOS detection and handling
                active_mask_Bx = eos_countdown_Bx != 0
                eos_trigger_Bx = torch.zeros_like(active_mask_Bx)
                
                if active_mask_Bx.any():
                    is_eos_token = (~eos_detected_Bx[active_mask_Bx]) & (pred_BxC[active_mask_Bx, 0] == audio_eos_value)
                    is_max_len = current_step_idx >= max_tokens - max_delay_pattern
                    eos_trigger_Bx[active_mask_Bx] = is_eos_token | is_max_len
                
                eos_detected_Bx |= eos_trigger_Bx
                start_countdown_mask_Bx = eos_trigger_Bx & (eos_countdown_Bx < 0)
                
                if start_countdown_mask_Bx.any():
                    eos_countdown_Bx[start_countdown_mask_Bx] = max_delay_pattern
                    finished_step_Bx[start_countdown_mask_Bx] = current_step_idx

                # Padding handling
                padding_mask_Bx = eos_countdown_Bx > 0
                if padding_mask_Bx.any():
                    pred_active_BxC = pred_BxC[padding_mask_Bx].clone()
                    countdown_active_Bx = eos_countdown_Bx[padding_mask_Bx]
                    step_after_eos_Bx = max_delay_pattern - countdown_active_Bx
                    step_after_eos_Bx_ = step_after_eos_Bx.unsqueeze(1)
                    delay_pattern_Cx_ = delay_pattern_Cx.unsqueeze(0)
                    eos_mask_NxC = step_after_eos_Bx_ == delay_pattern_Cx_
                    pad_mask_NxC = step_after_eos_Bx_ > delay_pattern_Cx_
                    pred_active_BxC[eos_mask_NxC] = audio_eos_value
                    pred_active_BxC[pad_mask_NxC] = audio_pad_value
                    pred_BxC[padding_mask_Bx] = pred_active_BxC
                    eos_countdown_Bx[padding_mask_Bx] -= 1

                # Update BOS flag
                if not bos_over:
                    bos_over = all(
                        dec_step - prefill_step > max_delay_pattern 
                        for prefill_step in dec_output.prefill_steps
                    )

                dec_output.update_one(pred_BxC, current_step_idx, not bos_over)
                dec_step += 1

                # Progress reporting
                if verbose and dec_step % 86 == 0:
                    duration = time.time() - start_time
                    if duration > 0:
                        print(
                            f"generate step {dec_step}: speed={86 * batch_size / duration:.3f} tokens/s, "
                            f"realtime factor={batch_size / duration:.3f}x"
                        )
                    start_time = time.time()
        
        except Exception as e:
            if isinstance(e, GenerationError):
                raise
            raise GenerationError(f"Unexpected error during generation loop: {e}")
    
    return eos_detected_Bx, eos_countdown_Bx, finished_step_Bx


def finalize_generation_output(
    dec_output,
    finished_step_Bx: torch.Tensor,
    max_delay_pattern: int,
    batch_size: int,
    config,
    device: torch.device,
    generate_output_fn,
    verbose: bool = False,
    total_start_time: Optional[float] = None,
) -> List[Optional[np.ndarray]]:
    """Finalize and extract generation output.
    
    Args:
        dec_output: Decoder output object.
        finished_step_Bx: Tensor of finished steps per batch item.
        max_delay_pattern: Maximum delay in the pattern.
        batch_size: Batch size.
        config: Model configuration.
        device: Target device.
        generate_output_fn: Function to generate final output.
        verbose: Whether to print progress.
        total_start_time: Start time for total duration calculation.
        
    Returns:
        List of generated audio arrays.
    """
    try:
        # Calculate final step and lengths
        final_step = finished_step_Bx.max().item() + 1
        finished_step_Bx[finished_step_Bx == -1] = final_step - max_delay_pattern
        
        prefill_steps_tensor = torch.tensor(dec_output.prefill_steps, device=device)
        lengths_Bx = finished_step_Bx - prefill_steps_tensor
        lengths_Bx = torch.clamp(lengths_Bx, min=0)
        
        max_len = lengths_Bx.max().item() + max_delay_pattern
        
        if max_len > 0:
            num_channels = config.decoder_config.num_channels
            audio_pad_value = config.pad_token_id
            
            generated_codes = torch.full(
                (batch_size, max_len, num_channels),
                fill_value=audio_pad_value,
                dtype=torch.long,
                device=device,
            )
            
            # Copy generated tokens
            for i in range(batch_size):
                start_step = dec_output.prefill_steps[i]
                actual_len = lengths_Bx[i].item() + max_delay_pattern
                if actual_len > 0:
                    tokens_to_copy = dec_output.generated_tokens[i, start_step : start_step + actual_len, :]
                    generated_codes[i, :actual_len, :] = tokens_to_copy
            
            if verbose and total_start_time:
                avg_steps = lengths_Bx.float().mean().item()
                total_duration = time.time() - total_start_time
                print(f"generate: avg steps={avg_steps:.1f}, total duration={total_duration:.3f}s")
            
            # Generate final output
            outputs = generate_output_fn(generated_codes, lengths_Bx)
        else:
            print("Warning: Nothing generated for any sequence in the batch.")
            outputs = [None] * batch_size
        
        return outputs
        
    except Exception as e:
        raise GenerationError(f"Failed to finalize generation output: {e}")


def check_compilation_cache(
    use_torch_compile: bool,
    prepare_generation_fn,
    decoder_step_fn,
    compiled_flag_attr: str = "_compiled"
) -> Tuple:
    """Check and set up torch compilation if requested.
    
    Args:
        use_torch_compile: Whether to use torch.compile.
        prepare_generation_fn: Preparation function to compile.
        decoder_step_fn: Decoder step function to compile.
        compiled_flag_attr: Attribute name for compilation flag.
        
    Returns:
        Tuple of (compiled_prepare_fn, compiled_decoder_fn).
    """
    if use_torch_compile and not hasattr(prepare_generation_fn.__self__, compiled_flag_attr):
        print("Compiling generation functions (this may take a while on first run)...")
        
        # Compile the functions
        compiled_prepare = torch.compile(prepare_generation_fn, dynamic=True, fullgraph=True)
        compiled_decoder = torch.compile(decoder_step_fn, fullgraph=True, mode="max-autotune")
        
        # Set compilation flag
        setattr(prepare_generation_fn.__self__, compiled_flag_attr, True)
        
        return compiled_prepare, compiled_decoder
    elif use_torch_compile:
        # Already compiled, use compiled versions
        return prepare_generation_fn, decoder_step_fn
    else:
        # No compilation requested
        return prepare_generation_fn, decoder_step_fn