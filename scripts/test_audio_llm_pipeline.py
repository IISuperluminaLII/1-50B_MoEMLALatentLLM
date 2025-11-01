#!/usr/bin/env python
"""
Test Script for Audio-LLM Pipeline.

This script tests the complete encoder-LLM-iSTFT pipeline:
1. Audio input → wav2vec2 encoder → embeddings
2. Embeddings → DeepSeekV3 LLM → audio tokens
3. Audio tokens → iSTFT decoder → audio output

Usage:
    # Test with sample audio
    python scripts/test_audio_llm_pipeline.py \
        --audio_path "path/to/test.mp3" \
        --config "configs/deepseek_v3_500m_audio_llm.json"

    # Test with dummy audio
    python scripts/test_audio_llm_pipeline.py \
        --dummy \
        --config "configs/deepseek_v3_500m_audio_llm.json"
"""

import argparse
import torch
import json
import sys
import time
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.audio_llm_wrapper import AudioLLMWrapper
from src.config.model_config import DeepSeekV3Config, MLAConfig, MoEConfig, AudioConfig
from src.data.audio_tokenizer import AudioTokenizer


def test_pipeline(
    audio_path: str = None,
    config_path: str = None,
    use_dummy: bool = False,
    device: str = "auto",
    save_output: str = None,
):
    """
    Test the complete audio-LLM pipeline.

    Args:
        audio_path: Path to test audio file
        config_path: Path to model configuration
        use_dummy: Use dummy audio for testing
        device: Device to use
        save_output: Path to save output audio
    """
    print("=" * 70)
    print("TESTING AUDIO-LLM PIPELINE")
    print("=" * 70)

    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        model_cfg = config_dict["model"]
        wrapper_cfg = model_cfg.get("wrapper", {})
    else:
        # Use minimal test config
        model_cfg = {
            "num_layers": 4,
            "vocab_size": 51540,
            "mla": {
                "d_model": 512,
                "d_latent": 256,
                "num_heads": 8,
            },
            "moe": {
                "num_experts": 4,
                "num_experts_per_token": 2,
                "expert_intermediate_size": 1024,
            },
            "audio": {
                "audio_mode": "spec",
            }
        }
        wrapper_cfg = {
            "audio_encoder_model": "facebook/wav2vec2-base",
            "freeze_audio_encoder": True,
            "enable_two_stage": False,
            "audio_tokenizer_mode": "spec",
        }

    # Create model config
    config = DeepSeekV3Config(
        num_layers=model_cfg.get("num_layers", 4),
        vocab_size=model_cfg.get("vocab_size", 51540),
        mla=MLAConfig(
            d_model=model_cfg["mla"]["d_model"],
            d_latent=model_cfg["mla"]["d_latent"],
            num_heads=model_cfg["mla"]["num_heads"],
        ),
        moe=MoEConfig(
            num_experts=model_cfg["moe"]["num_experts"],
            num_experts_per_token=model_cfg["moe"]["num_experts_per_token"],
            expert_intermediate_size=model_cfg["moe"]["expert_intermediate_size"],
        ),
    )

    print("\nModel Configuration:")
    print(f"  Layers: {config.num_layers}")
    print(f"  d_model: {config.mla.d_model}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Audio encoder: {wrapper_cfg.get('audio_encoder_model', 'wav2vec2-base')}")
    print(f"  Two-stage: {wrapper_cfg.get('enable_two_stage', False)}")
    print(f"  Audio mode: {wrapper_cfg.get('audio_tokenizer_mode', 'spec')}")

    # Create model
    print("\n[1/7] Creating AudioLLMWrapper...")
    wrapper = AudioLLMWrapper(
        config=config,
        audio_encoder_model=wrapper_cfg.get("audio_encoder_model", "facebook/wav2vec2-base"),
        freeze_audio_encoder=wrapper_cfg.get("freeze_audio_encoder", True),
        enable_two_stage=wrapper_cfg.get("enable_two_stage", False),
        audio_tokenizer_mode=wrapper_cfg.get("audio_tokenizer_mode", "spec"),
    )
    wrapper = wrapper.to(device)
    wrapper.eval()

    total_params = sum(p.numel() for p in wrapper.parameters())
    trainable_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    print(f"  Total params: {total_params/1e6:.1f}M")
    print(f"  Trainable params: {trainable_params/1e6:.1f}M")

    # Load or create test audio
    if use_dummy:
        print("\n[2/7] Creating dummy audio...")
        # Create a simple sine wave
        duration = 2.0  # seconds
        sample_rate = 16000
        t = torch.linspace(0, duration, int(duration * sample_rate))
        # Mix of frequencies to simulate speech
        audio_waveform = (
            0.3 * torch.sin(2 * np.pi * 440 * t) +  # A4 note
            0.2 * torch.sin(2 * np.pi * 880 * t) +  # A5 note
            0.1 * torch.sin(2 * np.pi * 220 * t)    # A3 note
        )
        # Add some noise
        audio_waveform += 0.05 * torch.randn_like(audio_waveform)
        audio_waveform = audio_waveform.unsqueeze(0)  # [1, samples]
    else:
        print(f"\n[2/7] Loading audio: {audio_path}")
        audio_tokenizer = AudioTokenizer(mode="spec", sample_rate=16000)
        audio_waveform = audio_tokenizer.load_and_resample(audio_path)

    print(f"  Audio shape: {audio_waveform.shape}")
    print(f"  Duration: {audio_waveform.shape[1]/16000:.2f}s")

    # Move to device
    audio_waveform = audio_waveform.to(device)

    # Test Stage 1: Audio to embeddings
    print("\n[3/7] Testing audio encoder (audio → embeddings)...")
    start_time = time.time()

    with torch.no_grad():
        encoder_output = wrapper.forward_audio_to_embeddings(audio_waveform)
        embeddings = encoder_output["embeddings"]
        embed_mask = encoder_output["attention_mask"]

    encode_time = time.time() - start_time
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Embeddings stats: mean={embeddings.mean():.3f}, std={embeddings.std():.3f}")
    print(f"  Encoding time: {encode_time:.3f}s")

    # Test Stage 2: Embeddings to tokens
    print("\n[4/7] Testing LLM processing (embeddings → tokens)...")
    start_time = time.time()

    with torch.no_grad():
        tokens = wrapper.forward_embeddings_to_tokens(
            embeddings,
            attention_mask=embed_mask,
            temperature=1.0
        )

    llm_time = time.time() - start_time
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Token range: [{tokens.min().item()}, {tokens.max().item()}]")
    print(f"  Unique tokens: {torch.unique(tokens).numel()}")
    print(f"  LLM time: {llm_time:.3f}s")

    # Verify tokens are in valid audio range
    if wrapper.audio_tokenizer.mode == "spec":
        valid_range = [wrapper.audio_tokenizer.SPEC_START, wrapper.audio_tokenizer.SPEC_END]
    else:
        valid_range = [wrapper.audio_tokenizer.MULAW_START, wrapper.audio_tokenizer.MULAW_END]

    in_range = (tokens >= valid_range[0]) & (tokens < valid_range[1])
    print(f"  Valid audio tokens: {in_range.sum().item()}/{tokens.numel()} ({in_range.float().mean()*100:.1f}%)")

    # Test Stage 3 (optional): Token refinement
    if wrapper.enable_two_stage:
        print("\n[5/7] Testing token refinement (coarse → fine)...")
        start_time = time.time()

        with torch.no_grad():
            refined_tokens = wrapper.forward_tokens_refinement(tokens, embed_mask)

        refine_time = time.time() - start_time
        print(f"  Refined tokens shape: {refined_tokens.shape}")
        print(f"  Changed tokens: {(refined_tokens != tokens).sum().item()}")
        print(f"  Refinement time: {refine_time:.3f}s")
        tokens = refined_tokens
    else:
        print("\n[5/7] Skipping token refinement (not enabled)")

    # Test Stage 4: Tokens to audio
    print("\n[6/7] Testing iSTFT decoder (tokens → audio)...")
    start_time = time.time()

    with torch.no_grad():
        output_audio = wrapper.forward_tokens_to_audio(tokens, vocoder=None)

    decode_time = time.time() - start_time
    print(f"  Output audio shape: {output_audio.shape}")
    print(f"  Output duration: {output_audio.shape[1]/16000:.2f}s")
    print(f"  Output range: [{output_audio.min():.3f}, {output_audio.max():.3f}]")
    print(f"  Decoding time: {decode_time:.3f}s")

    # Test full pipeline
    print("\n[7/7] Testing full pipeline...")
    start_time = time.time()

    with torch.no_grad():
        full_tokens, full_audio = wrapper.generate_from_audio(
            audio_waveform,
            temperature=1.0,
            vocoder=None
        )

    full_time = time.time() - start_time
    print(f"  Full pipeline time: {full_time:.3f}s")
    print(f"  RTF (Real-Time Factor): {full_time / (audio_waveform.shape[1]/16000):.2f}x")

    # Audio quality check
    print("\nAudio Quality Check:")

    # Check for silence
    silence_threshold = 0.001
    is_silent = output_audio.abs().max() < silence_threshold
    print(f"  [{'FAIL' if is_silent else 'OK'}] Not silent (max amplitude: {output_audio.abs().max():.4f})")

    # Check for DC offset
    dc_offset = output_audio.mean()
    has_dc_offset = abs(dc_offset) > 0.1
    print(f"  [{'FAIL' if has_dc_offset else 'OK'}] No DC offset (mean: {dc_offset:.4f})")

    # Check for clipping
    clipping_threshold = 0.99
    is_clipping = (output_audio.abs() > clipping_threshold).any()
    print(f"  [{'FAIL' if is_clipping else 'OK'}] No clipping (max: {output_audio.abs().max():.4f})")

    # Check spectral content (should have varied frequencies, not just beeps)
    # Simple check: compute FFT and check energy distribution
    fft = torch.fft.rfft(output_audio[0])
    fft_magnitude = fft.abs()

    # Split spectrum into low, mid, high
    n_bins = len(fft_magnitude)
    low_energy = fft_magnitude[:n_bins//3].mean()
    mid_energy = fft_magnitude[n_bins//3:2*n_bins//3].mean()
    high_energy = fft_magnitude[2*n_bins//3:].mean()

    print(f"  Spectral energy: Low={low_energy:.3f}, Mid={mid_energy:.3f}, High={high_energy:.3f}")

    # Check if energy is concentrated in narrow band (beeps/tones issue)
    energy_ratio = fft_magnitude.max() / (fft_magnitude.mean() + 1e-6)
    has_tones = energy_ratio > 100  # Very peaked spectrum suggests tones
    print(f"  [{'FAIL' if has_tones else 'OK'}] No dominant tones (peak/mean ratio: {energy_ratio:.1f})")

    # Save output if requested
    if save_output:
        print(f"\nSaving output to: {save_output}")
        import torchaudio
        output_path = Path(save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            str(output_path),
            output_audio.cpu(),
            16000
        )
        print(f"  [OK] Audio saved")

    print("\n" + "=" * 70)
    if is_silent or has_tones:
        print("[FAIL] Pipeline test failed - audio quality issues detected")
        print("\nRecommendations:")
        if has_tones:
            print("  1. Retrain VQ codebook with more samples:")
            print("     python scripts/retrain_spec_codebook.py --num_samples 50000")
        if is_silent:
            print("  2. Check model weights are properly initialized")
            print("  3. Verify audio encoder is working (embeddings should be non-zero)")
    else:
        print("[PASSED] Pipeline test successful!")
    print("=" * 70)

    return output_audio


def main():
    parser = argparse.ArgumentParser(description="Test Audio-LLM pipeline")

    parser.add_argument(
        "--audio_path",
        type=str,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model configuration JSON"
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy audio for testing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--save_output",
        type=str,
        help="Path to save output audio"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.dummy and not args.audio_path:
        parser.error("Either --audio_path or --dummy must be specified")

    test_pipeline(
        audio_path=args.audio_path,
        config_path=args.config,
        use_dummy=args.dummy,
        device=args.device,
        save_output=args.save_output,
    )


if __name__ == "__main__":
    main()