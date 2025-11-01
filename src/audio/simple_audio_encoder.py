"""
Simple Audio Encoder for LLM-Centric Audio Processing.

Converts audio waveforms to embeddings that the LLM can directly process.
Uses frozen wav2vec2 as feature extractor with simple projection to LLM space.

This maintains a pure LLM architecture where DeepSeekV3 handles all intelligence,
with minimal audio-specific components.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from typing import Optional, Dict, Tuple
import warnings


class SimpleAudioEncoder(nn.Module):
    """
    Lightweight audio encoder that outputs LLM-compatible embeddings.

    Architecture:
        Audio → wav2vec2 (frozen) → Linear projection → LLM embeddings

    The encoder is kept minimal to maintain LLM-centric processing.
    All intelligence remains in the DeepSeekV3 LLM.
    """

    def __init__(
        self,
        encoder_model: str = "facebook/wav2vec2-base",
        output_dim: int = 1536,  # Match DeepSeekV3 d_model
        freeze_encoder: bool = True,
        dropout: float = 0.1,
        max_audio_length: int = 16000 * 30,  # 30 seconds at 16kHz
    ):
        """
        Initialize simple audio encoder.

        Args:
            encoder_model: Pretrained wav2vec2 model name
            output_dim: Output dimension to match LLM d_model
            freeze_encoder: Whether to freeze wav2vec2 weights (recommended)
            dropout: Dropout rate for projection
            max_audio_length: Maximum audio length in samples
        """
        super().__init__()

        # Load pretrained wav2vec2
        print(f"Loading audio encoder: {encoder_model}")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(encoder_model)
        self.encoder_dim = self.wav2vec2.config.hidden_size  # 768 for base

        # Freeze encoder if specified (recommended for feature extraction)
        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            self.wav2vec2.eval()
            print("Audio encoder frozen - using as feature extractor only")

        # Simple linear projection to LLM embedding space
        self.projection = nn.Sequential(
            nn.Linear(self.encoder_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

        # Audio processor for preprocessing
        self.processor = Wav2Vec2Processor.from_pretrained(encoder_model)
        self.max_audio_length = max_audio_length
        self.output_dim = output_dim

        # Cache for inference optimization
        self._cache_enabled = False
        self._feature_cache = {}

    def preprocess_audio(
        self,
        audio_waveforms: torch.Tensor,
        sampling_rate: int = 16000
    ) -> torch.Tensor:
        """
        Preprocess audio waveforms for wav2vec2.

        Args:
            audio_waveforms: [batch, samples] or [batch, 1, samples]
            sampling_rate: Audio sample rate (must be 16kHz for wav2vec2)

        Returns:
            processed_audio: [batch, samples] normalized audio
        """
        # Ensure 2D shape [batch, samples]
        if audio_waveforms.dim() == 3:
            audio_waveforms = audio_waveforms.squeeze(1)

        # Truncate to max length
        if audio_waveforms.shape[1] > self.max_audio_length:
            audio_waveforms = audio_waveforms[:, :self.max_audio_length]

        # Normalize audio (wav2vec2 expects normalized input)
        # Using simple normalization to avoid CPU/GPU transfer
        audio_mean = audio_waveforms.mean(dim=1, keepdim=True)
        audio_std = audio_waveforms.std(dim=1, keepdim=True) + 1e-7
        normalized_audio = (audio_waveforms - audio_mean) / audio_std

        return normalized_audio

    def forward(
        self,
        audio_waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Convert audio to LLM embeddings.

        Args:
            audio_waveforms: [batch, samples] audio at 16kHz
            attention_mask: Optional [batch, samples] mask
            return_attention_mask: Whether to return embedding mask

        Returns:
            Dict containing:
                - embeddings: [batch, seq_len, output_dim] LLM-ready embeddings
                - attention_mask: [batch, seq_len] mask for embeddings
        """
        batch_size = audio_waveforms.shape[0]
        device = audio_waveforms.device

        # Preprocess audio
        processed_audio = self.preprocess_audio(audio_waveforms)

        # Extract features with wav2vec2
        with torch.no_grad() if self.wav2vec2.training == False else torch.enable_grad():
            # wav2vec2 forward pass
            outputs = self.wav2vec2(
                processed_audio,
                attention_mask=attention_mask
            )
            # outputs.last_hidden_state: [batch, time_steps, 768]
            audio_features = outputs.last_hidden_state

        # Project to LLM embedding space
        embeddings = self.projection(audio_features)

        # Generate attention mask for embeddings
        if return_attention_mask:
            # wav2vec2 downsamples by factor of 320 (16kHz → 50Hz)
            downsample_factor = 320
            embedding_length = audio_features.shape[1]

            if attention_mask is not None:
                # Downsample input mask
                embedding_mask = attention_mask[:, ::downsample_factor]
                embedding_mask = embedding_mask[:, :embedding_length]
            else:
                # All positions are valid
                embedding_mask = torch.ones(
                    batch_size, embedding_length,
                    device=device, dtype=torch.bool
                )
        else:
            embedding_mask = None

        return {
            "embeddings": embeddings,
            "attention_mask": embedding_mask
        }

    def encode_batch(
        self,
        audio_list: list,
        padding: bool = True,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of audio files with padding.

        Args:
            audio_list: List of audio tensors (varying lengths)
            padding: Whether to pad to same length
            max_length: Maximum sequence length for padding

        Returns:
            Batched embeddings with attention mask
        """
        if not padding:
            # Process each audio separately
            embeddings_list = []
            for audio in audio_list:
                output = self.forward(audio.unsqueeze(0))
                embeddings_list.append(output["embeddings"].squeeze(0))
            return {"embeddings": embeddings_list}

        # Pad audio to same length
        max_len = max(a.shape[-1] for a in audio_list)
        if max_length is not None:
            max_len = min(max_len, max_length)

        padded_audio = []
        attention_masks = []

        for audio in audio_list:
            audio_len = audio.shape[-1]
            if audio_len > max_len:
                audio = audio[..., :max_len]
                audio_len = max_len

            # Pad with zeros
            pad_len = max_len - audio_len
            if pad_len > 0:
                audio = torch.nn.functional.pad(audio, (0, pad_len))

            padded_audio.append(audio)

            # Create attention mask
            mask = torch.ones(max_len)
            if pad_len > 0:
                mask[-pad_len:] = 0
            attention_masks.append(mask)

        # Stack batch
        batch_audio = torch.stack(padded_audio)
        batch_mask = torch.stack(attention_masks)

        return self.forward(batch_audio, batch_mask)

    def enable_caching(self, enable: bool = True):
        """Enable/disable feature caching for inference."""
        self._cache_enabled = enable
        if not enable:
            self._feature_cache.clear()

    def clear_cache(self):
        """Clear feature cache."""
        self._feature_cache.clear()


class TwoTowerAudioEncoder(nn.Module):
    """
    Two-tower encoder for source and target audio (optional extension).

    Useful for translation tasks where we want separate encoders for
    source and target languages.
    """

    def __init__(
        self,
        source_encoder: str = "facebook/wav2vec2-base",
        target_encoder: str = "facebook/wav2vec2-large-xlsr-53-chinese-zh-cn",
        output_dim: int = 1536,
        freeze: bool = True
    ):
        """
        Initialize two-tower audio encoder.

        Args:
            source_encoder: Model for source language (English)
            target_encoder: Model for target language (Chinese)
            output_dim: Output dimension for LLM
            freeze: Whether to freeze encoders
        """
        super().__init__()

        # Source language encoder (English)
        self.source_encoder = SimpleAudioEncoder(
            encoder_model=source_encoder,
            output_dim=output_dim,
            freeze_encoder=freeze
        )

        # Target language encoder (Chinese)
        self.target_encoder = SimpleAudioEncoder(
            encoder_model=target_encoder,
            output_dim=output_dim,
            freeze_encoder=freeze
        )

        # Fusion layer for combining encoders (optional)
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        source_audio: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
        fusion_mode: str = "concat"
    ) -> Dict[str, torch.Tensor]:
        """
        Encode source and/or target audio.

        Args:
            source_audio: [batch, samples] source language audio
            target_audio: [batch, samples] target language audio
            fusion_mode: How to combine embeddings ("concat", "add", "separate")

        Returns:
            Dict with embeddings and masks
        """
        outputs = {}

        # Encode source audio
        if source_audio is not None:
            source_output = self.source_encoder(source_audio)
            outputs["source_embeddings"] = source_output["embeddings"]
            outputs["source_mask"] = source_output["attention_mask"]

        # Encode target audio
        if target_audio is not None:
            target_output = self.target_encoder(target_audio)
            outputs["target_embeddings"] = target_output["embeddings"]
            outputs["target_mask"] = target_output["attention_mask"]

        # Fuse embeddings if both present
        if source_audio is not None and target_audio is not None:
            if fusion_mode == "concat":
                # Concatenate along sequence dimension
                outputs["fused_embeddings"] = torch.cat([
                    outputs["source_embeddings"],
                    outputs["target_embeddings"]
                ], dim=1)
                outputs["fused_mask"] = torch.cat([
                    outputs["source_mask"],
                    outputs["target_mask"]
                ], dim=1)
            elif fusion_mode == "add":
                # Add embeddings (must be same length)
                min_len = min(
                    outputs["source_embeddings"].shape[1],
                    outputs["target_embeddings"].shape[1]
                )
                outputs["fused_embeddings"] = (
                    outputs["source_embeddings"][:, :min_len] +
                    outputs["target_embeddings"][:, :min_len]
                ) / 2
                outputs["fused_mask"] = outputs["source_mask"][:, :min_len]

        return outputs


# Utility functions
def test_encoder():
    """Test the audio encoder with dummy data."""
    print("Testing SimpleAudioEncoder...")

    # Create encoder
    encoder = SimpleAudioEncoder(
        encoder_model="facebook/wav2vec2-base",
        output_dim=1536
    )

    # Create dummy audio (2 seconds at 16kHz)
    batch_size = 2
    audio = torch.randn(batch_size, 16000 * 2)

    # Forward pass
    output = encoder(audio)
    embeddings = output["embeddings"]
    mask = output["attention_mask"]

    print(f"Input shape: {audio.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Output mask shape: {mask.shape}")
    print(f"Embeddings stats: mean={embeddings.mean():.3f}, std={embeddings.std():.3f}")

    # Check output dimensions
    assert embeddings.shape[0] == batch_size
    assert embeddings.shape[2] == 1536
    print("[PASSED] Encoder test")

    return encoder


if __name__ == "__main__":
    # Run test
    test_encoder()