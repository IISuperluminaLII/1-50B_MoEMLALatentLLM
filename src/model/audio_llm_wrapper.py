"""
Audio-LLM Wrapper for DeepSeekV3Model.

This wrapper adds audio encoding capabilities to the base DeepSeekV3 model,
implementing the LLM-centric architecture:
Audio → Encoder → Embeddings → LLM → Tokens → LLM (refinement) → Tokens → iSTFT → Audio

The wrapper keeps the base model unmodified and adds audio processing as a layer on top.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import warnings

from .deepseek_v3_model import DeepSeekV3Model
from ..audio.simple_audio_encoder import SimpleAudioEncoder
from ..data.audio_tokenizer import AudioTokenizer


class AudioLLMWrapper(nn.Module):
    """
    Wrapper that adds audio encoding to DeepSeekV3 for speech-to-speech translation.

    Architecture:
        1. Audio input → SimpleAudioEncoder → embeddings
        2. Embeddings → DeepSeekV3 LLM → coarse tokens
        3. (Optional) Coarse tokens → DeepSeekV3 LLM → refined tokens
        4. Tokens → iSTFT decoder → Audio output

    This maintains a pure LLM-centric approach where all intelligence
    resides in the DeepSeekV3 model.
    """

    def __init__(
        self,
        config,
        audio_encoder_model: str = "facebook/wav2vec2-base",
        freeze_audio_encoder: bool = True,
        enable_two_stage: bool = False,
        audio_tokenizer_mode: str = "spec",
    ):
        """
        Initialize Audio-LLM wrapper.

        Args:
            config: DeepSeekV3Config instance
            audio_encoder_model: Pretrained audio encoder to use
            freeze_audio_encoder: Whether to freeze encoder weights
            enable_two_stage: Whether to enable two-stage refinement
            audio_tokenizer_mode: Audio tokenization mode ("spec" or "mulaw")
        """
        super().__init__()

        self.config = config
        self.enable_two_stage = enable_two_stage

        # Core LLM model (unmodified)
        self.llm = DeepSeekV3Model(config)

        # Audio encoder for converting audio to embeddings
        d_model = config.mla.d_model
        self.audio_encoder = SimpleAudioEncoder(
            encoder_model=audio_encoder_model,
            output_dim=d_model,
            freeze_encoder=freeze_audio_encoder,
            dropout=0.1,
        )

        # Audio tokenizer for decoding (iSTFT)
        self.audio_tokenizer = AudioTokenizer(
            mode=audio_tokenizer_mode,
            sample_rate=16000 if audio_tokenizer_mode == "spec" else 8000,
        )

        # Optional: Projection from embeddings to token space for two-stage
        if enable_two_stage:
            self.embed_to_token = nn.Linear(d_model, config.vocab_size)
            self.token_to_embed = nn.Embedding(config.vocab_size, d_model)

        print(f"AudioLLMWrapper initialized:")
        print(f"  Audio encoder: {audio_encoder_model} (frozen={freeze_audio_encoder})")
        print(f"  LLM model: DeepSeekV3 ({sum(p.numel() for p in self.llm.parameters())/1e6:.1f}M params)")
        print(f"  Two-stage refinement: {enable_two_stage}")
        print(f"  Audio tokenizer: {audio_tokenizer_mode}")

    def forward_audio_to_embeddings(
        self,
        audio_waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 1: Convert audio to embeddings.

        Args:
            audio_waveforms: [batch, samples] audio at 16kHz
            attention_mask: Optional [batch, samples] mask

        Returns:
            Dict with embeddings and mask
        """
        return self.audio_encoder(
            audio_waveforms=audio_waveforms,
            attention_mask=attention_mask
        )

    def forward_embeddings_to_tokens(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Stage 2: Process embeddings through LLM to generate tokens.

        Args:
            embeddings: [batch, seq_len, d_model] from audio encoder
            attention_mask: [batch, seq_len] mask
            temperature: Sampling temperature for token generation

        Returns:
            tokens: [batch, seq_len] predicted audio tokens
        """
        # Forward through LLM with audio embeddings
        outputs = self.llm(
            audio_embeddings=embeddings,
            attention_mask=attention_mask,
        )

        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # Apply temperature and sample tokens
        if temperature > 0:
            logits = logits / temperature

        # Get tokens (argmax for now, can add sampling later)
        tokens = torch.argmax(logits, dim=-1)

        # Mask non-audio tokens to ensure valid audio generation
        # Audio tokens are in range [50262, 51286] for spec mode
        if self.audio_tokenizer.mode == "spec":
            min_token = self.audio_tokenizer.SPEC_START
            max_token = self.audio_tokenizer.SPEC_END
        else:  # mulaw
            min_token = self.audio_tokenizer.MULAW_START
            max_token = self.audio_tokenizer.MULAW_END

        # Clamp tokens to valid audio range
        tokens = torch.clamp(tokens, min_token, max_token - 1)

        return tokens

    def forward_tokens_refinement(
        self,
        coarse_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Stage 3 (Optional): Refine coarse tokens through second LLM pass.

        Args:
            coarse_tokens: [batch, seq_len] coarse audio tokens
            attention_mask: [batch, seq_len] mask

        Returns:
            refined_tokens: [batch, seq_len] refined audio tokens
        """
        if not self.enable_two_stage:
            return coarse_tokens

        # Convert tokens back to embeddings
        refined_embeddings = self.token_to_embed(coarse_tokens)

        # Second pass through LLM
        outputs = self.llm(
            audio_embeddings=refined_embeddings,
            attention_mask=attention_mask,
        )

        refined_logits = outputs.logits
        refined_tokens = torch.argmax(refined_logits, dim=-1)

        # Ensure valid audio tokens
        if self.audio_tokenizer.mode == "spec":
            min_token = self.audio_tokenizer.SPEC_START
            max_token = self.audio_tokenizer.SPEC_END
        else:
            min_token = self.audio_tokenizer.MULAW_START
            max_token = self.audio_tokenizer.MULAW_END

        refined_tokens = torch.clamp(refined_tokens, min_token, max_token - 1)

        return refined_tokens

    def forward_tokens_to_audio(
        self,
        tokens: torch.Tensor,
        vocoder=None,
    ) -> torch.Tensor:
        """
        Stage 4: Decode tokens to audio waveform using iSTFT.

        Args:
            tokens: [batch, seq_len] audio tokens
            vocoder: Optional neural vocoder (if None, uses iSTFT)

        Returns:
            waveforms: [batch, samples] audio waveforms
        """
        batch_size = tokens.shape[0]
        waveforms = []

        for i in range(batch_size):
            batch_tokens = tokens[i]

            # Decode based on tokenizer mode
            if self.audio_tokenizer.mode == "spec":
                waveform = self.audio_tokenizer.decode_spec(batch_tokens, vocoder=vocoder)
            else:  # mulaw
                waveform = self.audio_tokenizer.decode_mulaw(batch_tokens)

            waveforms.append(waveform)

        # Stack batch
        waveforms = torch.cat(waveforms, dim=0)

        return waveforms

    def forward(
        self,
        audio_waveforms: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        generate_audio: bool = False,
    ) -> Dict[str, Any]:
        """
        Full forward pass through the audio-LLM pipeline.

        Args:
            audio_waveforms: [batch, samples] input audio
            input_ids: [batch, seq_len] token IDs (for standard mode)
            attention_mask: Attention mask
            labels: Target labels for training
            return_dict: Whether to return dict or tuple
            generate_audio: Whether to generate audio output

        Returns:
            Dict with loss, logits, tokens, and optionally audio
        """
        # Branch based on input type
        if audio_waveforms is not None:
            # Audio input path: audio → encoder → embeddings → LLM

            # Stage 1: Audio to embeddings
            encoder_output = self.forward_audio_to_embeddings(
                audio_waveforms, attention_mask
            )
            embeddings = encoder_output["embeddings"]
            embed_mask = encoder_output["attention_mask"]

            # Stage 2: Process through LLM
            if labels is not None:
                # Training mode: compute loss
                outputs = self.llm(
                    audio_embeddings=embeddings,
                    attention_mask=embed_mask,
                    labels=labels,
                )
                loss = outputs.loss
                logits = outputs.logits
            else:
                # Inference mode: generate tokens
                tokens = self.forward_embeddings_to_tokens(
                    embeddings, embed_mask
                )

                # Stage 3: Optional refinement
                if self.enable_two_stage:
                    tokens = self.forward_tokens_refinement(tokens, embed_mask)

                loss = None
                logits = None

                # Stage 4: Generate audio if requested
                if generate_audio:
                    audio_output = self.forward_tokens_to_audio(tokens)
                else:
                    audio_output = None

        else:
            # Standard token input path
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits
            tokens = None
            audio_output = None

        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "tokens": tokens if 'tokens' in locals() else None,
                "audio": audio_output if 'audio_output' in locals() else None,
            }
        else:
            return (loss, logits)

    def generate_from_audio(
        self,
        audio_waveforms: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        vocoder=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate audio output from audio input.

        Args:
            audio_waveforms: [batch, samples] input audio
            max_length: Maximum token sequence length
            temperature: Sampling temperature
            vocoder: Optional neural vocoder

        Returns:
            tokens: Generated audio tokens
            audio: Generated audio waveform
        """
        # Stage 1: Encode audio
        encoder_output = self.forward_audio_to_embeddings(audio_waveforms)
        embeddings = encoder_output["embeddings"]

        # Stage 2: Generate tokens
        tokens = self.forward_embeddings_to_tokens(embeddings, temperature=temperature)

        # Stage 3: Refine if enabled
        if self.enable_two_stage:
            tokens = self.forward_tokens_refinement(tokens)

        # Stage 4: Decode to audio
        audio = self.forward_tokens_to_audio(tokens, vocoder=vocoder)

        return tokens, audio


def test_wrapper():
    """Test the AudioLLMWrapper with dummy data."""
    from ..config.model_config import DeepSeekV3Config, MLAConfig, MoEConfig

    print("Testing AudioLLMWrapper...")

    # Create minimal config
    config = DeepSeekV3Config(
        num_layers=4,
        vocab_size=51540,
        mla=MLAConfig(
            d_model=512,
            d_latent=256,
            num_heads=8,
        ),
        moe=MoEConfig(
            num_experts=4,
            num_experts_per_token=2,
            expert_intermediate_size=1024,
        ),
    )

    # Create wrapper
    wrapper = AudioLLMWrapper(
        config=config,
        audio_encoder_model="facebook/wav2vec2-base",
        enable_two_stage=True,
        audio_tokenizer_mode="spec",
    )

    # Test with dummy audio
    batch_size = 2
    audio = torch.randn(batch_size, 16000 * 2)  # 2 seconds

    # Test inference
    with torch.no_grad():
        tokens, audio_out = wrapper.generate_from_audio(audio)

    print(f"Input audio shape: {audio.shape}")
    print(f"Generated tokens shape: {tokens.shape}")
    print(f"Output audio shape: {audio_out.shape}")
    print("[PASSED] Wrapper test")

    return wrapper


if __name__ == "__main__":
    test_wrapper()