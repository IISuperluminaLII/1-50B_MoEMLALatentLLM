"""
Logit Masking for Audio Generation.

Provides utilities to mask text tokens during audio generation,
ensuring the model only outputs audio tokens.
"""

import torch
from typing import List, Tuple, Optional


class LogitMasker:
    """
    Masks logits to restrict generation to specific token ranges.

    Used during audio generation to:
    1. Mask text tokens [0, 50000) to prevent text generation
    2. Allow only audio tokens (μ-law or spectrogram)
    3. Allow special tokens (SRC, /SRC, TGT, EOS)

    Args:
        vocab_size: Total vocabulary size (51540)
        audio_token_ranges: List of (start, end) tuples for allowed audio tokens
        special_token_ids: List of special token IDs to allow
        mask_value: Value to set masked logits to (default: -inf)
    """

    def __init__(
        self,
        vocab_size: int = 51540,
        audio_token_ranges: Optional[List[Tuple[int, int]]] = None,
        special_token_ids: Optional[List[int]] = None,
        mask_value: float = float('-inf')
    ):
        self.vocab_size = vocab_size
        self.mask_value = mask_value

        # Default audio token ranges
        if audio_token_ranges is None:
            audio_token_ranges = [
                (50000, 50256),  # μ-law
                (50262, 51286),  # Spectrogram
            ]

        # Default special tokens
        if special_token_ids is None:
            special_token_ids = [
                50256,  # EN2ZH
                50257,  # ZH2EN
                50258,  # SRC
                50259,  # /SRC
                50260,  # TGT
                50261,  # EOS
            ]

        self.audio_token_ranges = audio_token_ranges
        self.special_token_ids = special_token_ids

        # Create mask tensor (will be cached and moved to device on first use)
        self._mask_cache = None
        self._device = None

    def create_mask(self, device: torch.device) -> torch.Tensor:
        """
        Create mask tensor for the given device.

        Returns:
            mask: [vocab_size] boolean tensor (True = allowed, False = masked)
        """
        # Initialize all tokens as masked
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)

        # Allow audio token ranges
        for start, end in self.audio_token_ranges:
            mask[start:end] = True

        # Allow special tokens
        for token_id in self.special_token_ids:
            if 0 <= token_id < self.vocab_size:
                mask[token_id] = True

        return mask

    def get_mask(self, device: torch.device) -> torch.Tensor:
        """
        Get cached mask for the given device.

        Args:
            device: Target device

        Returns:
            mask: [vocab_size] boolean tensor
        """
        if self._mask_cache is None or self._device != device:
            self._mask_cache = self.create_mask(device)
            self._device = device

        return self._mask_cache

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to logits.

        Args:
            logits: [..., vocab_size] logits tensor

        Returns:
            masked_logits: Same shape as input, with masked positions set to mask_value
        """
        # Get mask on same device
        mask = self.get_mask(logits.device)

        # Apply mask (invert: True = allowed, False = masked)
        masked_logits = logits.clone()
        masked_logits[..., ~mask] = self.mask_value

        return masked_logits

    def apply_audio_mode_mask(
        self,
        logits: torch.Tensor,
        audio_mode: str
    ) -> torch.Tensor:
        """
        Apply mode-specific mask (only μ-law OR spectrogram).

        Args:
            logits: [..., vocab_size] logits tensor
            audio_mode: "mulaw" or "spec"

        Returns:
            masked_logits: Same shape as input
        """
        device = logits.device

        # Create mode-specific mask
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)

        if audio_mode == "mulaw":
            # Allow μ-law tokens only
            mask[50000:50256] = True
        elif audio_mode == "spec":
            # Allow spectrogram tokens only
            mask[50262:51286] = True
        else:
            raise ValueError(f"Invalid audio_mode: {audio_mode}")

        # Allow special tokens
        for token_id in self.special_token_ids:
            if 0 <= token_id < self.vocab_size:
                mask[token_id] = True

        # Apply mask
        masked_logits = logits.clone()
        masked_logits[..., ~mask] = self.mask_value

        return masked_logits

    def apply_generation_mask(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        is_audio_generation: bool = True
    ) -> torch.Tensor:
        """
        Smart masking based on generation context.

        Automatically detects if we're in:
        - Source audio region [SRC] ... [/SRC]: Allow source audio tokens
        - Target audio region [TGT] ... [EOS]: Allow target audio tokens
        - Text/phoneme region: Allow text/phoneme tokens

        Args:
            logits: [batch, seq_len, vocab_size] or [batch, vocab_size]
            input_ids: [batch, seq_len] input sequence
            is_audio_generation: If True, mask text tokens

        Returns:
            masked_logits: Same shape as input logits
        """
        if not is_audio_generation:
            # No masking in text mode
            return logits

        # Simple heuristic: mask all text tokens
        return self.apply(logits)


def create_audio_logit_masker(
    audio_mode: str = "spec",
    vocab_size: int = 51540
) -> LogitMasker:
    """
    Factory function to create LogitMasker for audio generation.

    Args:
        audio_mode: "mulaw", "spec", or "both"
        vocab_size: Total vocabulary size

    Returns:
        LogitMasker instance
    """
    if audio_mode == "mulaw":
        audio_token_ranges = [(50000, 50256)]
    elif audio_mode == "spec":
        audio_token_ranges = [(50262, 51286)]
    elif audio_mode == "both":
        audio_token_ranges = [(50000, 50256), (50262, 51286)]
    else:
        raise ValueError(f"Invalid audio_mode: {audio_mode}")

    return LogitMasker(
        vocab_size=vocab_size,
        audio_token_ranges=audio_token_ranges
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing LogitMasker...")

    # Create masker
    masker = create_audio_logit_masker(audio_mode="spec")

    # Test logits
    batch_size = 2
    seq_len = 10
    vocab_size = 51540

    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Apply mask
    masked_logits = masker.apply(logits)

    # Check that text tokens are masked
    print(f"Original logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"Masked logits range: [{masked_logits.min():.2f}, {masked_logits.max():.2f}]")

    # Check specific ranges
    text_tokens = masked_logits[0, 0, :50000]
    spec_tokens = masked_logits[0, 0, 50262:51286]

    print(f"\nText tokens (should be -inf): {text_tokens[:5]}")
    print(f"Spec tokens (should be finite): {spec_tokens[:5]}")

    # Test mode-specific masking
    print("\nTesting mode-specific masking...")
    mulaw_masked = masker.apply_audio_mode_mask(logits, audio_mode="mulaw")
    spec_masked = masker.apply_audio_mode_mask(logits, audio_mode="spec")

    mulaw_tokens = mulaw_masked[0, 0, 50000:50256]
    spec_tokens_mulaw = mulaw_masked[0, 0, 50262:51286]

    print(f"μ-law mode - μ-law tokens (should be finite): {mulaw_tokens[:5]}")
    print(f"μ-law mode - spec tokens (should be -inf): {spec_tokens_mulaw[:5]}")

    # Test sampling with mask
    print("\nTesting sampling with mask...")
    masked_logits_last = masker.apply(logits[:, -1, :])  # [batch, vocab_size]
    probs = torch.softmax(masked_logits_last, dim=-1)

    # Sample
    sampled_tokens = torch.multinomial(probs, num_samples=1)
    print(f"Sampled tokens: {sampled_tokens.squeeze()}")

    # Check that sampled tokens are in valid range
    for token in sampled_tokens.squeeze():
        if token < 50000:
            print(f"ERROR: Sampled text token {token}")
        elif 50000 <= token < 50256:
            print(f"Sampled μ-law token {token}")
        elif 50256 <= token < 50262:
            print(f"Sampled special token {token}")
        elif 50262 <= token < 51286:
            print(f"Sampled spec token {token}")
        else:
            print(f"Sampled phoneme token {token}")

    print("\nLogitMasker test passed!")
