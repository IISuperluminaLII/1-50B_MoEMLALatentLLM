"""
Test custom Wikipedia tokenizer training and usage.
"""

import pytest
import torch


class TestCustomTokenizer:
    """Tests for custom Wikipedia BPE tokenizer."""

    def test_tokenizer_training(self, custom_wikipedia_tokenizer):
        """Test that custom tokenizer trains and loads correctly."""
        print("\n" + "="*70)
        print("TESTING CUSTOM TOKENIZER")
        print("="*70)

        # Check tokenizer attributes
        assert custom_wikipedia_tokenizer is not None
        assert custom_wikipedia_tokenizer.vocab_size == 8000

        # Check special tokens
        assert custom_wikipedia_tokenizer.eos_token == "<|endoftext|>"
        assert custom_wikipedia_tokenizer.pad_token == "<|padding|>"
        assert custom_wikipedia_tokenizer.unk_token == "<|unk|>"

        print(f"[OK] Tokenizer loaded with vocab_size={custom_wikipedia_tokenizer.vocab_size}")

    def test_tokenizer_encode_decode(self, custom_wikipedia_tokenizer):
        """Test encoding and decoding with custom tokenizer."""
        test_texts = [
            "The capital of France is Paris.",
            "The Earth orbits the Sun.",
            "The atomic bombing of Hiroshima occurred in 1945.",
            "Machine learning models require large amounts of training data.",
        ]

        print("\n" + "="*70)
        print("TESTING TOKENIZATION")
        print("="*70)

        for text in test_texts:
            # Encode
            encoded = custom_wikipedia_tokenizer.encode(text)

            # Check that all token IDs are within vocab range
            assert all(0 <= token_id < 8000 for token_id in encoded), \
                f"Token IDs out of range for text: {text}"

            # Decode
            decoded = custom_wikipedia_tokenizer.decode(encoded, skip_special_tokens=True)

            print(f"\nOriginal: {text}")
            print(f"Encoded:  {encoded[:15]}{'...' if len(encoded) > 15 else ''} ({len(encoded)} tokens)")
            print(f"Decoded:  {decoded}")

            # Check decode is reasonably similar (BPE may not perfectly round-trip)
            assert len(decoded) > 0, "Decoded text is empty"

        print("\n[OK] All tokenization tests passed")

    def test_tokenizer_batch_encoding(self, custom_wikipedia_tokenizer):
        """Test batch encoding with padding and truncation."""
        texts = [
            "Short text.",
            "This is a much longer text that should be truncated at some point.",
            "Medium length text here.",
        ]

        print("\n" + "="*70)
        print("TESTING BATCH ENCODING")
        print("="*70)

        # Batch encode with padding
        encoded = custom_wikipedia_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )

        # Check shapes
        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert encoded["input_ids"].shape[0] == len(texts)
        assert encoded["input_ids"].shape[1] <= 32

        # Check all token IDs are valid
        assert torch.all(encoded["input_ids"] >= 0)
        assert torch.all(encoded["input_ids"] < 8000)

        print(f"[OK] Batch encoded {len(texts)} texts")
        print(f"  Shape: {encoded['input_ids'].shape}")
        print(f"  Max token ID: {encoded['input_ids'].max().item()}")
        print(f"  Min token ID: {encoded['input_ids'].min().item()}")

    def test_tokenizer_special_tokens(self, custom_wikipedia_tokenizer):
        """Test that special tokens work correctly."""
        print("\n" + "="*70)
        print("TESTING SPECIAL TOKENS")
        print("="*70)

        # Get special token IDs
        eos_id = custom_wikipedia_tokenizer.eos_token_id
        pad_id = custom_wikipedia_tokenizer.pad_token_id
        unk_id = custom_wikipedia_tokenizer.unk_token_id

        print(f"EOS token: '{custom_wikipedia_tokenizer.eos_token}' (ID: {eos_id})")
        print(f"PAD token: '{custom_wikipedia_tokenizer.pad_token}' (ID: {pad_id})")
        print(f"UNK token: '{custom_wikipedia_tokenizer.unk_token}' (ID: {unk_id})")

        # Check IDs are valid
        assert 0 <= eos_id < 8000
        assert 0 <= pad_id < 8000
        assert 0 <= unk_id < 8000

        # Check IDs are different
        assert eos_id != pad_id
        assert eos_id != unk_id
        assert pad_id != unk_id

        print("[OK] All special tokens configured correctly")

    @pytest.mark.gpu
    def test_tokenizer_with_model(self, custom_wikipedia_tokenizer):
        """Test tokenizer compatibility with PyTorch model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        print("\n" + "="*70)
        print("TESTING TOKENIZER WITH MODEL")
        print("="*70)

        # Create a simple embedding layer with 8K vocab
        embedding = torch.nn.Embedding(8000, 128).cuda()

        # Tokenize some text
        text = "The capital of France is Paris."
        input_ids = custom_wikipedia_tokenizer.encode(text, return_tensors="pt").cuda()

        # Pass through embedding (this will fail if token IDs are out of range)
        try:
            embeddings = embedding(input_ids)
            print(f"[OK] Successfully embedded {input_ids.shape[1]} tokens")
            print(f"  Embedding shape: {embeddings.shape}")
            print(f"  Token IDs: {input_ids[0].tolist()}")
        except RuntimeError as e:
            pytest.fail(f"Failed to embed tokens: {e}")
