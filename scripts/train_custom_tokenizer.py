#!/usr/bin/env python
"""
Train a custom BPE tokenizer on Wikipedia data with 8000 vocab size.

This ensures the tokenizer vocabulary matches the model's embedding size,
enabling proper prompt adherence without token ID clamping or mismatches.
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


def train_wikipedia_tokenizer(
    vocab_size: int = 8000,
    output_dir: str = "./tokenizers",
    num_samples: int = 10000,
):
    """
    Train a BPE tokenizer on Wikipedia data.

    Args:
        vocab_size: Target vocabulary size
        output_dir: Directory to save tokenizer
        num_samples: Number of Wikipedia articles to use for training
    """
    print(f"Training tokenizer with vocab_size={vocab_size}")
    print(f"Using {num_samples} Wikipedia articles")

    # Load Wikipedia dataset
    print("Loading Wikipedia dataset...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    # Extract text from articles
    def get_training_corpus():
        """Generator for training texts."""
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            if i % 1000 == 0:
                print(f"Processed {i}/{num_samples} articles...")
            text = item.get("text", "")
            if text and len(text) > 100:  # Only use substantial articles
                yield text

    # Initialize BPE tokenizer
    print("Initializing BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE())

    # Set pre-tokenizer (split on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|padding|>", "<|unk|>"],
        show_progress=True,
        min_frequency=2,
    )

    # Train tokenizer
    print("Training tokenizer on Wikipedia data...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # Add post-processor for special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Set padding token
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("<|padding|>"),
        pad_token="<|padding|>",
    )

    # Set truncation
    tokenizer.enable_truncation(max_length=512)

    # Save tokenizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer_file = output_path / f"wikipedia_bpe_{vocab_size}.json"
    tokenizer.save(str(tokenizer_file))
    print(f"\nTokenizer saved to: {tokenizer_file}")

    # Test tokenizer
    print("\nTesting tokenizer...")
    test_text = "The capital of France is Paris. The Earth orbits the Sun."
    encoded = tokenizer.encode(test_text)
    print(f"Test text: {test_text}")
    print(f"Encoded: {encoded.ids[:20]}...")  # Show first 20 tokens
    print(f"Decoded: {tokenizer.decode(encoded.ids)}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom Wikipedia tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size")
    parser.add_argument("--output-dir", type=str, default="./tokenizers", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of training samples")

    args = parser.parse_args()

    train_wikipedia_tokenizer(
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )
