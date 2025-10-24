#!/usr/bin/env python
"""
Example script demonstrating Wikipedia training pipeline usage.

This script shows how to:
1. Setup and configure the training pipeline
2. Train a model on sanitized Wikipedia data
3. Test the trained model on factual prompts
4. Compare CPU vs GPU performance
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_wikipedia_unified import WikipediaTrainer
from scripts.test_hiroshima_prompt import HiroshimaPromptTester
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_training():
    """Example 1: Basic CPU training"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 1: Basic CPU Training")
    logger.info("="*60)

    # Create trainer
    trainer = WikipediaTrainer(
        config_path="configs/deepseek_v3_cpu_wikipedia.json",
        device="cpu",
        checkpoint_dir="./example_checkpoints/cpu"
    )

    # Train for 100 steps (quick demo)
    logger.info("Training for 100 steps...")
    trainer.config["training"]["train_steps"] = 100
    trainer.train()

    logger.info("Training complete! Checkpoint saved to ./example_checkpoints/cpu")


def example_2_gpu_training():
    """Example 2: GPU training with custom config"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: GPU Training")
    logger.info("="*60)

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU example")
        return

    # Create trainer
    trainer = WikipediaTrainer(
        config_path="configs/deepseek_v3_gpu_wikipedia.json",
        device="cuda",
        checkpoint_dir="./example_checkpoints/gpu"
    )

    # Modify config for quick demo
    trainer.config["training"]["train_steps"] = 100
    trainer.config["training"]["log_interval"] = 10

    # Train
    logger.info("Training for 100 steps on GPU...")
    trainer.train()

    logger.info("Training complete! Checkpoint saved to ./example_checkpoints/gpu")


def example_3_test_prompt():
    """Example 3: Test trained model on Hiroshima prompt"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: Testing Hiroshima Prompt")
    logger.info("="*60)

    checkpoint_path = "./example_checkpoints/cpu/checkpoint_final.pt"

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Please run example_1_basic_training() first")
        return

    # Create tester
    tester = HiroshimaPromptTester(checkpoint_path, device="cpu")

    # Test the prompt
    prompt = "The atomic bombing of Hiroshima occurred in"
    logger.info(f"\nPrompt: '{prompt}'")

    # Generate completions
    completions = tester.generate_completion(
        prompt,
        max_length=20,
        temperature=0.7,
        num_samples=3
    )

    logger.info("\nGenerated completions:")
    for i, completion in enumerate(completions, 1):
        logger.info(f"  {i}. {completion}")

    # Check for "1945"
    success, matched = tester.check_factual_accuracy(completions[0], "1945")
    if success:
        logger.info(f"\n✓ SUCCESS: Model correctly generated year 1945")
    else:
        logger.info(f"\n✗ FAILED: Model did not generate year 1945")


def example_4_custom_sanitization():
    """Example 4: Custom sanitization configuration"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 4: Custom Sanitization")
    logger.info("="*60)

    from src.data.wikipedia_sanitizer import WikipediaSanitizer, SanitizationConfig

    # Create custom sanitization config
    config = SanitizationConfig(
        target_language="en",
        min_language_confidence=0.95,
        min_article_length=100,  # More strict
        max_article_length=5000,
        min_quality_score=0.8,   # Higher quality threshold
        filter_toxic=True,
        filter_boilerplate=True,
    )

    # Create sanitizer
    sanitizer = WikipediaSanitizer(config)

    # Example text
    sample_text = """
    The atomic bombing of Hiroshima occurred on August 6, 1945, when the United States
    dropped an atomic bomb on the city during World War II. The bomb, nicknamed "Little Boy",
    was the first nuclear weapon used in warfare.

    [[Category:World War II]]
    {{cite web|url=example.com|title=Example}}

    == References ==
    1. Example reference

    The explosion killed an estimated 80,000 people instantly.
    """

    logger.info("Original text length:", len(sample_text))

    # Sanitize
    cleaned = sanitizer.sanitize(sample_text, article_id="hiroshima_example")

    if cleaned:
        logger.info(f"Cleaned text length: {len(cleaned)}")
        logger.info(f"\nCleaned text:\n{cleaned[:200]}...")
    else:
        logger.info("Text was filtered out")

    # Show statistics
    logger.info(f"\nSanitization statistics:")
    for key, value in sanitizer.get_statistics().items():
        logger.info(f"  {key}: {value}")


def example_5_streaming_data():
    """Example 5: Streaming Wikipedia data"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 5: Streaming Wikipedia Data")
    logger.info("="*60)

    from transformers import AutoTokenizer
    from src.data.wikipedia_loader import SanitizedWikipediaDataset, WikipediaDataConfig
    from src.data.wikipedia_sanitizer import SanitizationConfig

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create sanitization config
    san_config = SanitizationConfig(
        min_article_length=50,
        max_article_length=1000,
        min_quality_score=0.7,
    )

    # Create data config
    data_config = WikipediaDataConfig(
        streaming=True,
        sanitization_enabled=True,
        sanitization_config=san_config,
        seq_length=256,
        max_articles=10,  # Limit for demo
    )

    # Create dataset
    logger.info("Creating Wikipedia dataset with streaming...")
    dataset = SanitizedWikipediaDataset(
        tokenizer=tokenizer,
        config=data_config,
        device="cpu"
    )

    # Iterate through a few samples
    logger.info("\nIterating through first 3 articles:")
    for i, batch in enumerate(dataset):
        if i >= 3:
            break

        logger.info(f"\nArticle {i+1}:")
        logger.info(f"  Input shape: {batch['input_ids'].shape}")

        # Decode a portion of text
        text = tokenizer.decode(batch['input_ids'][:100], skip_special_tokens=True)
        logger.info(f"  Text preview: {text[:150]}...")

    # Show statistics
    stats = dataset.get_statistics()
    logger.info(f"\nDataset statistics:")
    logger.info(f"  Articles processed: {stats['articles_processed']}")
    logger.info(f"  Articles filtered: {stats['articles_filtered']}")
    logger.info(f"  Cache hits: {stats['cache_hits']}")
    logger.info(f"  Cache misses: {stats['cache_misses']}")


def example_6_compare_cpu_gpu():
    """Example 6: Compare CPU vs GPU training performance"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 6: CPU vs GPU Performance Comparison")
    logger.info("="*60)

    import time

    results = {}

    # Train on CPU
    logger.info("\n--- CPU Training ---")
    cpu_start = time.time()

    trainer_cpu = WikipediaTrainer(
        config_path="configs/deepseek_v3_cpu_wikipedia.json",
        device="cpu",
        checkpoint_dir="./example_checkpoints/comparison_cpu"
    )
    trainer_cpu.config["training"]["train_steps"] = 50
    trainer_cpu.config["training"]["log_interval"] = 10
    trainer_cpu.train()

    cpu_time = time.time() - cpu_start
    results["cpu_time"] = cpu_time
    logger.info(f"CPU training completed in {cpu_time:.2f} seconds")

    # Train on GPU if available
    if torch.cuda.is_available():
        logger.info("\n--- GPU Training ---")
        gpu_start = time.time()

        trainer_gpu = WikipediaTrainer(
            config_path="configs/deepseek_v3_gpu_wikipedia.json",
            device="cuda",
            checkpoint_dir="./example_checkpoints/comparison_gpu"
        )
        trainer_gpu.config["training"]["train_steps"] = 50
        trainer_gpu.config["training"]["log_interval"] = 10
        trainer_gpu.train()

        gpu_time = time.time() - gpu_start
        results["gpu_time"] = gpu_time
        logger.info(f"GPU training completed in {gpu_time:.2f} seconds")

        # Compare
        speedup = cpu_time / gpu_time
        logger.info(f"\n--- Comparison ---")
        logger.info(f"CPU Time: {cpu_time:.2f}s")
        logger.info(f"GPU Time: {gpu_time:.2f}s")
        logger.info(f"Speedup: {speedup:.2f}x")
    else:
        logger.warning("CUDA not available, skipping GPU comparison")


def main():
    """Run all examples"""
    logger.info("="*60)
    logger.info("Wikipedia Training Examples")
    logger.info("="*60)
    logger.info("\nAvailable examples:")
    logger.info("  1. Basic CPU training")
    logger.info("  2. GPU training")
    logger.info("  3. Test Hiroshima prompt")
    logger.info("  4. Custom sanitization")
    logger.info("  5. Streaming data")
    logger.info("  6. CPU vs GPU comparison")
    logger.info("\nRunning all examples...\n")

    # Create example checkpoints directory
    os.makedirs("./example_checkpoints", exist_ok=True)

    try:
        # Run examples
        example_4_custom_sanitization()  # Start with non-training example
        example_5_streaming_data()        # Another non-training example

        # Training examples
        logger.info("\nStarting training examples (this may take some time)...")

        # Uncomment to run training examples:
        # example_1_basic_training()
        # example_2_gpu_training()
        # example_3_test_prompt()
        # example_6_compare_cpu_gpu()

        logger.info("\n" + "="*60)
        logger.info("All examples completed!")
        logger.info("="*60)
        logger.info("\nNote: Training examples are commented out by default.")
        logger.info("Uncomment them in the main() function to run full training.")

    except KeyboardInterrupt:
        logger.info("\nExamples interrupted by user")
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
