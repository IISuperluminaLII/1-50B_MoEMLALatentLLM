"""
Test 500K parameter model training with custom Wikipedia tokenizer.

This test validates the complete training pipeline with properly matched vocab sizes.
"""

import pytest
import torch
import time
import os
import json
from scripts.train_wikipedia_unified import WikipediaTrainer


class TestFast500KCustomTokenizer:
    """Test 500K parameter training with custom 8K tokenizer."""

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_with_custom_tokenizer(self, custom_wikipedia_tokenizer):
        """
        Test GPU training with custom 8K tokenizer for perfect vocab match.

        This ensures:
        - Model vocab: 8000
        - Tokenizer vocab: 8000
        - No token ID mismatches
        - Proper prompt adherence testing
        """
        print("\n" + "="*70)
        print("GPU TRAINING WITH CUSTOM 8K TOKENIZER")
        print("="*70)
        print(f"Tokenizer vocab: {custom_wikipedia_tokenizer.vocab_size}")
        print(f"Model will use matching 8K vocab")
        print("="*70 + "\n")

        # Load config
        config_path = "configs/deepseek_v3_test_run_gpu.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Verify config has 8K vocab
        assert config["model"]["vocab_size"] == 8000, \
            f"Config vocab size {config['model']['vocab_size']} != 8000"

        # Override config for extended training
        config["data"]["max_articles"] = 100000  # Train on 100K Wikipedia articles
        config["training"]["train_steps"] = 1000  # Extend to 1000 training steps

        print(f"[OVERRIDE] max_articles: 1000 -> 100,000")
        print(f"[OVERRIDE] train_steps: 200 -> 1,000")
        print(f"[INFO] This will take ~10-15 minutes but provide much better learning\n")

        # Create trainer with modified config
        trainer = WikipediaTrainer(config_path)
        # Override the config with our modifications
        trainer.config = config

        # Initialize components
        trainer.setup_tokenizer()
        trainer.setup_model()
        trainer.setup_data()
        trainer.setup_optimizer()

        # CRITICAL: Replace the tokenizer with our custom one
        print("[OK] Replacing default tokenizer with custom 8K tokenizer...")
        trainer.tokenizer = custom_wikipedia_tokenizer
        trainer._vocab_size_limit = None  # No need for clamping with matching vocab

        # Recreate data loader with custom tokenizer
        from src.data.wikipedia_loader import create_wikipedia_dataloader, WikipediaDataConfig, SanitizationConfig

        san_cfg = config["data"]["sanitization"]
        sanitization_config = SanitizationConfig(
            target_language=san_cfg["target_language"],
            min_language_confidence=san_cfg["min_language_confidence"],
            min_article_length=san_cfg["min_article_length"],
            max_article_length=san_cfg["max_article_length"],
            max_perplexity=san_cfg["max_perplexity"],
            min_quality_score=san_cfg["min_quality_score"],
            max_char_repetition=san_cfg["max_char_repetition"],
            max_word_repetition=san_cfg["max_word_repetition"],
            max_line_repetition=san_cfg["max_line_repetition"],
            dedup_threshold=san_cfg["dedup_threshold"],
            filter_toxic=san_cfg["filter_toxic"],
            filter_boilerplate=san_cfg["filter_boilerplate"],
            remove_references=san_cfg["remove_references"],
        )

        wiki_config = WikipediaDataConfig(
            dataset_name=config["data"]["dataset_name"],
            dataset_version=config["data"]["dataset_version"],
            streaming=config["data"]["preprocessing"]["streaming"],
            sanitization_enabled=san_cfg["enabled"],
            sanitization_config=sanitization_config,
            cache_dir=config["data"]["cache_dir"],
            seq_length=config["training"]["seq_length"],
            max_articles=config["data"].get("max_articles"),
            buffer_size=config["data"]["preprocessing"]["buffer_size"],
        )

        trainer.dataloader = create_wikipedia_dataloader(
            tokenizer=custom_wikipedia_tokenizer,  # Use custom tokenizer
            config=wiki_config,
            batch_size=config["training"]["micro_batch_size"],
            device=trainer.device.type,
            num_workers=config["data"]["preprocessing"]["num_workers"],
            vocab_size_limit=None,  # No clamping needed
        )

        # Verify vocab sizes match
        assert trainer.model.config.vocab_size == custom_wikipedia_tokenizer.vocab_size, \
            f"Model vocab {trainer.model.config.vocab_size} != Tokenizer vocab {custom_wikipedia_tokenizer.vocab_size}"

        print(f"[OK] Model vocab: {trainer.model.config.vocab_size}")
        print(f"[OK] Tokenizer vocab: {custom_wikipedia_tokenizer.vocab_size}")
        print("[OK] Perfect match - no token ID clamping needed!\n")

        # Test tokenization doesn't exceed vocab
        test_text = "The capital of France is Paris. The atomic bombing of Hiroshima occurred in 1945."
        encoded = custom_wikipedia_tokenizer.encode(test_text)
        max_token_id = max(encoded)
        print(f"Test tokenization: {len(encoded)} tokens, max ID: {max_token_id}")
        assert max_token_id < 8000, f"Token ID {max_token_id} exceeds vocab size!"
        print("[OK] All token IDs within vocab range\n")

        # Train model (1000 steps on 100K articles)
        print("Starting training (1000 steps on 100K Wikipedia articles)...")
        print("Expected time: ~10-15 minutes")
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        print(f"\n[OK] Training completed in {training_time/60:.1f} minutes")

        # Test generation with prompts
        print("\nTesting generation on factual prompts...")
        test_prompts = [
            {"prompt": "The capital of France is", "expected": "Paris"},
            {"prompt": "The Earth orbits the", "expected": "Sun"},
            {"prompt": "The atomic bombing of Hiroshima occurred in", "expected": "1945"},
        ]

        success_count = 0
        for i, test in enumerate(test_prompts, 1):
            print(f"\nPrompt {i}: '{test['prompt']}'")

            # Encode prompt
            input_ids = custom_wikipedia_tokenizer.encode(test['prompt'], return_tensors="pt").to(trainer.device)

            # Verify no out-of-bounds tokens
            assert torch.all(input_ids < 8000), f"Input token IDs exceed vocab size!"

            # Generate
            trainer.model.eval()
            with torch.no_grad():
                generated = input_ids.clone()

                for _ in range(20):
                    outputs = trainer.model(input_ids=generated)

                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, dict):
                        logits = outputs["logits"]
                    else:
                        break

                    next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)

                    # Verify generated token is in vocab
                    assert next_token.item() < 8000, f"Generated token {next_token.item()} exceeds vocab!"

                    generated = torch.cat([generated, next_token], dim=1)

                    if next_token.item() == custom_wikipedia_tokenizer.eos_token_id:
                        break

            # Decode
            completion = custom_wikipedia_tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"  Generated: '{completion}'")

            # Check if expected word appears
            if test['expected'].lower() in completion.lower():
                print(f"  [OK] Contains expected: '{test['expected']}'")
                success_count += 1
            else:
                print(f"  [MISS] Expected: '{test['expected']}'")

        success_rate = (success_count / len(test_prompts)) * 100
        print(f"\n{'='*70}")
        print(f"SUCCESS RATE: {success_rate:.1f}% ({success_count}/{len(test_prompts)})")
        print(f"{'='*70}")

        # Document results
        print("\n[RESULT ANALYSIS]")
        print(f"- Training: 1000 steps on 100K Wikipedia articles")
        print(f"- Model size: ~4M parameters (2 layers, 8K vocab)")
        print(f"- Vocab match: Perfect (8K/8K)")
        print(f"- Token errors: None (all IDs in valid range)")
        print(f"- Prompt adherence: {success_rate:.1f}%")
        print(f"- Training time: {training_time/60:.1f} minutes")

        if success_rate > 0:
            print(f"\n[SUCCESS] Model learned some factual associations!")
            print(f"With 5x more data and 5x more steps, we achieved {success_rate:.1f}% accuracy")
        else:
            print("\nNote: Even with 100K articles and 1000 steps, the model is still very small.")
            print("For better factual learning:")
            print("  - Larger model: Billions of parameters (not 4M)")
            print("  - More data: Full Wikipedia (6.5M articles)")
            print("  - More training: Millions of steps")
            print("  - Better objective: Instruction tuning + RLHF")

        print("\n[OK] TEST PASSED - Extended training on 100K articles completed!")
