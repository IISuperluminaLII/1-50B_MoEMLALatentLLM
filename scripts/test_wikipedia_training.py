#!/usr/bin/env python
"""
Test framework for Wikipedia training with sparse assertion.

Tests both CPU and GPU training pipelines with timing and validation.
Includes sparse assertion for the Hiroshima prompt and other historical facts.

Usage:
    # Test both CPU and GPU
    python scripts/test_wikipedia_training.py

    # Test only CPU
    python scripts/test_wikipedia_training.py --cpu-only

    # Test only GPU
    python scripts/test_wikipedia_training.py --gpu-only

    # Quick test with fewer steps
    python scripts/test_wikipedia_training.py --quick --steps 50
"""
import argparse
import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_wikipedia_unified import WikipediaTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikipediaTestFramework:
    """
    Test framework for training and evaluation with timing and sparse assertion.
    """

    def __init__(
        self,
        test_steps: int = 100,
        quick_mode: bool = False,
        save_results: bool = True,
    ):
        """
        Initialize test framework.

        Args:
            test_steps: Number of training steps for testing
            quick_mode: Run quick tests with minimal steps
            save_results: Save test results to file
        """
        self.test_steps = test_steps
        self.quick_mode = quick_mode
        self.save_results = save_results

        # Test prompts for sparse assertion
        self.test_prompts = [
            {
                "prompt": "The atomic bombing of Hiroshima occurred in",
                "expected_patterns": ["1945", "nineteen forty-five", "August 1945", "August 6, 1945"],
                "context": "historical_date",
            },
            {
                "prompt": "World War II ended in the year",
                "expected_patterns": ["1945", "nineteen forty-five", "September 1945"],
                "context": "historical_date",
            },
            {
                "prompt": "The first atomic bomb was dropped on",
                "expected_patterns": ["Hiroshima", "Japan", "August", "1945"],
                "context": "historical_event",
            },
        ]

        # Results storage
        self.results = {
            "cpu": None,
            "gpu": None,
            "comparison": {},
            "timing": {},
        }

    def train_mini_model(
        self,
        config_path: str,
        device: str,
        steps: Optional[int] = None,
    ) -> Tuple[WikipediaTrainer, float]:
        """
        Train a mini model for testing.

        Args:
            config_path: Path to configuration file
            device: Device to use ("cpu" or "cuda")
            steps: Number of training steps (overrides config)

        Returns:
            Tuple of (trainer, training_time_seconds)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training mini model on {device.upper()}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        # Load and modify config for testing
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Override training steps for testing
        if steps is None:
            steps = self.test_steps

        config["training"]["train_steps"] = steps
        config["training"]["eval_interval"] = max(10, steps // 10)
        config["training"]["save_interval"] = steps  # Only save at end
        config["training"]["log_interval"] = max(5, steps // 20)

        # Limit data for quick testing
        if self.quick_mode:
            config["data"]["max_articles"] = 100

        # Save modified config
        test_config_path = f"test_config_{device}.json"
        with open(test_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Create trainer
        checkpoint_dir = f"./test_checkpoints/{device}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trainer = WikipediaTrainer(
            config_path=test_config_path,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        # Train model
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"Training failed on {device}: {e}")
            raise

        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {self._format_time(training_time)}")

        # Clean up temp config
        os.remove(test_config_path)

        return trainer, training_time

    def generate_completion(
        self,
        trainer: WikipediaTrainer,
        prompt: str,
        max_length: int = 20,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text completion for a prompt.

        Args:
            trainer: Trained model trainer
            prompt: Input prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        trainer.model.eval()

        # Tokenize prompt
        input_ids = trainer.tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(trainer.device)

        # Generate
        with torch.no_grad():
            # Simple greedy/sampling generation
            generated = input_ids.clone()

            for _ in range(max_length):
                # Get model output
                outputs = trainer.model(input_ids=generated)
                logits = outputs["logits"]

                # Get next token logits
                next_token_logits = logits[0, -1, :]

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        0, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float('inf')

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                # Stop if EOS token
                if next_token.item() == trainer.tokenizer.eos_token_id:
                    break

        # Decode generated text
        generated_text = trainer.tokenizer.decode(
            generated[0], skip_special_tokens=True
        )

        return generated_text

    def sparse_assert(
        self,
        text: str,
        patterns: List[str],
        context: str = "default",
    ) -> Tuple[bool, List[str]]:
        """
        Perform sparse assertion on generated text.

        Args:
            text: Generated text to check
            patterns: List of expected patterns
            context: Context type for matching strategy

        Returns:
            Tuple of (success, matched_patterns)
        """
        matched_patterns = []
        text_lower = text.lower()

        for pattern in patterns:
            pattern_lower = pattern.lower()

            # Try different matching strategies based on context
            if context == "historical_date":
                # For dates, be more flexible
                # Check for year patterns
                if re.search(r'\b' + re.escape(pattern_lower) + r'\b', text_lower):
                    matched_patterns.append(pattern)
                # Check for spelled-out numbers
                elif pattern == "1945" and any(
                    x in text_lower for x in ["forty-five", "forty five", "1945"]
                ):
                    matched_patterns.append(pattern)
            else:
                # Standard pattern matching
                if pattern_lower in text_lower:
                    matched_patterns.append(pattern)

        success = len(matched_patterns) > 0
        return success, matched_patterns

    def run_test_suite(
        self,
        trainer: WikipediaTrainer,
        device: str,
    ) -> Dict:
        """
        Run complete test suite on trained model.

        Args:
            trainer: Trained model trainer
            device: Device being tested

        Returns:
            Test results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test suite for {device.upper()}")
        logger.info(f"{'='*60}")

        test_results = {
            "device": device,
            "prompts": {},
            "overall_success": False,
            "success_rate": 0.0,
        }

        successes = 0
        total_tests = len(self.test_prompts)

        for test_case in self.test_prompts:
            prompt = test_case["prompt"]
            logger.info(f"\nTesting prompt: '{prompt}'")

            # Generate completion
            start_time = time.time()
            generated = self.generate_completion(trainer, prompt)
            generation_time = time.time() - start_time

            # Extract completion (remove prompt)
            completion = generated[len(prompt):].strip()

            # Perform sparse assertion
            success, matched = self.sparse_assert(
                completion,
                test_case["expected_patterns"],
                test_case["context"],
            )

            if success:
                successes += 1
                logger.info(f"  ✓ PASSED - Found: {matched}")
            else:
                logger.info(f"  ✗ FAILED - Expected one of: {test_case['expected_patterns']}")

            logger.info(f"  Generated: '{completion[:100]}{'...' if len(completion) > 100 else ''}'")
            logger.info(f"  Generation time: {generation_time:.3f}s")

            # Store results
            test_results["prompts"][prompt] = {
                "generated": completion,
                "success": success,
                "matched_patterns": matched,
                "expected_patterns": test_case["expected_patterns"],
                "generation_time": generation_time,
            }

        # Calculate overall metrics
        test_results["success_rate"] = (successes / total_tests) * 100
        test_results["overall_success"] = successes == total_tests

        logger.info(f"\n{'='*40}")
        logger.info(f"Test Summary for {device.upper()}:")
        logger.info(f"  Passed: {successes}/{total_tests}")
        logger.info(f"  Success Rate: {test_results['success_rate']:.1f}%")
        logger.info(f"  Status: {'✓ ALL TESTS PASSED' if test_results['overall_success'] else '✗ SOME TESTS FAILED'}")

        return test_results

    def compare_results(self):
        """Compare CPU and GPU results for divergence analysis."""
        if not (self.results["cpu"] and self.results["gpu"]):
            logger.warning("Cannot compare - missing results from CPU or GPU")
            return

        logger.info(f"\n{'='*60}")
        logger.info("COMPARISON: CPU vs GPU")
        logger.info(f"{'='*60}")

        comparison = {}

        # Compare each prompt
        for prompt in self.test_prompts:
            prompt_text = prompt["prompt"]

            cpu_result = self.results["cpu"]["prompts"].get(prompt_text, {})
            gpu_result = self.results["gpu"]["prompts"].get(prompt_text, {})

            if cpu_result and gpu_result:
                # Check if both succeeded or both failed
                agreement = cpu_result["success"] == gpu_result["success"]

                # Calculate text similarity (simple approach)
                cpu_text = cpu_result.get("generated", "")[:50]
                gpu_text = gpu_result.get("generated", "")[:50]
                similarity = self._calculate_similarity(cpu_text, gpu_text)

                comparison[prompt_text] = {
                    "agreement": agreement,
                    "cpu_success": cpu_result["success"],
                    "gpu_success": gpu_result["success"],
                    "text_similarity": similarity,
                }

                status = "✓ AGREE" if agreement else "✗ DIVERGE"
                logger.info(f"\nPrompt: '{prompt_text}'")
                logger.info(f"  Status: {status}")
                logger.info(f"  CPU: {'PASSED' if cpu_result['success'] else 'FAILED'}")
                logger.info(f"  GPU: {'PASSED' if gpu_result['success'] else 'FAILED'}")
                logger.info(f"  Text Similarity: {similarity:.1f}%")

        self.results["comparison"] = comparison

        # Overall agreement
        if comparison:
            overall_agreement = sum(1 for c in comparison.values() if c["agreement"]) / len(comparison)
            logger.info(f"\n{'='*40}")
            logger.info(f"Overall Agreement: {overall_agreement * 100:.1f}%")

            # Timing comparison
            logger.info(f"\nTiming Comparison:")
            logger.info(f"  CPU Training: {self._format_time(self.results['timing'].get('cpu_training', 0))}")
            logger.info(f"  GPU Training: {self._format_time(self.results['timing'].get('gpu_training', 0))}")

            if self.results['timing'].get('gpu_training', 0) > 0:
                speedup = self.results['timing']['cpu_training'] / self.results['timing']['gpu_training']
                logger.info(f"  GPU Speedup: {speedup:.2f}x")

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity percentage."""
        if not text1 or not text2:
            return 0.0

        # Simple character-level similarity
        matches = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
        max_len = max(len(text1), len(text2))

        return (matches / max_len) * 100 if max_len > 0 else 100.0

    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"

    def save_test_results(self):
        """Save test results to file."""
        if not self.save_results:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"

        # Prepare serializable results
        save_data = {
            "timestamp": timestamp,
            "test_steps": self.test_steps,
            "quick_mode": self.quick_mode,
            "results": self.results,
        }

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        logger.info(f"\nTest results saved to: {results_file}")

    def run_full_test(
        self,
        test_cpu: bool = True,
        test_gpu: bool = True,
    ):
        """
        Run full test suite for CPU and/or GPU.

        Args:
            test_cpu: Whether to test CPU training
            test_gpu: Whether to test GPU training
        """
        logger.info(f"\n{'='*80}")
        logger.info("WIKIPEDIA TRAINING TEST SUITE")
        logger.info(f"{'='*80}")
        logger.info(f"Test Configuration:")
        logger.info(f"  Training Steps: {self.test_steps}")
        logger.info(f"  Quick Mode: {self.quick_mode}")
        logger.info(f"  Test CPU: {test_cpu}")
        logger.info(f"  Test GPU: {test_gpu}")

        overall_start = time.time()

        # Test CPU
        if test_cpu:
            try:
                cpu_start = time.time()
                cpu_trainer, cpu_train_time = self.train_mini_model(
                    "configs/deepseek_v3_cpu_wikipedia.json",
                    device="cpu",
                )
                self.results["timing"]["cpu_training"] = cpu_train_time

                # Run tests
                self.results["cpu"] = self.run_test_suite(cpu_trainer, "cpu")

                cpu_total = time.time() - cpu_start
                self.results["timing"]["cpu_total"] = cpu_total

                logger.info(f"\nCPU Total Time: {self._format_time(cpu_total)}")

            except Exception as e:
                logger.error(f"CPU testing failed: {e}")
                self.results["cpu"] = {"error": str(e)}

        # Test GPU
        if test_gpu and torch.cuda.is_available():
            try:
                gpu_start = time.time()
                gpu_trainer, gpu_train_time = self.train_mini_model(
                    "configs/deepseek_v3_gpu_wikipedia.json",
                    device="cuda",
                )
                self.results["timing"]["gpu_training"] = gpu_train_time

                # Run tests
                self.results["gpu"] = self.run_test_suite(gpu_trainer, "gpu")

                gpu_total = time.time() - gpu_start
                self.results["timing"]["gpu_total"] = gpu_total

                logger.info(f"\nGPU Total Time: {self._format_time(gpu_total)}")

            except Exception as e:
                logger.error(f"GPU testing failed: {e}")
                self.results["gpu"] = {"error": str(e)}
        elif test_gpu:
            logger.warning("GPU testing requested but CUDA is not available")

        # Compare results
        if test_cpu and test_gpu:
            self.compare_results()

        # Save results
        self.save_test_results()

        # Final summary
        overall_time = time.time() - overall_start
        self.results["timing"]["overall"] = overall_time

        logger.info(f"\n{'='*80}")
        logger.info("FINAL TEST SUMMARY")
        logger.info(f"{'='*80}")

        if self.results.get("cpu"):
            if "error" not in self.results["cpu"]:
                logger.info(f"CPU: {self.results['cpu'].get('success_rate', 0):.1f}% success rate")
                logger.info(f"  Training time: {self._format_time(self.results['timing'].get('cpu_training', 0))}")
            else:
                logger.info(f"CPU: FAILED - {self.results['cpu']['error']}")

        if self.results.get("gpu"):
            if "error" not in self.results["gpu"]:
                logger.info(f"GPU: {self.results['gpu'].get('success_rate', 0):.1f}% success rate")
                logger.info(f"  Training time: {self._format_time(self.results['timing'].get('gpu_training', 0))}")
            else:
                logger.info(f"GPU: FAILED - {self.results['gpu']['error']}")

        logger.info(f"\nTotal Test Time: {self._format_time(overall_time)}")

        # Determine overall success
        overall_success = True
        if test_cpu and self.results.get("cpu"):
            if "error" in self.results["cpu"] or not self.results["cpu"].get("overall_success"):
                overall_success = False
        if test_gpu and self.results.get("gpu"):
            if "error" in self.results["gpu"] or not self.results["gpu"].get("overall_success"):
                overall_success = False

        if overall_success:
            logger.info("\n✓ ALL TESTS PASSED")
        else:
            logger.info("\n✗ SOME TESTS FAILED")

        return overall_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Wikipedia training pipeline")

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps for testing",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests with minimal data",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Test only CPU training",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Test only GPU training",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save test results to file",
    )

    args = parser.parse_args()

    # Determine what to test
    test_cpu = not args.gpu_only
    test_gpu = not args.cpu_only

    # Create test framework
    framework = WikipediaTestFramework(
        test_steps=args.steps,
        quick_mode=args.quick,
        save_results=not args.no_save,
    )

    # Run tests
    success = framework.run_full_test(
        test_cpu=test_cpu,
        test_gpu=test_gpu,
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()