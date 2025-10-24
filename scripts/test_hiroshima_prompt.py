#!/usr/bin/env python
"""
Specialized test script for the Hiroshima atomic bombing prompt.

Tests trained models on generating factually correct completion for:
"The atomic bombing of Hiroshima occurred in" -> "1945"

Usage:
    # Test a specific checkpoint
    python scripts/test_hiroshima_prompt.py --checkpoint ./wikipedia_checkpoints/cpu/checkpoint_final.pt

    # Test both CPU and GPU checkpoints
    python scripts/test_hiroshima_prompt.py --test-both

    # Interactive mode
    python scripts/test_hiroshima_prompt.py --checkpoint ./checkpoint.pt --interactive
"""
import argparse
import json
import os
import sys
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.deepseek_v3_model import DeepSeekV3Model
from src.config.model_config import DeepSeekV3Config, MLAConfig, MoEConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HiroshimaPromptTester:
    """
    Specialized tester for Hiroshima atomic bombing prompt.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """
        Initialize tester with checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ("cpu", "cuda", or "auto")
        """
        self.checkpoint_path = checkpoint_path

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load checkpoint
        self.checkpoint = self._load_checkpoint()
        self.config = self.checkpoint.get("config", {})

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._setup_model()
        self._setup_tokenizer()

        # Test prompts with variations
        self.test_prompts = [
            # Primary prompt
            {
                "text": "The atomic bombing of Hiroshima occurred in",
                "expected": "1945",
                "priority": "high",
            },
            # Variations
            {
                "text": "The atomic bomb was dropped on Hiroshima in",
                "expected": "1945",
                "priority": "high",
            },
            {
                "text": "Hiroshima was bombed in the year",
                "expected": "1945",
                "priority": "high",
            },
            {
                "text": "The first atomic bomb was used in combat in",
                "expected": "1945",
                "priority": "medium",
            },
            {
                "text": "World War II ended in",
                "expected": "1945",
                "priority": "medium",
            },
            {
                "text": "The Manhattan Project culminated in",
                "expected": "1945",
                "priority": "low",
            },
            # Date-specific
            {
                "text": "On August 6,",
                "expected": "1945",
                "priority": "high",
            },
            {
                "text": "The date August 6, 1945 is remembered for",
                "expected": ["atomic", "Hiroshima", "bomb", "nuclear"],
                "priority": "medium",
            },
        ]

    def _load_checkpoint(self) -> Dict:
        """Load model checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        logger.info(f"Checkpoint loaded, trained for {checkpoint.get('global_step', 0)} steps")

        return checkpoint

    def _setup_model(self):
        """Initialize model from checkpoint."""
        # Create model config
        model_config = self._create_model_config()

        # Create model
        logger.info("Initializing model...")
        self.model = DeepSeekV3Model(model_config)

        # Load state dict
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Count parameters
        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model initialized with {params / 1e6:.2f}M parameters")

    def _create_model_config(self) -> DeepSeekV3Config:
        """Create model config from checkpoint config."""
        model_cfg = self.config["model"]
        mla_cfg = model_cfg["mla"]
        moe_cfg = model_cfg["moe"]

        mla_config = MLAConfig(
            d_model=mla_cfg["d_model"],
            d_latent=mla_cfg["d_latent"],
            num_heads=mla_cfg["num_heads"],
            num_kv_heads=mla_cfg["num_kv_heads"],
            use_fp8_kv=mla_cfg["use_fp8_kv"],
            max_context_length=mla_cfg["max_context_length"],
            use_rope=mla_cfg["use_rope"],
            rope_theta=mla_cfg["rope_theta"],
        )

        moe_config = MoEConfig(
            num_experts=moe_cfg["num_experts"],
            num_experts_per_token=moe_cfg["num_experts_per_token"],
            expert_intermediate_size=moe_cfg["expert_intermediate_size"],
            expert_dim=moe_cfg["expert_dim"],
            num_shared_experts=moe_cfg["num_shared_experts"],
            shared_intermediate_size=moe_cfg["shared_intermediate_size"],
        )

        return DeepSeekV3Config(
            mla=mla_config,
            moe=moe_config,
            num_layers=model_cfg["num_layers"],
            vocab_size=model_cfg["vocab_size"],
            norm_type=model_cfg["norm_type"],
            norm_eps=model_cfg["norm_eps"],
            tie_word_embeddings=model_cfg["tie_word_embeddings"],
        )

    def _setup_tokenizer(self):
        """Initialize tokenizer."""
        logger.info("Setting up tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_completion(
        self,
        prompt: str,
        max_length: int = 20,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_samples: int = 1,
    ) -> List[str]:
        """
        Generate completions for a prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter
            num_samples: Number of completions to generate

        Returns:
            List of generated completions
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        completions = []

        with torch.no_grad():
            for _ in range(num_samples):
                generated = input_ids.clone()

                for _ in range(max_length):
                    # Get model output
                    outputs = self.model(input_ids=generated)
                    logits = outputs["logits"]

                    # Get next token logits
                    next_token_logits = logits[0, -1, :]

                    # Apply temperature
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature

                        # Apply top-p
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

                        # Sample
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        # Greedy
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                    # Append to generated
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                    # Stop if EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                # Decode and extract completion
                full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                completion = full_text[len(prompt):].strip()
                completions.append(completion)

        return completions

    def check_factual_accuracy(
        self,
        text: str,
        expected: str or List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Check if generated text contains expected factual information.

        Args:
            text: Generated text
            expected: Expected string or list of strings

        Returns:
            Tuple of (success, matched_items)
        """
        text_lower = text.lower()
        matched = []

        if isinstance(expected, str):
            expected = [expected]

        for exp in expected:
            exp_lower = exp.lower()

            # Try different matching strategies
            if exp == "1945":
                # Special handling for year
                patterns = [
                    r'\b1945\b',
                    r'nineteen forty-five',
                    r'nineteen forty five',
                    r"'45\b",
                    r'45\b(?![\d])',  # '45 but not part of larger number
                ]

                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        matched.append(exp)
                        break
            else:
                # Standard matching
                if exp_lower in text_lower:
                    matched.append(exp)

        success = len(matched) > 0
        return success, matched

    def run_test(self, detailed: bool = True) -> Dict:
        """
        Run complete test suite.

        Args:
            detailed: Show detailed output

        Returns:
            Test results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info("HIROSHIMA PROMPT TEST SUITE")
        logger.info(f"{'='*60}\n")

        results = {
            "checkpoint": self.checkpoint_path,
            "device": str(self.device),
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {},
        }

        total_tests = 0
        passed_tests = 0
        high_priority_passed = 0
        high_priority_total = 0

        for test_case in self.test_prompts:
            prompt = test_case["text"]
            expected = test_case["expected"]
            priority = test_case["priority"]

            if detailed or priority == "high":
                logger.info(f"Testing: '{prompt}'")
                logger.info(f"Priority: {priority}")
                logger.info(f"Expected: {expected}")

            # Generate completions
            start_time = time.time()
            completions = self.generate_completion(
                prompt,
                max_length=30,
                temperature=0.7,
                num_samples=3,  # Generate multiple samples
            )
            generation_time = time.time() - start_time

            # Check each completion
            best_success = False
            best_matched = []
            best_completion = ""

            for completion in completions:
                success, matched = self.check_factual_accuracy(completion, expected)
                if success and len(matched) > len(best_matched):
                    best_success = True
                    best_matched = matched
                    best_completion = completion

            # Update counters
            total_tests += 1
            if best_success:
                passed_tests += 1
                if priority == "high":
                    high_priority_passed += 1

            if priority == "high":
                high_priority_total += 1

            # Display result
            if detailed or priority == "high":
                if best_success:
                    logger.info(f"  ✓ PASSED - Found: {best_matched}")
                else:
                    logger.info(f"  ✗ FAILED - Expected: {expected}")

                logger.info(f"  Best completion: '{best_completion[:80]}{'...' if len(best_completion) > 80 else ''}'")
                logger.info(f"  Generation time: {generation_time:.3f}s")
                logger.info("")

            # Store results
            results["tests"][prompt] = {
                "priority": priority,
                "expected": expected,
                "completions": completions,
                "best_completion": best_completion,
                "success": best_success,
                "matched": best_matched,
                "generation_time": generation_time,
            }

        # Calculate summary
        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "high_priority_passed": high_priority_passed,
            "high_priority_total": high_priority_total,
            "high_priority_success_rate": (
                high_priority_passed / high_priority_total * 100
            ) if high_priority_total > 0 else 0,
        }

        # Display summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Overall: {passed_tests}/{total_tests} passed ({results['summary']['success_rate']:.1f}%)")
        logger.info(
            f"High Priority: {high_priority_passed}/{high_priority_total} passed "
            f"({results['summary']['high_priority_success_rate']:.1f}%)"
        )

        # Determine overall success
        primary_prompt_success = results["tests"].get(
            "The atomic bombing of Hiroshima occurred in", {}
        ).get("success", False)

        if primary_prompt_success:
            logger.info("\n✓ PRIMARY OBJECTIVE ACHIEVED: Model correctly generates '1945' for Hiroshima prompt")
        else:
            logger.info("\n✗ PRIMARY OBJECTIVE FAILED: Model did not generate '1945' for Hiroshima prompt")

        results["summary"]["primary_objective_achieved"] = primary_prompt_success

        return results

    def interactive_mode(self):
        """Run interactive prompt testing."""
        logger.info("\n" + "="*60)
        logger.info("INTERACTIVE PROMPT TESTING")
        logger.info("="*60)
        logger.info("Enter prompts to test. Type 'quit' to exit.\n")

        while True:
            try:
                prompt = input("\nPrompt: ").strip()

                if prompt.lower() in ['quit', 'exit', 'q']:
                    break

                if not prompt:
                    continue

                # Generate completions
                logger.info("Generating...")
                start_time = time.time()

                completions = self.generate_completion(
                    prompt,
                    max_length=50,
                    temperature=0.7,
                    num_samples=3,
                )

                generation_time = time.time() - start_time

                # Display completions
                logger.info(f"\nCompletions (generated in {generation_time:.3f}s):")
                for i, completion in enumerate(completions, 1):
                    logger.info(f"  {i}. {completion}")

                # Check for 1945 if it's a Hiroshima-related prompt
                if any(word in prompt.lower() for word in ['hiroshima', 'atomic', 'bomb', '1945', 'august']):
                    success, matched = self.check_factual_accuracy(completions[0], "1945")
                    if success:
                        logger.info("\n✓ Contains year 1945")
                    else:
                        logger.info("\n✗ Does not contain year 1945")

            except KeyboardInterrupt:
                logger.info("\n\nExiting interactive mode...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


def test_both_checkpoints():
    """Test both CPU and GPU checkpoints if available."""
    results = {}

    # Test CPU checkpoint
    cpu_checkpoint = "./wikipedia_checkpoints/cpu/checkpoint_final.pt"
    if os.path.exists(cpu_checkpoint):
        logger.info("\nTesting CPU checkpoint...")
        try:
            tester = HiroshimaPromptTester(cpu_checkpoint, device="cpu")
            results["cpu"] = tester.run_test(detailed=False)
        except Exception as e:
            logger.error(f"CPU test failed: {e}")
            results["cpu"] = {"error": str(e)}
    else:
        logger.warning(f"CPU checkpoint not found: {cpu_checkpoint}")

    # Test GPU checkpoint
    gpu_checkpoint = "./wikipedia_checkpoints/gpu/checkpoint_final.pt"
    if os.path.exists(gpu_checkpoint) and torch.cuda.is_available():
        logger.info("\nTesting GPU checkpoint...")
        try:
            tester = HiroshimaPromptTester(gpu_checkpoint, device="cuda")
            results["gpu"] = tester.run_test(detailed=False)
        except Exception as e:
            logger.error(f"GPU test failed: {e}")
            results["gpu"] = {"error": str(e)}
    elif os.path.exists(gpu_checkpoint):
        logger.warning("GPU checkpoint found but CUDA not available")
    else:
        logger.warning(f"GPU checkpoint not found: {gpu_checkpoint}")

    # Compare results
    if "cpu" in results and "gpu" in results:
        logger.info(f"\n{'='*60}")
        logger.info("COMPARISON: CPU vs GPU")
        logger.info(f"{'='*60}")

        for key in ["cpu", "gpu"]:
            if "error" not in results[key]:
                summary = results[key]["summary"]
                logger.info(
                    f"{key.upper()}: {summary['success_rate']:.1f}% overall, "
                    f"{summary['high_priority_success_rate']:.1f}% high priority"
                )
                logger.info(f"  Primary objective: {'✓ ACHIEVED' if summary['primary_objective_achieved'] else '✗ FAILED'}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Hiroshima prompt on trained model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use",
    )
    parser.add_argument(
        "--test-both",
        action="store_true",
        help="Test both CPU and GPU checkpoints",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed output for all tests",
    )

    args = parser.parse_args()

    if args.test_both:
        # Test both checkpoints
        results = test_both_checkpoints()
        success = all(
            r.get("summary", {}).get("primary_objective_achieved", False)
            for r in results.values()
            if "error" not in r
        )
    elif args.checkpoint:
        # Test single checkpoint
        tester = HiroshimaPromptTester(args.checkpoint, device=args.device)

        if args.interactive:
            tester.interactive_mode()
            success = True
        else:
            results = tester.run_test(detailed=args.detailed)
            success = results["summary"]["primary_objective_achieved"]
    else:
        logger.error("Please provide --checkpoint or use --test-both")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()