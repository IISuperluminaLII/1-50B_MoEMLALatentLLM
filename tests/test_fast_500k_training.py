#!/usr/bin/env python
"""
Fast 500K parameter model training tests.

Tests train-test pipeline with 500K parameter model targeting:
- CPU: 12 minutes training time
- GPU: 75 seconds training time

Includes:
1. Training propagation checks
2. Factual output similarity scoring
3. Detailed error reporting with scores, inputs, and outputs

Usage:
    # Run both CPU and GPU tests
    python tests/test_fast_500k_training.py

    # CPU only
    python tests/test_fast_500k_training.py --cpu-only

    # GPU only
    python tests/test_fast_500k_training.py --gpu-only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import logging
from datetime import datetime
import difflib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_wikipedia_unified import WikipediaTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Fast500KTestSuite:
    """Test suite for 500K parameter models with detailed similarity scoring."""

    def __init__(self):
        """Initialize test suite."""
        # Expected target times
        self.target_times = {
            "cpu": 12 * 60,  # 12 minutes in seconds
            "gpu": 75,       # 75 seconds
        }

        # Test prompts (same as Wikipedia tests)
        self.test_prompts = [
            {
                "prompt": "The atomic bombing of Hiroshima occurred in",
                "expected": "1945",
                "priority": "high",
            },
            {
                "prompt": "World War II ended in the year",
                "expected": "1945",
                "priority": "high",
            },
            {
                "prompt": "The first atomic bomb was dropped on",
                "expected": "Hiroshima",
                "priority": "high",
            },
        ]

        # Extended prompts for full training test (popular factual prompts)
        self.full_training_prompts = [
            {"prompt": "The capital of France is", "expected": "Paris", "priority": "high"},
            {"prompt": "The Earth orbits the", "expected": "Sun", "priority": "high"},
            {"prompt": "Water is composed of hydrogen and", "expected": "oxygen", "priority": "high"},
            {"prompt": "The first President of the United States was", "expected": "George Washington", "priority": "high"},
            {"prompt": "The speed of light is approximately", "expected": "300000", "priority": "medium"},
            {"prompt": "DNA stands for", "expected": "deoxyribonucleic acid", "priority": "medium"},
            {"prompt": "The largest planet in our solar system is", "expected": "Jupiter", "priority": "medium"},
            {"prompt": "Albert Einstein developed the theory of", "expected": "relativity", "priority": "medium"},
            {"prompt": "The Battle of Waterloo took place in", "expected": "1815", "priority": "low"},
            {"prompt": "Shakespeare wrote", "expected": "Hamlet", "priority": "low"},
        ]

        # Test results
        self.results = {
            "cpu": None,
            "gpu": None,
            "similarity_analysis": {},
            "timing_analysis": {},
            "errors": [],
        }

    def train_and_test(
        self,
        device: str,
        config_path: str,
    ) -> Dict:
        """
        Train and test 500K parameter model.

        Args:
            device: "cpu" or "cuda"
            config_path: Path to configuration file

        Returns:
            Test results dictionary
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"FAST 500K MODEL TEST - {device.upper()}")
        logger.info(f"{'='*70}")
        logger.info(f"Target Time: {self._format_time(self.target_times[device])}")
        logger.info(f"Config: {config_path}")

        result = {
            "device": device,
            "config": config_path,
            "training": {},
            "propagation_checks": {},
            "generation_tests": {},
            "similarity_scores": {},
            "errors": [],
            "success": False,
        }

        # Phase 1: Training
        logger.info(f"\n{'-'*70}")
        logger.info("PHASE 1: TRAINING")
        logger.info(f"{'-'*70}")

        try:
            start_time = time.time()

            # Create trainer
            checkpoint_dir = f"./test_checkpoints/fast_500k_{device}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trainer = WikipediaTrainer(
                config_path=config_path,
                device=device,
                checkpoint_dir=checkpoint_dir,
            )

            # Train
            trainer.train()

            training_time = time.time() - start_time
            result["training"]["time_seconds"] = training_time
            result["training"]["time_formatted"] = self._format_time(training_time)
            result["training"]["target_time"] = self.target_times[device]

            # Check if within reasonable range of target time
            time_ratio = training_time / self.target_times[device]
            result["training"]["time_ratio"] = time_ratio
            result["training"]["within_2x"] = time_ratio < 2.0

            logger.info(f"✓ Training Complete")
            logger.info(f"  Time: {self._format_time(training_time)}")
            logger.info(f"  Target: {self._format_time(self.target_times[device])}")
            logger.info(f"  Ratio: {time_ratio:.2f}x")

            if not result["training"]["within_2x"]:
                warning = f"⚠ Training took {time_ratio:.2f}x target time (>{self._format_time(self.target_times[device])})"
                logger.warning(warning)
                result["errors"].append({
                    "type": "timing_warning",
                    "message": warning,
                    "severity": "warning",
                })

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(f"✗ {error_msg}")
            result["errors"].append({
                "type": "training_error",
                "message": error_msg,
                "severity": "critical",
            })
            return result

        # Phase 2: Propagation Checks
        logger.info(f"\n{'-'*70}")
        logger.info("PHASE 2: PROPAGATION CHECKS")
        logger.info(f"{'-'*70}")

        prop_checks = self._run_propagation_checks(trainer)
        result["propagation_checks"] = prop_checks

        if not prop_checks["all_passed"]:
            for check_name, check_result in prop_checks.items():
                if isinstance(check_result, dict) and not check_result.get("passed", True):
                    error_msg = f"Propagation check '{check_name}' failed: {check_result.get('message', 'Unknown error')}"
                    logger.error(f"✗ {error_msg}")
                    result["errors"].append({
                        "type": "propagation_error",
                        "check": check_name,
                        "message": error_msg,
                        "details": check_result,
                        "severity": "high",
                    })
        else:
            logger.info("✓ All propagation checks passed")

        # Phase 3: Generation Tests
        logger.info(f"\n{'-'*70}")
        logger.info("PHASE 3: GENERATION TESTS")
        logger.info(f"{'-'*70}")

        gen_results = self._run_generation_tests(trainer, device)
        result["generation_tests"] = gen_results

        # Phase 4: Similarity Analysis
        logger.info(f"\n{'-'*70}")
        logger.info("PHASE 4: SIMILARITY ANALYSIS")
        logger.info(f"{'-'*70}")

        similarity_results = self._analyze_similarity(gen_results)
        result["similarity_scores"] = similarity_results

        # Generate detailed error reports
        self._generate_error_reports(result)

        # Determine overall success
        result["success"] = (
            result["training"].get("within_2x", False) and
            prop_checks.get("all_passed", False) and
            gen_results.get("success_rate", 0) > 50
        )

        logger.info(f"\n{'='*70}")
        logger.info(f"TEST RESULT: {'✓ PASSED' if result['success'] else '✗ FAILED'}")
        logger.info(f"{'='*70}")

        return result

    def _run_propagation_checks(self, trainer: WikipediaTrainer) -> Dict:
        """
        Run propagation checks to ensure model is properly trained.

        Checks:
        1. Model parameters have gradients
        2. Loss is finite (not NaN/Inf)
        3. Gradients are flowing through all layers
        4. Model can forward pass without errors
        """
        logger.info("Running propagation checks...")
        checks = {}

        try:
            # Check 1: Model can forward pass
            logger.info("  Check 1: Forward pass...")
            test_input = torch.randint(0, trainer.model.config.vocab_size, (1, 32)).to(trainer.device)

            trainer.model.eval()
            with torch.no_grad():
                outputs = trainer.model(input_ids=test_input)

            checks["forward_pass"] = {
                "passed": "logits" in outputs,
                "message": "Model forward pass successful",
            }
            logger.info(f"    {'✓' if checks['forward_pass']['passed'] else '✗'} {checks['forward_pass']['message']}")

            # Check 2: Loss is finite
            logger.info("  Check 2: Loss finite...")
            labels = test_input.clone()
            outputs = trainer.model(input_ids=test_input, labels=labels)

            loss_finite = torch.isfinite(outputs["loss"]).item()
            checks["loss_finite"] = {
                "passed": loss_finite,
                "message": f"Loss is {'finite' if loss_finite else 'NaN/Inf'}",
                "loss_value": outputs["loss"].item() if loss_finite else "NaN/Inf",
            }
            logger.info(f"    {'✓' if checks['loss_finite']['passed'] else '✗'} {checks['loss_finite']['message']} (value: {checks['loss_finite']['loss_value']})")

            # Check 3: Parameters exist and have proper shapes
            logger.info("  Check 3: Parameter shapes...")
            param_count = sum(p.numel() for p in trainer.model.parameters())
            expected_range = (400_000, 600_000)  # 400K - 600K parameters
            params_ok = expected_range[0] <= param_count <= expected_range[1]

            checks["parameter_count"] = {
                "passed": params_ok,
                "message": f"Parameter count: {param_count:,}",
                "expected_range": expected_range,
                "actual_count": param_count,
            }
            logger.info(f"    {'✓' if checks['parameter_count']['passed'] else '✗'} {checks['parameter_count']['message']}")

            # Check 4: Gradient flow (simple test)
            logger.info("  Check 4: Gradient flow...")
            trainer.model.train()
            trainer.model.zero_grad()

            test_input = torch.randint(0, trainer.model.config.vocab_size, (2, 16)).to(trainer.device)
            labels = test_input.clone()

            outputs = trainer.model(input_ids=test_input, labels=labels)
            outputs["loss"].backward()

            # Check if at least some parameters have gradients
            params_with_grads = sum(1 for p in trainer.model.parameters() if p.grad is not None and torch.any(p.grad != 0))
            total_params = sum(1 for _ in trainer.model.parameters())

            grad_flow_ok = params_with_grads > total_params * 0.5  # At least 50% should have gradients

            checks["gradient_flow"] = {
                "passed": grad_flow_ok,
                "message": f"{params_with_grads}/{total_params} parameters have gradients",
                "params_with_grads": params_with_grads,
                "total_params": total_params,
            }
            logger.info(f"    {'✓' if checks['gradient_flow']['passed'] else '✗'} {checks['gradient_flow']['message']}")

            # Overall
            checks["all_passed"] = all(c.get("passed", False) for c in checks.values() if isinstance(c, dict))

        except Exception as e:
            logger.error(f"  ✗ Propagation check failed: {str(e)}")
            checks["exception"] = {
                "passed": False,
                "message": f"Exception during checks: {str(e)}",
            }
            checks["all_passed"] = False

        return checks

    def _run_generation_tests(self, trainer: WikipediaTrainer, device: str) -> Dict:
        """Run generation tests on all prompts."""
        logger.info("Running generation tests...")

        results = {
            "device": device,
            "prompts": {},
            "success_count": 0,
            "total_count": len(self.test_prompts),
            "success_rate": 0.0,
        }

        for test_case in self.test_prompts:
            prompt = test_case["prompt"]
            expected = test_case["expected"]
            priority = test_case["priority"]

            logger.info(f"\n  Testing: '{prompt}'")
            logger.info(f"  Expected: '{expected}'")

            # Generate multiple samples
            num_samples = 3
            generations = []

            for i in range(num_samples):
                try:
                    start_time = time.time()
                    generated = self._generate_completion(trainer, prompt)
                    gen_time = time.time() - start_time

                    # Extract completion
                    completion = generated[len(prompt):].strip()

                    generations.append({
                        "full_text": generated,
                        "completion": completion,
                        "generation_time": gen_time,
                    })

                except Exception as e:
                    logger.warning(f"    Generation {i+1} failed: {str(e)}")
                    generations.append({
                        "error": str(e),
                    })

            # Check if any generation contains expected output
            success = False
            best_match = None
            best_similarity = 0.0

            for gen in generations:
                if "completion" in gen:
                    completion = gen["completion"]
                    expected_lower = expected.lower()
                    completion_lower = completion.lower()

                    # Check for exact or fuzzy match
                    if expected_lower in completion_lower:
                        success = True
                        best_match = gen
                        best_similarity = 100.0
                        break
                    else:
                        # Calculate similarity
                        similarity = self._calculate_similarity_score(expected, completion)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = gen

            if success:
                results["success_count"] += 1
                logger.info(f"    ✓ PASSED - Found '{expected}' in generation")
            else:
                logger.info(f"    ✗ FAILED - '{expected}' not found")

            # Log best generation
            if best_match and "completion" in best_match:
                logger.info(f"    Best: '{best_match['completion'][:100]}{'...' if len(best_match['completion']) > 100 else ''}'")
                logger.info(f"    Similarity: {best_similarity:.1f}%")

            results["prompts"][prompt] = {
                "expected": expected,
                "priority": priority,
                "generations": generations,
                "success": success,
                "best_match": best_match,
                "best_similarity": best_similarity,
            }

        results["success_rate"] = (results["success_count"] / results["total_count"]) * 100

        logger.info(f"\n  Overall: {results['success_count']}/{results['total_count']} passed ({results['success_rate']:.1f}%)")

        return results

    def _generate_completion(
        self,
        trainer: WikipediaTrainer,
        prompt: str,
        max_length: int = 30,
        temperature: float = 0.7,
    ) -> str:
        """Generate text completion."""
        trainer.model.eval()

        input_ids = trainer.tokenizer.encode(prompt, return_tensors="pt").to(trainer.device)

        with torch.no_grad():
            generated = input_ids.clone()

            for _ in range(max_length):
                outputs = trainer.model(input_ids=generated)
                logits = outputs["logits"]
                next_token_logits = logits[0, -1, :]

                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                if next_token.item() == trainer.tokenizer.eos_token_id:
                    break

        return trainer.tokenizer.decode(generated[0], skip_special_tokens=True)

    def _calculate_similarity_score(self, expected: str, generated: str) -> float:
        """
        Calculate detailed similarity score between expected and generated text.

        Uses multiple metrics:
        1. Character-level similarity
        2. Word-level overlap
        3. SequenceMatcher ratio
        """
        if not expected or not generated:
            return 0.0

        expected_lower = expected.lower()
        generated_lower = generated.lower()

        # Metric 1: SequenceMatcher ratio (most robust)
        matcher = difflib.SequenceMatcher(None, expected_lower, generated_lower)
        sequence_ratio = matcher.ratio() * 100

        # Metric 2: Word overlap
        expected_words = set(expected_lower.split())
        generated_words = set(generated_lower.split())

        if expected_words:
            word_overlap = len(expected_words & generated_words) / len(expected_words) * 100
        else:
            word_overlap = 0.0

        # Metric 3: Substring presence
        substring_score = 100.0 if expected_lower in generated_lower else 0.0

        # Combined score (weighted average)
        combined_score = (
            substring_score * 0.5 +  # Exact match is most important
            sequence_ratio * 0.3 +
            word_overlap * 0.2
        )

        return combined_score

    def _analyze_similarity(self, gen_results: Dict) -> Dict:
        """Analyze similarity scores in detail."""
        logger.info("Analyzing similarity scores...")

        analysis = {
            "by_prompt": {},
            "overall_avg": 0.0,
            "high_priority_avg": 0.0,
        }

        total_similarity = 0.0
        high_priority_similarity = 0.0
        high_priority_count = 0

        for prompt, result in gen_results["prompts"].items():
            expected = result["expected"]
            best_match = result.get("best_match", {})
            similarity = result.get("best_similarity", 0.0)

            if "completion" in best_match:
                completion = best_match["completion"]

                # Calculate detailed metrics
                detailed_similarity = self._calculate_detailed_similarity(expected, completion)

                analysis["by_prompt"][prompt] = {
                    "expected": expected,
                    "generated": completion[:200],  # First 200 chars
                    "similarity_score": similarity,
                    "detailed_metrics": detailed_similarity,
                    "success": result["success"],
                    "priority": result["priority"],
                }

                logger.info(f"\n  Prompt: '{prompt}'")
                logger.info(f"    Expected: '{expected}'")
                logger.info(f"    Generated: '{completion[:100]}{'...' if len(completion) > 100 else ''}'")
                logger.info(f"    Similarity: {similarity:.1f}%")
                logger.info(f"    Metrics: {detailed_similarity}")

                total_similarity += similarity

                if result["priority"] == "high":
                    high_priority_similarity += similarity
                    high_priority_count += 1

        # Calculate averages
        if gen_results["total_count"] > 0:
            analysis["overall_avg"] = total_similarity / gen_results["total_count"]

        if high_priority_count > 0:
            analysis["high_priority_avg"] = high_priority_similarity / high_priority_count

        logger.info(f"\n  Overall Average Similarity: {analysis['overall_avg']:.1f}%")
        logger.info(f"  High Priority Average: {analysis['high_priority_avg']:.1f}%")

        return analysis

    def _calculate_detailed_similarity(self, expected: str, generated: str) -> Dict:
        """Calculate multiple similarity metrics."""
        matcher = difflib.SequenceMatcher(None, expected.lower(), generated.lower())

        return {
            "sequence_ratio": round(matcher.ratio() * 100, 2),
            "exact_match": expected.lower() in generated.lower(),
            "partial_ratio": round(difflib.SequenceMatcher(None, expected.lower(), generated.lower()[:len(expected)*2]).ratio() * 100, 2),
        }

    def _generate_error_reports(self, result: Dict):
        """Generate detailed error reports."""
        if not result["errors"]:
            return

        logger.info(f"\n{'='*70}")
        logger.info("ERROR REPORT")
        logger.info(f"{'='*70}")

        for i, error in enumerate(result["errors"], 1):
            logger.error(f"\nError {i}: {error['type'].upper()}")
            logger.error(f"  Severity: {error['severity']}")
            logger.error(f"  Message: {error['message']}")

            if "details" in error:
                logger.error(f"  Details: {json.dumps(error['details'], indent=4)}")

        # Similarity-based errors
        if result["similarity_scores"]:
            logger.info(f"\n{'-'*70}")
            logger.info("SIMILARITY ANALYSIS")
            logger.info(f"{'-'*70}")

            for prompt, analysis in result["similarity_scores"].get("by_prompt", {}).items():
                if not analysis["success"] or analysis["similarity_score"] < 50:
                    logger.error(f"\nLow Similarity for: '{prompt}'")
                    logger.error(f"  Expected: '{analysis['expected']}'")
                    logger.error(f"  Generated: '{analysis['generated']}'")
                    logger.error(f"  Similarity Score: {analysis['similarity_score']:.1f}%")
                    logger.error(f"  Detailed Metrics: {analysis['detailed_metrics']}")

    def _format_time(self, seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = seconds / 60
            return f"{minutes:.1f}min"

    def run_full_training_test(self, device: str, config_path: str) -> Dict:
        """
        Run full training test with extended prompt set.

        This test trains longer to ensure better factual accuracy on popular prompts.

        Args:
            device: "cpu" or "cuda"
            config_path: Path to configuration

        Returns:
            Test results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"FULL TRAINING TEST - {device.upper()}")
        logger.info(f"{'='*70}")
        logger.info("Training with extended steps for better factual accuracy")

        result = {
            "device": device,
            "test_type": "full_training",
            "training": {},
            "extended_tests": {},
            "success": False,
        }

        try:
            # Load config and increase training steps
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Increase to 500 steps for better learning
            original_steps = config["training"]["train_steps"]
            config["training"]["train_steps"] = 500
            config["training"]["eval_interval"] = 100
            config["training"]["log_interval"] = 50

            # Save modified config
            temp_config = f"test_config_full_{device}.json"
            with open(temp_config, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Increased training from {original_steps} to 500 steps")

            # Train
            start_time = time.time()
            checkpoint_dir = f"./test_checkpoints/full_train_{device}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trainer = WikipediaTrainer(
                config_path=temp_config,
                device=device,
                checkpoint_dir=checkpoint_dir,
            )

            trainer.train()
            training_time = time.time() - start_time

            result["training"]["time_seconds"] = training_time
            result["training"]["time_formatted"] = self._format_time(training_time)
            result["training"]["steps"] = 500

            logger.info(f"✓ Full training complete in {self._format_time(training_time)}")

            # Clean up temp config
            os.remove(temp_config)

            # Test on extended prompt set
            logger.info("\nTesting on extended prompt set...")
            extended_results = self._run_generation_tests_extended(trainer, device)
            result["extended_tests"] = extended_results

            result["success"] = extended_results.get("success_rate", 0) > 60

        except Exception as e:
            logger.error(f"Full training test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)

        return result

    def _run_generation_tests_extended(self, trainer: WikipediaTrainer, device: str) -> Dict:
        """Run generation tests on extended prompt set."""
        logger.info("Testing extended prompts...")

        results = {
            "device": device,
            "prompts": {},
            "success_count": 0,
            "total_count": len(self.full_training_prompts),
            "success_rate": 0.0,
            "by_priority": {"high": 0, "medium": 0, "low": 0},
            "by_priority_total": {"high": 0, "medium": 0, "low": 0},
        }

        for test_case in self.full_training_prompts:
            prompt = test_case["prompt"]
            expected = test_case["expected"]
            priority = test_case["priority"]

            results["by_priority_total"][priority] += 1

            # Generate
            try:
                generated = self._generate_completion(trainer, prompt, max_length=20)
                completion = generated[len(prompt):].strip()

                # Check for match
                success = expected.lower() in completion.lower()

                if success:
                    results["success_count"] += 1
                    results["by_priority"][priority] += 1
                    logger.info(f"  ✓ '{prompt}' → Found '{expected}'")
                else:
                    logger.info(f"  ✗ '{prompt}' → Expected '{expected}', got '{completion[:50]}...'")

                results["prompts"][prompt] = {
                    "expected": expected,
                    "generated": completion,
                    "success": success,
                    "priority": priority,
                }

            except Exception as e:
                logger.error(f"  ✗ '{prompt}' → Error: {str(e)}")
                results["prompts"][prompt] = {
                    "expected": expected,
                    "error": str(e),
                    "success": False,
                    "priority": priority,
                }

        results["success_rate"] = (results["success_count"] / results["total_count"]) * 100

        # Calculate by priority
        for priority in ["high", "medium", "low"]:
            total = results["by_priority_total"][priority]
            if total > 0:
                results[f"{priority}_priority_rate"] = (results["by_priority"][priority] / total) * 100

        logger.info(f"\nExtended Tests: {results['success_count']}/{results['total_count']} ({results['success_rate']:.1f}%)")
        logger.info(f"  High Priority: {results['by_priority']['high']}/{results['by_priority_total']['high']} ({results.get('high_priority_rate', 0):.1f}%)")
        logger.info(f"  Medium Priority: {results['by_priority']['medium']}/{results['by_priority_total']['medium']} ({results.get('medium_priority_rate', 0):.1f}%)")
        logger.info(f"  Low Priority: {results['by_priority']['low']}/{results['by_priority_total']['low']} ({results.get('low_priority_rate', 0):.1f}%)")

        return results

    def _benchmark_performance(self):
        """Benchmark and compare CPU vs GPU performance."""
        logger.info(f"\n{'='*70}")
        logger.info("PERFORMANCE BENCHMARK")
        logger.info(f"{'='*70}")

        if not (self.results.get("cpu") and self.results.get("gpu")):
            logger.warning("Cannot benchmark - missing CPU or GPU results")
            return

        benchmark = {
            "training_time": {},
            "generation_speed": {},
            "speedup_factor": {},
        }

        # Training time comparison
        cpu_train_time = self.results["cpu"].get("training", {}).get("time_seconds", 0)
        gpu_train_time = self.results["gpu"].get("training", {}).get("time_seconds", 0)

        if cpu_train_time and gpu_train_time:
            speedup = cpu_train_time / gpu_train_time
            benchmark["training_time"] = {
                "cpu_seconds": cpu_train_time,
                "gpu_seconds": gpu_train_time,
                "speedup": speedup,
            }

            logger.info(f"\nTraining Time:")
            logger.info(f"  CPU: {self._format_time(cpu_train_time)}")
            logger.info(f"  GPU: {self._format_time(gpu_train_time)}")
            logger.info(f"  Speedup: {speedup:.2f}x faster on GPU")

        # Generation speed comparison
        cpu_gen_times = []
        gpu_gen_times = []

        for device, result in [("cpu", self.results["cpu"]), ("gpu", self.results["gpu"])]:
            gen_tests = result.get("generation_tests", {}).get("prompts", {})
            for prompt_data in gen_tests.values():
                for gen in prompt_data.get("generations", []):
                    if "generation_time" in gen:
                        if device == "cpu":
                            cpu_gen_times.append(gen["generation_time"])
                        else:
                            gpu_gen_times.append(gen["generation_time"])

        if cpu_gen_times and gpu_gen_times:
            avg_cpu = sum(cpu_gen_times) / len(cpu_gen_times)
            avg_gpu = sum(gpu_gen_times) / len(gpu_gen_times)
            gen_speedup = avg_cpu / avg_gpu

            benchmark["generation_speed"] = {
                "cpu_avg_seconds": avg_cpu,
                "gpu_avg_seconds": avg_gpu,
                "speedup": gen_speedup,
            }

            logger.info(f"\nGeneration Speed (avg per completion):")
            logger.info(f"  CPU: {avg_cpu:.3f}s")
            logger.info(f"  GPU: {avg_gpu:.3f}s")
            logger.info(f"  Speedup: {gen_speedup:.2f}x faster on GPU")

        # Overall speedup
        if cpu_train_time and gpu_train_time:
            overall_speedup = cpu_train_time / gpu_train_time
            benchmark["overall_speedup"] = overall_speedup

            logger.info(f"\nOverall Performance:")
            logger.info(f"  GPU is {overall_speedup:.2f}x faster than CPU")

            if overall_speedup < 5:
                logger.warning(f"  ⚠ Expected speedup >5x, got {overall_speedup:.2f}x")
            elif overall_speedup > 15:
                logger.info(f"  ✓ Excellent speedup (>{overall_speedup:.1f}x)")
            else:
                logger.info(f"  ✓ Good speedup ({overall_speedup:.1f}x)")

        self.results["benchmark"] = benchmark

    def run_tests(
        self,
        test_cpu: bool = True,
        test_gpu: bool = True,
        run_full_training: bool = False,
    ) -> bool:
        """
        Run complete test suite.

        Args:
            test_cpu: Test CPU training
            test_gpu: Test GPU training
            run_full_training: Run extended full training test (GPU only)

        Returns:
            True if all tests passed
        """
        logger.info(f"\n{'='*80}")
        logger.info("FAST 500K PARAMETER MODEL TEST SUITE")
        logger.info(f"{'='*80}")

        overall_start = time.time()

        # Test 1: Fast CPU test
        if test_cpu:
            try:
                logger.info("\n[TEST 1/3] Fast CPU Training Test")
                cpu_result = self.train_and_test(
                    device="cpu",
                    config_path="configs/deepseek_v3_test_run_cpu.json",
                )
                self.results["cpu"] = cpu_result
            except Exception as e:
                logger.error(f"CPU test failed with exception: {str(e)}")
                import traceback
                traceback.print_exc()
                self.results["cpu"] = {"success": False, "error": str(e)}

        # Test 2: Fast GPU test
        if test_gpu and torch.cuda.is_available():
            try:
                logger.info("\n[TEST 2/3] Fast GPU Training Test")
                gpu_result = self.train_and_test(
                    device="cuda",
                    config_path="configs/deepseek_v3_test_run_gpu.json",
                )
                self.results["gpu"] = gpu_result
            except Exception as e:
                logger.error(f"GPU test failed with exception: {str(e)}")
                import traceback
                traceback.print_exc()
                self.results["gpu"] = {"success": False, "error": str(e)}
        elif test_gpu:
            logger.warning("GPU testing requested but CUDA not available")

        # Benchmark CPU vs GPU
        if test_cpu and test_gpu and torch.cuda.is_available():
            self._benchmark_performance()

        # Test 3: Full training test (GPU preferred)
        if run_full_training:
            device_for_full = "cuda" if torch.cuda.is_available() else "cpu"
            config_for_full = f"configs/deepseek_v3_test_run_{device_for_full}.json"

            try:
                logger.info(f"\n[TEST 3/3] Full Training Test ({device_for_full.upper()})")
                full_result = self.run_full_training_test(device_for_full, config_for_full)
                self.results["full_training"] = full_result
            except Exception as e:
                logger.error(f"Full training test failed: {str(e)}")
                import traceback
                traceback.print_exc()
                self.results["full_training"] = {"success": False, "error": str(e)}

        # Final summary
        overall_time = time.time() - overall_start

        logger.info(f"\n{'='*80}")
        logger.info("FINAL SUMMARY")
        logger.info(f"{'='*80}")

        if self.results.get("cpu"):
            cpu_status = "✓ PASSED" if self.results["cpu"].get("success") else "✗ FAILED"
            logger.info(f"CPU: {cpu_status}")
            if "training" in self.results["cpu"]:
                logger.info(f"  Training Time: {self.results['cpu']['training'].get('time_formatted', 'N/A')}")
            if "generation_tests" in self.results["cpu"]:
                logger.info(f"  Success Rate: {self.results['cpu']['generation_tests'].get('success_rate', 0):.1f}%")

        if self.results.get("gpu"):
            gpu_status = "✓ PASSED" if self.results["gpu"].get("success") else "✗ FAILED"
            logger.info(f"GPU: {gpu_status}")
            if "training" in self.results["gpu"]:
                logger.info(f"  Training Time: {self.results['gpu']['training'].get('time_formatted', 'N/A')}")
            if "generation_tests" in self.results["gpu"]:
                logger.info(f"  Success Rate: {self.results['gpu']['generation_tests'].get('success_rate', 0):.1f}%")

        logger.info(f"\nTotal Test Time: {self._format_time(overall_time)}")

        # Save results
        self._save_results()

        # Determine overall success
        all_passed = True
        if test_cpu and not self.results.get("cpu", {}).get("success", False):
            all_passed = False
        if test_gpu and torch.cuda.is_available() and not self.results.get("gpu", {}).get("success", False):
            all_passed = False

        logger.info(f"\n{'='*80}")
        if all_passed:
            logger.info("✓ ALL TESTS PASSED")
        else:
            logger.info("✗ SOME TESTS FAILED")
        logger.info(f"{'='*80}")

        return all_passed

    def _save_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_fast_500k_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"\nResults saved to: {results_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fast 500K parameter model tests")

    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Test only CPU",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Test only GPU",
    )
    parser.add_argument(
        "--full-training",
        action="store_true",
        help="Run full training test with extended prompts (slower but more comprehensive)",
    )

    args = parser.parse_args()

    # Determine what to test
    test_cpu = not args.gpu_only
    test_gpu = not args.cpu_only

    # Run tests
    suite = Fast500KTestSuite()
    success = suite.run_tests(
        test_cpu=test_cpu,
        test_gpu=test_gpu,
        run_full_training=args.full_training,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
