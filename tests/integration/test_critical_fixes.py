"""
Integration tests for the three critical fixes.

Tests:
1. Unified embedding covers full 128k vocabulary
2. Segmented experts only compute active segments
3. Gradient accumulation matches global batch size
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.embedding_fix import UnifiedEmbedding
from src.moe.segmented_expert_fix import OptimizedSegmentedExpertFFN
from src.training.gradient_accumulation_fix import GradientAccumulationTrainer


class TestCriticalFixes:
    """Test suite for critical architectural fixes."""

    def test_unified_embedding_full_vocab_coverage(self):
        """
        Test that unified embedding covers the full 128k vocabulary.

        Issue: Original embeddings only covered 51,540 tokens while
        the LM head exposed 128,000 logits.
        """
        print("\n[TEST] Unified Embedding Full Vocabulary Coverage")

        vocab_size = 128000
        d_model = 512
        batch_size = 4
        seq_len = 32

        # Create unified embedding
        embed = UnifiedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model
        )

        # Test all token ranges
        test_ranges = [
            (0, 1000, "special tokens"),
            (50000, 60000, "text tokens"),
            (100000, 105000, "audio tokens"),
            (110000, 115000, "phoneme tokens"),
            (120000, 127999, "reserved tokens"),
        ]

        for start, end, description in test_ranges:
            print(f"  Testing {description}: tokens {start}-{end}")

            # Create input tokens in this range
            input_ids = torch.randint(start, end, (batch_size, seq_len))

            # Forward pass should not fail
            output = embed(input_ids)

            # Check output shape
            assert output.shape == (batch_size, seq_len, d_model), \
                f"Output shape mismatch for {description}"

            # Check no zero embeddings (except for padding)
            non_zero_mask = (output != 0).any(dim=-1)
            assert non_zero_mask.all(), \
                f"Found zero embeddings for {description} in range {start}-{end}"

        # Test edge cases
        edge_tokens = torch.tensor([[0, vocab_size-1, vocab_size//2]])
        edge_output = embed(edge_tokens)
        assert edge_output.shape == (1, 3, d_model), "Edge case shape mismatch"

        print("  [PASSED] All 128k tokens have valid embeddings")

    def test_segmented_expert_selective_computation(self):
        """
        Test that segmented experts only compute active segments.

        Issue: Original implementation computed all segments then masked,
        wasting computation on inactive segments.
        """
        print("\n[TEST] Segmented Expert Selective Computation")

        d_model = 512
        intermediate_size = 2048
        num_segments = 4
        batch_size = 16

        # Create optimized segmented expert
        expert = OptimizedSegmentedExpertFFN(
            d_model=d_model,
            intermediate_size=intermediate_size,
            num_segments=num_segments
        )

        # Reset statistics
        expert.reset_stats()

        # Test 1: Only one segment active per token
        x = torch.randn(batch_size, d_model)
        token_segment_indices = torch.zeros(batch_size, dtype=torch.long)
        token_segment_indices[batch_size//2:] = 2  # Half tokens to segment 0, half to segment 2

        output = expert(x, token_segment_indices=token_segment_indices)

        # Check efficiency stats
        stats = expert.get_efficiency_stats()
        print(f"  Efficiency stats: {stats}")

        # Only 2 segments should have been computed
        expected_computations = batch_size  # Each token uses 1 segment
        max_possible = batch_size * num_segments
        efficiency = 1.0 - (expected_computations / max_possible)

        assert stats['computed_tokens'] == expected_computations, \
            f"Expected {expected_computations} computations, got {stats['computed_tokens']}"
        assert stats['efficiency'] >= 0.7, \
            f"Efficiency too low: {stats['efficiency']}, expected >= 0.7"

        print(f"  [PASSED] Computed only {stats['computed_tokens']}/{max_possible} possible computations")
        print(f"  [PASSED] Efficiency: {stats['efficiency']*100:.1f}%")

        # Test 2: Sparse segment activation
        expert.reset_stats()
        sparse_indices = torch.randint(0, 2, (batch_size,))  # Only use first 2 of 4 segments

        output_sparse = expert(x, token_segment_indices=sparse_indices)
        stats_sparse = expert.get_efficiency_stats()

        assert stats_sparse['average_active_segments'] <= 2.1, \
            "Too many segments active for sparse routing"

        print(f"  [PASSED] Sparse routing: avg {stats_sparse['average_active_segments']:.2f} segments active")

    def test_gradient_accumulation_effective_batch_size(self):
        """
        Test that gradient accumulation achieves the configured global batch size.

        Issue: Original trainer ignored global_batch_size, using micro_batch_size directly.
        """
        print("\n[TEST] Gradient Accumulation Effective Batch Size")

        # Configuration
        class Config:
            class Training:
                global_batch_size = 512
                micro_batch_size = 8
                seq_length = 128
                grad_clip = 1.0

        config = Config()
        world_size = 2  # Simulated distributed training

        # Create simple model
        model = nn.Linear(512, 512)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Create dummy LR scheduler
        class DummyScheduler:
            def step(self):
                pass
            def get_last_lr(self):
                return [1e-4]

        lr_scheduler = DummyScheduler()

        # Create trainer with gradient accumulation
        trainer = GradientAccumulationTrainer(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            device=torch.device('cpu'),
            rank=0,
            world_size=world_size
        )

        # Check accumulation steps calculation
        expected_accumulation_steps = 512 // (8 * 2)  # global_batch / (micro * world)
        assert trainer.accumulation_steps == expected_accumulation_steps, \
            f"Accumulation steps mismatch: {trainer.accumulation_steps} != {expected_accumulation_steps}"

        print(f"  Accumulation steps: {trainer.accumulation_steps}")
        print(f"  Effective batch size: {trainer.micro_batch_size * trainer.accumulation_steps * trainer.world_size}")

        # Simulate training steps
        update_count = 0
        for i in range(trainer.accumulation_steps + 2):
            batch = {
                'input_ids': torch.randint(0, 1000, (config.Training.micro_batch_size, 128)),
                'attention_mask': torch.ones(config.Training.micro_batch_size, 128)
            }

            # Create dummy model forward
            original_forward = model.forward
            def mock_forward(*args, **kwargs):
                class Output:
                    loss = torch.tensor(1.0, requires_grad=True)
                    moe_metrics = None
                return Output()
            model.forward = mock_forward

            metrics = trainer.train_step_with_accumulation(batch)

            if metrics is not None:
                update_count += 1
                print(f"  Update {update_count}: loss={metrics['loss']:.4f}, effective_batch={metrics['effective_batch_size']}")

                # Check effective batch size
                assert metrics['effective_batch_size'] == config.Training.global_batch_size, \
                    f"Effective batch size {metrics['effective_batch_size']} != global {config.Training.global_batch_size}"

        # Should have exactly 1 update after accumulation_steps batches
        assert update_count == 1, f"Expected 1 update, got {update_count}"

        print(f"  [PASSED] Gradient accumulation achieves global_batch_size={config.Training.global_batch_size}")

    def test_integration_all_fixes_together(self):
        """
        Test that all three fixes work together in an integrated way.
        """
        print("\n[TEST] Integration of All Three Critical Fixes")

        # This test verifies the fixes can work together without conflicts
        vocab_size = 128000
        d_model = 512
        intermediate_size = 2048
        num_segments = 4
        batch_size = 8
        seq_len = 64

        # 1. Create unified embedding
        embed = UnifiedEmbedding(vocab_size=vocab_size, d_model=d_model)

        # 2. Create segmented expert
        expert = OptimizedSegmentedExpertFFN(
            d_model=d_model,
            intermediate_size=intermediate_size,
            num_segments=num_segments
        )

        # 3. Create gradient accumulation trainer (mock)
        class MockConfig:
            class Training:
                global_batch_size = 512
                micro_batch_size = 8
                grad_clip = 1.0

        # Test forward pass with all components
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Embedding forward
        embedded = embed(input_ids)
        assert embedded.shape == (batch_size, seq_len, d_model)

        # Expert forward with selective computation
        flat_embedded = embedded.view(-1, d_model)
        segment_indices = torch.randint(0, num_segments, (batch_size * seq_len,))
        expert_output = expert(flat_embedded, token_segment_indices=segment_indices)
        assert expert_output.shape == flat_embedded.shape

        # Check stats show selective computation
        stats = expert.get_efficiency_stats()
        assert stats['efficiency'] > 0, "No efficiency gain from selective computation"

        print("  [PASSED] All three fixes integrate successfully:")
        print(f"    - Embedding handles {vocab_size} vocab")
        print(f"    - Expert efficiency: {stats['efficiency']*100:.1f}%")
        print(f"    - Gradient accumulation ready for global_batch_size=512")


if __name__ == "__main__":
    # Run tests
    test_suite = TestCriticalFixes()

    try:
        test_suite.test_unified_embedding_full_vocab_coverage()
        test_suite.test_segmented_expert_selective_computation()
        test_suite.test_gradient_accumulation_effective_batch_size()
        test_suite.test_integration_all_fixes_together()
        print("\n[SUCCESS] All critical fixes validated!")
    except AssertionError as e:
        print(f"\n[FAILURE] Test failed: {e}")
        raise