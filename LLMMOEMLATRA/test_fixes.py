"""Quick test script to verify all fixes are working."""
import torch
import sys
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model.mla import RMSNorm
from src.model.mtp import MTPHead
from src.model.deepseek_v3_model import DeepSeekV3Model
from src.config.model_config import DeepSeekV3Config, MLAConfig, MoEConfig, TrainingConfig, ParallelConfig

def test_rmsnorm():
    """Test RMSNorm implementation."""
    print("Testing RMSNorm...")
    dimension = 128
    norm = RMSNorm(dimension)

    # Create test input
    x = torch.randn(2, 8, dimension)

    # Compute expected output manually (RMS normalization)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    expected_output = x / torch.sqrt(variance + norm.eps)
    expected_output = expected_output * norm.weight

    # Get actual output
    actual_output = norm(x)

    # Check outputs match
    assert torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-6)
    print("[PASS] RMSNorm correctly computes x / sqrt(mean(x^2) + eps)")

def test_moe_metrics():
    """Test MoE metrics attribute naming."""
    print("\nTesting MoE metrics...")

    # Create small config
    mla_config = MLAConfig(d_model=256, num_heads=4, d_latent=64)
    moe_config = MoEConfig(num_experts=4, expert_dim=512, num_experts_per_token=2, router_aux_loss_weight=0.01)
    training_config = TrainingConfig(
        global_batch_size=2, micro_batch_size=2, seq_length=128,
        learning_rate=1e-4, train_steps=10, log_interval=1,
        eval_interval=5, save_interval=10, lr_warmup_steps=2
    )
    parallel_config = ParallelConfig()

    config = DeepSeekV3Config(
        mla=mla_config,
        moe=moe_config,
        training=training_config,
        parallel=parallel_config,
        num_layers=2,
        vocab_size=1000
    )

    model = DeepSeekV3Model(config)

    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 16))

    # Forward pass
    output = model(input_ids)

    # Check that moe_metrics attribute exists (not expert_metrics)
    assert hasattr(output, 'moe_metrics'), "Output should have 'moe_metrics' attribute"
    assert not hasattr(output, 'expert_metrics'), "Output should not have 'expert_metrics' attribute"
    print("[PASS] MoE metrics correctly named as 'moe_metrics'")

def test_mtp_sequential():
    """Test MTP sequential conditioning."""
    print("\nTesting MTP sequential conditioning...")

    d_model = 256
    vocab_size = 1000
    mtp_tokens = 2

    # Create embedding layer
    embedding = torch.nn.Embedding(vocab_size, d_model)

    # Create MTP with embedding
    mtp = MTPHead(d_model, vocab_size, mtp_tokens, embedding_layer=embedding)

    # Check embedding is stored
    assert mtp.embedding is not None
    print("[PASS] MTP head properly receives embedding layer")

    # Test forward pass with sequential conditioning
    hidden = torch.randn(2, 8, d_model)
    mtp_labels = torch.randint(0, vocab_size, (2, 8, mtp_tokens))

    mtp.train()
    logits, loss = mtp(hidden, mtp_labels=mtp_labels)

    assert logits.shape == (2, 8, mtp_tokens, vocab_size)
    assert loss is not None
    print("[PASS] MTP sequential conditioning works with teacher forcing")

    # Test inference mode
    mtp.eval()
    with torch.no_grad():
        logits_inf, _ = mtp(hidden)

    assert logits_inf.shape == (2, 8, mtp_tokens, vocab_size)
    print("[PASS] MTP sequential conditioning works in inference mode")

def test_full_model_integration():
    """Test that all fixes work together in the full model."""
    print("\nTesting full model integration...")

    # Create config
    mla_config = MLAConfig(d_model=256, num_heads=4, d_latent=64)
    moe_config = MoEConfig(num_experts=4, expert_dim=512, num_experts_per_token=2, router_aux_loss_weight=0.01)
    training_config = TrainingConfig(
        global_batch_size=2, micro_batch_size=2, seq_length=128,
        learning_rate=1e-4, train_steps=10, log_interval=1,
        eval_interval=5, save_interval=10, lr_warmup_steps=2,
        mtp_tokens=2
    )
    parallel_config = ParallelConfig()

    config = DeepSeekV3Config(
        mla=mla_config,
        moe=moe_config,
        training=training_config,
        parallel=parallel_config,
        num_layers=2,
        vocab_size=1000
    )

    model = DeepSeekV3Model(config)

    # Check that MTP head has embedding layer
    assert model.mtp_head.embedding is not None
    print("[PASS] MTP head receives embedding from main model")

    # Create inputs
    input_ids = torch.randint(0, 1000, (2, 16))
    labels = torch.randint(0, 1000, (2, 16))
    mtp_labels = torch.randint(0, 1000, (2, 16, 2))

    # Forward pass
    output = model(input_ids, labels=labels, mtp_labels=mtp_labels)

    # Check all outputs
    assert hasattr(output, 'logits')
    assert hasattr(output, 'mtp_logits')
    assert hasattr(output, 'loss')
    assert hasattr(output, 'moe_metrics')
    assert output.loss is not None
    assert output.loss.item() > 0

    print("[PASS] Full model integration successful with all fixes")

if __name__ == "__main__":
    print("="*60)
    print("Testing DeepSeek-V3 Implementation Fixes")
    print("="*60)

    try:
        test_rmsnorm()
        test_moe_metrics()
        test_mtp_sequential()
        test_full_model_integration()

        print("\n" + "="*60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("="*60)
        print("\nSummary of fixes:")
        print("1. [PASS] RMSNorm now uses correct formula: x / sqrt(mean(x²) + eps)")
        print("2. [PASS] MoE metrics renamed from 'expert_metrics' to 'moe_metrics'")
        print("3. [PASS] MTP implements sequential conditioning with embeddings")
        print("4. [PASS] Full model integration works with all fixes")

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)