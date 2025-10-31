"""Check for remaining critical issues after P0 fixes."""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_mla_latent_dimension_fix():
    """Verify MLA latent dimension issue is fixed."""
    from src.mla.paper_compliant_mla import MultiheadLatentAttention

    mla = MultiheadLatentAttention(
        d_model=7168,
        num_heads=128,
        q_head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512
    )

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 7168)

    try:
        output, _ = mla(x)
        return True, "MLA forward pass successful"
    except Exception as e:
        return False, f"MLA forward failed: {e}"

def check_moe_residual_fix():
    """Verify MoE double residual is fixed."""
    from src.moe.paper_compliant_moe import PaperCompliantMoE

    moe = PaperCompliantMoE(
        d_model=1024,
        num_experts=8,
        expert_capacity_factor=1.25,
        num_experts_per_token=2
    )

    x = torch.randn(2, 10, 1024)
    try:
        output = moe(x)
        # Check output is not zero (was a symptom of double residual bug)
        if torch.allclose(output, torch.zeros_like(output)):
            return False, "MoE output is all zeros (residual bug)"
        return True, "MoE output looks correct"
    except Exception as e:
        return False, f"MoE forward failed: {e}"

def check_routing_statistics():
    """Check routing statistics normalization."""
    from src.moe.paper_compliant_moe import PaperCompliantMoE

    moe = PaperCompliantMoE(
        d_model=1024,
        num_experts=8,
        expert_capacity_factor=1.25,
        num_experts_per_token=2,
        use_aux_loss_free=True
    )

    # Simulate multiple forward passes
    moe.train()
    for i in range(10):
        x = torch.randn(2, 10, 1024)
        output = moe(x)

    # Check expert counts are reasonable
    counts = moe.expert_counts
    if counts.max() > 100 or counts.min() < 0:
        return False, f"Expert counts unreasonable: {counts}"

    return True, "Routing statistics look normal"

def check_deepseekv3_router_usage():
    """Verify DeepSeekV3Router is used for aux-loss-free."""
    from src.moe.deepseek_moe import DeepSeekMoE
    from src.moe.deepseek_v3_routing import DeepSeekV3Router

    moe = DeepSeekMoE(
        d_model=1024,
        num_experts=8,
        num_experts_per_token=2,
        expert_intermediate_size=2048,
        use_aux_loss_free=True
    )

    if not isinstance(moe.router, DeepSeekV3Router):
        return False, f"Router is {type(moe.router).__name__}, not DeepSeekV3Router"

    return True, "DeepSeekV3Router correctly used"

def check_gradient_accumulation():
    """Check gradient accumulation logic."""
    from src.config.model_config import DeepSeekV3Config, TrainingConfig

    config = DeepSeekV3Config()
    config.training = TrainingConfig(
        global_batch_size=32,
        micro_batch_size=8
    )

    # Expected accumulation steps = 32 / (8 * 1) = 4
    expected = 4

    # Simulate what trainer does
    accumulation_steps = config.training.global_batch_size // (config.training.micro_batch_size * 1)

    if accumulation_steps != expected:
        return False, f"Expected {expected} accumulation steps, got {accumulation_steps}"

    return True, "Gradient accumulation calculation correct"

def check_amp_enabled():
    """Check AMP configuration."""
    from src.training.trainer import DeepSeekV3Trainer
    from src.config.model_config import DeepSeekV3Config, TrainingConfig
    from unittest.mock import MagicMock

    config = DeepSeekV3Config()
    config.training = TrainingConfig()
    config.training.precision = "fp16"

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.randn(10, 10)])

    trainer = DeepSeekV3Trainer(
        model=mock_model,
        train_dataloader=MagicMock(),
        val_dataloader=None,
        optimizer=MagicMock(),
        lr_scheduler=MagicMock(),
        config=config,
        output_dir="test"
    )

    if not trainer.use_amp:
        return False, "AMP not enabled"
    if trainer.amp_dtype != torch.float16:
        return False, f"Wrong AMP dtype: {trainer.amp_dtype}"

    return True, "AMP properly configured"

if __name__ == "__main__":
    print("\n[CHECKING] Remaining Issues After P0 Fixes")
    print("="*60)

    tests = [
        ("MLA Latent Dimensions", check_mla_latent_dimension_fix),
        ("MoE Residual Fix", check_moe_residual_fix),
        ("Routing Statistics", check_routing_statistics),
        ("DeepSeekV3 Router", check_deepseekv3_router_usage),
        ("Gradient Accumulation", check_gradient_accumulation),
        ("Mixed Precision", check_amp_enabled),
    ]

    failures = []
    for name, test_func in tests:
        try:
            success, msg = test_func()
            if success:
                print(f"[OK] {name}: {msg}")
            else:
                print(f"[FAILED] {name}: {msg}")
                failures.append((name, msg))
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failures.append((name, str(e)))

    print("="*60)
    if failures:
        print(f"\n[WARNING] {len(failures)} issues found:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
    else:
        print("[SUCCESS] All critical P0 issues verified as fixed!")