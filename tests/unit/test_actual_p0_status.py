"""Actual verification of P0 fixes with correct class names and parameters."""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("\n[VERIFYING] Actual P0 Fix Status")
print("="*60)

# Test 1: MLA Forward Pass
print("\n1. Testing MLA forward pass...")
try:
    from src.mla.paper_compliant_mla import LatentSpaceMLA  # Correct class name

    mla = LatentSpaceMLA(
        d_model=7168,
        num_heads=128,
        q_head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512
    )

    x = torch.randn(2, 10, 7168)
    output, _ = mla(x)

    if output.shape == x.shape:
        print("   [OK] MLA forward pass works correctly")
    else:
        print(f"   [FAILED] MLA output shape mismatch: {output.shape} vs {x.shape}")
except Exception as e:
    print(f"   [FAILED] MLA error: {e}")

# Test 2: MoE with correct config
print("\n2. Testing MoE layer...")
try:
    from src.moe.paper_compliant_moe import PaperCompliantMoE, PaperCompliantMoEConfig

    config = PaperCompliantMoEConfig(
        num_experts=8,
        num_experts_per_token=2,
        balance_loss_type="aux_free",
        use_aux_loss_free=True
    )

    moe = PaperCompliantMoE(config, d_model=1024)
    x = torch.randn(2, 10, 1024)
    output = moe(x)

    # Check that output is not all zeros
    if not torch.allclose(output, torch.zeros_like(output)):
        print("   [OK] MoE produces non-zero output")
    else:
        print("   [FAILED] MoE output is all zeros")

    # Check expert statistics
    if hasattr(moe, 'expert_counts'):
        if moe.expert_counts.max() < 100:
            print("   [OK] Expert counts look reasonable")
        else:
            print(f"   [FAILED] Expert counts suspicious: {moe.expert_counts}")
except Exception as e:
    print(f"   [FAILED] MoE error: {e}")

# Test 3: DeepSeekV3 Router for aux-loss-free
print("\n3. Testing DeepSeekV3Router usage...")
try:
    from src.moe.deepseek_moe import DeepSeekMoE
    from src.moe.deepseek_v3_routing import DeepSeekV3Router

    moe = DeepSeekMoE(
        d_model=1024,
        num_experts=8,
        num_experts_per_token=2,
        expert_intermediate_size=2048,
        use_aux_loss_free=True
    )

    if isinstance(moe.router, DeepSeekV3Router):
        print("   [OK] DeepSeekV3Router correctly used for aux-loss-free")

        # Check bias is a buffer
        if 'expert_bias' in moe.router._buffers:
            print("   [OK] Router bias is a non-trainable buffer")
        else:
            print("   [FAILED] Router bias is not a buffer")
    else:
        print(f"   [FAILED] Router is {type(moe.router).__name__}, not DeepSeekV3Router")
except Exception as e:
    print(f"   [FAILED] DeepSeekV3Router error: {e}")

# Test 4: Tokenizer requirement
print("\n4. Testing tokenizer enforcement...")
try:
    train_file = Path(__file__).parent.parent.parent / "src/training/train.py"
    content = train_file.read_text()

    checks = [
        ("deepseek-ai/DeepSeek-V3-Base" in content, "DeepSeek-V3 tokenizer required"),
        ("sys.exit(1)" in content, "Exit on wrong tokenizer"),
        ("128000" in content or "128k" in content.lower(), "Vocab size validation")
    ]

    all_passed = True
    for check, desc in checks:
        if check:
            print(f"   [OK] {desc}")
        else:
            print(f"   [FAILED] {desc}")
            all_passed = False

except Exception as e:
    print(f"   [FAILED] Tokenizer check error: {e}")
    all_passed = False

# Test 5: Gradient accumulation
print("\n5. Testing gradient accumulation...")
try:
    from src.config.model_config import DeepSeekV3Config, TrainingConfig
    from src.training.trainer import DeepSeekV3Trainer
    from unittest.mock import MagicMock

    config = DeepSeekV3Config()
    config.training = TrainingConfig(
        global_batch_size=32,
        micro_batch_size=8
    )

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

    if trainer.accumulation_steps == 4:
        print("   [OK] Gradient accumulation steps calculated correctly")
    else:
        print(f"   [FAILED] Expected 4 accumulation steps, got {trainer.accumulation_steps}")
except Exception as e:
    print(f"   [FAILED] Gradient accumulation error: {e}")

# Test 6: Mixed precision
print("\n6. Testing mixed precision...")
try:
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

    if trainer.use_amp and trainer.amp_dtype == torch.float16:
        print("   [OK] Mixed precision (FP16) enabled")
    else:
        print(f"   [FAILED] AMP not properly configured")
except Exception as e:
    print(f"   [FAILED] Mixed precision error: {e}")

# Test 7: PPO trainer
print("\n7. Testing PPO trainer implementation...")
try:
    from src.alignment.ppo_trainer import PPOTrainer, PPOConfig

    # Check for key methods
    methods = ['generate_completions', 'compute_rewards_and_advantages',
               '_compute_gae', '_compute_kl_div', 'ppo_step', 'train']

    all_present = True
    for method in methods:
        if hasattr(PPOTrainer, method):
            print(f"   [OK] PPOTrainer.{method} exists")
        else:
            print(f"   [FAILED] PPOTrainer.{method} missing")
            all_present = False

    # Check config parameters
    config = PPOConfig()
    if hasattr(config, 'clip_range') and hasattr(config, 'gae_lambda'):
        print("   [OK] PPO config has required parameters")
    else:
        print("   [FAILED] PPO config missing parameters")
except Exception as e:
    print(f"   [FAILED] PPO trainer error: {e}")

print("\n" + "="*60)
print("[SUMMARY] P0 Fix Verification Complete")
print("Check above for any [FAILED] items that need attention.")
print("="*60)