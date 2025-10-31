"""
Test suite for official DeepSeek-V3 architecture integration.

Verifies that the official HuggingFace-compatible implementation works correctly
and matches expected specifications.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestOfficialArchitecture:
    """Test official DeepSeek-V3 architecture components."""

    def test_official_model_imports(self):
        """Test that official model components can be imported."""
        try:
            from src.model.deepseek_v3_official import (
                DeepseekV3Config,
                DeepseekV3Model,
                DeepseekV3ForCausalLM,
            )
            from src.mla.deepseek_v3_attention import DeepseekV3Attention
            from src.moe.moe_gate import MoEGate, DeepseekV3MLP, DeepseekV3MoE

            assert DeepseekV3Config is not None
            assert DeepseekV3Model is not None
            assert DeepseekV3ForCausalLM is not None
            assert DeepseekV3Attention is not None
            assert MoEGate is not None
            print("[PASSED] All official components imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import official components: {e}")

    def test_official_config_creation(self):
        """Test creation of official configuration."""
        from src.model.deepseek_v3_official import DeepseekV3Config

        config = DeepseekV3Config()

        # Verify critical parameters match HuggingFace release
        assert config.vocab_size == 129280, f"Expected vocab_size=129280, got {config.vocab_size}"
        assert config.hidden_size == 7168, f"Expected hidden_size=7168, got {config.hidden_size}"
        assert config.num_hidden_layers == 61, f"Expected num_hidden_layers=61, got {config.num_hidden_layers}"
        assert config.num_attention_heads == 128, f"Expected num_attention_heads=128, got {config.num_attention_heads}"

        # MLA parameters
        assert config.q_lora_rank == 1536, f"Expected q_lora_rank=1536, got {config.q_lora_rank}"
        assert config.kv_lora_rank == 512, f"Expected kv_lora_rank=512, got {config.kv_lora_rank}"
        assert config.qk_nope_head_dim == 128, f"Expected qk_nope_head_dim=128, got {config.qk_nope_head_dim}"
        assert config.qk_rope_head_dim == 64, f"Expected qk_rope_head_dim=64, got {config.qk_rope_head_dim}"

        # MoE parameters
        assert config.num_experts == 256, f"Expected num_experts=256, got {config.num_experts}"
        assert config.n_shared_experts == 1, f"Expected n_shared_experts=1, got {config.n_shared_experts}"
        assert config.n_group == 8, f"Expected n_group=8, got {config.n_group}"
        assert config.topk_group == 1, f"Expected topk_group=1, got {config.topk_group}"
        assert config.scoring_func == "sigmoid", f"Expected scoring_func='sigmoid', got {config.scoring_func}"

        # Layer configuration
        assert config.first_k_dense_replace == 3, f"Expected first_k_dense_replace=3, got {config.first_k_dense_replace}"
        assert config.moe_layer_freq == 1, f"Expected moe_layer_freq=1, got {config.moe_layer_freq}"

        print("[PASSED] Official config parameters match HuggingFace release")

    def test_model_creation(self):
        """Test creation of official model."""
        from src.model.deepseek_v3_official import DeepseekV3Config, DeepseekV3ForCausalLM

        # Create small test config
        config = DeepseekV3Config(
            num_hidden_layers=4,  # Small for testing
            hidden_size=512,
            num_experts=4,
            num_attention_heads=8,
        )

        model = DeepseekV3ForCausalLM(config)
        assert model is not None

        # Check model structure
        assert hasattr(model, 'model'), "Model should have base model"
        assert hasattr(model, 'lm_head'), "Model should have language modeling head"
        assert len(model.model.layers) == 4, f"Expected 4 layers, got {len(model.model.layers)}"

        # Check embedding
        assert model.model.embed_tokens.num_embeddings == 129280, "Should have official vocab size"

        print("[PASSED] Official model created successfully")

    def test_layer_configuration(self):
        """Test that layers are configured correctly (dense vs MoE)."""
        from src.model.deepseek_v3_official import DeepseekV3Config, DeepseekV3Model

        config = DeepseekV3Config(
            num_hidden_layers=10,
            hidden_size=512,
            num_experts=4,
            num_attention_heads=8,
            first_k_dense_replace=3,
            moe_layer_freq=1,
        )

        model = DeepseekV3Model(config)

        # Check first 3 layers are dense
        for i in range(3):
            layer = model.layers[i]
            assert not layer.is_moe, f"Layer {i} should be dense but is MoE"
            assert layer.mlp is not None, f"Layer {i} should have dense MLP"

        # Check layers 3+ are MoE
        for i in range(3, 10):
            layer = model.layers[i]
            assert layer.is_moe, f"Layer {i} should be MoE but is dense"
            assert hasattr(layer.mlp, 'gate'), f"Layer {i} should have MoE gate"
            assert hasattr(layer.mlp, 'experts'), f"Layer {i} should have experts"

        print("[PASSED] Layer configuration correct (first 3 dense, rest MoE)")

    def test_moe_gate_routing(self):
        """Test MoEGate with sigmoid scoring and grouped top-k."""
        from src.moe.moe_gate import MoEGate

        config = {
            'n_group': 2,
            'topk_group': 2,
            'scoring_func': 'sigmoid',
            'norm_topk_prob': True,
            'routed_scaling_factor': 1.0,
        }

        gate = MoEGate(config, num_experts=8, d_model=256)

        # Test forward pass
        batch_size = 4
        hidden_states = torch.randn(batch_size, 256)

        topk_idx, topk_weight, aux_loss = gate(hidden_states, training=False)

        # Verify outputs
        assert topk_idx.shape == (batch_size, 4), f"Expected shape (4, 4), got {topk_idx.shape}"  # n_group * topk_group
        assert topk_weight.shape == (batch_size, 4), f"Expected shape (4, 4), got {topk_weight.shape}"
        assert aux_loss is None, "NoAux routing should not return aux loss"

        # Verify weights are normalized
        weight_sums = topk_weight.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), "Weights should sum to 1"

        print("[PASSED] MoEGate routing works correctly")

    def test_attention_with_nope_rope(self):
        """Test DeepseekV3Attention with NOPE/ROPE splitting."""
        from src.mla.deepseek_v3_attention import DeepseekV3Attention

        config = {
            'hidden_size': 512,
            'num_attention_heads': 8,
            'q_lora_rank': 128,
            'kv_lora_rank': 64,
            'qk_nope_head_dim': 32,
            'qk_rope_head_dim': 16,
            'v_head_dim': 32,
            'max_position_embeddings': 2048,
            'rope_theta': 10000.0,
            'attention_dropout': 0.0,
        }

        attention = DeepseekV3Attention(config)

        # Test forward pass
        batch_size, seq_len = 2, 64
        hidden_states = torch.randn(batch_size, seq_len, 512)

        output = attention(hidden_states)

        assert output.hidden_states.shape == (batch_size, seq_len, 512), \
            f"Output shape mismatch: {output.hidden_states.shape}"

        print("[PASSED] DeepseekV3Attention with NOPE/ROPE works")

    def test_forward_pass(self):
        """Test complete forward pass through official model."""
        from src.model.deepseek_v3_official import DeepseekV3Config, DeepseekV3ForCausalLM

        # Create small test config
        config = DeepseekV3Config(
            num_hidden_layers=2,
            hidden_size=256,
            num_experts=2,
            num_attention_heads=4,
            q_lora_rank=64,
            kv_lora_rank=32,
        )

        model = DeepseekV3ForCausalLM(config)
        model.eval()

        # Test input
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            output = model(input_ids)

        # Verify output
        assert 'logits' in output, "Output should contain logits"
        assert output['logits'].shape == (batch_size, seq_len, 129280), \
            f"Expected logits shape (2, 32, 129280), got {output['logits'].shape}"

        # Check that output is finite
        assert torch.isfinite(output['logits']).all(), "Output contains NaN or Inf"

        print("[PASSED] Forward pass through official model works")

    def test_migration_helper(self):
        """Test migration from custom to official architecture."""
        from src.model.migrate_to_official import (
            create_official_config_from_existing,
            ModelMigrator,
        )

        # Create test config
        config_dict = {
            'model': {
                'd_model': 512,
                'num_layers': 4,
                'max_context_length': 4096,
                'rope_base': 10000.0,
                'dropout': 0.1,
                'norm_eps': 1e-6,
                'mtp_enabled': True,
                'mtp_num_experts': 1,
            },
            'moe': {
                'num_experts': 8,
                'expert_intermediate_size': 1024,
                'n_group': 2,
                'topk_group': 1,
                'routed_scaling_factor': 1.0,
            },
            'mla': {
                'num_heads': 8,
                'q_lora_rank': 128,
                'kv_lora_rank': 64,
                'qk_nope_head_dim': 32,
                'qk_rope_head_dim': 16,
                'v_head_dim': 32,
                'dropout': 0.1,
            }
        }

        # Convert to official config
        official_config = create_official_config_from_existing(config_dict)

        assert official_config.vocab_size == 129280, "Should set official vocab size"
        assert official_config.hidden_size == 512, "Should preserve hidden size"
        assert official_config.num_experts == 8, "Should preserve num experts"
        assert official_config.scoring_func == 'sigmoid', "Should use sigmoid scoring"
        assert official_config.first_k_dense_replace == 3, "Should set dense layers"

        # Test migrator
        migrator = ModelMigrator()
        migrator.load_config(config_dict)
        official_model = migrator.create_official_model(with_lm_head=True)

        assert official_model is not None, "Should create official model"

        print("[PASSED] Migration helper works correctly")

    def test_official_config_file(self):
        """Test that official config file exists and is valid."""
        config_path = Path("configs/deepseek_v3_official.json")

        if not config_path.exists():
            pytest.skip("Official config file not found")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Verify critical fields
        assert config_dict['model']['vocab_size'] == 129280
        assert config_dict['model']['first_k_dense_replace'] == 3
        assert config_dict['moe']['scoring_func'] == 'sigmoid'
        assert config_dict['mla']['q_lora_rank'] == 1536
        assert config_dict['mla']['kv_lora_rank'] == 512

        print("[PASSED] Official config file is valid")


if __name__ == "__main__":
    # Run tests
    test = TestOfficialArchitecture()

    print("\n" + "="*60)
    print("Testing Official DeepSeek-V3 Architecture Integration")
    print("="*60 + "\n")

    try:
        test.test_official_model_imports()
        test.test_official_config_creation()
        test.test_model_creation()
        test.test_layer_configuration()
        test.test_moe_gate_routing()
        test.test_attention_with_nope_rope()
        test.test_forward_pass()
        test.test_migration_helper()
        test.test_official_config_file()

        print("\n" + "="*60)
        print("[SUCCESS] All official architecture tests passed!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback
        traceback.print_exc()