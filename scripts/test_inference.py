#!/usr/bin/env python
"""
Test inference script for trained DeepSeek-V3 model.

Usage:
    python scripts/test_inference.py --checkpoint "S:\DL+Diffusion Models\LLM\DL\Train_Dataset\LLMs\150BLLM\wikimedia___wikipedia\ckpts\checkpoint_final.pt" --prompt "The atomic bombing of Hiroshima"
"""
import argparse
import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.deepseek_v3_model import DeepSeekV3Model
from src.config.model_config import DeepSeekV3Config, MLAConfig, MoEConfig


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    print(f"Checkpoint info:")
    print(f"  - Global step: {checkpoint['global_step']}")
    print(f"  - Model: {config['model']['num_layers']} layers, {config['model']['vocab_size']} vocab")

    # Create model config
    model_cfg = config["model"]
    mla_cfg = model_cfg["mla"]
    moe_cfg = model_cfg["moe"]

    mla_config = MLAConfig(
        d_model=mla_cfg["d_model"],
        d_latent=mla_cfg["d_latent"],
        num_heads=mla_cfg["num_heads"],
        num_kv_heads=mla_cfg["num_kv_heads"],
        use_fp8_kv=mla_cfg["use_fp8_kv"],
        max_context_length=mla_cfg["max_context_length"],
        use_flash_mla=mla_cfg["use_flash_mla"] and device == "cuda",
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
        router_aux_loss_weight=moe_cfg["router_aux_loss_weight"],
    )

    model_config = DeepSeekV3Config(
        mla=mla_config,
        moe=moe_config,
        num_layers=model_cfg["num_layers"],
        vocab_size=model_cfg["vocab_size"],
        norm_type=model_cfg["norm_type"],
        norm_eps=model_cfg["norm_eps"],
        tie_word_embeddings=model_cfg["tie_word_embeddings"],
        init_method_std=model_cfg["init_method_std"],
    )

    # Create and load model
    print("Creating model...")
    model = DeepSeekV3Model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.1f}M")

    return model, config


def generate_text(
    model: DeepSeekV3Model,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda",
):
    """Generate text from prompt."""
    print(f"\n{'='*70}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*70}\n")

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Clamp token IDs to model vocab size
    vocab_size = model.config.vocab_size
    input_ids = torch.clamp(input_ids, max=vocab_size - 1)

    print(f"Input tokens: {input_ids.shape[1]}")

    # Generate
    generated_tokens = input_ids[0].tolist()

    with torch.no_grad():
        for i in range(max_length):
            # Forward pass
            outputs = model(input_ids=input_ids)

            # Get logits for next token
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            next_token_logits = logits[0, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Print progress every 10 tokens
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1} tokens...")

    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"\n{'='*70}")
    print(f"GENERATED TEXT ({len(generated_tokens)} tokens):")
    print(f"{'='*70}")
    print(generated_text)
    print(f"{'='*70}\n")

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Test inference on trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The atomic bombing of Hiroshima",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k filtering",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) filtering",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, args.device)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate text
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )


if __name__ == "__main__":
    main()
