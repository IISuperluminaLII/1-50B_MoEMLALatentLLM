#!/usr/bin/env python
"""
Generate model configs from 1B to 50B in 5B increments.

Scales:
- Model dimensions (d_model, num_heads, num_layers)
- Number of experts
- Batch sizes
- GPU requirements
"""
import json
from pathlib import Path


def get_model_config(size_b):
    """
    Generate model configuration based on target parameter size.

    Scaling follows: params ≈ 12 * d_model² * num_layers + expert_params
    """
    configs = {
        1: {"layers": 16, "d_model": 1536, "heads": 24, "experts": 16, "exp_per_tok": 4, "gpus": 4},
        5: {"layers": 24, "d_model": 2560, "heads": 40, "experts": 32, "exp_per_tok": 6, "gpus": 8},
        10: {"layers": 32, "d_model": 3072, "heads": 48, "experts": 48, "exp_per_tok": 6, "gpus": 16},
        15: {"layers": 36, "d_model": 3584, "heads": 56, "experts": 64, "exp_per_tok": 8, "gpus": 32},
        20: {"layers": 40, "d_model": 4096, "heads": 64, "experts": 80, "exp_per_tok": 8, "gpus": 48},
        25: {"layers": 44, "d_model": 4608, "heads": 72, "experts": 96, "exp_per_tok": 8, "gpus": 64},
        30: {"layers": 48, "d_model": 5120, "heads": 80, "experts": 112, "exp_per_tok": 8, "gpus": 80},
        35: {"layers": 52, "d_model": 5632, "heads": 88, "experts": 128, "exp_per_tok": 8, "gpus": 96},
        40: {"layers": 56, "d_model": 6144, "heads": 96, "experts": 144, "exp_per_tok": 8, "gpus": 112},
        45: {"layers": 60, "d_model": 6656, "heads": 104, "experts": 160, "exp_per_tok": 8, "gpus": 128},
        50: {"layers": 64, "d_model": 7168, "heads": 112, "experts": 176, "exp_per_tok": 8, "gpus": 144}
    }

    return configs[size_b]


def generate_config(size_b):
    """Generate complete training configuration."""
    model = get_model_config(size_b)

    # Calculate derived values
    d_latent = model["d_model"] // 4
    expert_dim = int(model["d_model"] * 2.75)
    shared_experts = max(1, model["experts"] // 16)

    # Batch size scales with model size
    global_batch_size = min(4096, 256 * (size_b // 5 + 1))
    micro_batch_size = 1 if size_b >= 15 else 2

    # Parallelism scaling
    if size_b <= 5:
        tp, pp, ep = 1, 1, 1
    elif size_b <= 15:
        tp, pp, ep = 2, 1, 2
    else:
        tp, pp, ep = 4, 2, 2

    nodes = model["gpus"] // 8
    gpus_per_node = min(8, model["gpus"])

    config = {
        "experiment_name": f"deepseek_v3_{size_b}b",
        "output_dir": f"./outputs/{size_b}b",
        "seed": 42,

        "model": {
            "num_layers": model["layers"],
            "vocab_size": 128000,
            "norm_type": "rmsnorm",
            "norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "init_method_std": 0.006,
            "dense_layer_interval": 4 if size_b <= 10 else 3,

            "mla": {
                "d_model": model["d_model"],
                "d_latent": d_latent,
                "num_heads": model["heads"],
                "num_kv_heads": model["heads"],
                "use_fp8_kv": size_b >= 15,
                "max_context_length": 32768 if size_b >= 20 else 16384 if size_b >= 10 else 8192,
                "use_flash_mla": True,
                "flash_mla_backend": "auto",
                "fallback_to_dense": True,
                "use_rope": True,
                "rope_theta": 10000.0,
                "sliding_window": None,
                "attn_dropout": 0.0
            },

            "moe": {
                "num_experts": model["experts"],
                "num_experts_per_token": model["exp_per_tok"],
                "expert_intermediate_size": expert_dim,
                "expert_dim": expert_dim,
                "dropout": 0.0,
                "num_shared_experts": shared_experts,
                "shared_intermediate_size": expert_dim,
                "router_aux_loss_weight": 0.001 if size_b < 20 else 0.0,
                "router_temperature": 1.0,
                "router_noise_std": 0.0,
                "capacity_factor": 1.0,
                "use_aux_loss_free": size_b >= 20,
                "balance_loss_type": "entropy",
                "min_expert_capacity": 4,
                "use_deep_ep": size_b >= 15,
                "deep_ep_fp8": size_b >= 20,
                "deep_ep_async": size_b >= 25
            }
        },

        "training": {
            "global_batch_size": global_batch_size,
            "micro_batch_size": micro_batch_size,
            "seq_length": 4096,
            "tokens_per_parameter_ratio": 20.0,
            "total_training_tokens": None,
            "learning_rate": max(1.5e-4, 3e-4 / (size_b / 5)),
            "min_learning_rate": max(1.5e-5, 3e-5 / (size_b / 5)),
            "lr_warmup_steps": 2000,
            "lr_decay_style": "cosine",
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "use_fp16": False,
            "use_bf16": True,
            "use_fp8": size_b >= 20,
            "use_mtp": True,
            "num_predict_tokens": 2,
            "mtp_tokens": 2,
            "train_steps": max(50000, 500000 // (size_b // 5 + 1)),
            "eval_interval": 1000,
            "save_interval": 5000,
            "log_interval": 10,
            "optimizer": "adamw",
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1e-8
        },

        "data": {
            "dataset_name": "allenai/dolma",
            "version": "v1_7",
            "_comment_version": "Dolma v1.7 is the latest version (~3T tokens, pre-mixed from 6 source categories)",
            "cache_dir": "./data/cache",
            "chinchilla_tokens": int(size_b * 1e9 * 20),
            "_comment_chinchilla": f"Chinchilla optimal: 20 tokens per parameter for {size_b}B params = {size_b * 20}B tokens (Hoffmann et al., 2022, arXiv:2203.15556)",
            "_comment_composition": "Dolma is pre-mixed from: Common Crawl, GitHub, Reddit, Semantic Scholar, Project Gutenberg, Wikipedia/Wikibooks",
            "preprocessing": {
                "num_workers": min(16, model["gpus"] // 4),
                "shuffle": True,
                "shuffle_seed": 42
            },
            "pipeline": {
                "_comment_pipeline": "SOTA data processing pipeline for LLM training quality",
                "enabled": True,
                "stages": ["deduplication", "quality_filtering", "heuristic_filtering", "ranking_filtering", "domain_mixing"]
            },
            "deduplication": {
                "_comment_dedup": "MinHash LSH deduplication (Lee et al., 2022, arXiv:2107.06499 - Deduplicating Training Data)",
                "enabled": True,
                "method": "minhash_lsh",
                "ngram_size": 13,
                "num_perm": 128,
                "threshold": 0.8,
                "citation": "Lee et al. (2022). Deduplicating Training Data Makes Language Models Better. arXiv:2107.06499"
            },
            "quality_filter": {
                "_comment_quality": "Quality filtering based on CCNet (Wenzek et al., 2020) and RefinedWeb (Penedo et al., 2023)",
                "enabled": True,
                "methods": [
                    {
                        "name": "ccnet_quality",
                        "min_score": 0.5,
                        "citation": "Wenzek et al. (2020). CCNet: Extracting High Quality Monolingual Datasets. arXiv:1911.00359"
                    },
                    {
                        "name": "perplexity_filter",
                        "model": "kenlm",
                        "max_perplexity": 1500,
                        "citation": "Penedo et al. (2023). The RefinedWeb Dataset for Falcon LLM. arXiv:2306.01116"
                    }
                ]
            },
            "heuristic_filters": {
                "_comment_heuristics": "Rule-based filtering from Gopher (Rae et al., 2021) and C4 (Raffel et al., 2020)",
                "enabled": True,
                "rules": [
                    {
                        "name": "repetition_filter",
                        "max_char_repetition": 0.2,
                        "max_word_repetition": 0.2,
                        "max_line_repetition": 0.3,
                        "citation": "Rae et al. (2021). Scaling Language Models: Methods, Analysis & Insights. arXiv:2112.11446"
                    },
                    {
                        "name": "length_filter",
                        "min_doc_length": 50,
                        "max_doc_length": 100000,
                        "min_avg_word_length": 3,
                        "max_avg_word_length": 10
                    },
                    {
                        "name": "language_filter",
                        "target_language": "en",
                        "min_language_score": 0.65,
                        "citation": "Raffel et al. (2020). Exploring the Limits of Transfer Learning with T5. JMLR"
                    },
                    {
                        "name": "toxicity_filter",
                        "enabled": False,
                        "_comment": "Disabled for pre-cleaned Dolma data"
                    }
                ]
            },
            "ranking_filter": {
                "_comment_ranking": "Educational quality ranking from FineWeb-Edu (HuggingFace, 2024) using LLM-as-judge",
                "enabled": True,
                "method": "fineweb_edu",
                "min_educational_score": 2,
                "max_educational_score": 5,
                "score_distribution": "favor_high_quality",
                "citation": "Lozhkov et al. (2024). FineWeb-Edu: LLM-based Educational Quality Filtering. HuggingFace"
            },
            "domain_mixer": {
                "_comment_mixer": "DoReMi adaptive domain weighting (Xie et al., 2023) for optimal domain mixture",
                "enabled": False,
                "_comment_dolma": "Dolma is already pre-mixed optimally. Domain mixing would require raw source files, not the HuggingFace API",
                "method": "doremi",
                "reference_model_size": "280m",
                "num_epochs": 1,
                "reweight_temperature": 0.5,
                "citation": "Xie et al. (2023). DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. arXiv:2305.10429"
            }
        },

        "distributed": {
            "backend": "deepspeed",
            "launcher": "deepspeed",
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "expert_parallel_size": ep,
            "data_parallel_size": -1,
            "zero_stage": 2 if size_b >= 20 else 1,
            "zero_offload": size_b >= 30,
            "overlap_grad_reduce": True,
            "overlap_param_gather": True,
            "deepspeed": {
                "enabled": True,
                "config_file": "configs/deepspeed_config.json"
            },
            "slurm": {
                "enabled": False,
                "partition": "gpu_a100" if size_b >= 20 else "gpu",
                "nodes": nodes,
                "ntasks_per_node": gpus_per_node,
                "gpus_per_node": gpus_per_node,
                "cpus_per_task": 16,
                "time": "72:00:00" if size_b >= 30 else "48:00:00",
                "mem": f"{64 * gpus_per_node}G",
                "job_name": f"deepseek_{size_b}b",
                "output": "logs/slurm-%j.out",
                "error": "logs/slurm-%j.err"
            }
        },

        "checkpointing": {
            "save_interval": 5000,
            "save_total_limit": 3 if size_b >= 20 else 5,
            "resume_from_checkpoint": None,
            "checkpoint_format": "deepspeed",
            "save_optimizer_states": True
        },

        "logging": {
            "log_interval": 10,
            "wandb": {
                "enabled": False,
                "project": f"deepseek-v3-{size_b}b",
                "entity": None,
                "name": None,
                "tags": [f"{size_b}b"]
            },
            "tensorboard": {
                "enabled": True,
                "log_dir": "./logs/tensorboard"
            }
        },

        "validation": {
            "enabled": True,
            "eval_interval": 1000,
            "eval_samples": min(5000, 1000 * (size_b // 5 + 1)),
            "metrics": ["loss", "perplexity"]
        }
    }

    return config


def main():
    """Generate all configs."""
    output_dir = Path("configs")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("Generating DeepSeek-V3 Model Configurations")
    print("="*80)

    sizes = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for size in sizes:
        config = generate_config(size)
        filename = output_dir / f"deepseek_v3_{size}b.json"

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

        model = get_model_config(size)
        print(f"\n[OK] {size:2d}B: {model['layers']} layers, {model['d_model']} hidden, "
              f"{model['experts']} experts, {model['gpus']} GPUs")
        print(f"      Saved to: {filename}")

    print("\n" + "="*80)
    print(f"Generated {len(sizes)} configurations")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
