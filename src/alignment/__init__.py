"""
Alignment module for SFT and preference optimization.

This module implements the alignment phase of DeepSeek-V3 training:
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO) with stable-baselines3
"""

from .sft_trainer import SFTTrainer, SFTConfig
from .preference_optimization import (
    DPOTrainer,
    DPOConfig,
    PPOTrainerSB3,
    PPOTrainerCustom,
    PPOConfig,
)

__all__ = [
    "SFTTrainer",
    "SFTConfig",
    "DPOTrainer",
    "DPOConfig",
    "PPOTrainerSB3",
    "PPOTrainerCustom",
    "PPOConfig",
]