"""
Proper PPO (Proximal Policy Optimization) implementation for RLHF.

Replaces placeholder code with functional PPO training following
InstructGPT/DeepSeek methodologies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List, Tuple
import logging
import numpy as np
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # Model paths
    policy_model_path: str = None  # SFT model to start from
    reward_model_path: str = None  # Trained reward model
    reference_model_path: str = None  # Reference for KL penalty

    # PPO hyperparameters
    learning_rate: float = 1.41e-5
    batch_size: int = 32
    mini_batch_size: int = 4
    ppo_epochs: int = 4  # Number of PPO epochs per batch
    gradient_accumulation_steps: int = 8

    # PPO specific
    clip_range: float = 0.2  # PPO clipping parameter
    value_clip_range: float = 0.2
    gamma: float = 1.0  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    target_kl: float = 0.01  # KL divergence threshold
    kl_penalty: float = 0.2  # KL penalty coefficient

    # Generation parameters
    max_prompt_length: int = 512
    max_generation_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

    # Reward normalization
    reward_scale: float = 1.0
    reward_baseline_alpha: float = 0.99  # EMA for reward baseline
    whiten_rewards: bool = True

    # Training
    num_train_steps: int = 10000
    save_interval: int = 100
    eval_interval: int = 50

    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True


class RewardModel(nn.Module):
    """
    Reward model for scoring completions.

    Based on the SFT model with a scalar head.
    """

    def __init__(self, base_model: nn.Module, hidden_size: int = 7168):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

        # Initialize reward head
        nn.init.normal_(self.reward_head.weight, std=0.01)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reward for input sequences.

        Returns:
            Rewards of shape [batch_size]
        """
        # Get model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last token's hidden state
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]

        # Get last non-padding token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_idx, sequence_lengths]

        # Compute reward
        rewards = self.reward_head(last_hidden).squeeze(-1)

        return rewards


class PPOTrainer:
    """
    Functional PPO trainer for RLHF.

    Implements the complete PPO algorithm with:
    - Advantage estimation using GAE
    - PPO clipping objective
    - Value function training
    - KL penalty from reference model
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: RewardModel,
        config: PPOConfig,
        tokenizer: Any,
        device: str = "cuda",
    ):
        """
        Initialize PPO trainer.

        Args:
            policy_model: Model to train (actor)
            ref_model: Reference model for KL penalty
            reward_model: Reward model for scoring
            config: PPO configuration
            tokenizer: Tokenizer for text processing
            device: Device to use
        """
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        # Freeze reference and reward models
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # Value head for advantage estimation
        self.value_head = nn.Linear(
            policy_model.config.hidden_size, 1, bias=False
        ).to(device)
        nn.init.normal_(self.value_head.weight, std=0.01)

        # Optimizer for policy and value
        self.optimizer = torch.optim.AdamW(
            list(self.policy_model.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Reward baseline for normalization
        self.reward_baseline = 0.0

        # Metrics tracking
        self.global_step = 0
        self.episode_rewards = deque(maxlen=100)

    def generate_completions(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate completions for prompts using the policy model.

        Args:
            prompts: List of prompt strings

        Returns:
            input_ids: Full sequences (prompt + completion)
            attention_mask: Attention mask
            action_mask: Mask for generated tokens (1 for generated, 0 for prompt)
        """
        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
            return_tensors="pt",
        ).to(self.device)

        prompt_input_ids = prompt_encodings["input_ids"]
        prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1)

        # Generate completions
        self.policy_model.eval()
        with torch.no_grad():
            generated = self.policy_model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_encodings["attention_mask"],
                max_new_tokens=self.config.max_generation_length,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Create action mask (1 for generated tokens, 0 for prompt)
        action_mask = torch.zeros_like(generated)
        for i, prompt_len in enumerate(prompt_lengths):
            action_mask[i, prompt_len:] = 1

        attention_mask = (generated != self.tokenizer.pad_token_id).long()

        return generated, attention_mask, action_mask

    def compute_rewards_and_advantages(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute rewards, values, and advantages.

        Args:
            input_ids: Generated sequences
            attention_mask: Attention mask
            action_mask: Mask for generated tokens

        Returns:
            rewards: Reward for each sequence
            advantages: Advantage estimates
            returns: Return estimates
        """
        batch_size = input_ids.size(0)

        # Get rewards from reward model
        with torch.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)

        # Get value estimates
        self.policy_model.eval()
        with torch.no_grad():
            outputs = self.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]

            # Value for each position
            values = self.value_head(hidden_states).squeeze(-1)  # [batch, seq]

        # Compute KL penalty from reference model
        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            policy_logits = outputs.logits

            # Compute KL divergence
            kl_div = self._compute_kl_div(
                policy_logits, ref_logits, action_mask
            )

        # Apply KL penalty to rewards
        rewards = rewards - self.config.kl_penalty * kl_div

        # Update reward baseline (EMA)
        self.reward_baseline = (
            self.config.reward_baseline_alpha * self.reward_baseline +
            (1 - self.config.reward_baseline_alpha) * rewards.mean().item()
        )

        # Normalize rewards
        rewards = (rewards - self.reward_baseline) * self.config.reward_scale

        # Compute advantages using GAE
        advantages = self._compute_gae(
            rewards.unsqueeze(1).expand(-1, input_ids.size(1)),
            values,
            action_mask,
        )

        # Compute returns
        returns = advantages + values

        return rewards, advantages, returns

    def _compute_kl_div(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference.

        Args:
            policy_logits: Policy model logits
            ref_logits: Reference model logits
            action_mask: Mask for generated tokens

        Returns:
            KL divergence per sequence
        """
        # Compute log probs
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # Get probs for actual tokens
        tokens = torch.cat([
            torch.zeros_like(policy_logits[:, :1, 0]).long(),
            policy_logits.argmax(dim=-1)[:, :-1]
        ], dim=1)

        policy_token_log_probs = torch.gather(
            policy_log_probs, 2, tokens.unsqueeze(-1)
        ).squeeze(-1)
        ref_token_log_probs = torch.gather(
            ref_log_probs, 2, tokens.unsqueeze(-1)
        ).squeeze(-1)

        # KL divergence
        kl = (torch.exp(policy_token_log_probs) *
              (policy_token_log_probs - ref_token_log_probs))

        # Mask and sum
        kl = (kl * action_mask).sum(dim=1)

        return kl

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Rewards at each timestep [batch, seq]
            values: Value estimates [batch, seq]
            masks: Action masks [batch, seq]

        Returns:
            Advantages [batch, seq]
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)

        # Compute advantages backwards
        gae = 0
        for t in reversed(range(seq_len - 1)):
            # Temporal difference
            if t == seq_len - 2:
                next_value = 0
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.config.gamma * next_value - values[:, t]

            # GAE
            gae = delta + self.config.gamma * self.config.gae_lambda * gae * masks[:, t + 1]
            advantages[:, t] = gae

        # Normalize advantages
        if self.config.whiten_rewards:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def ppo_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single PPO optimization step.

        Args:
            input_ids: Input sequences
            attention_mask: Attention mask
            action_mask: Action mask
            old_log_probs: Log probs from generation
            advantages: Advantage estimates
            returns: Return estimates

        Returns:
            Loss components
        """
        self.policy_model.train()

        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            outputs = self.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]

            # Compute log probs
            log_probs = F.log_softmax(logits, dim=-1)
            tokens = torch.cat([
                torch.zeros_like(logits[:, :1, 0]).long(),
                logits.argmax(dim=-1)[:, :-1]
            ], dim=1)
            action_log_probs = torch.gather(
                log_probs, 2, tokens.unsqueeze(-1)
            ).squeeze(-1)

            # Value predictions
            values = self.value_head(hidden_states).squeeze(-1)

            # PPO loss
            ratio = torch.exp(action_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
            ) * advantages

            policy_loss = -torch.min(surr1, surr2)
            policy_loss = (policy_loss * action_mask).sum() / action_mask.sum()

            # Value loss
            value_loss = F.mse_loss(values, returns, reduction="none")
            value_loss = (value_loss * action_mask).sum() / action_mask.sum()

            # Total loss
            loss = policy_loss + 0.5 * value_loss

        # Backward pass
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_model.parameters()) + list(self.value_head.parameters()),
                max_norm=0.5
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_model.parameters()) + list(self.value_head.parameters()),
                max_norm=0.5
            )
            self.optimizer.step()

        self.optimizer.zero_grad()

        # Compute KL for monitoring
        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            kl_div = self._compute_kl_div(logits, ref_logits, action_mask).mean()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "kl_divergence": kl_div.item(),
            "clip_fraction": ((ratio - 1).abs() > self.config.clip_range).float().mean().item(),
        }

    def train(
        self,
        prompts_dataset: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        Run PPO training.

        Args:
            prompts_dataset: DataLoader providing prompts

        Returns:
            Training history
        """
        history = {
            "rewards": [],
            "policy_loss": [],
            "value_loss": [],
            "kl_divergence": [],
        }

        for step, batch in enumerate(prompts_dataset):
            if step >= self.config.num_train_steps:
                break

            prompts = batch["prompts"]

            # Generate completions
            input_ids, attention_mask, action_mask = self.generate_completions(prompts)

            # Compute rewards and advantages
            rewards, advantages, returns = self.compute_rewards_and_advantages(
                input_ids, attention_mask, action_mask
            )

            # Store old log probs for PPO
            self.policy_model.eval()
            with torch.no_grad():
                old_logits = self.policy_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                old_log_probs = F.log_softmax(old_logits, dim=-1)
                tokens = torch.cat([
                    torch.zeros_like(old_logits[:, :1, 0]).long(),
                    old_logits.argmax(dim=-1)[:, :-1]
                ], dim=1)
                old_action_log_probs = torch.gather(
                    old_log_probs, 2, tokens.unsqueeze(-1)
                ).squeeze(-1)

            # PPO epochs
            for ppo_epoch in range(self.config.ppo_epochs):
                # Create mini-batches
                indices = torch.randperm(input_ids.size(0))
                for start_idx in range(0, input_ids.size(0), self.config.mini_batch_size):
                    end_idx = min(start_idx + self.config.mini_batch_size, input_ids.size(0))
                    mb_indices = indices[start_idx:end_idx]

                    metrics = self.ppo_step(
                        input_ids[mb_indices],
                        attention_mask[mb_indices],
                        action_mask[mb_indices],
                        old_action_log_probs[mb_indices],
                        advantages[mb_indices],
                        returns[mb_indices],
                    )

                # Early stopping if KL too large
                if metrics["kl_divergence"] > self.config.target_kl:
                    logger.info(f"Early stopping at epoch {ppo_epoch}, KL={metrics['kl_divergence']:.4f}")
                    break

            # Update history
            history["rewards"].append(rewards.mean().item())
            history["policy_loss"].append(metrics["policy_loss"])
            history["value_loss"].append(metrics["value_loss"])
            history["kl_divergence"].append(metrics["kl_divergence"])

            # Log progress
            if step % 10 == 0:
                logger.info(
                    f"Step {step}/{self.config.num_train_steps} | "
                    f"Reward: {rewards.mean():.3f} | "
                    f"KL: {metrics['kl_divergence']:.4f} | "
                    f"Loss: {metrics['loss']:.4f}"
                )

            # Save checkpoint
            if step % self.config.save_interval == 0:
                self.save_checkpoint(f"ppo_checkpoint_step_{step}.pt")

            self.global_step = step

        return history

    def save_checkpoint(self, path: str):
        """Save PPO checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "policy_model_state": self.policy_model.state_dict(),
            "value_head_state": self.value_head.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "reward_baseline": self.reward_baseline,
        }

        if self.scaler is not None:
            checkpoint["scaler_state"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved PPO checkpoint to {path}")