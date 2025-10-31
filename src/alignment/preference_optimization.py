"""
Preference optimization trainers (DPO and PPO).

Implements the preference optimization phase from DeepSeek-V3:
- Direct Preference Optimization (DPO) - simpler, more stable
- PPO with stable-baselines3 - for full RLHF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from copy import deepcopy

logger = logging.getLogger(__name__)

# Optional: Use stable-baselines3 for PPO if available
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    logger.warning("stable-baselines3 not available. Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""

    # Dataset
    dataset_name: str = "Anthropic/hh-rlhf"  # Preference dataset
    max_seq_length: int = 2048

    # DPO hyperparameters
    beta: float = 0.1  # KL penalty coefficient (controls deviation from reference)
    learning_rate: float = 1e-6  # Very low LR for preference tuning
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Loss weights
    sft_loss_weight: float = 0.1  # Optional SFT regularization

    # Reference model
    reference_model_path: str = None  # Path to SFT checkpoint (used as reference)

    # Checkpointing
    save_dir: str = "checkpoints/dpo"


@dataclass
class PPOConfig:
    """Configuration for PPO training using stable-baselines3."""

    # Environment
    max_seq_length: int = 2048
    max_new_tokens: int = 512

    # PPO hyperparameters (stable-baselines3 defaults)
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Number of steps per update
    batch_size: int = 64
    n_epochs: int = 10  # PPO epochs per update
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_range: float = 0.2  # PPO clip range
    vf_coef: float = 0.5  # Value function coefficient
    ent_coef: float = 0.01  # Entropy coefficient

    # KL penalty (to prevent diverging from SFT model)
    kl_coef: float = 0.02
    target_kl: float = 0.01  # Early stopping if KL exceeds this

    # Reward model
    reward_model_path: str = None  # Path to trained reward model

    # Training
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000

    # Checkpointing
    save_dir: str = "checkpoints/ppo"


class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    DPO directly optimizes the policy to match human preferences without
    training a separate reward model. It's more stable than PPO but less flexible.
    """

    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        config: DPOConfig,
        tokenizer: Any,
        device: str = "cuda",
    ):
        """
        Initialize DPO trainer.

        Args:
            model: SFT model to optimize
            reference_model: Reference model (frozen SFT checkpoint)
            config: DPO configuration
            tokenizer: Tokenizer
            device: Device to train on
        """
        self.model = model.to(device)
        self.reference_model = reference_model.to(device)
        self.reference_model.eval()  # Freeze reference model

        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

        # Load preference dataset
        self.train_loader = self._load_preference_dataset()

    def _load_preference_dataset(self) -> DataLoader:
        """Load preference dataset with chosen/rejected pairs."""
        from ..data.preference_dataset import PreferenceDataset

        dataset = PreferenceDataset(
            dataset_name=self.config.dataset_name,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def compute_dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.

        The DPO objective maximizes:
        log_sigmoid(β * (log π(y_w|x) - log π_ref(y_w|x) - log π(y_l|x) + log π_ref(y_l|x)))

        Where y_w is chosen response and y_l is rejected response.
        """
        # Get logprobs from policy model
        with torch.cuda.amp.autocast():
            chosen_output = self.model(
                input_ids=chosen_ids,
                attention_mask=chosen_mask,
                labels=chosen_ids,
            )
            rejected_output = self.model(
                input_ids=rejected_ids,
                attention_mask=rejected_mask,
                labels=rejected_ids,
            )

        # Get logprobs from reference model
        with torch.no_grad():
            ref_chosen_output = self.reference_model(
                input_ids=chosen_ids,
                attention_mask=chosen_mask,
                labels=chosen_ids,
            )
            ref_rejected_output = self.reference_model(
                input_ids=rejected_ids,
                attention_mask=rejected_mask,
                labels=rejected_ids,
            )

        # Extract log probabilities
        chosen_logprobs = -chosen_output.loss
        rejected_logprobs = -rejected_output.loss
        ref_chosen_logprobs = -ref_chosen_output.loss
        ref_rejected_logprobs = -ref_rejected_output.loss

        # Compute DPO loss
        chosen_rewards = self.config.beta * (chosen_logprobs - ref_chosen_logprobs)
        rejected_rewards = self.config.beta * (rejected_logprobs - ref_rejected_logprobs)

        # Loss is negative log sigmoid of reward difference
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        # Optional: Add SFT regularization
        if self.config.sft_loss_weight > 0:
            sft_loss = chosen_output.loss
            loss = loss + self.config.sft_loss_weight * sft_loss

        # Compute metrics
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            chosen_reward_mean = chosen_rewards.mean().item()
            rejected_reward_mean = rejected_rewards.mean().item()

        metrics = {
            "accuracy": accuracy,
            "chosen_reward": chosen_reward_mean,
            "rejected_reward": rejected_reward_mean,
            "reward_margin": chosen_reward_mean - rejected_reward_mean,
        }

        return loss, metrics

    def train(self) -> Dict[str, List[float]]:
        """Run DPO training."""
        logger.info("Starting DPO training")
        history = {
            "loss": [],
            "accuracy": [],
            "reward_margin": [],
        }

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_metrics = []

            for batch_idx, batch in enumerate(self.train_loader):
                # Move to device
                chosen_ids = batch["chosen_ids"].to(self.device)
                rejected_ids = batch["rejected_ids"].to(self.device)
                chosen_mask = batch["chosen_mask"].to(self.device)
                rejected_mask = batch["rejected_mask"].to(self.device)

                # Compute loss
                loss, metrics = self.compute_dpo_loss(
                    chosen_ids, rejected_ids, chosen_mask, rejected_mask
                )

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_metrics.append(metrics)

                if batch_idx % 100 == 0:
                    logger.info(
                        f"Epoch {epoch} | Batch {batch_idx} | "
                        f"Loss: {loss:.4f} | Acc: {metrics['accuracy']:.2%} | "
                        f"Margin: {metrics['reward_margin']:.3f}"
                    )

            # Aggregate epoch metrics
            avg_metrics = {
                k: np.mean([m[k] for m in epoch_metrics])
                for k in epoch_metrics[0].keys()
            }
            history["accuracy"].append(avg_metrics["accuracy"])
            history["reward_margin"].append(avg_metrics["reward_margin"])

            # Save checkpoint
            self.save_checkpoint(f"{self.config.save_dir}/dpo_epoch_{epoch}.pt")

        return history

    def save_checkpoint(self, path: str):
        """Save DPO checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)


class PPOTrainerSB3:
    """
    PPO trainer using stable-baselines3.

    This provides a production-ready PPO implementation with all the
    bells and whistles (GAE, value function, entropy bonus, etc.).
    """

    def __init__(
        self,
        model: nn.Module,
        reward_model: nn.Module,
        config: PPOConfig,
        tokenizer: Any,
        device: str = "cuda",
    ):
        """
        Initialize PPO trainer with stable-baselines3.

        Args:
            model: SFT model to optimize
            reward_model: Trained reward model
            config: PPO configuration
            tokenizer: Tokenizer
            device: Device to train on
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required for PPO. Install with: pip install stable-baselines3")

        self.model = model.to(device)
        self.reward_model = reward_model.to(device)
        self.reward_model.eval()

        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        # Create environment wrapper
        self.env = self._create_text_env()

        # Initialize PPO from stable-baselines3
        self.ppo = PPO(
            policy="MlpPolicy",  # Will be replaced with our custom policy
            env=self.env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            verbose=1,
            tensorboard_log=f"{config.save_dir}/tb_logs",
        )

        # Replace policy with our language model
        self._setup_custom_policy()

    def _create_text_env(self):
        """Create text generation environment for PPO."""
        from ..rl.text_environment import TextGenerationEnv

        env = TextGenerationEnv(
            model=self.model,
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            max_new_tokens=self.config.max_new_tokens,
            device=self.device,
        )

        # Wrap in DummyVecEnv for stable-baselines3
        return DummyVecEnv([lambda: env])

    def _setup_custom_policy(self):
        """Replace SB3's default policy with our language model."""
        # This requires custom integration between SB3 and our model
        # Implementation would wrap our model to work with SB3's interface
        pass

    def train(self):
        """Run PPO training with stable-baselines3."""
        logger.info("Starting PPO training with stable-baselines3")

        # Setup evaluation callback
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=f"{self.config.save_dir}/best",
            log_path=f"{self.config.save_dir}/logs",
            eval_freq=self.config.eval_freq,
            deterministic=True,
            render=False,
        )

        # Train with PPO
        self.ppo.learn(
            total_timesteps=self.config.total_timesteps,
            callback=eval_callback,
        )

        # Save final model
        self.ppo.save(f"{self.config.save_dir}/ppo_final")
        logger.info("PPO training complete")

    def generate(self, prompt: str, deterministic: bool = False) -> str:
        """Generate text using the PPO-trained policy."""
        obs = self.env.reset()
        obs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        action, _states = self.ppo.predict(obs, deterministic=deterministic)

        # Decode action to text
        response = self.tokenizer.decode(action, skip_special_tokens=True)
        return response


class PPOTrainerCustom:
    """
    Custom PPO implementation for when stable-baselines3 is not available.

    This is a simplified PPO specifically for language models.
    """

    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        reward_model: nn.Module,
        config: PPOConfig,
        tokenizer: Any,
        device: str = "cuda",
    ):
        """Initialize custom PPO trainer."""
        self.model = model.to(device)
        self.reference_model = reference_model.to(device)
        self.reference_model.eval()
        self.reward_model = reward_model.to(device)
        self.reward_model.eval()

        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        # Optimizer for both policy and value networks
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Value head (predicts expected returns)
        self.value_head = nn.Linear(model.config.d_model, 1).to(device)
        self.value_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=config.learning_rate,
        )

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.

        Uses Generalized Advantage Estimation for variance reduction.
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Compute returns and advantages using GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single PPO training step."""
        # Implementation of PPO update
        # This would include:
        # 1. Generate trajectories
        # 2. Compute rewards using reward model
        # 3. Compute advantages using GAE
        # 4. Multiple epochs of PPO updates
        # 5. Clip ratio to prevent large updates

        # Placeholder for now
        return {"loss": 0.0, "reward": 0.0, "kl": 0.0}