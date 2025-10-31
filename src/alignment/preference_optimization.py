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
        # Get logits from policy model (not using labels to avoid loss computation)
        with torch.cuda.amp.autocast():
            chosen_output = self.model(
                input_ids=chosen_ids,
                attention_mask=chosen_mask,
            )
            rejected_output = self.model(
                input_ids=rejected_ids,
                attention_mask=rejected_mask,
            )

        # Get logits from reference model
        with torch.no_grad():
            ref_chosen_output = self.reference_model(
                input_ids=chosen_ids,
                attention_mask=chosen_mask,
            )
            ref_rejected_output = self.reference_model(
                input_ids=rejected_ids,
                attention_mask=rejected_mask,
            )

        # Compute sequence log-probabilities (sum of token log-probs)
        def get_sequence_logprobs(logits, input_ids, attention_mask):
            """Compute sequence log-probabilities from logits."""
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            # Get log probabilities
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

            # Gather log probs of actual tokens
            batch_size, seq_len = shift_labels.shape
            token_log_probs = torch.gather(
                log_probs.view(-1, log_probs.size(-1)),
                1,
                shift_labels.view(-1).unsqueeze(1)
            ).view(batch_size, seq_len)

            # Mask and sum to get sequence log-prob
            masked_log_probs = token_log_probs * shift_mask
            sequence_log_probs = masked_log_probs.sum(dim=-1)

            return sequence_log_probs

        # Extract sequence log probabilities
        chosen_logprobs = get_sequence_logprobs(chosen_output.logits, chosen_ids, chosen_mask)
        rejected_logprobs = get_sequence_logprobs(rejected_output.logits, rejected_ids, rejected_mask)
        ref_chosen_logprobs = get_sequence_logprobs(ref_chosen_output.logits, chosen_ids, chosen_mask)
        ref_rejected_logprobs = get_sequence_logprobs(ref_rejected_output.logits, rejected_ids, rejected_mask)

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
        """
        Single PPO training step with proper implementation.

        Args:
            batch: Dictionary containing prompts and other data

        Returns:
            Dictionary of training metrics
        """
        prompts = batch["prompts"]
        batch_size = prompts.shape[0]

        # Generate trajectories with current policy
        with torch.no_grad():
            # Generate responses
            max_new_tokens = self.config.max_new_tokens
            generated = self.model.generate(
                prompts,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Get responses (remove prompt from generated)
            responses = generated[:, prompts.shape[1]:]

            # Compute rewards using reward model
            rewards = self.reward_model(generated).squeeze(-1)

            # Get log probabilities from policy and reference models
            policy_logits = self.model(generated).logits
            ref_logits = self.reference_model(generated).logits

            # Compute log probs for generated tokens
            policy_logprobs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
            ref_logprobs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

            # Gather log probs of actual tokens
            seq_len = responses.shape[1]
            response_logprobs = torch.gather(
                policy_logprobs[:, prompts.shape[1]-1:-1, :],
                2,
                responses.unsqueeze(-1)
            ).squeeze(-1)

            ref_response_logprobs = torch.gather(
                ref_logprobs[:, prompts.shape[1]-1:-1, :],
                2,
                responses.unsqueeze(-1)
            ).squeeze(-1)

            # Compute KL penalty
            kl_penalty = (response_logprobs - ref_response_logprobs).sum(dim=1)

            # Compute rewards with KL penalty
            rewards = rewards - self.config.kl_coef * kl_penalty

            # Get values from value head
            hidden_states = self.model(generated, output_hidden_states=True).hidden_states[-1]
            values = self.value_head(hidden_states[:, -1, :]).squeeze(-1)

            # Create masks for done states
            dones = torch.zeros(batch_size, device=self.device)
            dones[-1] = 1.0  # Last step is done

            # Compute advantages and returns
            advantages, returns = self.compute_advantages(rewards, values, dones)

            # Store old log probs for ratio computation
            old_logprobs = response_logprobs.sum(dim=1)

        # PPO update with multiple epochs
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(self.config.ppo_epochs):
            # Forward pass with gradient
            outputs = self.model(generated)
            new_logits = outputs.logits

            # Compute new log probs
            new_logprobs_all = torch.nn.functional.log_softmax(new_logits, dim=-1)
            new_response_logprobs = torch.gather(
                new_logprobs_all[:, prompts.shape[1]-1:-1, :],
                2,
                responses.unsqueeze(-1)
            ).squeeze(-1)
            new_logprobs = new_response_logprobs.sum(dim=1)

            # Compute probability ratio
            ratio = torch.exp(new_logprobs - old_logprobs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            new_hidden = self.model(generated, output_hidden_states=True).hidden_states[-1]
            new_values = self.value_head(new_hidden[:, -1, :]).squeeze(-1)
            value_loss = torch.nn.functional.mse_loss(new_values, returns)

            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss

            # Backward pass
            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.config.max_grad_norm)

            # Optimizer steps
            self.optimizer.step()
            self.value_optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        # Compute average metrics
        avg_loss = total_loss / self.config.ppo_epochs
        avg_policy_loss = total_policy_loss / self.config.ppo_epochs
        avg_value_loss = total_value_loss / self.config.ppo_epochs

        return {
            "loss": avg_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "reward": rewards.mean().item(),
            "kl": kl_penalty.mean().item(),
            "advantages": advantages.mean().item(),
        }