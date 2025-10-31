"""
Text generation environment for PPO training.

This module provides a Gym-compatible environment for training
language models with PPO by treating text generation as an RL task.
"""

import gym
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any, List
from dataclasses import dataclass
from gym import spaces


@dataclass
class TextGenerationConfig:
    """Configuration for text generation environment."""
    max_seq_length: int = 512
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    penalty_alpha: float = 0.0  # For contrastive search
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class TextGenerationEnv(gym.Env):
    """
    Gym environment for text generation with language models.

    This environment allows PPO to optimize text generation by:
    1. Taking prompts as initial states
    2. Generating tokens as actions
    3. Receiving rewards from a reward model
    4. Supporting batched generation for efficiency
    """

    def __init__(
        self,
        model: torch.nn.Module,
        reward_model: torch.nn.Module,
        tokenizer: Any,
        max_seq_length: int = 512,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        device: str = "cuda",
    ):
        """
        Initialize the text generation environment.

        Args:
            model: Language model for text generation
            reward_model: Model for computing rewards
            tokenizer: Tokenizer for text processing
            max_seq_length: Maximum sequence length
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            device: Device to run models on
        """
        super().__init__()

        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device

        # Generation config
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        # Token IDs
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.vocab_size = len(tokenizer)

        # Define action and observation spaces
        # Action space: discrete (token IDs)
        self.action_space = spaces.Discrete(self.vocab_size)

        # Observation space: sequence of token IDs
        self.observation_space = spaces.Box(
            low=0,
            high=self.vocab_size - 1,
            shape=(max_seq_length,),
            dtype=np.int32
        )

        # Current state
        self.current_prompt = None
        self.current_tokens = None
        self.current_position = 0
        self.done = False

    def reset(self, prompt_text: Optional[str] = None) -> np.ndarray:
        """
        Reset environment with a new prompt.

        Args:
            prompt_text: Initial prompt text

        Returns:
            Initial observation (tokenized prompt)
        """
        if prompt_text is None:
            # Use a default prompt for training
            prompt_text = "The following is a helpful response:"

        # Tokenize prompt
        tokens = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=True,
            max_length=self.max_seq_length - self.max_new_tokens,
            truncation=True,
            return_tensors="pt"
        ).squeeze(0)

        # Initialize state
        self.current_prompt = prompt_text
        self.current_tokens = tokens.to(self.device)
        self.current_position = len(tokens)
        self.done = False

        # Pad to max length for observation
        obs = torch.zeros(self.max_seq_length, dtype=torch.long)
        obs[:len(tokens)] = tokens

        return obs.cpu().numpy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment by generating a token.

        Args:
            action: Token ID to generate

        Returns:
            observation: Updated token sequence
            reward: Reward for this action
            done: Whether episode is complete
            info: Additional information
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")

        # Add action (token) to sequence
        new_token = torch.tensor([action], device=self.device)
        self.current_tokens = torch.cat([self.current_tokens, new_token])
        self.current_position += 1

        # Check if done
        if action == self.eos_token_id or self.current_position >= self.max_seq_length:
            self.done = True

        # Compute reward (only at the end of generation)
        reward = 0.0
        if self.done:
            reward = self._compute_reward()

        # Create observation
        obs = torch.zeros(self.max_seq_length, dtype=torch.long)
        obs[:len(self.current_tokens)] = self.current_tokens.cpu()

        # Additional info
        info = {
            "tokens_generated": self.current_position - len(self.tokenizer.encode(self.current_prompt)),
            "sequence_length": len(self.current_tokens),
            "done_reason": "eos" if action == self.eos_token_id else "max_length" if self.done else None
        }

        return obs.cpu().numpy(), reward, self.done, info

    def _compute_reward(self) -> float:
        """
        Compute reward using the reward model.

        Returns:
            Reward score for the generated sequence
        """
        with torch.no_grad():
            # Create attention mask
            attention_mask = (self.current_tokens != self.pad_token_id).float()

            # Get reward from reward model
            if hasattr(self.reward_model, 'compute_reward'):
                reward = self.reward_model.compute_reward(
                    self.current_tokens.unsqueeze(0),
                    attention_mask.unsqueeze(0)
                )
            else:
                # Fallback: use forward pass
                reward = self.reward_model(
                    self.current_tokens.unsqueeze(0),
                    attention_mask.unsqueeze(0)
                )

            # Convert to scalar
            if isinstance(reward, torch.Tensor):
                reward = reward.item()

        return float(reward)

    def generate_trajectory(
        self,
        prompt: str,
        use_model_sampling: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a complete trajectory for the given prompt.

        Args:
            prompt: Input prompt
            use_model_sampling: Whether to use model's sampling

        Returns:
            Dictionary containing trajectory data
        """
        obs = self.reset(prompt)

        observations = [obs]
        actions = []
        rewards = []
        log_probs = []

        with torch.no_grad():
            while not self.done:
                # Get model predictions
                input_ids = self.current_tokens.unsqueeze(0)
                attention_mask = (input_ids != self.pad_token_id).float()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )

                logits = outputs.logits[0, -1, :]  # Get last token logits

                # Apply temperature
                if self.temperature != 1.0:
                    logits = logits / self.temperature

                # Compute probabilities
                probs = F.softmax(logits, dim=-1)

                # Sample action
                if use_model_sampling:
                    # Top-k sampling
                    if self.top_k > 0:
                        top_k_probs, top_k_indices = torch.topk(probs, min(self.top_k, probs.size(-1)))
                        action = top_k_indices[torch.multinomial(top_k_probs, 1)]
                    else:
                        action = torch.multinomial(probs, 1)
                else:
                    # Greedy
                    action = torch.argmax(probs)

                # Log probability of action
                log_prob = torch.log(probs[action])

                # Take step
                obs, reward, done, info = self.step(action.item())

                observations.append(obs)
                actions.append(action.item())
                rewards.append(reward)
                log_probs.append(log_prob.item())

        return {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "log_probs": np.array(log_probs),
            "generated_text": self.tokenizer.decode(self.current_tokens, skip_special_tokens=True)
        }

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the current state.

        Args:
            mode: Rendering mode

        Returns:
            Rendered output (text)
        """
        if self.current_tokens is None:
            return "Environment not initialized. Call reset() first."

        text = self.tokenizer.decode(self.current_tokens, skip_special_tokens=True)

        if mode == "human":
            print(f"Current text: {text}")
            print(f"Position: {self.current_position}/{self.max_seq_length}")
            print(f"Done: {self.done}")

        return text


class BatchedTextGenerationEnv:
    """
    Batched version of TextGenerationEnv for efficient PPO training.

    This handles multiple text generation episodes in parallel
    for better GPU utilization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        reward_model: torch.nn.Module,
        tokenizer: Any,
        batch_size: int = 8,
        **kwargs
    ):
        """
        Initialize batched environment.

        Args:
            model: Language model
            reward_model: Reward model
            tokenizer: Tokenizer
            batch_size: Number of parallel environments
            **kwargs: Additional arguments for TextGenerationEnv
        """
        self.batch_size = batch_size
        self.envs = [
            TextGenerationEnv(model, reward_model, tokenizer, **kwargs)
            for _ in range(batch_size)
        ]

    def reset(self, prompts: Optional[List[str]] = None) -> np.ndarray:
        """Reset all environments."""
        if prompts is None:
            prompts = [None] * self.batch_size
        elif len(prompts) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} prompts, got {len(prompts)}")

        observations = []
        for env, prompt in zip(self.envs, prompts):
            obs = env.reset(prompt)
            observations.append(obs)

        return np.stack(observations)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Take a step in all environments."""
        observations = []
        rewards = []
        dones = []
        infos = []

        for env, action in zip(self.envs, actions):
            if env.done:
                # If done, return zeros (will be masked in PPO)
                obs = np.zeros_like(env.observation_space.sample())
                reward = 0.0
                done = True
                info = {"done": True}
            else:
                obs, reward, done, info = env.step(action)

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(observations),
            np.array(rewards),
            np.array(dones),
            infos
        )


# Create module-level __init__ file
__all__ = [
    "TextGenerationEnv",
    "BatchedTextGenerationEnv",
    "TextGenerationConfig",
]