"""
SFT (Supervised Fine-Tuning) dataset loaders.

Handles instruction-following datasets for the SFT phase.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any
import json
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning on instruction-following data.

    Supports multiple formats:
    - Single-turn Q&A
    - Multi-turn conversations
    - System prompts
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        max_seq_length: int = 4096,
        mask_prompt: bool = True,
        response_template: str = "### Assistant:",
        instruction_template: str = "### Human:",
        system_template: str = "### System:",
        split: str = "train",
    ):
        """
        Initialize SFT dataset.

        Args:
            dataset_name: HuggingFace dataset name or local path
            tokenizer: Tokenizer for processing text
            max_seq_length: Maximum sequence length
            mask_prompt: If True, only compute loss on responses
            response_template: Template marking response start
            instruction_template: Template marking instruction start
            system_template: Template marking system prompt
            split: Dataset split to use
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.response_template = response_template
        self.instruction_template = instruction_template
        self.system_template = system_template

        # Load dataset
        self.dataset = self._load_dataset(dataset_name, split)
        logger.info(f"Loaded {len(self.dataset)} examples from {dataset_name}")

    def _load_dataset(self, dataset_name: str, split: str) -> List[Dict]:
        """Load and format dataset."""
        if dataset_name == "HuggingFaceH4/ultrachat_200k":
            return self._load_ultrachat(split)
        elif dataset_name == "teknium/OpenHermes-2.5":
            return self._load_openhermes(split)
        elif dataset_name == "Open-Orca/SlimOrca":
            return self._load_slimorca(split)
        elif dataset_name.endswith(".jsonl"):
            return self._load_jsonl(dataset_name)
        else:
            # Generic HuggingFace dataset loader
            return self._load_generic_hf(dataset_name, split)

    def _load_ultrachat(self, split: str) -> List[Dict]:
        """Load UltraChat dataset."""
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)

        formatted = []
        for item in dataset:
            # UltraChat has multi-turn conversations
            conversation = []
            for msg in item["messages"]:
                role = msg["role"]
                content = msg["content"]

                if role == "system":
                    conversation.append({"role": "system", "content": content})
                elif role == "user":
                    conversation.append({"role": "user", "content": content})
                elif role == "assistant":
                    conversation.append({"role": "assistant", "content": content})

            if conversation:
                formatted.append({"conversation": conversation})

        return formatted

    def _load_openhermes(self, split: str) -> List[Dict]:
        """Load OpenHermes dataset."""
        dataset = load_dataset("teknium/OpenHermes-2.5", split=split)

        formatted = []
        for item in dataset:
            conversation = []

            # Add system message if present
            if "system" in item and item["system"]:
                conversation.append({"role": "system", "content": item["system"]})

            # Add instruction and response
            conversation.append({"role": "user", "content": item["instruction"]})
            conversation.append({"role": "assistant", "content": item["output"]})

            formatted.append({"conversation": conversation})

        return formatted

    def _load_slimorca(self, split: str) -> List[Dict]:
        """Load SlimOrca dataset."""
        dataset = load_dataset("Open-Orca/SlimOrca", split=split)

        formatted = []
        for item in dataset:
            conversation = []

            # SlimOrca includes system prompts
            if "system_prompt" in item:
                conversation.append({"role": "system", "content": item["system_prompt"]})

            conversation.append({"role": "user", "content": item["question"]})
            conversation.append({"role": "assistant", "content": item["response"]})

            formatted.append({"conversation": conversation})

        return formatted

    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load custom JSONL dataset."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Expect format: {"conversation": [{"role": "...", "content": "..."}]}
                data.append(item)
        return data

    def _load_generic_hf(self, dataset_name: str, split: str) -> List[Dict]:
        """Load generic HuggingFace dataset."""
        dataset = load_dataset(dataset_name, split=split)

        formatted = []
        for item in dataset:
            # Try to auto-detect format
            conversation = []

            if "instruction" in item and "output" in item:
                # Alpaca-style format
                if "input" in item and item["input"]:
                    instruction = f"{item['instruction']}\n\nInput: {item['input']}"
                else:
                    instruction = item["instruction"]

                conversation.append({"role": "user", "content": instruction})
                conversation.append({"role": "assistant", "content": item["output"]})

            elif "prompt" in item and "response" in item:
                # Simple prompt-response format
                conversation.append({"role": "user", "content": item["prompt"]})
                conversation.append({"role": "assistant", "content": item["response"]})

            elif "text" in item:
                # Try to parse as conversation
                # This is a fallback - might need custom parsing
                conversation = self._parse_text_conversation(item["text"])

            if conversation:
                formatted.append({"conversation": conversation})

        return formatted

    def _parse_text_conversation(self, text: str) -> List[Dict]:
        """Parse free-form text into conversation format."""
        # Simple heuristic-based parsing
        conversation = []

        # Look for common patterns
        if self.instruction_template in text and self.response_template in text:
            parts = text.split(self.response_template)
            if len(parts) >= 2:
                instruction = parts[0].replace(self.instruction_template, "").strip()
                response = parts[1].strip()
                conversation.append({"role": "user", "content": instruction})
                conversation.append({"role": "assistant", "content": response})

        return conversation

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - labels: Labels for loss computation (-100 for masked positions)
        """
        item = self.dataset[idx]
        conversation = item["conversation"]

        # Format conversation into text
        text = self._format_conversation(conversation)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels
        if self.mask_prompt:
            labels = self._create_masked_labels(text, input_ids)
        else:
            labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _format_conversation(self, conversation: List[Dict]) -> str:
        """Format conversation into training text."""
        formatted_parts = []

        for turn in conversation:
            role = turn["role"]
            content = turn["content"]

            if role == "system":
                formatted_parts.append(f"{self.system_template} {content}")
            elif role == "user":
                formatted_parts.append(f"{self.instruction_template} {content}")
            elif role == "assistant":
                formatted_parts.append(f"{self.response_template} {content}")

        return "\n\n".join(formatted_parts)

    def _create_masked_labels(self, text: str, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create labels with prompts masked out.

        Only compute loss on assistant responses.
        """
        labels = input_ids.clone()

        # Find response positions
        response_template_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )

        # Mask everything before each response
        for i in range(len(input_ids)):
            # Simple approach: look for response template
            # In practice, would use more sophisticated masking
            if i > 0 and self._is_response_start(input_ids, i, response_template_ids):
                # Don't mask from here onwards until next instruction
                break
            labels[i] = -100  # Mask token for loss computation

        return labels

    def _is_response_start(
        self,
        input_ids: torch.Tensor,
        position: int,
        template_ids: List[int],
    ) -> bool:
        """Check if position is the start of a response."""
        if position + len(template_ids) > len(input_ids):
            return False

        for i, template_id in enumerate(template_ids):
            if input_ids[position + i] != template_id:
                return False

        return True