"""
Preference dataset for DPO and reward model training.

Handles paired preference data (chosen vs rejected responses).
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
import json
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """
    Dataset for preference-based training (DPO, reward modeling).

    Each example contains:
    - A prompt/context
    - A chosen (preferred) response
    - A rejected (non-preferred) response
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        max_seq_length: int = 2048,
        max_prompt_length: Optional[int] = None,
        split: str = "train",
    ):
        """
        Initialize preference dataset.

        Args:
            dataset_name: HuggingFace dataset name or local path
            tokenizer: Tokenizer for processing text
            max_seq_length: Maximum total sequence length
            max_prompt_length: Maximum prompt length (if None, use max_seq_length // 2)
            split: Dataset split to use
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length or max_seq_length // 2

        # Load dataset
        self.dataset = self._load_dataset(dataset_name, split)
        logger.info(f"Loaded {len(self.dataset)} preference pairs from {dataset_name}")

    def _load_dataset(self, dataset_name: str, split: str) -> List[Dict]:
        """Load and format preference dataset."""
        if dataset_name == "Anthropic/hh-rlhf":
            return self._load_hh_rlhf(split)
        elif dataset_name == "stanfordnlp/SHP":
            return self._load_shp(split)
        elif dataset_name == "openai/summarize_from_feedback":
            return self._load_summarize_feedback(split)
        elif dataset_name == "Dahoas/rm-static":
            return self._load_rm_static(split)
        elif dataset_name.endswith(".jsonl"):
            return self._load_jsonl(dataset_name)
        else:
            # Try generic loading
            return self._load_generic_preference(dataset_name, split)

    def _load_hh_rlhf(self, split: str) -> List[Dict]:
        """Load Anthropic's HH-RLHF dataset."""
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)

        formatted = []
        for item in dataset:
            # HH-RLHF has "chosen" and "rejected" fields
            prompt = self._extract_prompt(item["chosen"])
            chosen_response = self._extract_response(item["chosen"])
            rejected_response = self._extract_response(item["rejected"])

            formatted.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
            })

        return formatted

    def _load_shp(self, split: str) -> List[Dict]:
        """Load Stanford Human Preferences dataset."""
        dataset = load_dataset("stanfordnlp/SHP", split=split)

        formatted = []
        for item in dataset:
            # SHP has score_A and score_B to determine preference
            prompt = item["history"]
            response_a = item["human_ref_A"]
            response_b = item["human_ref_B"]

            # Determine which is chosen based on scores
            if item["score_A"] > item["score_B"]:
                chosen = response_a
                rejected = response_b
            else:
                chosen = response_b
                rejected = response_a

            formatted.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

        return formatted

    def _load_summarize_feedback(self, split: str) -> List[Dict]:
        """Load OpenAI's summarization with feedback dataset."""
        dataset = load_dataset("openai/summarize_from_feedback", "comparisons", split=split)

        formatted = []
        for item in dataset:
            prompt = item["info"]["article"]
            summaries = item["summaries"]

            # Find chosen summary (higher score)
            if summaries[0]["choice"] == 0:
                chosen = summaries[0]["text"]
                rejected = summaries[1]["text"]
            else:
                chosen = summaries[1]["text"]
                rejected = summaries[0]["text"]

            formatted.append({
                "prompt": f"Summarize the following article:\n\n{prompt}",
                "chosen": chosen,
                "rejected": rejected,
            })

        return formatted

    def _load_rm_static(self, split: str) -> List[Dict]:
        """Load Dahoas/rm-static dataset."""
        dataset = load_dataset("Dahoas/rm-static", split=split)

        formatted = []
        for item in dataset:
            formatted.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })

        return formatted

    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load custom JSONL preference dataset."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Expected format: {"prompt": "...", "chosen": "...", "rejected": "..."}
                data.append(item)
        return data

    def _load_generic_preference(self, dataset_name: str, split: str) -> List[Dict]:
        """Try to load a generic preference dataset."""
        dataset = load_dataset(dataset_name, split=split)

        formatted = []
        for item in dataset:
            # Try common field names
            prompt = None
            chosen = None
            rejected = None

            # Prompt fields
            for field in ["prompt", "context", "question", "input", "text"]:
                if field in item:
                    prompt = item[field]
                    break

            # Chosen/rejected fields
            for chosen_field, rejected_field in [
                ("chosen", "rejected"),
                ("response_chosen", "response_rejected"),
                ("positive", "negative"),
                ("better", "worse"),
            ]:
                if chosen_field in item and rejected_field in item:
                    chosen = item[chosen_field]
                    rejected = item[rejected_field]
                    break

            if prompt and chosen and rejected:
                formatted.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })

        if not formatted:
            logger.warning(f"Could not parse preference dataset {dataset_name}")

        return formatted

    def _extract_prompt(self, conversation: str) -> str:
        """Extract prompt from a conversation string."""
        # Simple heuristic: take everything before the last assistant response
        if "\n\nAssistant:" in conversation:
            parts = conversation.split("\n\nAssistant:")
            return "\n\nAssistant:".join(parts[:-1])
        elif "\n\nHuman:" in conversation:
            # Take the last human message as prompt
            parts = conversation.split("\n\nHuman:")
            return "\n\nHuman:" + parts[-1].split("\n\nAssistant:")[0]
        else:
            # Fallback: use first half as prompt
            return conversation[:len(conversation)//2]

    def _extract_response(self, conversation: str) -> str:
        """Extract response from a conversation string."""
        # Simple heuristic: take the last assistant response
        if "\n\nAssistant:" in conversation:
            parts = conversation.split("\n\nAssistant:")
            return parts[-1]
        else:
            # Fallback: use second half as response
            return conversation[len(conversation)//2:]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single preference pair.

        Returns:
            Dictionary with:
            - prompt_ids: Token IDs for prompt
            - chosen_ids: Token IDs for prompt + chosen response
            - rejected_ids: Token IDs for prompt + rejected response
            - chosen_mask: Attention mask for chosen
            - rejected_mask: Attention mask for rejected
        """
        item = self.dataset[idx]

        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )

        # Tokenize full sequences (prompt + response)
        chosen_text = f"{prompt}\n\n{chosen}"
        rejected_text = f"{prompt}\n\n{rejected}"

        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "prompt_ids": prompt_encoding["input_ids"].squeeze(0),
            "chosen_ids": chosen_encoding["input_ids"].squeeze(0),
            "rejected_ids": rejected_encoding["input_ids"].squeeze(0),
            "chosen_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_mask": rejected_encoding["attention_mask"].squeeze(0),
        }


class RewardModelingDataset(PreferenceDataset):
    """
    Dataset specifically for training reward models.

    Similar to PreferenceDataset but returns individual examples
    with binary labels instead of pairs.
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get examples for reward model training.

        Returns both chosen and rejected as separate examples with labels.
        """
        # Get base preference pair
        base_item = super().__getitem__(idx)

        # Return as individual examples with labels
        # In practice, you'd return these in separate calls or restructure
        return {
            "chosen": {
                "input_ids": base_item["chosen_ids"],
                "attention_mask": base_item["chosen_mask"],
                "labels": torch.tensor(1.0),  # Positive reward
            },
            "rejected": {
                "input_ids": base_item["rejected_ids"],
                "attention_mask": base_item["rejected_mask"],
                "labels": torch.tensor(0.0),  # Negative/lower reward
            },
        }