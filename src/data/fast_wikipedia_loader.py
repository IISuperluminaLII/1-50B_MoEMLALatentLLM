"""
High-performance Wikipedia data loader with aggressive CPU utilization.

Optimizations:
1. Local dataset loading (no streaming latency)
2. Multiprocessing for preprocessing
3. Prefetching and background processing
4. Memory-efficient batching
"""

import torch
import multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, List
import logging
from dataclasses import dataclass
from datasets import load_from_disk, load_dataset
import queue
import threading

logger = logging.getLogger(__name__)


@dataclass
class FastWikipediaConfig:
    """Configuration for fast Wikipedia loader."""
    dataset_path: str  # Path to downloaded dataset
    cache_dir: Optional[str] = None
    batch_size: int = 8
    max_length: int = 2048
    num_workers: int = 8  # Multiprocessing workers
    prefetch_factor: int = 4  # Batches to prefetch per worker
    shuffle_buffer: int = 10000  # Shuffle buffer size
    streaming: bool = False  # Always False for local datasets

    # Sanitization
    sanitization_enabled: bool = True
    min_article_length: int = 100
    max_article_length: int = 10000

    # MTP settings
    mtp_depth: int = 2  # Predict 2 tokens ahead


class FastWikipediaDataset(IterableDataset):
    """
    High-performance Wikipedia dataset with multiprocessing support.

    This dataset is designed for MULTI-WORKER DataLoader usage.
    Each worker processes a different shard of the data automatically.
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        sanitizer,
        max_length: int = 2048,
        mtp_depth: int = 2,
        split: str = "train",
    ):
        """
        Initialize fast Wikipedia dataset.

        Args:
            dataset_path: Path to downloaded Wikipedia dataset
            tokenizer: HuggingFace tokenizer
            sanitizer: WikipediaSanitizer instance
            max_length: Maximum sequence length
            mtp_depth: Multi-token prediction depth
            split: Dataset split ('train' or 'validation')
        """
        self.dataset_path = Path(dataset_path)
        self.tokenizer = tokenizer
        self.sanitizer = sanitizer
        self.max_length = max_length
        self.mtp_depth = mtp_depth
        self.split = split

        # Dataset will be loaded in __iter__ to support multiprocessing
        self.dataset = None

    def _load_dataset(self):
        """Load dataset (called per worker)."""
        if self.dataset is None:
            logger.info(f"Worker loading dataset from {self.dataset_path}")

            # Try to load from disk first (faster)
            try:
                self.dataset = load_from_disk(str(self.dataset_path))
                if self.split in self.dataset:
                    self.dataset = self.dataset[self.split]
            except:
                # Fall back to loading with cache_dir
                logger.info("Loading from HuggingFace cache...")
                # CRITICAL FIX #22: Add keep_in_memory=False to use memory-mapped files
                self.dataset = load_dataset(
                    "wikimedia/wikipedia",
                    "20231101.en",
                    split=self.split,
                    cache_dir=str(self.dataset_path),
                    streaming=False,
                    keep_in_memory=False,
                )

    def _process_article(self, article: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process a single article: sanitize, tokenize, create MTP labels.

        Returns None if article fails sanitization.
        """
        text = article.get("text", "")

        # Sanitize
        if self.sanitizer:
            sanitized = self.sanitizer.sanitize(text)
            if sanitized is None:
                return None
            text = sanitized

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create MTP labels (vectorized)
        seq_len = input_ids.size(0)
        mtp_labels = torch.full(
            (seq_len, self.mtp_depth),
            -100,
            dtype=torch.long
        )

        # First prediction: next token (check current position only)
        if seq_len > 1:
            mtp_labels[:-1, 0] = input_ids[1:]
            current_mask = attention_mask[:-1] == 0
            mtp_labels[:-1, 0][current_mask] = -100
            del current_mask  # CRITICAL FIX #21: Delete mask tensor

        # Second prediction: next-next token (check current AND target)
        if seq_len > 2:
            mtp_labels[:-2, 1] = input_ids[2:]
            current_mask = attention_mask[:-2] == 0
            target_mask = attention_mask[2:] == 0
            mtp_labels[:-2, 1][current_mask | target_mask] = -100
            del current_mask, target_mask  # CRITICAL FIX #21: Delete mask tensors

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # Standard LM labels
            "mtp_labels": mtp_labels,
        }

        # CRITICAL FIX #23: Delete intermediate tensors before returning
        del encoded, input_ids, attention_mask, mtp_labels

        return result

    def __iter__(self):
        """Iterate through dataset with per-worker sharding."""
        self._load_dataset()

        # Get worker info for sharding
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process mode
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process mode - each worker gets a shard
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Calculate this worker's shard
        total_samples = len(self.dataset)
        per_worker = total_samples // num_workers
        start_idx = worker_id * per_worker
        end_idx = start_idx + per_worker if worker_id < num_workers - 1 else total_samples

        logger.info(f"Worker {worker_id}/{num_workers} processing indices {start_idx}-{end_idx}")

        # Process this worker's shard
        for idx in range(start_idx, end_idx):
            article = self.dataset[idx]
            processed = self._process_article(article)

            # CRITICAL FIX #24: Delete article after processing
            del article

            if processed is not None:
                yield processed
                # CRITICAL FIX #24: Delete processed after yielding
                del processed


def create_fast_wikipedia_dataloader(
    dataset_path: str,
    tokenizer,
    sanitizer,
    batch_size: int = 8,
    max_length: int = 2048,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    device: str = "cuda",
    split: str = "train",
) -> DataLoader:
    """
    Create optimized DataLoader with multiprocessing support.

    Args:
        dataset_path: Path to downloaded Wikipedia dataset
        tokenizer: HuggingFace tokenizer
        sanitizer: WikipediaSanitizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of worker processes (recommended: 4-16)
        prefetch_factor: Batches to prefetch per worker
        device: Device type ('cuda' or 'cpu')
        split: Dataset split

    Returns:
        DataLoader with optimized settings
    """
    dataset = FastWikipediaDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        sanitizer=sanitizer,
        max_length=max_length,
        split=split,
    )

    # Custom collate function to handle MTP labels
    def collate_fn(batch):
        """Collate batch with proper stacking."""
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "mtp_labels": torch.stack([item["mtp_labels"] for item in batch]),
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # Use multiprocessing!
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True if device == "cuda" else False,
        collate_fn=collate_fn,
        persistent_workers=False,  # CRITICAL FIX #20: Disable to prevent unbounded memory growth
        multiprocessing_context='spawn' if num_workers > 0 else None,  # Windows compatibility
    )

    logger.info(f"Created fast DataLoader with {num_workers} workers, prefetch={prefetch_factor}")

    return dataloader
