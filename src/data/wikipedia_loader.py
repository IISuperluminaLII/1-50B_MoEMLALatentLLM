"""
Sanitized Wikipedia data loader for DeepSeek-V3 training.

Provides streaming access to Wikipedia data with on-the-fly sanitization
and caching for efficient training on both CPU and GPU.
"""
import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import logging

from .wikipedia_sanitizer import WikipediaSanitizer, SanitizationConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WikipediaDataConfig:
    """Configuration for Wikipedia data loading."""

    # Dataset settings
    dataset_name: str = "wikimedia/wikipedia"
    dataset_version: str = "20231101.en"  # November 2023 English Wikipedia
    streaming: bool = True

    # Sanitization
    sanitization_enabled: bool = True
    sanitization_config: Optional[SanitizationConfig] = None

    # Caching
    cache_dir: str = "./wikipedia_cache"
    cache_sanitized: bool = True
    cache_batch_size: int = 1000

    # Processing
    seq_length: int = 512
    max_articles: Optional[int] = None  # Limit for testing

    # Memory management
    prefetch_factor: int = 2
    buffer_size: int = 10000  # For streaming shuffle


class SanitizedWikipediaDataset(IterableDataset):
    """
    Streaming Wikipedia dataset with sanitization.

    Features:
    - On-the-fly sanitization with caching
    - Memory-efficient streaming
    - Support for both CPU and GPU training
    - Automatic handling of large dataset
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[WikipediaDataConfig] = None,
        split: str = "train",
        device: str = "cpu",
        vocab_size_limit: Optional[int] = None,
    ):
        """
        Initialize sanitized Wikipedia dataset.

        Args:
            tokenizer: HuggingFace tokenizer
            config: Data loading configuration
            split: Dataset split ("train" or "validation")
            device: Target device ("cpu" or "cuda")
            vocab_size_limit: Maximum vocab size for clamping token IDs
        """
        self.tokenizer = tokenizer
        self.config = config or WikipediaDataConfig()
        self.split = split
        self.device = device
        self.vocab_size_limit = vocab_size_limit

        # Initialize sanitizer
        if self.config.sanitization_enabled:
            self.sanitizer = WikipediaSanitizer(self.config.sanitization_config)
        else:
            self.sanitizer = None

        # Setup cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sanitized_cache_dir = self.cache_dir / "sanitized"
        self.sanitized_cache_dir.mkdir(exist_ok=True)

        # Load dataset
        self._load_dataset()

        # Statistics
        self.stats = {
            'articles_processed': 0,
            'articles_filtered': 0,
            'tokens_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    def _load_dataset(self):
        """Load Wikipedia dataset from HuggingFace."""
        logger.info(f"Loading Wikipedia dataset: {self.config.dataset_version}")

        try:
            # Try to load the wikimedia/wikipedia dataset
            self.dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_version,
                split=self.split,
                streaming=self.config.streaming,
            )
            logger.info("Successfully loaded wikimedia/wikipedia dataset")
        except Exception as e:
            logger.warning(f"Failed to load wikimedia/wikipedia: {e}")
            logger.info("Falling back to legacy wikipedia dataset")

            # Fallback to legacy dataset
            try:
                self.dataset = load_dataset(
                    "wikipedia",
                    "20220301.en",
                    split=self.split,
                    streaming=self.config.streaming,
                )
            except Exception as e2:
                logger.error(f"Failed to load fallback dataset: {e2}")
                logger.info("Using simplified Wikipedia subset")

                # Final fallback: use simplified Wikipedia
                self.dataset = load_dataset(
                    "wikipedia",
                    "20220301.simple",
                    split=self.split,
                    streaming=self.config.streaming,
                )

        # Shuffle if streaming
        if self.config.streaming and self.split == "train":
            self.dataset = self.dataset.shuffle(
                seed=42,
                buffer_size=self.config.buffer_size
            )

    def _get_cache_path(self, article_id: str) -> Path:
        """Generate cache file path for article."""
        hash_id = hashlib.md5(article_id.encode()).hexdigest()
        return self.sanitized_cache_dir / f"{hash_id[:2]}" / f"{hash_id}.pkl"

    def _load_from_cache(self, article_id: str) -> Optional[str]:
        """Try to load sanitized article from cache."""
        if not self.config.cache_sanitized:
            return None

        cache_path = self._get_cache_path(article_id)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    self.stats['cache_hits'] += 1
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed for {article_id}: {e}")

        self.stats['cache_misses'] += 1
        return None

    def _save_to_cache(self, article_id: str, text: str):
        """Save sanitized article to cache."""
        if not self.config.cache_sanitized:
            return

        cache_path = self._get_cache_path(article_id)
        cache_path.parent.mkdir(exist_ok=True)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(text, f)
        except Exception as e:
            logger.warning(f"Cache save failed for {article_id}: {e}")

    def _extract_text(self, article: Dict[str, Any]) -> Tuple[str, str]:
        """
        Extract text and ID from Wikipedia article.

        Returns:
            Tuple of (text, article_id)
        """
        # Handle different dataset formats
        if isinstance(article, dict):
            # Try different field names
            text = article.get('text', article.get('content', ''))
            article_id = article.get('id', article.get('title', ''))

            # For wikimedia format, combine title and text
            if 'title' in article and 'text' in article:
                text = f"{article['title']}\n\n{article['text']}"
        else:
            text = str(article)
            article_id = hashlib.md5(text.encode()).hexdigest()

        return text, article_id

    def _sanitize_article(self, text: str, article_id: str) -> Optional[str]:
        """Sanitize article with caching."""
        # Check cache first
        cached = self._load_from_cache(article_id)
        if cached is not None:
            return cached

        # Apply sanitization
        if self.sanitizer:
            sanitized = self.sanitizer.sanitize(text, article_id)
            if sanitized:
                # Save to cache
                self._save_to_cache(article_id, sanitized)
                return sanitized
            else:
                # Save empty marker to cache to avoid re-processing
                self._save_to_cache(article_id, "")
                return None

        return text

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text for model input."""
        # Tokenize with truncation and padding
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.seq_length,
            padding='max_length',
            return_tensors='pt',
        )

        # Clamp token IDs to model's vocab size if needed
        if self.vocab_size_limit is not None:
            encoded['input_ids'] = torch.clamp(
                encoded['input_ids'],
                max=self.vocab_size_limit - 1
            )

        # Create labels (shifted input_ids for language modeling)
        labels = encoded['input_ids'].clone()
        labels[encoded['attention_mask'] == 0] = -100  # Mask padding

        # Prepare MTP labels if needed (multi-token prediction)
        batch_size = encoded['input_ids'].shape[0]
        mtp_labels = torch.full(
            (batch_size, self.config.seq_length, 2),
            -100,
            dtype=torch.long
        )

        # Generate MTP labels
        for b in range(batch_size):
            for i in range(self.config.seq_length - 2):
                if encoded['attention_mask'][b, i] == 1:
                    mtp_labels[b, i, 0] = encoded['input_ids'][b, i + 1]
                    if i + 2 < self.config.seq_length and encoded['attention_mask'][b, i + 2] == 1:
                        mtp_labels[b, i, 1] = encoded['input_ids'][b, i + 2]

        # Don't move to device here - let DataLoader handle it with pin_memory
        # This avoids "cannot pin CUDA tensor" errors
        # The training loop will move batches to device

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'mtp_labels': mtp_labels.squeeze(0),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over sanitized and tokenized articles."""
        articles_processed = 0

        for article in self.dataset:
            # Check limit
            if self.config.max_articles and articles_processed >= self.config.max_articles:
                break

            # Extract text
            text, article_id = self._extract_text(article)

            if not text:
                continue

            self.stats['articles_processed'] += 1

            # Sanitize text
            sanitized_text = self._sanitize_article(text, article_id)

            if not sanitized_text:
                self.stats['articles_filtered'] += 1
                continue

            # Focus on historical content for Hiroshima test
            # (Optional: can be removed for general training)
            if "hiroshima" in sanitized_text.lower() or "1945" in sanitized_text:
                # Boost historical content by yielding it multiple times
                repeat_count = 3
            else:
                repeat_count = 1

            for _ in range(repeat_count):
                # Tokenize and yield
                tokens = self._tokenize_text(sanitized_text)
                self.stats['tokens_generated'] += self.config.seq_length

                yield tokens

                # CRITICAL: Delete tokens after yielding to prevent memory accumulation
                del tokens

            articles_processed += 1

            # Clean up text references
            del sanitized_text, text

            # Log progress periodically
            if articles_processed % 100 == 0:
                logger.info(f"Processed {articles_processed} articles, "
                           f"filtered {self.stats['articles_filtered']}, "
                           f"cache hits: {self.stats['cache_hits']}")

    def get_statistics(self) -> Dict[str, int]:
        """Get dataset statistics."""
        if self.sanitizer:
            self.stats.update(self.sanitizer.get_statistics())
        return self.stats


def create_wikipedia_dataloader(
    tokenizer,
    config: WikipediaDataConfig,
    batch_size: int = 4,
    device: str = "cpu",
    num_workers: int = 0,
    vocab_size_limit: Optional[int] = None,
) -> DataLoader:
    """
    Create DataLoader for Wikipedia dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        config: Data configuration
        batch_size: Batch size
        device: Target device
        num_workers: Number of data loading workers
        vocab_size_limit: Maximum vocab size for clamping token IDs

    Returns:
        DataLoader instance
    """
    dataset = SanitizedWikipediaDataset(
        tokenizer=tokenizer,
        config=config,
        split="train",
        device=device,
        vocab_size_limit=vocab_size_limit,
    )

    # Create dataloader
    # Note: num_workers=0 for IterableDataset to avoid issues on Windows
    # Note: Using default collate_fn - PyTorch's default_collate already handles this efficiently
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,  # Disabled to save GPU memory - data will be moved to GPU in training loop
        prefetch_factor=config.prefetch_factor if num_workers > 0 else None,
    )

    return dataloader


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create config
    config = WikipediaDataConfig(
        max_articles=10,  # Limit for testing
        seq_length=256,
        sanitization_enabled=True,
    )

    # Create dataset
    dataset = SanitizedWikipediaDataset(
        tokenizer=tokenizer,
        config=config,
        device="cpu",
    )

    # Test iteration
    print("Testing Wikipedia data loader...")
    for i, batch in enumerate(dataset):
        if i >= 3:
            break

        print(f"\nBatch {i}:")
        print(f"  Input shape: {batch['input_ids'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  First 50 tokens: {batch['input_ids'][:50]}")

        # Decode to see actual text
        text = tokenizer.decode(batch['input_ids'][:100], skip_special_tokens=True)
        print(f"  Decoded text (first 100 tokens): {text[:200]}...")

    print("\nStatistics:")
    for key, value in dataset.get_statistics().items():
        print(f"  {key}: {value}")