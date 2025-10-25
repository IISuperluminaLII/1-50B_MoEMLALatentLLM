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
from functools import lru_cache
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
    cache_to_disk: bool = False  # OPTIMIZATION: Disable slow disk writes, use memory cache only
    cache_batch_size: int = 1000

    # Processing
    seq_length: int = 512
    max_articles: Optional[int] = None  # Limit for testing

    # Memory management - AGGRESSIVE settings for high-throughput
    prefetch_factor: int = 8  # Each worker prefetches 8 batches (deep pipeline)
    buffer_size: int = 50000  # Large shuffle buffer for better randomization


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
        # CRITICAL FIX: Disable deduplication for multi-worker mode
        # Deduplication will be handled at dataset preprocessing level instead
        if self.config.sanitization_enabled:
            self.sanitizer = WikipediaSanitizer(
                self.config.sanitization_config,
                enable_dedup=False  # Disable to prevent unbounded LSH/seen_hashes growth per worker
            )
        else:
            self.sanitizer = None

        # Setup cache directory (keep for backward compatibility, but prefer memory cache)
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sanitized_cache_dir = self.cache_dir / "sanitized"
        self.sanitized_cache_dir.mkdir(exist_ok=True)

        # In-memory LRU cache for sanitized articles (much faster than disk I/O)
        # CRITICAL FIX: Disable memory cache in multi-worker mode to prevent 16× duplication
        # Each worker would create its own 100K-article cache = 16GB waste!
        # We rely on SSD speed instead (local dataset is fast enough)
        self._memory_cache = {}
        self._memory_cache_maxsize = 0  # DISABLED for multi-worker mode
        self._memory_cache_enabled = False  # Will be set based on worker context

        # Load dataset
        self._load_dataset()

        # Statistics
        self.stats = {
            'articles_processed': 0,
            'articles_filtered': 0,
            'tokens_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_cache_hits': 0,
            'disk_cache_hits': 0,
        }

    def _load_dataset(self):
        """Load Wikipedia dataset from HuggingFace."""
        logger.info(f"Loading Wikipedia dataset: {self.config.dataset_version}")

        try:
            # Try to load the wikimedia/wikipedia dataset
            # CRITICAL FIX: Pass cache_dir to use local downloaded dataset!
            # CRITICAL FIX #13: Use keep_in_memory=False for memory-mapped access
            # This allows 16 workers to share the SAME disk-backed data without duplicating in RAM!
            # Each worker will memory-map the same Arrow files on disk instead of loading into RAM
            self.dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_version,
                split=self.split,
                streaming=self.config.streaming,
                cache_dir=self.config.cache_dir,  # Use local dataset from S: drive
                keep_in_memory=False,  # Use memory-mapped files to avoid 16× RAM duplication!
            )
            logger.info("Successfully loaded wikimedia/wikipedia dataset from cache (memory-mapped)")
        except Exception as e:
            logger.warning(f"Failed to load wikimedia/wikipedia: {e}")
            logger.info("Falling back to legacy wikipedia dataset")

            # Fallback to legacy dataset
            # CRITICAL FIX #18: Must pass cache_dir and keep_in_memory=False to fallback too!
            try:
                self.dataset = load_dataset(
                    "wikipedia",
                    "20220301.en",
                    split=self.split,
                    streaming=self.config.streaming,
                    cache_dir=self.config.cache_dir,
                    keep_in_memory=False,
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
                    cache_dir=self.config.cache_dir,
                    keep_in_memory=False,
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
        """
        Try to load sanitized article from cache.
        Uses in-memory LRU cache first (fast), then falls back to disk (slow).
        """
        if not self.config.cache_sanitized:
            return None

        # Check memory cache first (instant lookup)
        if article_id in self._memory_cache:
            self.stats['cache_hits'] += 1
            self.stats['memory_cache_hits'] += 1
            return self._memory_cache[article_id]

        # Fall back to disk cache (slow)
        cache_path = self._get_cache_path(article_id)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_text = pickle.load(f)

                    # Store in memory cache for next time
                    self._add_to_memory_cache(article_id, cached_text)

                    self.stats['cache_hits'] += 1
                    self.stats['disk_cache_hits'] += 1
                    return cached_text
            except Exception as e:
                logger.warning(f"Cache load failed for {article_id}: {e}")

        self.stats['cache_misses'] += 1
        return None

    def _add_to_memory_cache(self, article_id: str, text: str):
        """Add article to in-memory LRU cache."""
        # CRITICAL FIX: Skip caching if disabled (multi-worker mode)
        if self._memory_cache_maxsize == 0:
            return

        # Simple LRU: if cache is full, remove oldest entry
        if len(self._memory_cache) >= self._memory_cache_maxsize:
            # Remove first (oldest) entry
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

        self._memory_cache[article_id] = text

    def _save_to_cache(self, article_id: str, text: str):
        """Save sanitized article to both memory and disk cache."""
        if not self.config.cache_sanitized:
            return

        # Add to memory cache (fast) - ALWAYS do this
        self._add_to_memory_cache(article_id, text)

        # Save to disk cache (slow, but persistent) - OPTIONAL
        # Since we have 100k article memory cache, disk writes are often unnecessary
        if self.config.cache_to_disk:
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
        # CRITICAL: Tokenize WITHOUT return_tensors first, then manually create CPU tensors!
        # This prevents workers from accidentally creating GPU tensors if CUDA is available
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.seq_length,
            padding='max_length',
            return_tensors=None,  # Return lists, NOT torch tensors yet!
        )

        # CRITICAL: Manually create tensors EXPLICITLY on CPU device='cpu'
        # This ensures workers NEVER allocate GPU memory for data tensors
        # Note: tokenizer returns 1D list when return_tensors=None, so add batch dimension with unsqueeze
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long, device='cpu').unsqueeze(0)
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long, device='cpu').unsqueeze(0)

        # Clamp token IDs to model's vocab size if needed
        if self.vocab_size_limit is not None:
            input_ids = torch.clamp(input_ids, max=self.vocab_size_limit - 1)

        # Create labels (shifted input_ids for language modeling)
        labels = input_ids.clone()
        padding_mask = attention_mask == 0
        labels[padding_mask] = -100  # Mask padding
        del padding_mask  # CRITICAL FIX: Delete boolean mask tensor

        # Prepare MTP labels if needed (multi-token prediction)
        # OPTIMIZED: Vectorized version - 10-20x faster than nested loops
        batch_size = input_ids.shape[0]
        seq_len = self.config.seq_length
        mtp_labels = torch.full((batch_size, seq_len, 2), -100, dtype=torch.long, device='cpu')

        # Vectorized MTP label generation - shifts input_ids by 1 and 2 positions
        # Reference dolma_loader logic (lines 199-204):
        # for i in range(seq_len-2):
        #     if attention_mask[i] == 1:  # CURRENT position must have attention
        #         mtp_labels[i, 0] = input_ids[i+1]
        #         if attention_mask[i+2] == 1:  # TARGET position must have attention
        #             mtp_labels[i, 1] = input_ids[i+2]

        if seq_len >= 2:
            # First prediction: next token (from positions 0 to seq_len-2)
            # Only check CURRENT position (i+1 will be validated as label independently)
            mtp_labels[:, :-1, 0] = input_ids[:, 1:]
            current_mask = attention_mask[:, :-1] == 0  # Current position only
            mtp_labels[:, :-1, 0][current_mask] = -100
            del current_mask  # CRITICAL FIX: Delete boolean mask tensor

        if seq_len >= 3:
            # Second prediction: token after next (from positions 0 to seq_len-3)
            # Check BOTH current position AND target position (i+2)
            mtp_labels[:, :-2, 1] = input_ids[:, 2:]
            current_mask = attention_mask[:, :-2] == 0  # Current position
            target_mask = attention_mask[:, 2:] == 0    # Target position (i+2)
            mtp_labels[:, :-2, 1][current_mask | target_mask] = -100
            del current_mask, target_mask  # CRITICAL FIX: Delete boolean mask tensors

        # CRITICAL: All tensors already created on CPU (device='cpu' in torch.tensor() calls)
        # Squeeze to remove batch dimension and return
        result = {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'labels': labels.squeeze(0),
            'mtp_labels': mtp_labels.squeeze(0),
        }

        # CRITICAL FIX: Delete intermediate tensors to prevent memory accumulation
        del input_ids, attention_mask, labels, mtp_labels

        return result

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over sanitized and tokenized articles.

        With multi-worker DataLoader, this method is called once per worker.
        For non-streaming datasets, we shard the data across workers.
        For streaming datasets, we process sequentially (single worker).
        """
        # Get worker info for sharding (if using num_workers > 0)
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process mode - process all data
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process mode - each worker gets a shard
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        articles_processed = 0

        # For non-streaming datasets, we can shard by index
        if not self.config.streaming and hasattr(self.dataset, '__len__'):
            total_samples = len(self.dataset)
            per_worker = total_samples // num_workers
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker if worker_id < num_workers - 1 else total_samples

            if num_workers > 1:
                logger.info(f"Worker {worker_id}/{num_workers} processing indices {start_idx}-{end_idx}")

            # Process this worker's shard
            for idx in range(start_idx, end_idx):
                # Check limit
                if self.config.max_articles and articles_processed >= self.config.max_articles:
                    break

                article = self.dataset[idx]

                # Extract text
                text, article_id = self._extract_text(article)

                # CRITICAL FIX: Delete article after extraction to free memory
                del article

                if not text:
                    continue

                self.stats['articles_processed'] += 1

                # Sanitize text
                sanitized_text = self._sanitize_article(text, article_id)

                if not sanitized_text:
                    self.stats['articles_filtered'] += 1
                    continue

                # Tokenize and yield
                tokens = self._tokenize_text(sanitized_text)
                self.stats['tokens_generated'] += self.config.seq_length

                # CRITICAL: Force ALL tensors to CPU before yielding
                # This prevents any accidental GPU allocation in worker processes
                tokens_cpu = {k: v.to('cpu') if torch.is_tensor(v) else v for k, v in tokens.items()}

                yield tokens_cpu

                # CRITICAL: Delete tokens after yielding to prevent memory accumulation
                del tokens, tokens_cpu

                articles_processed += 1

                # Clean up text references
                del sanitized_text, text

                # Log progress periodically
                if articles_processed % 100 == 0:
                    logger.info(f"Worker {worker_id}: Processed {articles_processed} articles, "
                               f"filtered {self.stats['articles_filtered']}, "
                               f"cache hits: {self.stats['cache_hits']}")
        else:
            # Streaming mode - process sequentially (no sharding)
            # WARNING: With num_workers > 1, this will cause duplicates!
            if num_workers > 1:
                logger.warning(f"Streaming mode with num_workers={num_workers} may cause duplicate processing!")

            for article in self.dataset:
                # Check limit
                if self.config.max_articles and articles_processed >= self.config.max_articles:
                    break

                # Extract text
                text, article_id = self._extract_text(article)

                # CRITICAL FIX: Delete article after extraction to free memory
                del article

                if not text:
                    continue

                self.stats['articles_processed'] += 1

                # Sanitize text
                sanitized_text = self._sanitize_article(text, article_id)

                if not sanitized_text:
                    self.stats['articles_filtered'] += 1
                    continue

                # Tokenize and yield
                tokens = self._tokenize_text(sanitized_text)
                self.stats['tokens_generated'] += self.config.seq_length

                # CRITICAL: Force ALL tensors to CPU before yielding
                # This prevents any accidental GPU allocation in worker processes
                tokens_cpu = {k: v.to('cpu') if torch.is_tensor(v) else v for k, v in tokens.items()}

                yield tokens_cpu

                # CRITICAL: Delete tokens after yielding to prevent memory accumulation
                del tokens, tokens_cpu

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


def _worker_init_fn_no_cuda(worker_id):
    """
    CRITICAL: Worker initialization to force CPU-only mode.

    KEY FINDING: Workers start with 0MB GPU memory. The 550MB allocation happens
    LATER during data processing, not during init. This is likely unavoidable
    with num_workers > 0 when torch is already imported in parent.

    Args:
        worker_id: Worker ID assigned by DataLoader
    """
    import os
    import torch
    import logging

    logger = logging.getLogger(__name__)

    # Set environment variable (may not help but try anyway)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Force default device to CPU
    try:
        torch.set_default_device('cpu')
        logger.info(f"Worker {worker_id} - Default device set to CPU")
    except Exception as e:
        logger.warning(f"Worker {worker_id} - Could not set default device: {e}")

    # Override torch.cuda.is_available
    torch.cuda.is_available = lambda: False

    logger.info(f"Worker {worker_id} initialized (CPU-only mode)")


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

    # Create dataloader with multiprocessing support
    # For non-streaming datasets (streaming=False), we implement worker sharding in __iter__
    # For streaming datasets (streaming=True), we force num_workers=0 to avoid duplicates

    if config.streaming and num_workers > 0:
        logger.warning(f"Streaming mode with num_workers={num_workers} may cause duplicates. "
                      f"Forcing num_workers=0 for safety.")
        num_workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # Use provided num_workers for non-streaming
        prefetch_factor=config.prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,  # CRITICAL FIX: Disable to prevent unbounded memory growth
        # persistent_workers=True keeps workers alive between epochs, but their caches grow unbounded
        # With 16 workers, this causes massive RAM usage. We accept the restart overhead.
        pin_memory=False,  # ALWAYS FALSE - data loading happens on CPU!
        multiprocessing_context='spawn' if num_workers > 0 else None,  # Windows compatibility
        worker_init_fn=_worker_init_fn_no_cuda if num_workers > 0 else None,  # CRITICAL: Disable CUDA in workers!
    )

    logger.info(f"DataLoader created with num_workers={num_workers}, "
                f"batch_size={batch_size}, prefetch_factor={config.prefetch_factor if num_workers > 0 else 'N/A'}, "
                f"pin_memory=False (CPU-only data loading), streaming={config.streaming}")

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