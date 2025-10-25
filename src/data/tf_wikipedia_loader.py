"""
High-performance TensorFlow tf.data pipeline for Wikipedia dataset.

This replaces the HuggingFace DataLoader with TensorFlow's optimized data pipeline:
- Proper multiprocessing with tf.data.AUTOTUNE
- Efficient memory-mapped Arrow file reading
- Better prefetching and caching
- Optimal CPU utilization
"""

import tensorflow as tf
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging
from glob import glob

logger = logging.getLogger(__name__)


class TFWikipediaDataset:
    """
    TensorFlow-based Wikipedia dataset loader with optimal multiprocessing.

    Uses tf.data API for high-performance data loading with:
    - Memory-mapped Arrow file reading
    - Proper worker sharding
    - Prefetching and parallelism
    - Zero memory duplication across workers
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        sanitizer,
        seq_length: int = 2048,
        num_parallel_calls: int = 16,
        prefetch_buffer_size: int = 128,
        shuffle_buffer_size: int = 10000,
        batch_size: int = 8,
    ):
        """
        Initialize TensorFlow data pipeline.

        Args:
            data_dir: Directory containing Arrow files
            tokenizer: HuggingFace tokenizer
            sanitizer: WikipediaSanitizer instance
            seq_length: Maximum sequence length
            num_parallel_calls: Number of parallel preprocessing workers (use tf.data.AUTOTUNE)
            prefetch_buffer_size: Number of batches to prefetch
            shuffle_buffer_size: Size of shuffle buffer
            batch_size: Batch size
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.sanitizer = sanitizer
        self.seq_length = seq_length
        self.num_parallel_calls = num_parallel_calls if num_parallel_calls > 0 else tf.data.AUTOTUNE
        self.prefetch_buffer_size = prefetch_buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size

        # Find all Arrow files
        self.arrow_files = self._find_arrow_files()
        logger.info(f"Found {len(self.arrow_files)} Arrow files in {data_dir}")

    def _find_arrow_files(self) -> List[str]:
        """Find all Arrow files in the dataset directory."""
        pattern = str(self.data_dir / "**" / "*.arrow")
        files = glob(pattern, recursive=True)
        if not files:
            raise ValueError(f"No Arrow files found in {self.data_dir}")
        return sorted(files)

    def _read_arrow_file(self, filename: bytes) -> tf.data.Dataset:
        """
        Read a single Arrow file and yield articles.

        This function is called by tf.data.Dataset.interleave() and runs
        in parallel across multiple worker threads.
        """
        filename_str = filename.decode('utf-8')

        def generator():
            # Memory-mapped read of Arrow file
            with pa.memory_map(filename_str, 'r') as source:
                reader = pa.ipc.open_file(source)
                table = reader.read_all()

                # Iterate through rows
                for i in range(len(table)):
                    row = table.slice(i, 1).to_pydict()
                    text = row.get('text', [''])[0]
                    article_id = row.get('id', [''])[0] or row.get('title', [''])[0]

                    # CRITICAL FIX: Delete row dict after extraction
                    del row

                    if text:
                        yield (text, article_id)

                # CRITICAL FIX: Delete table after processing
                del table, reader

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
            )
        )

    def _process_article(self, text: tf.Tensor, article_id: tf.Tensor):
        """
        Process a single article: sanitize, tokenize, create labels.

        This runs in parallel across multiple threads via tf.data.map().
        """
        def process_fn(text_bytes, article_id_bytes):
            text_str = text_bytes.numpy().decode('utf-8')
            article_id_str = article_id_bytes.numpy().decode('utf-8')

            # Sanitize
            if self.sanitizer:
                sanitized = self.sanitizer.sanitize(text_str, article_id_str)
                if sanitized is None:
                    return None
                text_str = sanitized

            # Tokenize
            encoded = self.tokenizer(
                text_str,
                max_length=self.seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='np',  # NumPy for TensorFlow compatibility
            )

            input_ids = encoded['input_ids'][0]
            attention_mask = encoded['attention_mask'][0]

            # Create labels
            labels = input_ids.copy()
            labels[attention_mask == 0] = -100

            # Create MTP labels
            mtp_labels = np.full((self.seq_length, 2), -100, dtype=np.int64)

            if self.seq_length >= 2:
                mtp_labels[:-1, 0] = input_ids[1:]
                current_mask = attention_mask[:-1] == 0
                mtp_labels[:-1, 0][current_mask] = -100
                del current_mask  # CRITICAL FIX: Delete mask

            if self.seq_length >= 3:
                mtp_labels[:-2, 1] = input_ids[2:]
                current_mask = attention_mask[:-2] == 0
                target_mask = attention_mask[2:] == 0
                mtp_labels[:-2, 1][current_mask | target_mask] = -100
                del current_mask, target_mask  # CRITICAL FIX: Delete masks

            result = {
                'input_ids': input_ids.astype(np.int64),
                'attention_mask': attention_mask.astype(np.int64),
                'labels': labels.astype(np.int64),
                'mtp_labels': mtp_labels.astype(np.int64),
            }

            # CRITICAL FIX: Delete intermediate arrays
            del encoded, input_ids, attention_mask, labels, mtp_labels

            return result

        # Use py_function to call Python code
        result = tf.py_function(
            func=process_fn,
            inp=[text, article_id],
            Tout={
                'input_ids': tf.int64,
                'attention_mask': tf.int64,
                'labels': tf.int64,
                'mtp_labels': tf.int64,
            }
        )

        # Set shapes for TensorFlow
        result['input_ids'].set_shape([self.seq_length])
        result['attention_mask'].set_shape([self.seq_length])
        result['labels'].set_shape([self.seq_length])
        result['mtp_labels'].set_shape([self.seq_length, 2])

        return result

    def build_dataset(self, shuffle: bool = True) -> tf.data.Dataset:
        """
        Build optimized TensorFlow dataset with parallelism and prefetching.

        Returns:
            tf.data.Dataset that yields batches ready for PyTorch
        """
        # Create dataset of filenames
        dataset = tf.data.Dataset.from_tensor_slices(self.arrow_files)

        # Shuffle files
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.arrow_files))

        # Interleave: read multiple files in parallel
        # cycle_length controls how many files are read simultaneously
        dataset = dataset.interleave(
            self._read_arrow_file,
            cycle_length=self.num_parallel_calls,
            block_length=1,
            num_parallel_calls=self.num_parallel_calls,
            deterministic=False,  # Allow non-deterministic for better performance
        )

        # Shuffle articles
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        # Process articles in parallel
        dataset = dataset.map(
            self._process_article,
            num_parallel_calls=self.num_parallel_calls,
            deterministic=False,
        )

        # Filter out None results (failed sanitization)
        dataset = dataset.filter(lambda x: x is not None)

        # Batch
        dataset = dataset.batch(self.batch_size, drop_remainder=False)

        # Prefetch batches
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)

        logger.info(f"Built TensorFlow dataset with {self.num_parallel_calls} parallel workers, "
                   f"prefetch={self.prefetch_buffer_size}, shuffle={self.shuffle_buffer_size}")

        return dataset

    def to_pytorch_iterator(self, dataset: tf.data.Dataset):
        """
        Convert TensorFlow dataset to PyTorch-compatible iterator.

        Yields batches as PyTorch tensors.
        """
        for batch in dataset:
            # Convert to NumPy first, then PyTorch
            result = {
                'input_ids': torch.from_numpy(batch['input_ids'].numpy()),
                'attention_mask': torch.from_numpy(batch['attention_mask'].numpy()),
                'labels': torch.from_numpy(batch['labels'].numpy()),
                'mtp_labels': torch.from_numpy(batch['mtp_labels'].numpy()),
            }

            # CRITICAL FIX: Delete TensorFlow batch after conversion
            del batch

            yield result


def create_tf_wikipedia_dataloader(
    data_dir: str,
    tokenizer,
    sanitizer,
    batch_size: int = 8,
    seq_length: int = 2048,
    num_workers: int = 16,
    prefetch_buffer: int = 128,
    shuffle: bool = True,
):
    """
    Create high-performance TensorFlow-based data loader.

    Args:
        data_dir: Path to Wikipedia Arrow files
        tokenizer: HuggingFace tokenizer
        sanitizer: WikipediaSanitizer
        batch_size: Batch size
        seq_length: Maximum sequence length
        num_workers: Number of parallel workers (use tf.data.AUTOTUNE for automatic)
        prefetch_buffer: Number of batches to prefetch
        shuffle: Whether to shuffle data

    Returns:
        Iterator yielding PyTorch tensor batches
    """
    tf_dataset = TFWikipediaDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        sanitizer=sanitizer,
        seq_length=seq_length,
        num_parallel_calls=tf.data.AUTOTUNE if num_workers == 0 else num_workers,
        prefetch_buffer_size=prefetch_buffer,
        batch_size=batch_size,
    )

    dataset = tf_dataset.build_dataset(shuffle=shuffle)

    return tf_dataset.to_pytorch_iterator(dataset)
