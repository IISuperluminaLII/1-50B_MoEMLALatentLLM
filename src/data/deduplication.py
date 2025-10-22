"""
Deduplication utilities for data sanitization.

Implements both MinHash LSH for near-duplicate detection and exact
deduplication using hash-based methods.

References:
    Lee et al. (2022). "Deduplicating Training Data Makes Language Models Better."
    arXiv:2107.06499

    Broder (1997). "On the resemblance and containment of documents."
    Proceedings of the Compression and Complexity of Sequences.
"""
import hashlib
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Iterator, Set


@dataclass
class DeduplicationStats:
    """Statistics from deduplication process."""
    total_documents: int = 0
    unique_documents: int = 0
    duplicate_documents: int = 0
    deduplication_ratio: float = 0.0
    processing_time_seconds: float = 0.0


class MinHashDeduplicator:
    """
    MinHash LSH deduplicator for near-duplicate detection.

    Uses MinHash with Locality-Sensitive Hashing to efficiently
    find near-duplicate documents based on Jaccard similarity.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        n_gram: int = 13,
        seed: int = 42,
    ):
        """
        Initialize MinHash deduplicator.

        Args:
            num_perm: Number of hash functions (permutations)
            threshold: Jaccard similarity threshold for duplicates
            n_gram: Size of character n-grams
            seed: Random seed for hash functions
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.n_gram = n_gram
        self.seed = seed

        # Initialize MinHash if datasketch is available
        try:
            from datasketch import MinHash, MinHashLSH
            self.has_datasketch = True
            self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        except ImportError:
            self.has_datasketch = False
            self.lsh = None
            # Use simple hash-based deduplication as fallback
            self.seen_hashes = set()

        self.stats = None

    def generate_n_grams(self, text: str) -> List[str]:
        """
        Generate character n-grams from text.

        Args:
            text: Input text

        Returns:
            List of n-grams
        """
        if len(text) < self.n_gram:
            return [text]

        return [
            text[i : i + self.n_gram]
            for i in range(len(text) - self.n_gram + 1)
        ]

    def compute_minhash(self, text: str):
        """
        Compute MinHash signature for text.

        Args:
            text: Input text

        Returns:
            MinHash object or hash string
        """
        if self.has_datasketch:
            from datasketch import MinHash

            minhash = MinHash(num_perm=self.num_perm, seed=self.seed)
            for gram in self.generate_n_grams(text):
                minhash.update(gram.encode("utf-8"))
            return minhash
        else:
            # Fallback to simple hash
            n_grams = self.generate_n_grams(text)
            combined = "".join(sorted(n_grams))
            return hashlib.sha256(combined.encode()).hexdigest()

    def compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Compute exact Jaccard similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity (0-1)
        """
        ngrams1 = set(self.generate_n_grams(text1))
        ngrams2 = set(self.generate_n_grams(text2))

        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def estimate_jaccard_similarity(self, minhash1, minhash2) -> float:
        """
        Estimate Jaccard similarity using MinHash signatures.

        Args:
            minhash1: First MinHash signature
            minhash2: Second MinHash signature

        Returns:
            Estimated Jaccard similarity
        """
        if self.has_datasketch:
            return minhash1.jaccard(minhash2)
        else:
            # For fallback, just check exact match
            return 1.0 if minhash1 == minhash2 else 0.0

    def deduplicate(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        return_duplicates: bool = False,
    ) -> Tuple[List[str], List[str], Optional[Dict]]:
        """
        Deduplicate a list of documents using MinHash LSH with Jaccard verification.

        Implements the two-phase deduplication algorithm from Lee et al. (2022):
        1. Query LSH index for candidate duplicates (approximate)
        2. Verify candidates by computing exact Jaccard similarity
        3. Only mark as duplicate if Jaccard >= threshold

        This ensures unrelated documents that merely collide in LSH buckets
        are not incorrectly filtered.

        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs
            return_duplicates: Whether to return duplicate mapping

        Returns:
            Tuple of (unique_docs, unique_ids, duplicate_map)

        References:
            Lee et al. (2022). "Deduplicating Training Data Makes Language Models Better."
            Broder (1997). "On the resemblance and containment of documents."
        """
        start_time = time.time()

        if not doc_ids:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]

        unique_docs = []
        unique_ids = []
        duplicate_map = {} if return_duplicates else None

        if self.has_datasketch:
            # Store document texts for Jaccard verification
            # Maps doc_id -> text for already-indexed documents
            stored_texts = {}

            # Use LSH for efficient near-duplicate detection
            for i, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
                if not doc:
                    continue

                minhash = self.compute_minhash(doc)

                # Phase 1: Query LSH for candidate duplicates (approximate)
                candidates = self.lsh.query(minhash)

                # Phase 2: Verify candidates with exact Jaccard similarity
                is_duplicate = False
                if candidates:
                    for candidate_id in candidates:
                        if candidate_id in stored_texts:
                            # Compute exact Jaccard similarity
                            jaccard_sim = self.compute_jaccard_similarity(
                                doc, stored_texts[candidate_id]
                            )

                            # Only mark as duplicate if similarity >= threshold
                            if jaccard_sim >= self.threshold:
                                is_duplicate = True
                                if return_duplicates:
                                    duplicate_map[doc_id] = candidate_id
                                break

                if not is_duplicate:
                    # Not a duplicate (either no candidates or all below threshold)
                    # Add to index and store text for future verification
                    self.lsh.insert(doc_id, minhash)
                    stored_texts[doc_id] = doc
                    unique_docs.append(doc)
                    unique_ids.append(doc_id)
        else:
            # Fallback to simple hash-based deduplication
            for doc, doc_id in zip(documents, doc_ids):
                if not doc:
                    continue

                doc_hash = self.compute_minhash(doc)

                if doc_hash not in self.seen_hashes:
                    self.seen_hashes.add(doc_hash)
                    unique_docs.append(doc)
                    unique_ids.append(doc_id)
                elif return_duplicates:
                    duplicate_map[doc_id] = "duplicate"

        # Update statistics
        self.stats = DeduplicationStats(
            total_documents=len(documents),
            unique_documents=len(unique_docs),
            duplicate_documents=len(documents) - len(unique_docs),
            deduplication_ratio=(len(documents) - len(unique_docs)) / len(documents)
            if documents
            else 0.0,
            processing_time_seconds=time.time() - start_time,
        )

        return unique_docs, unique_ids, duplicate_map

    def deduplicate_streaming(
        self,
        document_stream: Iterator[Tuple[str, str]],
        batch_size: int = 1000,
    ) -> Iterator[Tuple[str, str]]:
        """
        Deduplicate documents in streaming fashion.

        Args:
            document_stream: Iterator yielding (doc_id, doc_text) tuples
            batch_size: Batch size for processing

        Yields:
            Unique (doc_id, doc_text) tuples
        """
        batch_docs = []
        batch_ids = []

        for doc_id, doc_text in document_stream:
            batch_docs.append(doc_text)
            batch_ids.append(doc_id)

            if len(batch_docs) >= batch_size:
                # Process batch
                unique_docs, unique_ids, _ = self.deduplicate(batch_docs, batch_ids)

                for uid, udoc in zip(unique_ids, unique_docs):
                    yield uid, udoc

                # Clear batch
                batch_docs = []
                batch_ids = []

        # Process remaining documents
        if batch_docs:
            unique_docs, unique_ids, _ = self.deduplicate(batch_docs, batch_ids)
            for uid, udoc in zip(unique_ids, unique_docs):
                yield uid, udoc

    def get_stats(self) -> Optional[DeduplicationStats]:
        """Get deduplication statistics from last run."""
        return self.stats

    def clear(self):
        """Clear internal state and statistics."""
        if self.has_datasketch:
            from datasketch import MinHashLSH
            self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        else:
            self.seen_hashes.clear()
        self.stats = None


class ExactDeduplicator:
    """
    Exact deduplicator using hash-based matching.

    Faster than MinHash but only detects exact duplicates.
    """

    def __init__(self, hash_algorithm: str = "sha256"):
        """
        Initialize exact deduplicator.

        Args:
            hash_algorithm: Hash algorithm to use (md5, sha1, sha256, sha512)
        """
        self.hash_algorithm = hash_algorithm
        self.seen_hashes: Set[str] = set()
        self.stats = None

    def compute_hash(self, text: str) -> str:
        """
        Compute hash of text.

        Args:
            text: Input text

        Returns:
            Hash string
        """
        hash_func = getattr(hashlib, self.hash_algorithm)
        return hash_func(text.encode("utf-8")).hexdigest()

    def deduplicate(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Deduplicate documents using exact matching.

        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs

        Returns:
            Tuple of (unique_docs, unique_ids)
        """
        start_time = time.time()

        if not doc_ids:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]

        unique_docs = []
        unique_ids = []

        for doc, doc_id in zip(documents, doc_ids):
            if not doc:
                continue

            doc_hash = self.compute_hash(doc)

            if doc_hash not in self.seen_hashes:
                self.seen_hashes.add(doc_hash)
                unique_docs.append(doc)
                unique_ids.append(doc_id)

        # Update statistics
        self.stats = DeduplicationStats(
            total_documents=len(documents),
            unique_documents=len(unique_docs),
            duplicate_documents=len(documents) - len(unique_docs),
            deduplication_ratio=(len(documents) - len(unique_docs)) / len(documents)
            if documents
            else 0.0,
            processing_time_seconds=time.time() - start_time,
        )

        return unique_docs, unique_ids

    def clear(self):
        """Clear internal state and statistics."""
        self.seen_hashes.clear()
        self.stats = None