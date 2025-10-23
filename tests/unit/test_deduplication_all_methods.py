"""
Unit tests for all deduplication methods (MinHash, Exact, FED, LSHBloom).

Tests factory function, method-specific behaviors, and paper compliance.
"""

import pytest
import torch

try:
    import cupy  # type: ignore
except ImportError:
    cupy = None

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None
from src.data.deduplication import create_deduplicator, MinHashDeduplicator, ExactDeduplicator


class TestDeduplicationFactory:
    """Test the create_deduplicator factory function."""

    def test_factory_creates_minhash(self):
        """Test factory creates MinHashDeduplicator."""
        dedup = create_deduplicator("minhash", threshold=0.8, num_perm=128)
        assert isinstance(dedup, MinHashDeduplicator)
        assert dedup.threshold == 0.8
        assert dedup.num_perm == 128

    def test_factory_creates_exact(self):
        """Test factory creates ExactDeduplicator."""
        dedup = create_deduplicator("exact")
        assert isinstance(dedup, ExactDeduplicator)

    def test_factory_creates_lshbloom(self):
        """Test factory creates LSHBloomDeduplicator."""
        try:
            dedup = create_deduplicator("lshbloom", threshold=0.8)
            from src.data.deduplication_lshbloom import LSHBloomDeduplicator
            assert isinstance(dedup, LSHBloomDeduplicator)
        except ImportError as e:
            if "pybloom-live" in str(e):
                pytest.skip("pybloom-live not installed")
            else:
                raise

    @pytest.mark.skipif(
        not torch.cuda.is_available() or cupy is None or faiss is None,
        reason="FED requires CUDA, CuPy, and FAISS",
    )
    def test_factory_creates_fed(self):
        """Test factory creates FEDDeduplicator (requires GPU)."""
        try:
            dedup = create_deduplicator("fed", threshold=0.8)
            from src.data.deduplication_fed import FEDDeduplicator
            assert isinstance(dedup, FEDDeduplicator)
        except (ImportError, RuntimeError) as e:
            if "CUDA" in str(e) or "cupy" in str(e) or "torch" in str(e):
                pytest.skip("GPU dependencies not available")
            else:
                raise

    def test_factory_rejects_unknown_method(self):
        """Test factory raises ValueError for unknown methods."""
        with pytest.raises(ValueError, match="Unknown deduplication method"):
            create_deduplicator("invalid_method")


class TestLSHBloomDeduplication:
    """Test LSHBloom memory-efficient deduplication."""

    @pytest.fixture
    def lshbloom_dedup(self):
        """Create LSHBloom deduplicator for testing."""
        try:
            from src.data.deduplication_lshbloom import LSHBloomDeduplicator, LSHBloomConfig
            config = LSHBloomConfig(
                num_bands=20,
                rows_per_band=6,
                similarity_threshold=0.8,
            )
            return LSHBloomDeduplicator(config)
        except ImportError:
            pytest.skip("pybloom-live not installed")

    def test_lshbloom_removes_duplicates(self, lshbloom_dedup):
        """Test LSHBloom removes duplicate documents."""
        docs = [
            "The quick brown fox jumps over the lazy dog. " * 10,  # doc_1
            "The quick brown fox jumps over the lazy dog. " * 10,  # doc_2: exact duplicate
            "A completely different document about AI. " * 10,      # doc_3: unique
        ]
        doc_ids = ["doc_1", "doc_2", "doc_3"]

        unique_docs, unique_ids, _ = lshbloom_dedup.deduplicate(docs, doc_ids)

        # Should remove exact duplicate
        assert len(unique_docs) == 2
        assert "doc_1" in unique_ids
        assert "doc_2" not in unique_ids  # Duplicate removed
        assert "doc_3" in unique_ids

    def test_lshbloom_jaccard_verification(self, lshbloom_dedup):
        """Test LSHBloom performs Jaccard verification per Lee et al. (2022)."""
        base_text = "The quick brown fox jumps over the lazy dog. " * 10

        docs = [
            base_text,
            base_text + "Extra text " * 5,  # Medium similarity
        ]

        # Compute actual Jaccard similarity
        jaccard_sim = lshbloom_dedup.compute_jaccard_similarity(docs[0], docs[1])

        unique_docs, _, _ = lshbloom_dedup.deduplicate(docs)

        # Behavior should depend on Jaccard >= threshold
        if jaccard_sim >= 0.8:
            assert len(unique_docs) == 1, "High similarity should be deduplicated"
        else:
            assert len(unique_docs) == 2, "Low similarity should be kept"

    def test_lshbloom_reports_memory_usage(self, lshbloom_dedup):
        """Test LSHBloom reports memory statistics."""
        docs = ["Document " + str(i) for i in range(100)]

        lshbloom_dedup.deduplicate(docs)
        stats = lshbloom_dedup.get_stats()

        assert stats is not None
        assert stats.bloom_memory_mb > 0, "Should report Bloom filter memory usage"
        assert stats.total_documents == 100


class TestFEDDeduplication:
    """Test FED GPU-accelerated deduplication."""

    @pytest.fixture
    def fed_dedup(self):
        """Create FED deduplicator for testing."""
        pytest.importorskip("torch", reason="PyTorch required for FED")

        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU required for FED")

        try:
            from src.data.deduplication_fed import FEDDeduplicator, FEDConfig
            config = FEDConfig(
                num_hash_functions=128,
                similarity_threshold=0.8,
                batch_size=100,
            )
            return FEDDeduplicator(config)
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"FED dependencies not available: {e}")

    def test_fed_removes_duplicates(self, fed_dedup):
        """Test FED removes duplicate documents on GPU."""
        docs = [
            "The quick brown fox jumps over the lazy dog. " * 10,
            "The quick brown fox jumps over the lazy dog. " * 10,  # Exact duplicate
            "A different document. " * 10,
        ]
        doc_ids = ["doc_1", "doc_2", "doc_3"]

        unique_docs, unique_ids, _ = fed_dedup.deduplicate(docs, doc_ids)

        assert len(unique_docs) == 2
        assert "doc_1" in unique_ids
        assert "doc_2" not in unique_ids
        assert "doc_3" in unique_ids

    def test_fed_uses_gpu(self, fed_dedup):
        """Test FED actually uses GPU for computation."""
        import torch

        # Verify device is GPU
        assert fed_dedup.device.type == "cuda"

        # Verify hash functions are on GPU
        assert fed_dedup.hash_a.device.type == "cuda"
        assert fed_dedup.hash_b.device.type == "cuda"

    def test_fed_reports_gpu_time(self, fed_dedup):
        """Test FED reports GPU timing statistics."""
        docs = ["Document " + str(i) for i in range(50)]

        fed_dedup.deduplicate(docs)
        stats = fed_dedup.get_stats()

        assert stats is not None
        assert stats.gpu_time_seconds > 0, "Should report GPU time"
        assert stats.cpu_time_seconds >= 0
        assert stats.processing_time_seconds > 0


class TestDeduplicationMethodComparison:
    """Test that all methods produce correct deduplication results."""

    @pytest.fixture
    def test_documents(self):
        """Standard test documents for comparing methods."""
        return [
            "The quick brown fox jumps over the lazy dog. " * 10,  # doc_1
            "The quick brown fox jumps over the lazy dog. " * 10,  # doc_2: exact duplicate
            "A completely different document. " * 10,              # doc_3
            "Another unique document. " * 10,                       # doc_4
        ]

    def test_minhash_and_exact_agree_on_exact_duplicates(self, test_documents):
        """Test MinHash and Exact both remove exact duplicates."""
        from src.data.deduplication import MinHashDeduplicator, ExactDeduplicator

        minhash_dedup = MinHashDeduplicator(threshold=0.8)
        exact_dedup = ExactDeduplicator()

        minhash_unique, minhash_ids, _ = minhash_dedup.deduplicate(
            test_documents, [f"doc_{i}" for i in range(len(test_documents))]
        )
        exact_unique, exact_ids = exact_dedup.deduplicate(
            test_documents, [f"doc_{i}" for i in range(len(test_documents))]
        )

        # Both should remove doc_1 (exact duplicate of doc_0)
        assert len(minhash_unique) == 3
        assert len(exact_unique) == 3

    def test_lshbloom_and_minhash_similar_results(self, test_documents):
        """Test LSHBloom produces similar results to MinHash for exact duplicates."""
        try:
            from src.data.deduplication_lshbloom import LSHBloomDeduplicator, LSHBloomConfig
            from src.data.deduplication import MinHashDeduplicator

            lshbloom_dedup = LSHBloomDeduplicator(LSHBloomConfig(similarity_threshold=0.8))
            minhash_dedup = MinHashDeduplicator(threshold=0.8)

            lshbloom_unique, _, _ = lshbloom_dedup.deduplicate(test_documents)
            minhash_unique, _, _ = minhash_dedup.deduplicate(test_documents)

            # Both should remove approximately the same number of duplicates
            assert len(lshbloom_unique) == len(minhash_unique)

        except ImportError:
            pytest.skip("pybloom-live not installed")
