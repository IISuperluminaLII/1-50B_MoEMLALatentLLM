"""
Unit tests for deduplication compliance with cited papers.

Tests that deduplication implementations properly enforce requirements
from cited academic papers, particularly Lee et al. (2022) for MinHash.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


class TestMinHashCompliance:
    """
    Test MinHashDeduplicator compliance with Lee et al. (2022).

    Reference:
        Lee et al. (2022). "Deduplicating Training Data Makes Language Models Better."
        arXiv:2107.06499
    """

    def test_minhash_requires_datasketch(self):
        """
        Test that MinHashDeduplicator fails without datasketch library.

        This ensures the Lee et al. (2022) compliant two-phase MinHash LSH +
        Jaccard verification algorithm is enforced and cannot silently fall back
        to a non-compliant hash-based method.
        """
        # Mock datasketch import failure
        with patch.dict('sys.modules', {'datasketch': None}):
            # Force reimport to trigger ImportError
            import importlib
            from src.data import deduplication
            importlib.reload(deduplication)

            # Attempt to create MinHashDeduplicator should raise ImportError
            with pytest.raises(ImportError, match="requires 'datasketch' library"):
                deduplication.MinHashDeduplicator(threshold=0.8)

    def test_minhash_with_datasketch_installed(self):
        """
        Test normal operation when datasketch is properly installed.

        Verifies that MinHashDeduplicator initializes correctly and performs
        Lee et al. (2022) compliant deduplication.
        """
        from src.data.deduplication import MinHashDeduplicator

        # Should initialize without error
        dedup = MinHashDeduplicator(
            num_perm=128,
            threshold=0.8,
            n_gram=13,
            seed=42
        )

        # Verify LSH index is initialized
        assert dedup.lsh is not None, "LSH index should be initialized"
        assert dedup.threshold == 0.8
        assert dedup.num_perm == 128

        # Test deduplication works correctly
        docs = [
            "The quick brown fox jumps over the lazy dog. " * 10,  # doc_1
            "The quick brown fox jumps over the lazy dog. " * 10,  # doc_2: exact duplicate
            "A completely different document about AI. " * 10,      # doc_3: unique
        ]
        doc_ids = ["doc_1", "doc_2", "doc_3"]

        unique_docs, unique_ids, _ = dedup.deduplicate(docs, doc_ids)

        # Should remove exact duplicate (doc_2)
        assert len(unique_docs) == 2, f"Expected 2 unique documents, got {len(unique_docs)}"
        assert "doc_1" in unique_ids, "Original document should be kept"
        assert "doc_2" not in unique_ids, "Exact duplicate should be removed"
        assert "doc_3" in unique_ids, "Unique document should be kept"

    def test_minhash_jaccard_verification_required(self):
        """
        Test that MinHash performs Jaccard verification, not just LSH matching.

        This is critical for Lee et al. (2022) compliance - LSH candidates
        must be verified with exact Jaccard similarity to avoid false positives.
        """
        from src.data.deduplication import MinHashDeduplicator

        dedup = MinHashDeduplicator(
            num_perm=128,
            threshold=0.8,
            n_gram=13,
            seed=42
        )

        # Create documents with varying Jaccard similarities
        base_text = "The quick brown fox jumps over the lazy dog. " * 10

        docs = [
            base_text,  # doc_1
            base_text + "Extra text " * 5,  # doc_2: medium similarity
        ]

        # Compute actual Jaccard similarity
        jaccard_sim = dedup.compute_jaccard_similarity(docs[0], docs[1])

        unique_docs, unique_ids, _ = dedup.deduplicate(docs, ["doc_1", "doc_2"])

        # Behavior should depend on whether Jaccard >= threshold
        if jaccard_sim >= 0.8:
            # High similarity - should be removed as duplicate
            assert len(unique_docs) == 1, \
                f"Documents with Jaccard={jaccard_sim:.3f} >= 0.8 should be deduplicated"
        else:
            # Low similarity - should be kept
            assert len(unique_docs) == 2, \
                f"Documents with Jaccard={jaccard_sim:.3f} < 0.8 should both be kept"

    def test_importerror_message_clarity(self):
        """
        Test that ImportError provides clear instructions for installing datasketch.

        Users should get actionable guidance, not a cryptic error message.
        """
        with patch.dict('sys.modules', {'datasketch': None}):
            import importlib
            from src.data import deduplication
            importlib.reload(deduplication)

            try:
                deduplication.MinHashDeduplicator()
                pytest.fail("Should have raised ImportError")
            except ImportError as e:
                error_msg = str(e)

                # Verify error message contains helpful information
                assert "datasketch" in error_msg.lower(), \
                    "Error should mention datasketch library"
                assert "pip install" in error_msg.lower(), \
                    "Error should provide install command"
                assert "Lee et al." in error_msg or "2022" in error_msg, \
                    "Error should reference the paper for context"
                assert "compliance" in error_msg.lower() or "algorithm" in error_msg.lower(), \
                    "Error should explain why datasketch is required"


class TestExactDeduplicatorIndependence:
    """
    Test that ExactDeduplicator has no external dependencies.

    Unlike MinHash, exact deduplication should work with only stdlib.
    """

    def test_exact_deduplicator_no_dependencies(self):
        """Test ExactDeduplicator works without any external libraries."""
        from src.data.deduplication import ExactDeduplicator

        # Should initialize without any external dependencies
        dedup = ExactDeduplicator(hash_algorithm="sha256")

        # Test deduplication works
        docs = [
            "Document A",
            "Document A",  # Exact duplicate
            "Document B",
        ]
        doc_ids = ["doc_1", "doc_2", "doc_3"]

        unique_docs, unique_ids = dedup.deduplicate(docs, doc_ids)

        assert len(unique_docs) == 2, "Should remove exact duplicate"
        assert "doc_1" in unique_ids
        assert "doc_2" not in unique_ids  # Duplicate removed
        assert "doc_3" in unique_ids
