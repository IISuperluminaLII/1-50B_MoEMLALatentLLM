"""
Unit tests for data preprocessing pipeline.

Tests the DataPipeline class and its various stages including cleaning,
deduplication, heuristic filtering, and quality filtering.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data.pipeline import (
    DataPipeline,
    PipelineConfig,
    PipelineStats,
)


class TestPipelineDeduplicationStats:
    """Test deduplication statistics calculation."""

    def test_deduplication_stats_without_cleaning(self):
        """
        Regression test: Verify deduplication stats are non-negative when cleaning is disabled.

        This test ensures that when preliminary cleaning is skipped, the deduplication
        statistics are calculated correctly using the input document count as the baseline,
        preventing negative values that misreport the dedup stage.

        Reference: Pipeline monitoring goals for preprocessing pipeline.
        """
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup config with cleaning disabled, deduplication enabled
            config = PipelineConfig(
                input_path="dummy.jsonl",  # Won't be used since we pass input_data
                output_dir=tmpdir,
                enable_cleaning=False,  # Cleaning disabled
                enable_deduplication=True,  # Deduplication enabled
                enable_heuristic_filters=False,
                enable_quality_filters=False,
                enable_domain_mixing=False,
                show_progress=False,  # Disable progress bars for cleaner test output
            )

            # Create pipeline
            pipeline = DataPipeline(config)

            # Create test data with duplicates
            # 5 documents total: 3 unique, 2 duplicates
            input_data = [
                {"text": "Document one content here", "id": "doc_1"},
                {"text": "Document two content here", "id": "doc_2"},
                {"text": "Document one content here", "id": "doc_3"},  # Duplicate of doc_1
                {"text": "Document three unique content", "id": "doc_4"},
                {"text": "Document two content here", "id": "doc_5"},  # Duplicate of doc_2
            ]

            # Run pipeline
            stats = pipeline.process_and_save(input_data=input_data)

            # Assertion 1: documents_deduplicated must be non-negative
            assert stats.documents_deduplicated >= 0, \
                f"Deduplication count should be non-negative, got {stats.documents_deduplicated}"

            # Assertion 2: documents_deduplicated should match actual duplicate count
            # We have 2 duplicates, so should see 2 documents removed
            assert stats.documents_deduplicated == 2, \
                f"Expected 2 documents deduplicated, got {stats.documents_deduplicated}"

            # Assertion 3: Total removed should equal deduplicated (no other filters)
            total_removed = stats.total_input_documents - stats.total_output_documents
            assert total_removed == stats.documents_deduplicated, \
                f"Total removed ({total_removed}) should equal deduplicated ({stats.documents_deduplicated})"

            # Verify input/output counts
            assert stats.total_input_documents == 5
            assert stats.total_output_documents == 3  # 3 unique documents

    def test_deduplication_stats_with_cleaning_enabled(self):
        """
        Test that deduplication stats are correct when cleaning is enabled.

        This ensures the fix doesn't break the normal case where cleaning is enabled.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup config with both cleaning and deduplication enabled
            config = PipelineConfig(
                input_path="dummy.jsonl",
                output_dir=tmpdir,
                enable_cleaning=True,  # Cleaning enabled
                enable_deduplication=True,  # Deduplication enabled
                enable_heuristic_filters=False,
                enable_quality_filters=False,
                enable_domain_mixing=False,
                show_progress=False,
            )

            # Create pipeline
            pipeline = DataPipeline(config)

            # Create test data with duplicates
            input_data = [
                {"text": "Document one", "id": "doc_1"},
                {"text": "Document two", "id": "doc_2"},
                {"text": "Document one", "id": "doc_3"},  # Duplicate
                {"text": "Document three", "id": "doc_4"},
            ]

            # Run pipeline
            stats = pipeline.process_and_save(input_data=input_data)

            # Verify stats are sensible
            assert stats.documents_deduplicated >= 0
            assert stats.documents_cleaned == 4  # All docs pass through cleaning
            assert stats.total_input_documents == 4
            assert stats.total_output_documents == 3  # 1 duplicate removed

    def test_no_duplicates_zero_dedup_count(self):
        """Test that zero duplicates results in zero deduplication count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                input_path="dummy.jsonl",
                output_dir=tmpdir,
                enable_cleaning=False,
                enable_deduplication=True,
                enable_heuristic_filters=False,
                enable_quality_filters=False,
                enable_domain_mixing=False,
                show_progress=False,
            )

            pipeline = DataPipeline(config)

            # All unique documents
            input_data = [
                {"text": "Unique document one", "id": "doc_1"},
                {"text": "Unique document two", "id": "doc_2"},
                {"text": "Unique document three", "id": "doc_3"},
            ]

            stats = pipeline.process_and_save(input_data=input_data)

            # Should have zero deduplications
            assert stats.documents_deduplicated == 0
            assert stats.total_input_documents == stats.total_output_documents

    def test_all_duplicates(self):
        """Test edge case where all documents are duplicates of first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                input_path="dummy.jsonl",
                output_dir=tmpdir,
                enable_cleaning=False,
                enable_deduplication=True,
                enable_heuristic_filters=False,
                enable_quality_filters=False,
                enable_domain_mixing=False,
                show_progress=False,
            )

            pipeline = DataPipeline(config)

            # All same text
            input_data = [
                {"text": "Same document", "id": f"doc_{i}"}
                for i in range(5)
            ]

            stats = pipeline.process_and_save(input_data=input_data)

            # Should deduplicate 4 documents (keeping 1)
            assert stats.documents_deduplicated == 4
            assert stats.total_output_documents == 1


class TestPipelineStatsConsistency:
    """Test that pipeline statistics remain consistent across different configurations."""

    def test_stats_consistency_no_stages(self):
        """Test stats when all stages are disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                input_path="dummy.jsonl",
                output_dir=tmpdir,
                enable_cleaning=False,
                enable_deduplication=False,
                enable_heuristic_filters=False,
                enable_quality_filters=False,
                enable_domain_mixing=False,
                show_progress=False,
            )

            pipeline = DataPipeline(config)
            input_data = [
                {"text": "Doc 1", "id": "1"},
                {"text": "Doc 2", "id": "2"},
            ]

            stats = pipeline.process_and_save(input_data=input_data)

            # With no processing stages, input should equal output
            assert stats.total_input_documents == stats.total_output_documents
            assert stats.documents_deduplicated == 0
            assert stats.documents_filtered_heuristic == 0
            assert stats.documents_filtered_quality == 0

    def test_empty_input(self):
        """Test handling of empty input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                input_path="dummy.jsonl",
                output_dir=tmpdir,
                enable_cleaning=False,
                enable_deduplication=True,
                enable_heuristic_filters=False,
                enable_quality_filters=False,
                enable_domain_mixing=False,
                show_progress=False,
            )

            pipeline = DataPipeline(config)
            input_data = []

            stats = pipeline.process_and_save(input_data=input_data)

            # All counts should be zero
            assert stats.total_input_documents == 0
            assert stats.total_output_documents == 0
            assert stats.documents_deduplicated == 0


class TestDeduplicationMethods:
    """Test different deduplication methods (minhash, exact, both)."""

    def test_deduplication_method_both(self):
        """
        Test that method='both' runs both MinHash and Exact deduplication.

        This ensures the pipeline correctly applies near-duplicate removal (MinHash)
        followed by exact-duplicate removal (SHA-256), as documented.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                input_path="dummy.jsonl",
                output_dir=tmpdir,
                enable_cleaning=False,
                enable_deduplication=True,
                dedup_config={
                    "method": "both",  # Run both MinHash and Exact
                    "threshold": 0.8,
                    "num_perm": 128,
                    "n_gram": 13,
                    "seed": 42,
                },
                enable_heuristic_filters=False,
                enable_quality_filters=False,
                enable_domain_mixing=False,
                show_progress=False,
            )

            pipeline = DataPipeline(config)

            # Create test data with:
            # 1. Near-duplicates (similar but not exact)
            # 2. Exact duplicates
            # 3. Unique documents
            input_data = [
                {"text": "The quick brown fox jumps over the lazy dog.", "id": "doc_1"},
                {"text": "The quick brown fox jumps over the lazy dog.", "id": "doc_2"},  # Exact duplicate of doc_1
                {"text": "The quick brown fox jumps over a lazy dog.", "id": "doc_3"},   # Near-duplicate (threshold-dependent)
                {"text": "Completely different unique content here.", "id": "doc_4"},
                {"text": "Another totally unique document content.", "id": "doc_5"},
                {"text": "Another totally unique document content.", "id": "doc_6"},     # Exact duplicate of doc_5
            ]

            stats = pipeline.process_and_save(input_data=input_data)

            # Verify both deduplicators were instantiated
            assert pipeline.deduplicator is not None, "MinHash deduplicator should be initialized"
            assert pipeline.exact_deduplicator is not None, "Exact deduplicator should be initialized for 'both' method"

            # Verify some documents were removed
            assert stats.documents_deduplicated > 0, "Should have removed some duplicates"
            assert stats.total_output_documents < stats.total_input_documents

            # At minimum, the 2 exact duplicates should be removed
            assert stats.documents_deduplicated >= 2, f"Should remove at least 2 exact duplicates, got {stats.documents_deduplicated}"

    def test_deduplication_method_minhash_only(self):
        """Test that method='minhash' only uses MinHash deduplication."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                input_path="dummy.jsonl",
                output_dir=tmpdir,
                enable_cleaning=False,
                enable_deduplication=True,
                dedup_config={"method": "minhash"},
                enable_heuristic_filters=False,
                enable_quality_filters=False,
                enable_domain_mixing=False,
                show_progress=False,
            )

            pipeline = DataPipeline(config)

            # Verify only MinHash deduplicator is initialized
            assert pipeline.deduplicator is not None
            assert pipeline.exact_deduplicator is None

    def test_deduplication_method_exact_only(self):
        """Test that method='exact' only uses exact deduplication."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                input_path="dummy.jsonl",
                output_dir=tmpdir,
                enable_cleaning=False,
                enable_deduplication=True,
                dedup_config={"method": "exact"},
                enable_heuristic_filters=False,
                enable_quality_filters=False,
                enable_domain_mixing=False,
                show_progress=False,
            )

            pipeline = DataPipeline(config)

            # Verify only Exact deduplicator is used
            assert pipeline.deduplicator is not None
            assert pipeline.exact_deduplicator is None

            # Also verify it's actually an ExactDeduplicator instance
            from src.data.deduplication import ExactDeduplicator
            assert isinstance(pipeline.deduplicator, ExactDeduplicator)

    def test_deduplication_method_invalid(self):
        """Test that invalid deduplication method raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                input_path="dummy.jsonl",
                output_dir=tmpdir,
                enable_cleaning=False,
                enable_deduplication=True,
                dedup_config={"method": "invalid_method"},
                enable_heuristic_filters=False,
                enable_quality_filters=False,
                enable_domain_mixing=False,
                show_progress=False,
            )

            # Should raise ValueError when initializing pipeline
            with pytest.raises(ValueError, match="Unknown deduplication method"):
                pipeline = DataPipeline(config)


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig(
            input_path="test.jsonl",
            output_dir="output"
        )

        # Check defaults
        assert config.output_format == "jsonl"
        assert config.save_intermediate is True
        assert config.enable_cleaning is True
        assert config.enable_deduplication is True
        assert config.enable_heuristic_filters is True
        assert config.enable_quality_filters is False
        assert config.enable_domain_mixing is False
        assert config.batch_size == 10000
        assert config.num_workers == 1
        assert config.show_progress is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            input_path="custom.jsonl",
            output_dir="custom_output",
            output_format="parquet",
            enable_cleaning=False,
            batch_size=5000,
            num_workers=4,
        )

        assert config.output_format == "parquet"
        assert config.enable_cleaning is False
        assert config.batch_size == 5000
        assert config.num_workers == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
