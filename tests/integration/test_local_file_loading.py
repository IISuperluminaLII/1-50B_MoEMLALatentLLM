"""
Tests for local file loading in data pipeline.

Verifies that local files with directory separators (like 'data/raw.jsonl')
are correctly loaded from the filesystem instead of being misinterpreted
as HuggingFace dataset names.

References:
    - Pipeline implementation: src/data/pipeline.py
"""
import pytest
from pathlib import Path
from src.data.pipeline import DataPipeline, PipelineConfig


class TestLocalFileLoading:
    """Test that local files are prioritized over HuggingFace dataset loading."""

    def test_local_file_with_directory_separator(self, tmp_path):
        """
        Test loading a local file with directory separator.

        Previously, paths like 'data/sample.jsonl' would be misclassified
        as HuggingFace datasets due to '/' check happening before file existence.
        """
        # Create a test file in a subdirectory
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "test.jsonl"

        # Write sample data
        with open(test_file, "w", encoding="utf-8") as f:
            f.write('{"text": "Document 1", "id": "1"}\n')
            f.write('{"text": "Document 2", "id": "2"}\n')

        # Configure pipeline to use this local file
        config = PipelineConfig(
            input_path=str(test_file),
            output_dir=str(tmp_path / "output"),
            enable_cleaning=False,
            enable_deduplication=False,
            enable_heuristic_filters=False,
            show_progress=False,
        )

        pipeline = DataPipeline(config)

        # Load data - should NOT attempt HuggingFace dataset loading
        documents = list(pipeline._load_data())

        # Verify correct loading
        assert len(documents) == 2
        assert documents[0]["text"] == "Document 1"
        assert documents[0]["id"] == "1"
        assert documents[1]["text"] == "Document 2"
        assert documents[1]["id"] == "2"

    def test_relative_path_with_slashes(self):
        """
        Test loading from tests/data/sample.jsonl (relative path with slashes).

        This is the exact scenario from the issue description.
        """
        # Use the actual test fixture
        test_file = Path("tests/data/sample.jsonl")
        assert test_file.exists(), "Test fixture tests/data/sample.jsonl not found"

        config = PipelineConfig(
            input_path=str(test_file),
            output_dir="./test_output",
            enable_cleaning=False,
            enable_deduplication=False,
            enable_heuristic_filters=False,
            show_progress=False,
        )

        pipeline = DataPipeline(config)
        documents = list(pipeline._load_data())

        # Should load 3 documents from the fixture
        assert len(documents) == 3
        assert all("text" in doc for doc in documents)
        assert all("id" in doc for doc in documents)

    def test_huggingface_dataset_still_works(self):
        """
        Test that known HuggingFace datasets still load correctly.

        After the fix, whitelisted dataset names should still work.
        """
        config = PipelineConfig(
            input_path="wikitext",  # Known HF dataset
            output_dir="./test_output",
            enable_cleaning=False,
            enable_deduplication=False,
            enable_heuristic_filters=False,
            show_progress=False,
        )

        pipeline = DataPipeline(config)

        # This should attempt HuggingFace loading
        # We won't actually load it (requires network), but we can verify
        # the code path doesn't raise FileNotFoundError immediately
        try:
            # Will attempt HF loading - may fail due to network/permissions
            # but should NOT raise FileNotFoundError for local path
            next(iter(pipeline._load_data()))
        except FileNotFoundError as e:
            pytest.fail(f"Should not raise FileNotFoundError for known HF dataset: {e}")
        except Exception:
            # Other exceptions (network, auth, etc.) are acceptable
            pass

    def test_nonexistent_path_raises_error(self):
        """
        Test that nonexistent paths raise clear error messages.

        Should indicate the path is neither local nor a known HF dataset.
        """
        config = PipelineConfig(
            input_path="nonexistent/path/to/file.jsonl",
            output_dir="./test_output",
            show_progress=False,
        )

        pipeline = DataPipeline(config)

        with pytest.raises(FileNotFoundError) as exc_info:
            list(pipeline._load_data())

        # Error message should be helpful
        assert "neither a local file nor a known HuggingFace dataset" in str(exc_info.value)

    def test_plain_text_file_loading(self, tmp_path):
        """
        Test loading plain text files (non-JSONL).

        Should handle .txt files with directory separators.
        """
        # Create a test text file in subdirectory
        data_dir = tmp_path / "text_data"
        data_dir.mkdir()
        test_file = data_dir / "sample.txt"

        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Line 1\n")
            f.write("Line 2\n")
            f.write("Line 3\n")

        config = PipelineConfig(
            input_path=str(test_file),
            output_dir=str(tmp_path / "output"),
            enable_cleaning=False,
            enable_deduplication=False,
            enable_heuristic_filters=False,
            show_progress=False,
        )

        pipeline = DataPipeline(config)
        documents = list(pipeline._load_data())

        assert len(documents) == 3
        assert documents[0]["text"] == "Line 1"
        assert documents[1]["text"] == "Line 2"
        assert documents[2]["text"] == "Line 3"
        # Auto-generated IDs
        assert documents[0]["id"] == "doc_0"
        assert documents[1]["id"] == "doc_1"
        assert documents[2]["id"] == "doc_2"
