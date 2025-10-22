"""
Data preprocessing pipeline for DeepSeek-V3 training.

Implements the complete SOTA data sanitization pipeline with multiple stages:
1. Preliminary text cleaning
2. Deduplication (MinHash/exact)
3. Heuristic filtering
4. Quality filtering
5. Domain mixing

References:
    Zhou et al. (2025). "A Survey of LLM × DATA."
    arXiv:2505.18458
"""
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Iterator
from tqdm import tqdm


@dataclass
class PipelineConfig:
    """Configuration for data preprocessing pipeline."""

    # Input/Output
    input_path: str
    output_dir: str
    output_format: str = "jsonl"
    save_intermediate: bool = True

    # Stage toggles
    enable_cleaning: bool = True
    enable_deduplication: bool = True
    enable_heuristic_filters: bool = True
    enable_quality_filters: bool = False
    enable_domain_mixing: bool = False

    # Stage configs
    cleaning_config: Dict[str, Any] = field(default_factory=dict)
    dedup_config: Dict[str, Any] = field(default_factory=dict)
    heuristic_config: Dict[str, Any] = field(default_factory=dict)
    quality_config: Dict[str, Any] = field(default_factory=dict)
    domain_config: Dict[str, Any] = field(default_factory=dict)

    # Processing params
    batch_size: int = 10000
    num_workers: int = 1
    show_progress: bool = True
    checkpoint_interval: int = 100000


@dataclass
class PipelineStats:
    """Statistics from pipeline execution."""
    total_input_documents: int = 0
    total_output_documents: int = 0
    documents_cleaned: int = 0
    documents_deduplicated: int = 0
    documents_filtered_heuristic: int = 0
    documents_filtered_quality: int = 0
    processing_time_seconds: float = 0.0

    # Stage-specific stats
    cleaning_stats: Optional[Dict] = None
    dedup_stats: Optional[Dict] = None
    heuristic_stats: Optional[Dict] = None
    quality_stats: Optional[Dict] = None
    domain_stats: Optional[Dict] = None


class DataPipeline:
    """
    Complete data preprocessing pipeline.

    Orchestrates multiple stages of data sanitization including cleaning,
    deduplication, filtering, and domain mixing.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.stats = PipelineStats()

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config.save_intermediate:
            self.intermediate_dir = self.output_dir / "intermediate"
            self.intermediate_dir.mkdir(exist_ok=True)

        # Initialize stages
        self._init_stages()

    def _init_stages(self):
        """Initialize processing stages based on configuration."""
        # Initialize cleaner
        if self.config.enable_cleaning:
            from .preliminary_cleaning import PreliminaryCleaner
            self.cleaner = PreliminaryCleaner(**self.config.cleaning_config)
        else:
            self.cleaner = None

        # Initialize deduplicator
        if self.config.enable_deduplication:
            from .deduplication import MinHashDeduplicator, ExactDeduplicator
            dedup_method = self.config.dedup_config.get("method", "minhash")

            if dedup_method == "minhash":
                self.deduplicator = MinHashDeduplicator(
                    num_perm=self.config.dedup_config.get("num_perm", 128),
                    threshold=self.config.dedup_config.get("threshold", 0.8),
                    n_gram=self.config.dedup_config.get("n_gram", 13),
                    seed=self.config.dedup_config.get("seed", 42),
                )
            elif dedup_method == "exact":
                self.deduplicator = ExactDeduplicator()
            else:
                # Both methods
                self.deduplicator = MinHashDeduplicator(
                    num_perm=self.config.dedup_config.get("num_perm", 128),
                    threshold=self.config.dedup_config.get("threshold", 0.8),
                    n_gram=self.config.dedup_config.get("n_gram", 13),
                    seed=self.config.dedup_config.get("seed", 42),
                )
        else:
            self.deduplicator = None

        # Initialize heuristic filters
        if self.config.enable_heuristic_filters:
            from .heuristic_filters import HeuristicFilter, HeuristicFilterConfig
            self.heuristic_filter = HeuristicFilter(
                HeuristicFilterConfig(**self.config.heuristic_config)
            )
        else:
            self.heuristic_filter = None

        # Initialize quality filter
        if self.config.enable_quality_filters:
            from .quality_filters import FastTextQualityClassifier
            self.quality_filter = FastTextQualityClassifier(
                model_path=self.config.quality_config.get("fasttext_model_path"),
                threshold=self.config.quality_config.get("fasttext_threshold", 0.5),
            )
        else:
            self.quality_filter = None

        # Initialize domain mixer
        if self.config.enable_domain_mixing:
            from .domain_mixing import DomainMixer
            self.domain_mixer = DomainMixer(
                composition=self.config.domain_config.get("composition", "deepseek_v3"),
                identification_method=self.config.domain_config.get("identification_method", "keyword"),
                target_tokens=self.config.domain_config.get("target_tokens"),
                temperature=self.config.domain_config.get("temperature", 0.5),
                num_iterations=self.config.domain_config.get("num_iterations", 1),
                random_seed=self.config.domain_config.get("random_seed", 42),
            )
        else:
            self.domain_mixer = None

    def _load_data(self, input_path: Optional[str] = None) -> Iterator[Dict]:
        """
        Load input data from file or HuggingFace dataset.

        Supports:
        - Local JSONL files
        - Local plain text files
        - HuggingFace dataset names (e.g., "allenai/dolma", "wikitext", "c4")

        Args:
            input_path: Path to input file or HuggingFace dataset name (overrides config)

        Yields:
            Document dictionaries with "text" and optional "id" fields
        """
        path = input_path or self.config.input_path
        path_obj = Path(path)

        # FIRST: Check if local file exists
        if path_obj.exists():
            # Load from local file
            if path_obj.suffix == ".jsonl":
                with open(path_obj, "r", encoding="utf-8") as f:
                    for line in f:
                        yield json.loads(line)
            else:
                # Try to read as plain text
                with open(path_obj, "r", encoding="utf-8") as f:
                    doc_id = 0
                    for line in f:
                        if line.strip():
                            yield {"text": line.strip(), "id": f"doc_{doc_id}"}
                            doc_id += 1
            return  # Early return after local file handling

        # SECOND: Check if it's a known HuggingFace dataset name
        # Whitelist known HF dataset aliases
        known_hf_datasets = [
            "allenai/dolma",
            "wikitext",
            "c4",
            "openwebtext",
            "dolma",
            "the_pile",
            "redpajama",
        ]

        if str(path) in known_hf_datasets:
            # Load from HuggingFace
            from datasets import load_dataset
            dataset = load_dataset(str(path), split="train", streaming=True)
            for example in dataset:
                yield {"text": example.get("text", ""), "id": example.get("id", None)}
            return

        # THIRD: If path wasn't found locally and isn't a known HF dataset, raise error
        raise FileNotFoundError(
            f"Input not found: '{path}' is neither a local file nor a known HuggingFace dataset. "
            f"Known datasets: {', '.join(known_hf_datasets)}"
        )

    def _save_data(self, documents: List[Dict], filename: str):
        """
        Save documents to file.

        Args:
            documents: List of document dictionaries
            filename: Output filename (without extension)
        """
        output_path = self.output_dir / f"{filename}.{self.config.output_format}"

        if self.config.output_format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for doc in documents:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        elif self.config.output_format == "parquet":
            import pandas as pd
            df = pd.DataFrame(documents)
            df.to_parquet(output_path, index=False)
        elif self.config.output_format == "hf_dataset":
            from datasets import Dataset
            dataset = Dataset.from_list(documents)
            dataset.save_to_disk(str(output_path))

    def process_and_save(self, input_data: Optional[List[Dict]] = None) -> PipelineStats:
        """
        Run complete preprocessing pipeline and save results.

        Args:
            input_data: Optional list of documents (for testing). If provided (including
                       empty list), uses in-memory data; if None, streams from disk.

        Returns:
            Pipeline statistics

        Note:
            When input_data is None, the pipeline streams data to maintain O(batch_size)
            memory usage regardless of corpus size. This enables processing multi-TB datasets
            like Dolma without exhausting RAM.
        """
        start_time = time.time()

        # Branch: Test path (in-memory) vs Production path (streaming)
        # Use 'is not None' to allow empty lists for testing
        if input_data is not None:
            # IN-MEMORY PATH: For testing with small datasets
            return self._process_and_save_inmemory(input_data, start_time)
        else:
            # STREAMING PATH: For production with large datasets
            return self._process_and_save_streaming(start_time)

    def _process_and_save_inmemory(self, documents: List[Dict], start_time: float) -> PipelineStats:
        """
        Process data in-memory (for testing and small datasets).

        This path maintains backward compatibility with existing tests that pass
        in-memory lists and expect list-based behavior.
        """
        self.stats.total_input_documents = len(documents)

        # Stage 1: Preliminary cleaning
        if self.config.enable_cleaning and self.cleaner:
            if self.config.show_progress:
                print("\nStage 1/5: Preliminary Cleaning")
                documents = [
                    {"text": self.cleaner.clean(doc["text"]), "id": doc.get("id")}
                    for doc in tqdm(documents, desc="Cleaning")
                ]
            else:
                documents = [
                    {"text": self.cleaner.clean(doc["text"]), "id": doc.get("id")}
                    for doc in documents
                ]

            self.stats.documents_cleaned = len(documents)

            if self.config.save_intermediate:
                self._save_data(documents, "intermediate/01_cleaned")
        else:
            # If cleaning is disabled, set documents_cleaned to input count
            self.stats.documents_cleaned = len(documents)

        # Stage 2: Deduplication
        if self.config.enable_deduplication and self.deduplicator:
            if self.config.show_progress:
                print("\nStage 2/5: Deduplication")

            # Record pre-deduplication count (use cleaned count if available, else current)
            # This ensures correct statistics even when cleaning is disabled
            pre_dedup_count = self.stats.documents_cleaned if self.config.enable_cleaning else len(documents)

            texts = [doc["text"] for doc in documents]
            doc_ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]

            unique_texts, unique_ids, _ = self.deduplicator.deduplicate(
                texts, doc_ids, return_duplicates=True
            )

            # Recreate documents with unique texts
            documents = [
                {"text": text, "id": doc_id}
                for text, doc_id in zip(unique_texts, unique_ids)
            ]

            # Calculate deduplicated count using pre-dedup count (not documents_cleaned)
            self.stats.documents_deduplicated = pre_dedup_count - len(documents)

            if self.config.save_intermediate:
                self._save_data(documents, "intermediate/02_deduplicated")

        # Stage 3: Heuristic filtering
        if self.config.enable_heuristic_filters and self.heuristic_filter:
            if self.config.show_progress:
                print("\nStage 3/5: Heuristic Filtering")

            # Apply heuristic filters
            filtered_docs = []
            for doc in tqdm(documents, desc="Heuristic filtering", disable=not self.config.show_progress):
                if self.heuristic_filter.filter(doc["text"]):
                    filtered_docs.append(doc)

            self.stats.documents_filtered_heuristic = len(documents) - len(filtered_docs)
            self.stats.heuristic_stats = self.heuristic_filter.get_stats().__dict__
            documents = filtered_docs

            if self.config.save_intermediate:
                self._save_data(documents, "intermediate/03_heuristic_filtered")

        # Stage 4: Quality filtering
        if self.config.enable_quality_filters and self.quality_filter:
            if self.config.show_progress:
                print("\nStage 4/5: Quality Filtering")

            filtered_docs = []
            for doc in tqdm(documents, desc="Quality filtering", disable=not self.config.show_progress):
                if self.quality_filter.predict(doc["text"]):
                    filtered_docs.append(doc)

            self.stats.documents_filtered_quality = len(documents) - len(filtered_docs)
            documents = filtered_docs

            if self.config.save_intermediate:
                self._save_data(documents, "intermediate/04_quality_filtered")

        # Stage 5: Domain mixing
        if self.config.enable_domain_mixing and self.domain_mixer:
            if self.config.show_progress:
                print("\nStage 5/5: Domain Mixing")

            # Apply domain mixing to balance dataset composition
            target_size = self.config.domain_config.get("target_size")
            documents = self.domain_mixer.mix_documents(
                documents,
                target_size=target_size,
            )

            # Collect domain statistics
            self.stats.domain_stats = self.domain_mixer.get_statistics()

            if self.config.save_intermediate:
                self._save_data(documents, "intermediate/05_domain_mixed")

                # Save domain mixing statistics
                stats_file = self.intermediate_dir / "domain_mixing_stats.json"
                self.domain_mixer.save_statistics(stats_file)

        # Save final output
        self._save_data(documents, "final")

        # Update statistics
        self.stats.total_output_documents = len(documents)
        self.stats.processing_time_seconds = time.time() - start_time

        # Save statistics
        stats_path = self.output_dir / "pipeline_stats.json"
        with open(stats_path, "w") as f:
            json.dump(asdict(self.stats), f, indent=2)

        if self.config.show_progress:
            print("\n" + "="*80)
            print("Pipeline Statistics:")
            print(f"  Input documents:  {self.stats.total_input_documents:,}")
            print(f"  Output documents: {self.stats.total_output_documents:,}")
            print(f"  Documents removed: {self.stats.total_input_documents - self.stats.total_output_documents:,}")
            print(f"  Processing time: {self.stats.processing_time_seconds:.2f}s")
            print("="*80)

        return self.stats

    def _process_and_save_streaming(self, start_time: float) -> PipelineStats:
        """
        Process data in streaming fashion (for production with large datasets).

        This path streams through the pipeline, maintaining O(batch_size) memory usage.
        Enables processing multi-TB corpora like Dolma without exhausting RAM.

        Note:
            Domain mixing is disabled in streaming mode as it requires knowing the
            full dataset distribution. Use in-memory mode for domain mixing.
        """
        if self.config.enable_domain_mixing:
            raise ValueError(
                "Domain mixing requires full dataset visibility and is not supported in "
                "streaming mode. Either:\n"
                "  1. Disable domain mixing for large-scale streaming\n"
                "  2. Use in-memory mode (pass input_data parameter) for small datasets\n"
                "  3. Pre-process in batches and apply domain mixing separately"
            )

        # Initialize counters (we don't know total count upfront in streaming mode)
        input_count = 0
        cleaned_count = 0
        dedup_input_count = 0
        dedup_output_count = 0
        heuristic_input_count = 0
        heuristic_output_count = 0
        quality_input_count = 0
        quality_output_count = 0

        # Open output file for streaming writes
        output_path = self.output_dir / f"final.{self.config.output_format}"

        if self.config.output_format != "jsonl":
            # For non-JSONL formats, we need to buffer in memory (can't stream write)
            # This is a limitation of Parquet/HF datasets that require full dataset
            buffered_docs = []

        # Setup intermediate output files
        intermediate_files = {}
        if self.config.save_intermediate:
            if self.config.enable_cleaning:
                intermediate_files['cleaned'] = open(
                    self.intermediate_dir / "01_cleaned.jsonl", "w", encoding="utf-8"
                )
            if self.config.enable_deduplication:
                intermediate_files['dedup'] = open(
                    self.intermediate_dir / "02_deduplicated.jsonl", "w", encoding="utf-8"
                )
            if self.config.enable_heuristic_filters:
                intermediate_files['heuristic'] = open(
                    self.intermediate_dir / "03_heuristic_filtered.jsonl", "w", encoding="utf-8"
                )
            if self.config.enable_quality_filters:
                intermediate_files['quality'] = open(
                    self.intermediate_dir / "04_quality_filtered.jsonl", "w", encoding="utf-8"
                )

        try:
            # Open final output for JSONL streaming writes
            if self.config.output_format == "jsonl":
                final_output = open(output_path, "w", encoding="utf-8")

            # Create progress bar for input stream
            doc_stream = self._load_data()
            if self.config.show_progress:
                print("\nProcessing stream (count unknown until complete)...")
                doc_stream = tqdm(doc_stream, desc="Documents")

            # Stream through pipeline stages
            for doc in doc_stream:
                input_count += 1

                # Stage 1: Preliminary cleaning
                if self.config.enable_cleaning and self.cleaner:
                    doc = {"text": self.cleaner.clean(doc["text"]), "id": doc.get("id")}
                    cleaned_count += 1

                    if self.config.save_intermediate and 'cleaned' in intermediate_files:
                        intermediate_files['cleaned'].write(
                            json.dumps(doc, ensure_ascii=False) + "\n"
                        )
                else:
                    cleaned_count += 1

                # Stage 2: Deduplication (stateful - maintains hash index)
                if self.config.enable_deduplication and self.deduplicator:
                    dedup_input_count += 1

                    # Check if this document is a duplicate
                    text = doc["text"]
                    doc_id = doc.get("id", f"doc_{input_count}")

                    if self.has_datasketch_dedup():
                        minhash = self.deduplicator.compute_minhash(text)
                        result = self.deduplicator.lsh.query(minhash)

                        if result:
                            # Duplicate found, skip this document
                            continue
                        else:
                            # Not a duplicate, add to index
                            self.deduplicator.lsh.insert(doc_id, minhash)
                    else:
                        # Fallback: exact hash deduplication
                        doc_hash = self.deduplicator.compute_minhash(text)
                        if doc_hash in self.deduplicator.seen_hashes:
                            # Duplicate found, skip
                            continue
                        else:
                            self.deduplicator.seen_hashes.add(doc_hash)

                    dedup_output_count += 1

                    if self.config.save_intermediate and 'dedup' in intermediate_files:
                        intermediate_files['dedup'].write(
                            json.dumps(doc, ensure_ascii=False) + "\n"
                        )

                # Stage 3: Heuristic filtering
                if self.config.enable_heuristic_filters and self.heuristic_filter:
                    heuristic_input_count += 1

                    if not self.heuristic_filter.filter(doc["text"]):
                        # Document filtered out
                        continue

                    heuristic_output_count += 1

                    if self.config.save_intermediate and 'heuristic' in intermediate_files:
                        intermediate_files['heuristic'].write(
                            json.dumps(doc, ensure_ascii=False) + "\n"
                        )
                else:
                    heuristic_output_count += 1

                # Stage 4: Quality filtering
                if self.config.enable_quality_filters and self.quality_filter:
                    quality_input_count += 1

                    if not self.quality_filter.predict(doc["text"]):
                        # Document filtered out
                        continue

                    quality_output_count += 1

                    if self.config.save_intermediate and 'quality' in intermediate_files:
                        intermediate_files['quality'].write(
                            json.dumps(doc, ensure_ascii=False) + "\n"
                        )
                else:
                    quality_output_count += 1

                # Write to final output
                if self.config.output_format == "jsonl":
                    final_output.write(json.dumps(doc, ensure_ascii=False) + "\n")
                else:
                    buffered_docs.append(doc)

            # Close all files
            for f in intermediate_files.values():
                f.close()

            if self.config.output_format == "jsonl":
                final_output.close()
            else:
                # Write buffered data for non-JSONL formats
                if self.config.output_format == "parquet":
                    import pandas as pd
                    df = pd.DataFrame(buffered_docs)
                    df.to_parquet(output_path, index=False)
                elif self.config.output_format == "hf_dataset":
                    from datasets import Dataset
                    dataset = Dataset.from_list(buffered_docs)
                    dataset.save_to_disk(str(output_path))
                quality_output_count = len(buffered_docs)

        finally:
            # Ensure all files are closed even on error
            for f in intermediate_files.values():
                if not f.closed:
                    f.close()
            if self.config.output_format == "jsonl" and 'final_output' in locals() and not final_output.closed:
                final_output.close()

        # Update statistics
        self.stats.total_input_documents = input_count
        self.stats.documents_cleaned = cleaned_count

        # Calculate filtered counts
        if self.config.enable_deduplication:
            self.stats.documents_deduplicated = dedup_input_count - dedup_output_count

        if self.config.enable_heuristic_filters:
            self.stats.documents_filtered_heuristic = heuristic_input_count - heuristic_output_count
            self.stats.heuristic_stats = self.heuristic_filter.get_stats().__dict__

        if self.config.enable_quality_filters:
            self.stats.documents_filtered_quality = quality_input_count - quality_output_count

        self.stats.total_output_documents = quality_output_count
        self.stats.processing_time_seconds = time.time() - start_time

        # Save statistics
        stats_path = self.output_dir / "pipeline_stats.json"
        with open(stats_path, "w") as f:
            json.dump(asdict(self.stats), f, indent=2)

        if self.config.show_progress:
            print("\n" + "="*80)
            print("Streaming Pipeline Statistics:")
            print(f"  Input documents:  {self.stats.total_input_documents:,}")
            print(f"  Output documents: {self.stats.total_output_documents:,}")
            print(f"  Documents removed: {self.stats.total_input_documents - self.stats.total_output_documents:,}")
            if self.config.enable_deduplication:
                print(f"  Duplicates removed: {self.stats.documents_deduplicated:,}")
            if self.config.enable_heuristic_filters:
                print(f"  Heuristic filtered: {self.stats.documents_filtered_heuristic:,}")
            if self.config.enable_quality_filters:
                print(f"  Quality filtered: {self.stats.documents_filtered_quality:,}")
            print(f"  Processing time: {self.stats.processing_time_seconds:.2f}s")
            print("="*80)

        return self.stats

    def has_datasketch_dedup(self) -> bool:
        """Check if deduplicator has datasketch library."""
        return (
            self.deduplicator is not None and
            hasattr(self.deduplicator, 'has_datasketch') and
            self.deduplicator.has_datasketch
        )