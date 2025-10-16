"""
Data preprocessing configuration.

Defines configuration schema for the data sanitization pipeline.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from ..data.pipeline import PipelineConfig


@dataclass
class CleaningConfig:
    """Configuration for preliminary cleaning stage."""
    enabled: bool = True
    unicode_normalization: str = "NFKC"
    fix_encoding: bool = True
    unescape_html: bool = True
    remove_control_chars: bool = True
    normalize_whitespace: bool = True


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication stage."""
    enabled: bool = True
    method: str = "minhash"  # minhash, exact, or both
    threshold: float = 0.8  # Jaccard threshold for MinHash
    num_perm: int = 128  # Number of hash functions
    n_gram: int = 13  # N-gram size (Gopher uses 13)
    seed: int = 42


@dataclass
class HeuristicFilterConfig:
    """Configuration for heuristic filtering stage."""
    enabled: bool = True

    # Document length filters
    min_doc_length: int = 200  # Gopher threshold
    max_doc_length: Optional[int] = None
    min_word_count: int = 50  # Gopher threshold
    max_word_count: Optional[int] = None

    # Word-level filters
    min_mean_word_length: float = 3.0  # Gopher threshold
    max_mean_word_length: float = 10.0  # Gopher threshold

    # Character-level filters
    min_alpha_ratio: float = 0.5  # C4 threshold
    max_digit_ratio: float = 0.3  # RefinedWeb threshold
    max_uppercase_ratio: float = 0.3  # C4 threshold
    max_special_char_ratio: float = 0.3

    # Repetition filters
    max_repetition_ratio: float = 0.2  # FineWeb threshold
    max_ellipsis_count: float = 30.0  # Per 100 words, Gopher
    max_bullet_count: float = 20.0  # Per 100 words

    # Natural language indicators
    min_stop_word_ratio: float = 0.1


@dataclass
class QualityFilterConfig:
    """Configuration for quality filtering stage."""
    enabled: bool = False  # Disabled by default (requires trained models)

    # FastText classifier
    use_fasttext: bool = True
    fasttext_model_path: Optional[str] = None
    fasttext_threshold: float = 0.5

    # KenLM perplexity filter
    use_kenlm: bool = False
    kenlm_model_path: Optional[str] = None
    kenlm_max_perplexity: float = 1000.0

    # Ensemble weights
    fasttext_weight: float = 0.6
    kenlm_weight: float = 0.4
    ensemble_threshold: float = 0.5


@dataclass
class DomainMixingConfig:
    """Configuration for domain mixing stage."""
    enabled: bool = True
    composition: str = "deepseek_v3"  # deepseek_v3, llama3, balanced, or custom
    target_tokens: Optional[int] = None  # Target corpus size in tokens
    shuffle_output: bool = True


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    batch_size: int = 10000
    num_workers: int = 1  # Future: multiprocessing support
    show_progress: bool = True
    checkpoint_interval: int = 100000  # Save checkpoint every N documents


@dataclass
class DataPreprocessingConfig:
    """
    Complete data preprocessing configuration.

    This is the top-level config that contains all sub-configs.

    Example:
        >>> config = DataPreprocessingConfig.from_yaml("config.yaml")
        >>> pipeline = DataPipeline(config.to_pipeline_config())
        >>> stats = pipeline.process_and_save()
    """
    # Input/Output
    input_path: str
    output_dir: str
    output_format: str = "jsonl"  # jsonl, parquet, hf_dataset
    save_intermediate: bool = True

    # Stage configurations
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    heuristic_filters: HeuristicFilterConfig = field(default_factory=HeuristicFilterConfig)
    quality_filters: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    domain_mixing: DomainMixingConfig = field(default_factory=DomainMixingConfig)

    # Processing parameters
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'DataPreprocessingConfig':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            DataPreprocessingConfig instance

        Example:
            >>> config = DataPreprocessingConfig.from_yaml("config.yaml")
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse sub-configs
        cleaning = CleaningConfig(**data.get("cleaning", {}))
        deduplication = DeduplicationConfig(**data.get("deduplication", {}))
        heuristic_filters = HeuristicFilterConfig(**data.get("heuristic_filters", {}))
        quality_filters = QualityFilterConfig(**data.get("quality_filters", {}))
        domain_mixing = DomainMixingConfig(**data.get("domain_mixing", {}))
        processing = ProcessingConfig(**data.get("processing", {}))

        return cls(
            input_path=data["input_path"],
            output_dir=data["output_dir"],
            output_format=data.get("output_format", "jsonl"),
            save_intermediate=data.get("save_intermediate", True),
            cleaning=cleaning,
            deduplication=deduplication,
            heuristic_filters=heuristic_filters,
            quality_filters=quality_filters,
            domain_mixing=domain_mixing,
            processing=processing,
        )

    def to_yaml(self, path: str):
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        data = {
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "output_format": self.output_format,
            "save_intermediate": self.save_intermediate,
            "cleaning": asdict(self.cleaning),
            "deduplication": asdict(self.deduplication),
            "heuristic_filters": asdict(self.heuristic_filters),
            "quality_filters": asdict(self.quality_filters),
            "domain_mixing": asdict(self.domain_mixing),
            "processing": asdict(self.processing),
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_pipeline_config(self) -> PipelineConfig:
        """
        Convert to PipelineConfig for use with DataPipeline.

        Returns:
            PipelineConfig instance
        """
        return PipelineConfig(
            input_path=self.input_path,
            output_dir=self.output_dir,
            output_format=self.output_format,
            save_intermediate=self.save_intermediate,
            enable_cleaning=self.cleaning.enabled,
            enable_deduplication=self.deduplication.enabled,
            enable_heuristic_filters=self.heuristic_filters.enabled,
            enable_quality_filters=self.quality_filters.enabled,
            enable_domain_mixing=self.domain_mixing.enabled,
            cleaning_config=self._get_cleaning_config(),
            dedup_config=self._get_dedup_config(),
            heuristic_config=self._get_heuristic_config(),
            quality_config=self._get_quality_config(),
            domain_config=self._get_domain_config(),
            batch_size=self.processing.batch_size,
            num_workers=self.processing.num_workers,
            show_progress=self.processing.show_progress,
            checkpoint_interval=self.processing.checkpoint_interval,
        )

    def _get_cleaning_config(self) -> Dict[str, Any]:
        """Get cleaning configuration as dict."""
        return {
            "unicode_normalization": self.cleaning.unicode_normalization,
            "fix_encoding": self.cleaning.fix_encoding,
            "unescape_html": self.cleaning.unescape_html,
            "remove_control_chars": self.cleaning.remove_control_chars,
            "normalize_whitespace": self.cleaning.normalize_whitespace,
        }

    def _get_dedup_config(self) -> Dict[str, Any]:
        """Get deduplication configuration as dict."""
        return {
            "method": self.deduplication.method,
            "threshold": self.deduplication.threshold,
            "num_perm": self.deduplication.num_perm,
            "n_gram": self.deduplication.n_gram,
            "seed": self.deduplication.seed,
        }

    def _get_heuristic_config(self) -> Dict[str, Any]:
        """Get heuristic filter configuration as dict."""
        config = asdict(self.heuristic_filters)
        config.pop("enabled")  # Remove enabled flag
        return config

    def _get_quality_config(self) -> Dict[str, Any]:
        """Get quality filter configuration as dict."""
        return {
            "use_fasttext": self.quality_filters.use_fasttext,
            "fasttext_model_path": self.quality_filters.fasttext_model_path,
            "fasttext_threshold": self.quality_filters.fasttext_threshold,
            "use_kenlm": self.quality_filters.use_kenlm,
            "kenlm_model_path": self.quality_filters.kenlm_model_path,
            "kenlm_max_perplexity": self.quality_filters.kenlm_max_perplexity,
            "fasttext_weight": self.quality_filters.fasttext_weight,
            "kenlm_weight": self.quality_filters.kenlm_weight,
            "ensemble_threshold": self.quality_filters.ensemble_threshold,
        }

    def _get_domain_config(self) -> Dict[str, Any]:
        """Get domain mixing configuration as dict."""
        return {
            "composition": self.domain_mixing.composition,
            "target_tokens": self.domain_mixing.target_tokens,
            "shuffle_output": self.domain_mixing.shuffle_output,
        }

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 80)
        print("DATA PREPROCESSING CONFIGURATION")
        print("=" * 80)
        print(f"Input:  {self.input_path}")
        print(f"Output: {self.output_dir}")
        print(f"Format: {self.output_format}")
        print(f"\nEnabled Stages:")
        print(f"  [{'✓' if self.cleaning.enabled else ' '}] Preliminary Cleaning")
        print(f"  [{'✓' if self.deduplication.enabled else ' '}] Deduplication ({self.deduplication.method})")
        print(f"  [{'✓' if self.heuristic_filters.enabled else ' '}] Heuristic Filters")
        print(f"  [{'✓' if self.quality_filters.enabled else ' '}] Quality Filters")
        print(f"  [{'✓' if self.domain_mixing.enabled else ' '}] Domain Mixing ({self.domain_mixing.composition})")
        print("=" * 80)
