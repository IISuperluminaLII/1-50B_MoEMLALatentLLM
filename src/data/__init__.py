"""
Data processing and loading modules for DeepSeek-V3 training.

This package provides utilities for:
- Loading and mixing data from Allen AI's Dolma dataset
- Data preprocessing and sanitization pipeline
- Text cleaning and normalization
- Deduplication (MinHash and exact)
- Quality filtering
- Domain mixing and composition

References:
    Soldaini et al. (2024). "Dolma: an Open Corpus of Three Trillion Tokens
    for Language Model Pretraining Research." arXiv:2402.00159

    Zhou et al. (2025). "Data × LLM: From Principles to Practices."
    arXiv:2505.18458
"""

from .dolma_loader import (
    DolmaSource,
    DolmaDataset,
    create_dolma_dataloaders,
    print_dolma_sources_info,
    print_dolma_info,
)

from .pipeline import (
    DataPipeline,
    PipelineConfig,
    PipelineStats,
)

from .preliminary_cleaning import (
    PreliminaryCleaner,
    CleaningStats,
)

from .deduplication import (
    MinHashDeduplicator,
    ExactDeduplicator,
    DeduplicationStats,
)

from .quality_filters import (
    FastTextQualityClassifier,
)

__all__ = [
    # Dolma loader
    "DolmaSource",
    "DolmaDataset",
    "create_dolma_dataloaders",
    "print_dolma_sources_info",
    "print_dolma_info",
    # Pipeline
    "DataPipeline",
    "PipelineConfig",
    "PipelineStats",
    # Cleaning
    "PreliminaryCleaner",
    "CleaningStats",
    # Deduplication
    "MinHashDeduplicator",
    "ExactDeduplicator",
    "DeduplicationStats",
    # Quality filters
    "FastTextQualityClassifier",
]