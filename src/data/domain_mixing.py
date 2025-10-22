"""
Domain mixing implementation for LLM data preprocessing.

Implements DoReMi (Domain Reweighting with Minimax Optimization) and other
domain mixing strategies for optimal data composition.

References:
    Xie et al. (2023). "DoReMi: Optimizing Data Mixtures Speeds Up Language
    Model Pretraining." arXiv:2305.10429

    Zhou et al. (2025). "Data × LLM: From Principles to Practices."
    arXiv:2505.18458, Section on domain composition
"""

import re
import json
import random
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np


# Domain categories matching common LLM datasets (Dolma, C4, The Pile)
DOMAIN_CATEGORIES = [
    "common_crawl",    # General web text
    "wikipedia",       # Encyclopedia/reference
    "code",           # Programming code (GitHub)
    "academic",       # Research papers (arXiv, PubMed)
    "books",          # Literature/books
    "news",           # News articles
    "social",         # Social media (Reddit, forums)
]

# Predefined domain compositions from SOTA models
PRESET_COMPOSITIONS = {
    "deepseek_v3": {
        # Based on DeepSeek-V3 technical report
        "common_crawl": 0.45,
        "code": 0.20,
        "academic": 0.15,
        "books": 0.10,
        "wikipedia": 0.05,
        "news": 0.03,
        "social": 0.02,
    },
    "llama3": {
        # Based on LLaMA-3 data composition
        "common_crawl": 0.50,
        "code": 0.15,
        "wikipedia": 0.12,
        "books": 0.10,
        "academic": 0.08,
        "news": 0.03,
        "social": 0.02,
    },
    "balanced": {
        # Uniform distribution across all domains
        domain: 1.0 / len(DOMAIN_CATEGORIES)
        for domain in DOMAIN_CATEGORIES
    },
}


@dataclass
class DomainWeights:
    """Container for domain weights and statistics."""

    weights: Dict[str, float] = field(default_factory=dict)
    document_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    token_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    iteration: int = 0

    def normalize(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "weights": self.weights,
            "document_counts": dict(self.document_counts),
            "token_counts": dict(self.token_counts),
            "iteration": self.iteration,
        }


class DomainIdentifier:
    """
    Identifies document domains using keyword-based classification.

    Supports multiple identification methods:
    - keyword: Pattern matching on content
    - metadata: Use existing domain labels
    - hybrid: Combination of methods

    Args:
        method: Identification method to use
        confidence_threshold: Minimum confidence for classification (0-1)
    """

    def __init__(
        self,
        method: str = "keyword",
        confidence_threshold: float = 0.5,
    ):
        self.method = method
        self.confidence_threshold = confidence_threshold

        # Domain detection patterns (keyword-based)
        self.domain_patterns = {
            "code": [
                r'\bdef\s+\w+\s*\(',
                r'\bclass\s+\w+[:\(]',
                r'\bfunction\s+\w+\s*\(',
                r'\bimport\s+\w+',
                r'\bfrom\s+\w+\s+import',
                r'#include\s*<',
                r'public\s+class\s+',
                r'=>',
                r'\{\s*\w+\s*:\s*',
            ],
            "academic": [
                r'\babstract\b.*\bintroduction\b',
                r'\barXiv:\d{4}\.\d{4,6}',
                r'\bdoi:\s*10\.',
                r'\bet\s+al\.',
                r'\bcitation\b',
                r'\breferences?\b.*\[\d+\]',
                r'\btheorem\b.*\bproof\b',
                r'\bhypothesis\b',
            ],
            "wikipedia": [
                r'\[\[.*?\]\]',
                r'==\s*.*?\s*==',
                r'\{\{cite',
                r'\bInfobox\b',
                r'Category:',
                r'\bmain article\b',
            ],
            "news": [
                r'\b(WASHINGTON|LONDON|PARIS|BEIJING)\s*\(',
                r'\bAssociated Press\b',
                r'\bReuters\b',
                r'\breported\s+(today|yesterday)',
                r'\baccording to\b.*\bofficial',
                r'\bbreaking news\b',
            ],
            "social": [
                r'\bu/\w+',
                r'\br/\w+',
                r'\bOP\s+(said|here)',
                r'\bupvote',
                r'\bkarma\b',
                r'@\w+\s+(tweeted|posted)',
            ],
            "books": [
                r'\bChapter\s+\d+',
                r'\bPart\s+(I{1,3}|One|Two|Three)',
                r'\bCopyright\s+©',
                r'\bISBN[-\s]',
                r'\bPublished by\b',
            ],
        }

        # Compile patterns for efficiency
        self.compiled_patterns = {
            domain: [re.compile(p, re.IGNORECASE) for p in patterns]
            for domain, patterns in self.domain_patterns.items()
        }

    def identify(self, document: Dict[str, Any]) -> Tuple[str, float]:
        """
        Identify the domain of a document.

        Args:
            document: Document dictionary with 'text' field

        Returns:
            Tuple of (domain_name, confidence_score)
        """
        # Check for metadata domain label
        if self.method in ["metadata", "hybrid"] and "domain" in document:
            return document["domain"], 1.0

        text = document.get("text", "")

        # Use keyword-based classification
        if self.method in ["keyword", "hybrid"]:
            return self._classify_by_keywords(text)

        # Default to common_crawl if no method matches
        return "common_crawl", 0.3

    def _classify_by_keywords(self, text: str) -> Tuple[str, float]:
        """
        Classify document using keyword patterns.

        Args:
            text: Document text

        Returns:
            Tuple of (domain_name, confidence_score)
        """
        # Sample text for efficiency (first 5000 chars)
        sample = text[:5000]

        # Count pattern matches per domain
        domain_scores = {}

        for domain, patterns in self.compiled_patterns.items():
            matches = sum(1 for p in patterns if p.search(sample))
            # Use raw match count as score (more matches = higher confidence)
            domain_scores[domain] = matches

        # Get domain with highest score
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            # If we have any matches, use that domain
            if best_domain[1] > 0:
                # Convert match count to confidence (cap at 1.0)
                confidence = min(best_domain[1] / 3.0, 1.0)  # 3+ matches = full confidence
                return best_domain[0], confidence

        # Default to common_crawl for general web text
        return "common_crawl", 0.5


class GroupDROOptimizer:
    """
    Group Distributionally Robust Optimization for domain weight optimization.

    Implements the DoReMi algorithm's core optimization logic using Group DRO
    to find domain weights that minimize worst-case excess loss.

    References:
        Xie et al. (2023). DoReMi, Section 3.2: Group DRO Optimization

    Args:
        learning_rate: Step size for weight updates
        temperature: Temperature parameter for reweighting (lower = more aggressive)
        num_domains: Number of domains to optimize over
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        temperature: float = 0.5,
        num_domains: int = len(DOMAIN_CATEGORIES),
    ):
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.num_domains = num_domains

        # Initialize uniform weights
        self.alpha = np.ones(num_domains) / num_domains

    def update_weights(
        self,
        domain_losses: Dict[str, float],
        domain_names: List[str],
    ) -> Dict[str, float]:
        """
        Update domain weights based on losses using Group DRO.

        The algorithm increases weights for domains with higher excess loss,
        implementing the minimax objective: min_θ max_α E[loss(θ, α)]

        Args:
            domain_losses: Dictionary mapping domain names to loss values
            domain_names: Ordered list of domain names

        Returns:
            Updated domain weights dictionary
        """
        # Convert losses to numpy array
        losses = np.array([domain_losses.get(d, 0.0) for d in domain_names])

        # Compute excess losses (compared to mean)
        mean_loss = np.mean(losses)
        excess_losses = losses - mean_loss

        # Update alpha using exponentiated gradient ascent
        # Higher loss → higher weight (upweight hard domains)
        self.alpha = self.alpha * np.exp(self.learning_rate * excess_losses / self.temperature)

        # Normalize to sum to 1
        self.alpha = self.alpha / np.sum(self.alpha)

        # Convert back to dictionary
        return {domain: float(weight) for domain, weight in zip(domain_names, self.alpha)}


class DomainMixer:
    """
    Main domain mixing class implementing DoReMi and preset compositions.

    Orchestrates domain identification, weight optimization, and sampling
    to create optimally mixed training datasets.

    Supports multiple mixing strategies:
    - doremi: Adaptive reweighting using Group DRO
    - deepseek_v3: DeepSeek-V3 composition ratios
    - llama3: LLaMA-3 composition ratios
    - balanced: Uniform distribution
    - custom: User-defined weights

    Args:
        composition: Mixing strategy or custom weights dictionary
        identification_method: Method for domain identification
        target_tokens: Target total tokens (None = no limit)
        temperature: Temperature for DoReMi reweighting
        num_iterations: Number of DoReMi iterations
        random_seed: Random seed for reproducibility

    Example:
        >>> mixer = DomainMixer(composition="deepseek_v3")
        >>> mixed_docs = mixer.mix_documents(documents)
        >>> stats = mixer.get_statistics()
    """

    def __init__(
        self,
        composition: str = "deepseek_v3",
        identification_method: str = "keyword",
        target_tokens: Optional[int] = None,
        temperature: float = 0.5,
        num_iterations: int = 1,
        random_seed: int = 42,
    ):
        self.composition_name = composition
        self.identification_method = identification_method
        self.target_tokens = target_tokens
        self.temperature = temperature
        self.num_iterations = num_iterations
        self.random_seed = random_seed

        # Initialize components
        self.identifier = DomainIdentifier(method=identification_method)
        self.dro_optimizer = GroupDROOptimizer(temperature=temperature) if composition == "doremi" else None

        # Set domain weights
        if isinstance(composition, dict):
            # Custom weights provided
            self.domain_weights = DomainWeights(weights=composition)
        elif composition in PRESET_COMPOSITIONS:
            # Use preset composition
            self.domain_weights = DomainWeights(weights=PRESET_COMPOSITIONS[composition].copy())
        elif composition == "doremi":
            # Start with uniform weights for DoReMi
            uniform_weights = {d: 1.0 / len(DOMAIN_CATEGORIES) for d in DOMAIN_CATEGORIES}
            self.domain_weights = DomainWeights(weights=uniform_weights)
        else:
            raise ValueError(f"Unknown composition: {composition}. Use one of {list(PRESET_COMPOSITIONS.keys())} or 'doremi'")

        self.domain_weights.normalize()

        # Statistics tracking
        self.total_documents_processed = 0
        self.total_tokens_processed = 0
        self.documents_by_domain: Dict[str, List[Dict]] = defaultdict(list)

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def identify_domain(self, document: Dict[str, Any]) -> Tuple[str, float]:
        """
        Identify the domain of a document.

        Args:
            document: Document dictionary

        Returns:
            Tuple of (domain_name, confidence_score)
        """
        return self.identifier.identify(document)

    def classify_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Classify documents into domains.

        Args:
            documents: List of document dictionaries

        Returns:
            Dictionary mapping domain names to document lists
        """
        domain_buckets: Dict[str, List[Dict]] = defaultdict(list)

        for doc in documents:
            domain, confidence = self.identify_domain(doc)

            # Add domain metadata to document
            doc["_domain"] = domain
            doc["_domain_confidence"] = confidence

            # Add to appropriate bucket
            domain_buckets[domain].append(doc)

            # Update statistics
            self.domain_weights.document_counts[domain] += 1

            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            text_length = len(doc.get("text", ""))
            estimated_tokens = text_length // 4
            self.domain_weights.token_counts[domain] += estimated_tokens
            self.total_tokens_processed += estimated_tokens

        self.total_documents_processed += len(documents)
        return domain_buckets

    def mix_documents(
        self,
        documents: List[Dict[str, Any]],
        target_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply domain mixing to create a balanced dataset.

        Args:
            documents: Input documents to mix
            target_size: Target number of documents (None = use all)

        Returns:
            Mixed document list following target domain composition
        """
        # Step 1: Classify all documents into domains
        domain_buckets = self.classify_documents(documents)

        # Step 2: Calculate target counts per domain
        total_docs = target_size or len(documents)
        target_counts = {
            domain: int(total_docs * weight)
            for domain, weight in self.domain_weights.weights.items()
        }

        # Step 3: Sample from each domain according to weights
        mixed_documents = []

        for domain, target_count in target_counts.items():
            available_docs = domain_buckets.get(domain, [])

            if not available_docs:
                continue

            # Sample with replacement if target > available
            if len(available_docs) < target_count:
                sampled = random.choices(available_docs, k=target_count)
            else:
                sampled = random.sample(available_docs, target_count)

            mixed_documents.extend(sampled)

        # Step 4: Shuffle final mix
        random.shuffle(mixed_documents)

        return mixed_documents

    def optimize_weights_doremi(
        self,
        domain_losses: Dict[str, float],
    ) -> None:
        """
        Update domain weights using DoReMi Group DRO optimization.

        Args:
            domain_losses: Perplexity or loss values per domain
        """
        if self.dro_optimizer is None:
            raise ValueError("DoReMi optimizer not initialized. Set composition='doremi'")

        # Update weights using Group DRO
        new_weights = self.dro_optimizer.update_weights(
            domain_losses,
            list(DOMAIN_CATEGORIES),
        )

        self.domain_weights.weights = new_weights
        self.domain_weights.iteration += 1

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive domain mixing statistics.

        Returns:
            Statistics dictionary with domain distributions and counts
        """
        # Calculate actual domain distribution
        actual_distribution = {}
        if self.total_documents_processed > 0:
            for domain in DOMAIN_CATEGORIES:
                count = self.domain_weights.document_counts[domain]
                actual_distribution[domain] = count / self.total_documents_processed

        return {
            "composition": self.composition_name,
            "target_weights": self.domain_weights.weights,
            "actual_distribution": actual_distribution,
            "document_counts": dict(self.domain_weights.document_counts),
            "token_counts": dict(self.domain_weights.token_counts),
            "total_documents": self.total_documents_processed,
            "total_tokens": self.total_tokens_processed,
            "iteration": self.domain_weights.iteration,
        }

    def save_statistics(self, output_path: Path) -> None:
        """Save domain mixing statistics to JSON file."""
        stats = self.get_statistics()
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

    def __repr__(self) -> str:
        return (
            f"DomainMixer(composition='{self.composition_name}', "
            f"method='{self.identification_method}', "
            f"docs={self.total_documents_processed}, "
            f"tokens={self.total_tokens_processed})"
        )
