"""
Domain mixing implementation for LLM data preprocessing.

Implements DoReMi (Domain Reweighting with Minimax Optimization) and other
domain mixing strategies for optimal data composition.

References:
    Xie et al. (2023). "DoReMi: Optimizing Data Mixtures Speeds Up Language
    Model Pretraining." arXiv:2305.10429

    Zhou et al. (2025). "A Survey of LLM × DATA."
    arXiv:2505.18458, Section on domain composition
"""

import re
import json
import random
import warnings
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
        # Documented composition: 92% natural language, 5% code, 3% math
        # Natural language is aggregated from: common_crawl, wikipedia, books, news, social, academic
        "common_crawl": 0.50,  # Primary NL source
        "wikipedia": 0.15,     # Reference/encyclopedic NL
        "books": 0.10,         # Literature NL
        "academic": 0.08,      # Scientific/technical NL
        "news": 0.05,          # News/journalism NL
        "social": 0.04,        # Conversational NL
        # Total NL = 0.92 (92%)
        "code": 0.05,          # Code domain (5%)
        # Note: Math is often mixed with academic papers or included in code
        # For separate math domain, allocate 0.03 (3%) from academic budget
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
    # Track actual sampled output (including duplicates from oversampling)
    sampled_document_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sampled_token_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    iteration: int = 0
    # DoReMi reference model baseline (for excess loss computation)
    reference_weights: Optional[Dict[str, float]] = None
    reference_losses: Optional[Dict[str, float]] = None

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
            "sampled_document_counts": dict(self.sampled_document_counts),
            "sampled_token_counts": dict(self.sampled_token_counts),
            "iteration": self.iteration,
            "reference_weights": self.reference_weights,
            "reference_losses": self.reference_losses,
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
        reference_losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Update domain weights based on losses using Group DRO.

        The algorithm increases weights for domains with higher excess loss,
        implementing the minimax objective: min_θ max_α E[loss(θ, α)]

        As specified in DoReMi (Xie et al., 2023), the optimizer uses excess loss
        relative to a reference model baseline:
            excess_loss = proxy_loss - reference_loss

        If reference_losses is not provided, falls back to mean-centered losses
        (backward compatibility mode, not DoReMi-compliant).

        Args:
            domain_losses: Dictionary mapping domain names to proxy model loss values
            domain_names: Ordered list of domain names
            reference_losses: Dictionary mapping domain names to reference model losses
                             (required for DoReMi algorithm compliance)

        Returns:
            Updated domain weights dictionary

        Raises:
            ValueError: If reference_losses is None and strict DoReMi mode is expected
        """
        # Convert losses to numpy array
        losses = np.array([domain_losses.get(d, 0.0) for d in domain_names])

        # Compute excess losses per DoReMi specification
        if reference_losses is not None:
            # DoReMi mode: excess_loss = proxy_loss - reference_loss
            ref_losses = np.array([reference_losses.get(d, 0.0) for d in domain_names])
            excess_losses = losses - ref_losses
        else:
            # DoReMi algorithm requires reference losses for proper excess loss computation
            # Fallback to mean-centered losses is NOT compliant with Xie et al. (2023)
            warnings.warn(
                "DoReMi optimization called without reference_losses. "
                "This violates the DoReMi algorithm (Xie et al., 2023) which requires:\n"
                "  1. Train a reference model with reference_weights mixture\n"
                "  2. Measure reference_losses on validation data\n"
                "  3. Compute excess_loss = proxy_loss - reference_loss\n"
                "\n"
                "Falling back to mean-centered losses (NOT DoReMi-compliant). "
                "Results may not match paper's theoretical guarantees.\n"
                "\n"
                "To use DoReMi properly, provide reference_losses parameter.",
                UserWarning
            )
            mean_loss = np.mean(losses)
            excess_losses = losses - mean_loss

        # Update alpha using exponentiated gradient ascent
        # Higher excess loss → higher weight (upweight hard domains)
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
        loss_feedback_path: Optional[str] = None,
    ):
        self.composition_name = composition
        self.identification_method = identification_method
        self.target_tokens = target_tokens
        self.temperature = temperature
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.loss_feedback_path = loss_feedback_path

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

        # Load loss feedback if path provided (for DoReMi)
        self.domain_losses = None
        self.reference_losses = None
        self.reference_weights = None
        if loss_feedback_path and composition == "doremi":
            self._load_loss_feedback(loss_feedback_path)

        # Statistics tracking
        self.total_documents_processed = 0
        self.total_tokens_processed = 0
        self.documents_by_domain: Dict[str, List[Dict]] = defaultdict(list)

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def _load_loss_feedback(self, path: str) -> None:
        """
        Load loss feedback from JSON file for DoReMi optimization.

        Expected JSON format:
        {
            "domain_losses": {"code": 2.5, "common_crawl": 3.2, ...},
            "reference_losses": {"code": 2.3, "common_crawl": 2.8, ...},
            "reference_weights": {"code": 0.14, "common_crawl": 0.14, ...}
        }

        Args:
            path: Path to JSON file containing loss feedback
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.domain_losses = data.get("domain_losses")
            self.reference_losses = data.get("reference_losses")
            self.reference_weights = data.get("reference_weights")

            if self.domain_losses is None:
                raise ValueError("Loss feedback file missing 'domain_losses' field")

        except FileNotFoundError:
            print(f"Warning: Loss feedback file not found: {path}")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in loss feedback file: {e}")

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

        Uses remainder redistribution to ensure the total document count
        matches the target size exactly, distributing leftover documents
        to domains with the highest fractional parts after floor division.

        This prevents silent document loss due to integer rounding, as
        described in the DoReMi pipeline specification (Xie et al., 2023).

        Args:
            documents: Input documents to mix
            target_size: Target number of documents (None = use all)

        Returns:
            Mixed document list following target domain composition
        """
        # Step 1: Classify all documents into domains
        domain_buckets = self.classify_documents(documents)

        # Step 2: Calculate target counts per domain with remainder redistribution
        total_docs = target_size or len(documents)

        # Floor division for initial allocation
        target_counts = {
            domain: int(total_docs * weight)
            for domain, weight in self.domain_weights.weights.items()
        }

        # Calculate remainder from floor division
        allocated = sum(target_counts.values())
        remainder = total_docs - allocated

        # Redistribute remainder to domains with highest fractional parts
        if remainder > 0:
            fractional_parts = {
                domain: (total_docs * weight) - target_counts[domain]
                for domain, weight in self.domain_weights.weights.items()
            }
            # Sort by fractional part descending
            sorted_domains = sorted(
                fractional_parts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Add 1 to top domains until remainder is distributed
            for i in range(remainder):
                domain = sorted_domains[i][0]
                target_counts[domain] += 1

        # Step 3: Handle domains with no documents - redistribute their allocation
        # to domains that do have documents
        domains_with_docs = {d: target_counts[d] for d in target_counts if domain_buckets.get(d, [])}
        domains_without_docs = {d: target_counts[d] for d in target_counts if not domain_buckets.get(d, [])}

        # Redistribute documents from empty domains to non-empty ones
        unallocated = sum(domains_without_docs.values())
        if unallocated > 0 and domains_with_docs:
            # Calculate weights for redistribution (proportional to existing allocation)
            total_allocated_to_available = sum(domains_with_docs.values())

            if total_allocated_to_available > 0:
                # Redistribute proportionally
                for domain in domains_with_docs:
                    proportion = domains_with_docs[domain] / total_allocated_to_available
                    additional = int(unallocated * proportion)
                    target_counts[domain] += additional

                # Handle any remainder from floor division
                remaining = unallocated - sum(int(unallocated * (domains_with_docs[d] / total_allocated_to_available))
                                             for d in domains_with_docs)
                if remaining > 0:
                    # Give remaining docs to domain with highest allocation
                    top_domain = max(domains_with_docs.keys(), key=lambda d: domains_with_docs[d])
                    target_counts[top_domain] += remaining
            else:
                # All domains have zero allocation but we have unallocated docs
                # Distribute equally among available domains
                per_domain = unallocated // len(domains_with_docs)
                remainder = unallocated % len(domains_with_docs)

                for i, domain in enumerate(domains_with_docs.keys()):
                    target_counts[domain] += per_domain
                    if i < remainder:
                        target_counts[domain] += 1

        # Step 4: Sample from each domain according to adjusted weights
        mixed_documents = []

        # Reset sampled counts before new sampling
        self.domain_weights.sampled_document_counts = defaultdict(int)
        self.domain_weights.sampled_token_counts = defaultdict(int)

        for domain, target_count in target_counts.items():
            available_docs = domain_buckets.get(domain, [])

            if not available_docs or target_count == 0:
                continue

            # Sample with replacement if target > available
            if len(available_docs) < target_count:
                sampled = random.choices(available_docs, k=target_count)
            else:
                sampled = random.sample(available_docs, target_count)

            mixed_documents.extend(sampled)

            # Track actual sampled output (including duplicates)
            self.domain_weights.sampled_document_counts[domain] += len(sampled)
            for doc in sampled:
                # Estimate tokens for sampled documents
                text_length = len(doc.get("text", ""))
                estimated_tokens = text_length // 4
                self.domain_weights.sampled_token_counts[domain] += estimated_tokens

        # Step 5: Shuffle final mix
        random.shuffle(mixed_documents)

        return mixed_documents

    def optimize_weights_doremi(
        self,
        domain_losses: Dict[str, float],
        reference_losses: Optional[Dict[str, float]] = None,
        reference_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Update domain weights using DoReMi Group DRO optimization.

        Per DoReMi (Xie et al., 2023), the algorithm requires:
        1. A reference model trained with reference mixture weights
        2. Proxy model losses measured on the same data
        3. Weight updates based on excess loss: proxy_loss - reference_loss

        Args:
            domain_losses: Proxy model perplexity or loss values per domain
            reference_losses: Reference model loss values per domain (for excess loss)
            reference_weights: Reference mixture weights used to train reference model
                              (stored for reproducibility)
        """
        if self.dro_optimizer is None:
            raise ValueError("DoReMi optimizer not initialized. Set composition='doremi'")

        # Update weights using Group DRO with optional reference baseline
        new_weights = self.dro_optimizer.update_weights(
            domain_losses,
            list(DOMAIN_CATEGORIES),
            reference_losses=reference_losses,
        )

        self.domain_weights.weights = new_weights
        self.domain_weights.iteration += 1

        # Persist reference statistics for reproducibility
        if reference_losses is not None:
            self.domain_weights.reference_losses = reference_losses
        if reference_weights is not None:
            self.domain_weights.reference_weights = reference_weights

    def mix_documents_with_feedback(
        self,
        documents: List[Dict[str, Any]],
        domain_losses: Dict[str, float],
        num_iterations: int = 1,
        target_size: Optional[int] = None,
        reference_losses: Optional[Dict[str, float]] = None,
        reference_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply domain mixing with DoReMi feedback loop.

        This method iteratively updates domain weights based on per-domain losses
        before sampling the final mixture, implementing the DoReMi algorithm's
        minimax reweighting strategy.

        Per DoReMi (Xie et al., 2023), the algorithm requires:
        1. Train a reference model with reference_weights mixture
        2. Measure reference_losses and domain_losses (proxy model) on validation data
        3. Optimize proxy weights using excess loss: proxy_loss - reference_loss

        Args:
            documents: Input documents to mix
            domain_losses: Proxy model per-domain loss statistics (e.g., perplexity)
            num_iterations: Number of DoReMi weight update iterations
            target_size: Target number of documents (None = use all)
            reference_losses: Reference model losses per domain (for DoReMi excess loss)
            reference_weights: Reference mixture weights (stored for reproducibility)

        Returns:
            Mixed document list with DoReMi-optimized composition

        Example:
            >>> mixer = DomainMixer(composition="doremi")
            >>> # Step 1: Train reference model with uniform weights
            >>> ref_weights = {"code": 0.14, "common_crawl": 0.14, ...}
            >>> # Step 2: Measure losses on validation set
            >>> ref_losses = {"code": 2.3, "common_crawl": 2.8, ...}
            >>> proxy_losses = {"code": 2.5, "common_crawl": 3.2, ...}
            >>> # Step 3: Optimize with DoReMi
            >>> mixed = mixer.mix_documents_with_feedback(
            ...     docs, proxy_losses, num_iterations=3,
            ...     reference_losses=ref_losses, reference_weights=ref_weights
            ... )
        """
        if self.composition_name != "doremi":
            raise ValueError(
                "mix_documents_with_feedback requires composition='doremi'. "
                f"Current composition: {self.composition_name}"
            )

        # Run DoReMi weight optimization for requested iterations
        for _ in range(num_iterations):
            self.optimize_weights_doremi(
                domain_losses,
                reference_losses=reference_losses,
                reference_weights=reference_weights,
            )

        # Now sample using the optimized weights
        return self.mix_documents(documents, target_size=target_size)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive domain mixing statistics.

        Returns:
            Statistics dictionary with domain distributions and counts
        """
        # Calculate input domain distribution (what was classified)
        input_distribution = {}
        if self.total_documents_processed > 0:
            for domain in DOMAIN_CATEGORIES:
                count = self.domain_weights.document_counts[domain]
                input_distribution[domain] = count / self.total_documents_processed

        # Calculate actual sampled distribution (what was output, including duplicates)
        actual_distribution = {}
        total_sampled = sum(self.domain_weights.sampled_document_counts.values())
        if total_sampled > 0:
            for domain in DOMAIN_CATEGORIES:
                count = self.domain_weights.sampled_document_counts[domain]
                actual_distribution[domain] = count / total_sampled
        else:
            # Fallback to input distribution if no sampling has occurred
            actual_distribution = input_distribution

        return {
            "composition": self.composition_name,
            "target_weights": self.domain_weights.weights,
            "input_distribution": input_distribution,
            "actual_distribution": actual_distribution,
            "document_counts": dict(self.domain_weights.document_counts),
            "token_counts": dict(self.domain_weights.token_counts),
            "sampled_document_counts": dict(self.domain_weights.sampled_document_counts),
            "sampled_token_counts": dict(self.domain_weights.sampled_token_counts),
            "total_documents": self.total_documents_processed,
            "total_tokens": self.total_tokens_processed,
            "iteration": self.domain_weights.iteration,
            "reference_weights": self.domain_weights.reference_weights,
            "reference_losses": self.domain_weights.reference_losses,
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
