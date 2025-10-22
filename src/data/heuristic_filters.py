"""
Heuristic filtering for data sanitization pipeline.

Implements rule-based filters to remove low-quality documents based on
structural and statistical properties.

References:
    Zhou et al. (2025). "A Survey of LLM × DATA."
    arXiv:2505.18458, Section 3.1.2

    Kim et al. (2024). "Rethinking KenLM: Good and Bad Model Ensembles
    for Efficient Text Quality Filtering."
    arXiv:2409.09613

    Gao et al. (2021). "The Pile: An 800GB Dataset of Diverse Text."
    arXiv:2101.00027
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import re


@dataclass
class HeuristicFilterConfig:
    """Configuration for heuristic filters."""

    # Document length filters
    min_doc_length: int = 100  # Minimum characters
    max_doc_length: int = 1_000_000  # Maximum characters

    # Line-level filters
    min_lines: int = 3  # Minimum number of lines
    max_line_length: int = 10_000  # Maximum characters per line

    # Repetition filters
    max_char_ngram_repetition: float = 0.2  # Max fraction of repeated 3-grams
    max_line_repetition: float = 0.3  # Max fraction of repeated lines

    # Special character filters
    max_special_char_ratio: float = 0.3  # Max fraction of non-alphanumeric chars
    min_alpha_ratio: float = 0.5  # Min fraction of alphabetic characters

    # Word-level filters
    min_avg_word_length: float = 2.0  # Minimum average word length
    max_avg_word_length: float = 20.0  # Maximum average word length

    # Structural filters
    check_balanced_brackets: bool = True  # Check for balanced brackets/quotes


@dataclass
class HeuristicFilterStats:
    """Statistics from heuristic filtering."""
    total_documents: int = 0
    filtered_documents: int = 0
    reason_counts: Dict[str, int] = None

    def __post_init__(self):
        if self.reason_counts is None:
            self.reason_counts = {}


class HeuristicFilter:
    """
    Apply rule-based heuristic filters to documents.

    This is a basic implementation of common heuristic filters used in
    data sanitization pipelines for LLM training.
    """

    def __init__(self, config: Optional[HeuristicFilterConfig] = None):
        """
        Initialize heuristic filter.

        Args:
            config: Filter configuration (uses defaults if None)
        """
        self.config = config or HeuristicFilterConfig()
        self.stats = HeuristicFilterStats()

    def filter(self, text: str) -> bool:
        """
        Check if document passes all heuristic filters.

        Args:
            text: Document text to filter

        Returns:
            True if document passes filters, False otherwise
        """
        self.stats.total_documents += 1

        # Check document length
        if not self._check_document_length(text):
            self.stats.filtered_documents += 1
            self.stats.reason_counts["doc_length"] = self.stats.reason_counts.get("doc_length", 0) + 1
            return False

        # Check line count and length
        lines = text.split('\n')
        if not self._check_lines(lines):
            self.stats.filtered_documents += 1
            self.stats.reason_counts["line_filters"] = self.stats.reason_counts.get("line_filters", 0) + 1
            return False

        # Check repetition
        if not self._check_repetition(text, lines):
            self.stats.filtered_documents += 1
            self.stats.reason_counts["repetition"] = self.stats.reason_counts.get("repetition", 0) + 1
            return False

        # Check special character ratio
        if not self._check_special_chars(text):
            self.stats.filtered_documents += 1
            self.stats.reason_counts["special_chars"] = self.stats.reason_counts.get("special_chars", 0) + 1
            return False

        # Check word statistics
        if not self._check_word_stats(text):
            self.stats.filtered_documents += 1
            self.stats.reason_counts["word_stats"] = self.stats.reason_counts.get("word_stats", 0) + 1
            return False

        # Check structural integrity
        if self.config.check_balanced_brackets and not self._check_balanced(text):
            self.stats.filtered_documents += 1
            self.stats.reason_counts["unbalanced"] = self.stats.reason_counts.get("unbalanced", 0) + 1
            return False

        return True

    def _check_document_length(self, text: str) -> bool:
        """Check if document length is within bounds."""
        length = len(text)
        return self.config.min_doc_length <= length <= self.config.max_doc_length

    def _check_lines(self, lines: List[str]) -> bool:
        """Check line count and maximum line length."""
        if len(lines) < self.config.min_lines:
            return False

        for line in lines:
            if len(line) > self.config.max_line_length:
                return False

        return True

    def _check_repetition(self, text: str, lines: List[str]) -> bool:
        """Check for excessive repetition at character and line level."""
        # Check character-level n-gram repetition (3-grams)
        if len(text) > 0:
            ngrams = [text[i:i+3] for i in range(len(text) - 2)]
            if ngrams:
                unique_ngrams = len(set(ngrams))
                total_ngrams = len(ngrams)
                repetition_ratio = 1.0 - (unique_ngrams / total_ngrams)
                if repetition_ratio > self.config.max_char_ngram_repetition:
                    return False

        # Check line-level repetition
        if len(lines) > 0:
            unique_lines = len(set(line.strip() for line in lines if line.strip()))
            total_lines = len([line for line in lines if line.strip()])
            if total_lines > 0:
                line_repetition = 1.0 - (unique_lines / total_lines)
                if line_repetition > self.config.max_line_repetition:
                    return False

        return True

    def _check_special_chars(self, text: str) -> bool:
        """Check special character and alphabetic ratios."""
        if len(text) == 0:
            return False

        # Count alphanumeric and alphabetic characters
        alphanum_count = sum(1 for c in text if c.isalnum())
        alpha_count = sum(1 for c in text if c.isalpha())

        special_ratio = 1.0 - (alphanum_count / len(text))
        alpha_ratio = alpha_count / len(text)

        return (special_ratio <= self.config.max_special_char_ratio and
                alpha_ratio >= self.config.min_alpha_ratio)

    def _check_word_stats(self, text: str) -> bool:
        """Check word-level statistics."""
        words = re.findall(r'\b\w+\b', text)

        if not words:
            return False

        avg_word_length = sum(len(word) for word in words) / len(words)

        return (self.config.min_avg_word_length <= avg_word_length <=
                self.config.max_avg_word_length)

    def _check_balanced(self, text: str) -> bool:
        """Check for balanced brackets and quotes."""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        quote_count = 0

        for char in text:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack or pairs[stack[-1]] != char:
                    return False
                stack.pop()
            elif char == '"':
                quote_count += 1

        # Stack should be empty and quotes should be even
        return len(stack) == 0 and quote_count % 2 == 0

    def get_stats(self) -> HeuristicFilterStats:
        """
        Get filtering statistics.

        Returns:
            Statistics about filtered documents
        """
        return self.stats

    def reset_stats(self):
        """Reset statistics."""
        self.stats = HeuristicFilterStats()
