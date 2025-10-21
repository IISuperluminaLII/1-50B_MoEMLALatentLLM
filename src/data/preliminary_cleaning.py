"""
Preliminary text cleaning utilities for data sanitization.

Implements text normalization, encoding fixes, and whitespace handling
following best practices from major LLM training pipelines.

References:
    Lee et al. (2022). "Deduplicating Training Data Makes Language Models Better."
    arXiv:2107.06499

    Li et al. (2025). "Data × LLM: From Principles to Practices."
    arXiv:2505.18458
"""
import re
import html
import unicodedata
from dataclasses import dataclass
from typing import Optional


@dataclass
class CleaningStats:
    """Statistics from text cleaning."""
    chars_removed: int = 0
    reduction_ratio: float = 0.0
    unicode_normalized: bool = False
    encoding_fixed: bool = False
    html_unescaped: bool = False
    control_chars_removed: bool = False
    whitespace_normalized: bool = False


class PreliminaryCleaner:
    """
    Preliminary text cleaner for data sanitization.

    Performs multiple cleaning steps including:
    - Unicode normalization
    - Encoding error fixes
    - HTML entity unescaping
    - Control character removal
    - Whitespace normalization
    """

    def __init__(
        self,
        unicode_normalization: str = "NFKC",
        fix_encoding: bool = True,
        unescape_html: bool = True,
        remove_control_chars: bool = True,
        normalize_whitespace: bool = True,
    ):
        """
        Initialize cleaner with configuration.

        Args:
            unicode_normalization: Unicode normalization form (NFC, NFD, NFKC, NFKD)
            fix_encoding: Whether to fix common encoding errors (mojibake)
            unescape_html: Whether to unescape HTML entities
            remove_control_chars: Whether to remove control characters
            normalize_whitespace: Whether to normalize whitespace
        """
        self.unicode_normalization = unicode_normalization
        self.fix_encoding = fix_encoding
        self.unescape_html = unescape_html
        self.remove_control_chars = remove_control_chars
        self.normalize_whitespace = normalize_whitespace

        # Compile regex patterns
        self.control_chars_pattern = re.compile(r"[\x00-\x1F\x7F-\x9F]")
        self.whitespace_pattern = re.compile(r"\s+")
        self.newline_pattern = re.compile(r"\n{3,}")

    def clean(self, text: str) -> str:
        """
        Clean text applying all configured steps.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Step 1: Unicode normalization
        if self.unicode_normalization:
            text = unicodedata.normalize(self.unicode_normalization, text)

        # Step 2: Fix encoding errors (mojibake)
        if self.fix_encoding:
            text = self._fix_encoding(text)

        # Step 3: Unescape HTML entities
        if self.unescape_html:
            text = html.unescape(text)

        # Step 4: Remove control characters
        if self.remove_control_chars:
            # Keep tabs and newlines
            text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

        # Step 5: Normalize whitespace
        if self.normalize_whitespace:
            # Replace multiple spaces with single space
            text = self.whitespace_pattern.sub(" ", text)
            # Replace multiple newlines with double newline
            text = self.newline_pattern.sub("\n\n", text)
            # Strip leading/trailing whitespace
            text = text.strip()

        return text

    def _fix_encoding(self, text: str) -> str:
        """
        Fix common encoding errors (mojibake).

        Args:
            text: Text with potential encoding errors

        Returns:
            Fixed text
        """
        try:
            # Try to use ftfy if available
            import ftfy
            return ftfy.fix_text(text)
        except ImportError:
            # Fallback to basic fixes
            # Fix common mojibake patterns
            replacements = {
                "\u00e2\u20ac\u2122": "'",  # â€™ -> '
                "\u00e2\u20ac\u0153": '"',  # â€œ -> "
                "\u00e2\u20ac": '"',         # â€ -> "
                "\u00e2\u20ac\u201d": "-",  # â€" -> -
                "\u00e2\u20ac\u201c": "-",  # â€" -> -
                "\u00e2\u20ac\u00a6": "...", # â€¦ -> ...
                "\u00ef\u00ac": "fi",       # ï¬ -> fi
                "\u00ef\u00ac\u201a": "fl", # ï¬‚ -> fl
            }
            for old, new in replacements.items():
                text = text.replace(old, new)

            return text

    def get_stats(self, original: str, cleaned: str) -> CleaningStats:
        """
        Get statistics about the cleaning process.

        Args:
            original: Original text before cleaning
            cleaned: Text after cleaning

        Returns:
            Cleaning statistics
        """
        stats = CleaningStats()

        # Calculate basic stats
        original_len = len(original)
        cleaned_len = len(cleaned)

        stats.chars_removed = max(0, original_len - cleaned_len)
        stats.reduction_ratio = (
            stats.chars_removed / original_len if original_len > 0 else 0.0
        )

        # Track which operations were performed
        stats.unicode_normalized = bool(self.unicode_normalization)
        stats.encoding_fixed = self.fix_encoding
        stats.html_unescaped = self.unescape_html
        stats.control_chars_removed = self.remove_control_chars
        stats.whitespace_normalized = self.normalize_whitespace

        return stats