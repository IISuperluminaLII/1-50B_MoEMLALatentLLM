"""
Wikipedia data sanitization pipeline for DeepSeek-V3 training.

This module provides comprehensive data cleaning for Wikipedia articles,
ensuring high-quality training data by removing noise, duplicates, and
low-quality content.
"""
import re
import hashlib
import unicodedata
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import langdetect
from datasketch import MinHash, MinHashLSH


@dataclass
class SanitizationConfig:
    """Configuration for Wikipedia sanitization pipeline."""

    # Language filtering
    target_language: str = "en"
    min_language_confidence: float = 0.95

    # Length constraints
    min_article_length: int = 50  # words
    max_article_length: int = 10000  # words
    min_avg_word_length: int = 3
    max_avg_word_length: int = 10

    # Quality thresholds
    max_perplexity: float = 1500.0
    min_quality_score: float = 0.7

    # Repetition thresholds
    max_char_repetition: float = 0.2
    max_word_repetition: float = 0.2
    max_line_repetition: float = 0.3

    # Deduplication
    dedup_threshold: float = 0.8
    dedup_num_perm: int = 128
    dedup_ngram_size: int = 13

    # Content filtering
    filter_toxic: bool = True
    filter_boilerplate: bool = True
    remove_references: bool = True


class WikipediaSanitizer:
    """
    Comprehensive sanitization pipeline for Wikipedia data.

    Applies multiple filtering stages:
    1. Preliminary cleaning (HTML/markdown removal)
    2. Language detection
    3. Length and format validation
    4. Quality scoring
    5. Repetition detection
    6. Deduplication
    7. Content safety filtering
    """

    def __init__(self, config: Optional[SanitizationConfig] = None, enable_dedup: bool = True):
        """
        Initialize sanitizer with configuration.

        Args:
            config: Sanitization configuration
            enable_dedup: Enable deduplication (CRITICAL: disable in multi-worker mode!)
        """
        self.config = config or SanitizationConfig()
        self.enable_dedup = enable_dedup

        # Initialize deduplication LSH (CRITICAL FIX: only if enabled)
        # In multi-worker mode, each worker would build its own LSH index = massive memory waste
        if self.enable_dedup:
            self.lsh = MinHashLSH(
                threshold=self.config.dedup_threshold,
                num_perm=self.config.dedup_num_perm
            )
            self.seen_hashes = set()
        else:
            self.lsh = None
            self.seen_hashes = None

        # Boilerplate patterns
        self.boilerplate_patterns = [
            r'\[\[Category:.*?\]\]',
            r'\{\{.*?\}\}',
            r'<ref.*?>.*?</ref>',
            r'<references.*?/>',
            r'\|\s*[a-z_]+\s*=',  # Infobox parameters
            r'File:.*?\|',
            r'thumb\|',
            r'{{cite.*?}}',
            r'{{Citation.*?}}',
            r'==\s*References\s*==.*',
            r'==\s*External links\s*==.*',
            r'==\s*See also\s*==.*',
        ]

        # Compile patterns for efficiency
        self.boilerplate_regex = re.compile(
            '|'.join(self.boilerplate_patterns),
            re.IGNORECASE | re.DOTALL
        )

        # Pre-compile ALL regex patterns used throughout sanitization
        self.html_tag_regex = re.compile(r'<[^>]+>')
        self.whitespace_regex = re.compile(r'\s+')
        self.references_regex = re.compile(r'==\s*References\s*==.*', re.DOTALL | re.IGNORECASE)
        self.bibliography_regex = re.compile(r'==\s*Bibliography\s*==.*', re.DOTALL | re.IGNORECASE)
        self.sources_regex = re.compile(r'==\s*Sources\s*==.*', re.DOTALL | re.IGNORECASE)
        self.sentence_split_regex = re.compile(r'[.!?]+')
        self.punctuation_regex = re.compile(r'[.!?,;:]')
        self.special_char_regex = re.compile(r'[^a-zA-Z0-9\s.!?,;:\'\"-]')

        # Statistics tracking
        self.stats = defaultdict(int)

    def sanitize(self, text: str, article_id: Optional[str] = None) -> Optional[str]:
        """
        Apply full sanitization pipeline to text with early-exit optimizations.

        Args:
            text: Raw Wikipedia article text
            article_id: Optional article identifier for deduplication

        Returns:
            Sanitized text or None if article should be filtered
        """
        self.stats['total'] += 1

        # FAST PRE-CHECK: Quick length check before expensive cleaning
        # Reject obviously too-short or too-long articles BEFORE cleaning
        word_estimate = len(text.split())
        if word_estimate < self.config.min_article_length or word_estimate > self.config.max_article_length * 1.5:
            self.stats['invalid_length'] += 1
            return None

        # Stage 1: Preliminary cleaning
        text = self._preliminary_clean(text)
        if not text or len(text) < 50:  # Early exit if cleaning left nothing
            self.stats['empty_after_cleaning'] += 1
            return None

        # Stage 2: Length validation (after cleaning)
        if not self._check_length(text):
            self.stats['invalid_length'] += 1
            return None

        # Stage 3: Language detection (expensive - do after length check)
        if not self._check_language(text):
            self.stats['wrong_language'] += 1
            return None

        # Stage 4: Quality check
        if not self._check_quality(text):
            self.stats['low_quality'] += 1
            return None

        # Stage 5: Repetition check
        if not self._check_repetition(text):
            self.stats['too_repetitive'] += 1
            return None

        # Stage 6: Deduplication (CRITICAL FIX: skip if disabled in multi-worker mode)
        if self.enable_dedup and article_id and not self._check_duplicate(text, article_id):
            self.stats['duplicate'] += 1
            return None

        # Stage 7: Content safety (if enabled)
        if self.config.filter_toxic and not self._check_safety(text):
            self.stats['unsafe_content'] += 1
            return None

        self.stats['passed'] += 1
        return text

    def _preliminary_clean(self, text: str) -> str:
        """Remove HTML, markdown, and Wikipedia markup using pre-compiled patterns."""
        # Remove HTML tags (pre-compiled)
        text = self.html_tag_regex.sub('', text)

        # Remove Wikipedia markup if configured (pre-compiled)
        if self.config.filter_boilerplate:
            text = self.boilerplate_regex.sub('', text)

        # Remove references section if configured (pre-compiled)
        if self.config.remove_references:
            text = self.references_regex.sub('', text)
            text = self.bibliography_regex.sub('', text)
            text = self.sources_regex.sub('', text)

        # Remove multiple spaces and normalize whitespace (pre-compiled)
        text = self.whitespace_regex.sub(' ', text)
        text = text.strip()

        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)

        return text

    def _check_language(self, text: str) -> bool:
        """
        Verify text is in target language with high confidence.
        OPTIMIZED: Uses smaller sample and single-language detection for speed.
        """
        try:
            # OPTIMIZATION: Use smaller sample (200 chars instead of 500)
            # Language is usually detectable from first few sentences
            sample_size = min(200, len(text))
            sample = text[:sample_size]

            # Skip if sample is too short
            if len(sample) < 50:
                return False

            # OPTIMIZATION: Use detect() instead of detect_langs() when we only need top language
            # This is faster as it doesn't compute probabilities for all languages
            detected_lang = langdetect.detect(sample)

            if detected_lang != self.config.target_language:
                return False

            # Only compute detailed probabilities if needed for confidence threshold
            if self.config.min_language_confidence < 1.0:
                detected = langdetect.detect_langs(sample)
                if not detected:
                    return False

                # Check if target language has high enough confidence
                for lang in detected:
                    if lang.lang == self.config.target_language:
                        return lang.prob >= self.config.min_language_confidence

                return False

            return True

        except Exception:
            # If detection fails, assume it's not the target language
            return False

    def _check_length(self, text: str) -> bool:
        """Check if text meets length requirements."""
        words = text.split()
        num_words = len(words)

        # Check total word count
        if num_words < self.config.min_article_length:
            return False
        if num_words > self.config.max_article_length:
            return False

        # Check average word length
        if num_words > 0:
            avg_word_length = sum(len(w) for w in words) / num_words
            if avg_word_length < self.config.min_avg_word_length:
                return False
            if avg_word_length > self.config.max_avg_word_length:
                return False

        return True

    def _check_quality(self, text: str) -> bool:
        """
        Check text quality using optimized heuristics with pre-compiled patterns.

        Quality score based on:
        - Sentence structure
        - Punctuation usage
        - Capitalization patterns
        - Special character ratio
        """
        quality_score = 1.0
        text_len = len(text)

        # Early exit for empty text
        if text_len == 0:
            return False

        # Check sentence structure (pre-compiled pattern)
        sentences = self.sentence_split_regex.split(text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
            # Penalize very short or very long sentences
            if avg_sentence_length < 5 or avg_sentence_length > 50:
                quality_score *= 0.8

        # Check punctuation ratio (pre-compiled pattern)
        words = text.split()
        if words:
            punct_count = len(self.punctuation_regex.findall(text))
            punct_ratio = punct_count / len(words)
            if punct_ratio < 0.05 or punct_ratio > 0.3:
                quality_score *= 0.9

        # Check capitalization (vectorized - faster than generator)
        caps_ratio = sum(1 for c in text if c.isupper()) / text_len
        if caps_ratio < 0.01 or caps_ratio > 0.1:
            quality_score *= 0.9

        # Check special character ratio (pre-compiled pattern)
        special_count = len(self.special_char_regex.findall(text))
        special_ratio = special_count / text_len
        if special_ratio > 0.1:
            quality_score *= 0.8

        return quality_score >= self.config.min_quality_score

    def _check_repetition(self, text: str) -> bool:
        """Check for excessive repetition at character, word, and line level."""
        # Character-level repetition
        char_counts = defaultdict(int)
        for char in text.lower():
            if char.isalpha():
                char_counts[char] += 1

        if char_counts:
            max_char_ratio = max(char_counts.values()) / sum(char_counts.values())
            if max_char_ratio > self.config.max_char_repetition:
                return False

        # Word-level repetition
        words = text.lower().split()
        if words:
            word_counts = defaultdict(int)
            for word in words:
                word_counts[word] += 1

            max_word_ratio = max(word_counts.values()) / len(words)
            if max_word_ratio > self.config.max_word_repetition:
                return False

        # Line-level repetition
        lines = text.split('\n')
        if len(lines) > 1:
            unique_lines = len(set(lines))
            line_ratio = 1 - (unique_lines / len(lines))
            if line_ratio > self.config.max_line_repetition:
                return False

        return True

    def _check_duplicate(self, text: str, article_id: str) -> bool:
        """Check for near-duplicates using MinHash LSH."""
        # Create MinHash for current text
        minhash = MinHash(num_perm=self.config.dedup_num_perm)

        # Generate n-grams
        words = text.lower().split()
        for i in range(len(words) - self.config.dedup_ngram_size + 1):
            ngram = ' '.join(words[i:i + self.config.dedup_ngram_size])
            minhash.update(ngram.encode('utf-8'))

        # Check for exact duplicates using hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return False

        # Check for near-duplicates using LSH
        similar = self.lsh.query(minhash)
        if similar:
            return False

        # Add to index
        self.lsh.insert(article_id, minhash)
        self.seen_hashes.add(text_hash)

        return True

    def _check_safety(self, text: str) -> bool:
        """
        Basic content safety check.

        Note: This is a simple implementation. For production,
        use proper toxicity detection models.
        """
        # Simple keyword-based filtering (very basic)
        # In production, use a proper toxicity classifier

        # For now, just return True as we don't want to filter too aggressively
        # without a proper model
        return True

    def get_statistics(self) -> Dict[str, int]:
        """Get sanitization statistics."""
        return dict(self.stats)

    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats.clear()


def create_sanitizer_from_config(config_path: str) -> WikipediaSanitizer:
    """
    Create sanitizer from JSON configuration file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Configured WikipediaSanitizer instance
    """
    import json

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = SanitizationConfig(**config_dict)
    return WikipediaSanitizer(config)


# Example usage
if __name__ == "__main__":
    # Create sanitizer with default config
    sanitizer = WikipediaSanitizer()

    # Example text
    sample_text = """
    The atomic bombing of Hiroshima occurred on August 6, 1945, when the United States
    dropped an atomic bomb on the city during World War II. The bomb, nicknamed "Little Boy",
    was the first nuclear weapon used in warfare.

    [[Category:World War II]]
    {{cite web|url=example.com|title=Example}}

    The explosion killed an estimated 80,000 people instantly, with total deaths
    reaching about 135,000 from injuries and radiation sickness.
    """

    # Sanitize text
    cleaned = sanitizer.sanitize(sample_text, article_id="hiroshima_001")

    if cleaned:
        print("Sanitized text:")
        print(cleaned)
        print("\nStatistics:")
        print(sanitizer.get_statistics())
    else:
        print("Text was filtered out")