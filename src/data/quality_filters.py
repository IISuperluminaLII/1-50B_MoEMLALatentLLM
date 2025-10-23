"""
Quality filtering utilities for data sanitization.

Implements FastText-based quality classification, KenLM perplexity filtering,
and ensemble methods for filtering low-quality documents from training data.

References:
    Joulin et al. (2016). "Bag of Tricks for Efficient Text Classification."
    arXiv:1607.01759

    Li et al. (2024). "DataComp-LM: In Search of the Next Generation of
    Training Sets for Language Models." arXiv:2406.11794

    Kim et al. (2024). "Rethinking KenLM: Good and Bad Model Ensembles for
    Efficient Text Quality Filtering in Large Web Corpora." arXiv:2409.09613

    Zhou et al. (2025). "A Survey of LLM × DATA."
    arXiv:2505.18458
"""
import warnings
from typing import Optional, List, Tuple
from pathlib import Path


class FastTextQualityClassifier:
    """
    FastText-based quality classifier for document filtering.

    Uses a trained FastText model to classify documents as high or low quality.
    This is a placeholder implementation that can be extended with actual
    FastText models when available.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        allow_fallback: bool = False,
    ):
        """
        Initialize FastText quality classifier.

        Args:
            model_path: Path to trained FastText model
            threshold: Quality score threshold (0-1)
            allow_fallback: If True, falls back to heuristic scoring when no model
                          is provided. If False (default), raises ValueError.

        Raises:
            ValueError: If model_path is None and allow_fallback is False.

        References:
            Joulin et al. (2016). "Bag of Tricks for Efficient Text Classification."
        """
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.allow_fallback = allow_fallback

        # Try to load model if path provided
        if model_path:
            self._load_model(model_path)
        else:
            if not allow_fallback:
                raise ValueError(
                    "FastTextQualityClassifier requires a trained model to comply with "
                    "Joulin et al. (2016) methodology. Either:\n"
                    "  1. Provide model_path to a trained FastText model (.bin file)\n"
                    "  2. Set allow_fallback=True to use heuristic scoring (NOT recommended)\n"
                    "  3. Disable quality filtering in your pipeline config\n"
                    "\n"
                    "Note: Heuristic fallback does not implement the cited SOTA quality "
                    "classification and may produce unreliable results."
                )
            warnings.warn(
                "No FastText model path provided. Using placeholder heuristic implementation. "
                "This does NOT implement the Joulin et al. (2016) FastText methodology.",
                UserWarning,
            )

    def _load_model(self, model_path: str):
        """
        Load FastText model from file.

        Args:
            model_path: Path to model file
        """
        try:
            import fasttext
            self.model = fasttext.load_model(model_path)
        except ImportError:
            warnings.warn(
                "fasttext package not installed. Install with: pip install fasttext",
                UserWarning,
            )
            self.model = None
        except Exception as e:
            warnings.warn(
                f"Could not load FastText model from {model_path}: {e}",
                UserWarning,
            )
            self.model = None

    def predict(self, text: str) -> bool:
        """
        Predict if document is high quality.

        Args:
            text: Document text

        Returns:
            True if document passes quality threshold
        """
        score = self.predict_score(text)
        return score >= self.threshold

    def predict_score(self, text: str) -> float:
        """
        Predict quality score for document.

        Args:
            text: Document text

        Returns:
            Quality score (0-1)
        """
        if self.model:
            # Use actual FastText model
            try:
                # FastText returns (labels, probabilities)
                labels, probs = self.model.predict(text)
                # Assume model is trained with __label__good and __label__bad
                if "__label__good" in labels[0]:
                    return float(probs[0])
                else:
                    return 1.0 - float(probs[0])
            except Exception:
                # Fallback to heuristics
                pass

        # Placeholder heuristic-based scoring
        return self._heuristic_score(text)

    def _heuristic_score(self, text: str) -> float:
        """
        Compute heuristic-based quality score.

        This is a placeholder that uses simple heuristics.
        In production, this should be replaced with a trained model.

        Args:
            text: Document text

        Returns:
            Heuristic quality score (0-1)
        """
        if not text:
            return 0.0

        score = 1.0

        # Length check
        if len(text) < 100:
            score *= 0.5
        elif len(text) < 200:
            score *= 0.8

        # Check for minimum word count
        words = text.split()
        if len(words) < 50:
            score *= 0.6

        # Check for repetition
        unique_words = set(words)
        if len(words) > 0:
            diversity_ratio = len(unique_words) / len(words)
            if diversity_ratio < 0.3:
                score *= 0.5

        # Check for alpha ratio (letters vs other characters)
        alpha_count = sum(1 for c in text if c.isalpha())
        if len(text) > 0:
            alpha_ratio = alpha_count / len(text)
            if alpha_ratio < 0.5:
                score *= 0.7

        # Check for excessive uppercase
        upper_count = sum(1 for c in text if c.isupper())
        if alpha_count > 0:
            upper_ratio = upper_count / alpha_count
            if upper_ratio > 0.3:
                score *= 0.8

        return min(1.0, max(0.0, score))

    def predict_batch(self, texts: List[str]) -> List[bool]:
        """
        Predict quality for batch of documents.

        Args:
            texts: List of document texts

        Returns:
            List of boolean predictions
        """
        return [self.predict(text) for text in texts]

    def predict_scores_batch(self, texts: List[str]) -> List[float]:
        """
        Predict quality scores for batch of documents.

        Args:
            texts: List of document texts

        Returns:
            List of quality scores
        """
        return [self.predict_score(text) for text in texts]

    def train(
        self,
        train_data: List[Tuple[str, str]],
        model_output_path: str,
        **kwargs,
    ):
        """
        Train FastText model on labeled data.

        Args:
            train_data: List of (text, label) tuples
            model_output_path: Path to save trained model
            **kwargs: Additional FastText training parameters
        """
        try:
            import fasttext
            import tempfile

            # Write training data to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                for text, label in train_data:
                    # FastText format: __label__X text
                    f.write(f"__label__{label} {text}\n")
                temp_path = f.name

            # Train model
            model = fasttext.train_supervised(
                input=temp_path,
                epoch=kwargs.get("epoch", 25),
                lr=kwargs.get("lr", 0.1),
                wordNgrams=kwargs.get("wordNgrams", 2),
                verbose=kwargs.get("verbose", 2),
                minCount=kwargs.get("minCount", 1),
            )

            # Save model
            model.save_model(model_output_path)

            # Load the trained model
            self.model = model
            self.model_path = model_output_path

            print(f"Model trained and saved to {model_output_path}")

        except ImportError:
            raise ImportError(
                "fasttext package required for training. Install with: pip install fasttext"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to train FastText model: {e}")


class KenLMQualityFilter:
    """
    KenLM-based perplexity filter for document quality assessment.

    Uses language model perplexity as a proxy for document quality. Low
    perplexity indicates text that matches the language model's learned
    distribution (typically high-quality, coherent text).

    References:
        Kim et al. (2024). "Rethinking KenLM." arXiv:2409.09613
        Li et al. (2024). "DataComp-LM." arXiv:2406.11794
        Wenzek et al. (2019). "CCNet." arXiv:1911.00359
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_perplexity: float = 1000.0,
        allow_fallback: bool = False,
    ):
        """
        Initialize KenLM quality filter.

        Args:
            model_path: Path to KenLM .arpa or .bin model file
            max_perplexity: Maximum perplexity threshold (higher = more permissive)
            allow_fallback: If True, accepts all documents when no model is provided.
                          If False (default), raises ValueError.

        Raises:
            ValueError: If model_path is None and allow_fallback is False.

        References:
            Kim et al. (2024). "Rethinking KenLM." arXiv:2409.09613
            Li et al. (2024). "DataComp-LM." arXiv:2406.11794
            Wenzek et al. (2019). "CCNet." arXiv:1911.00359
        """
        self.model_path = model_path
        self.max_perplexity = max_perplexity
        self.model = None
        self.allow_fallback = allow_fallback

        if model_path:
            self._load_model(model_path)
        else:
            if not allow_fallback:
                raise ValueError(
                    "KenLMQualityFilter requires a trained language model to comply with "
                    "Kim et al. (2024) and Wenzek et al. (2019) perplexity filtering. Either:\n"
                    "  1. Provide model_path to a KenLM model (.arpa or .bin file)\n"
                    "  2. Set allow_fallback=True to accept all documents (NOT recommended)\n"
                    "  3. Disable KenLM filtering in your pipeline config\n"
                    "\n"
                    "Note: Fallback mode does not implement perplexity-based filtering "
                    "and accepts all documents regardless of quality."
                )
            warnings.warn(
                "No KenLM model path provided. Filter will accept all documents. "
                "This does NOT implement perplexity-based filtering from Kim et al. (2024).",
                UserWarning,
            )

    def _load_model(self, model_path: str):
        """
        Load KenLM model from file.

        Args:
            model_path: Path to .arpa or .bin model file
        """
        try:
            import kenlm
            model_file = Path(model_path)

            if not model_file.exists():
                raise FileNotFoundError(f"KenLM model not found: {model_path}")

            self.model = kenlm.Model(str(model_file))

        except ImportError:
            warnings.warn(
                "kenlm package not installed. Install with: pip install https://github.com/kpu/kenlm/archive/master.zip",
                UserWarning,
            )
            self.model = None
        except Exception as e:
            warnings.warn(
                f"Could not load KenLM model from {model_path}: {e}",
                UserWarning,
            )
            self.model = None

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of text using KenLM model.

        Args:
            text: Document text

        Returns:
            Perplexity score (lower = better quality)
        """
        if not self.model:
            # No model loaded, return neutral perplexity
            return self.max_perplexity / 2

        try:
            # KenLM computes log10 probability
            # Perplexity = 10^(-log_prob / num_words)
            log_prob = self.model.score(text)
            words = text.split()
            num_words = max(len(words), 1)  # Avoid division by zero

            # Convert log10 prob to perplexity
            perplexity = 10 ** (-log_prob / num_words)

            return perplexity

        except Exception as e:
            warnings.warn(f"Error computing perplexity: {e}", UserWarning)
            return self.max_perplexity

    def predict(self, text: str) -> bool:
        """
        Predict if document passes quality threshold.

        Args:
            text: Document text

        Returns:
            True if perplexity is below max_perplexity threshold
        """
        perplexity = self.compute_perplexity(text)
        return perplexity <= self.max_perplexity

    def predict_score(self, text: str) -> float:
        """
        Compute quality score (inverse of normalized perplexity).

        Args:
            text: Document text

        Returns:
            Quality score in [0, 1], where 1 is best quality
        """
        perplexity = self.compute_perplexity(text)

        # Normalize perplexity to [0, 1] score
        # score = 1 - min(perplexity / max_perplexity, 1.0)
        # This maps perplexity=0 -> score=1, perplexity>=max -> score=0
        if perplexity >= self.max_perplexity:
            return 0.0

        return 1.0 - (perplexity / self.max_perplexity)


class QualityEnsemble:
    """
    Ensemble quality filter combining FastText and KenLM.

    Combines multiple quality signals (FastText classification, KenLM perplexity)
    using weighted averaging to make final quality decisions.

    This implements the ensemble approach described in DataComp-LM and used
    in DeepSeek data preprocessing.
    """

    def __init__(
        self,
        fasttext_filter: Optional[FastTextQualityClassifier] = None,
        kenlm_filter: Optional[KenLMQualityFilter] = None,
        fasttext_weight: float = 0.6,
        kenlm_weight: float = 0.4,
        ensemble_threshold: float = 0.5,
    ):
        """
        Initialize quality ensemble.

        Args:
            fasttext_filter: FastText quality classifier (optional)
            kenlm_filter: KenLM perplexity filter (optional)
            fasttext_weight: Weight for FastText score in ensemble
            kenlm_weight: Weight for KenLM score in ensemble
            ensemble_threshold: Final threshold for accepting documents
        """
        self.fasttext_filter = fasttext_filter
        self.kenlm_filter = kenlm_filter
        self.fasttext_weight = fasttext_weight
        self.kenlm_weight = kenlm_weight
        self.ensemble_threshold = ensemble_threshold

        # Normalize weights
        total_weight = 0.0
        if fasttext_filter:
            total_weight += fasttext_weight
        if kenlm_filter:
            total_weight += kenlm_weight

        if total_weight > 0:
            self.fasttext_weight = fasttext_weight / total_weight
            self.kenlm_weight = kenlm_weight / total_weight
        else:
            raise ValueError("At least one filter (FastText or KenLM) must be provided")

    def predict_score(self, text: str) -> float:
        """
        Compute ensemble quality score.

        Args:
            text: Document text

        Returns:
            Weighted average quality score in [0, 1]
        """
        score = 0.0

        if self.fasttext_filter:
            fasttext_score = self.fasttext_filter.predict_score(text)
            score += self.fasttext_weight * fasttext_score

        if self.kenlm_filter:
            kenlm_score = self.kenlm_filter.predict_score(text)
            score += self.kenlm_weight * kenlm_score

        return score

    def predict(self, text: str) -> bool:
        """
        Predict if document passes ensemble quality threshold.

        Args:
            text: Document text

        Returns:
            True if ensemble score >= threshold
        """
        score = self.predict_score(text)
        return score >= self.ensemble_threshold