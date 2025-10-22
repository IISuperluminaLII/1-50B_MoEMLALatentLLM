"""
Quality filtering utilities for data sanitization.

Implements FastText-based quality classification and other quality metrics
for filtering low-quality documents from training data.

References:
    Joulin et al. (2016). "Bag of Tricks for Efficient Text Classification."
    arXiv:1607.01759

    Zhou et al. (2025). "Data × LLM: From Principles to Practices."
    arXiv:2505.18458
"""
import warnings
from typing import Optional, List, Tuple


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
    ):
        """
        Initialize FastText quality classifier.

        Args:
            model_path: Path to trained FastText model
            threshold: Quality score threshold (0-1)
        """
        self.model_path = model_path
        self.threshold = threshold
        self.model = None

        # Try to load model if path provided
        if model_path:
            self._load_model(model_path)
        else:
            warnings.warn(
                "No FastText model path provided. Using placeholder implementation.",
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