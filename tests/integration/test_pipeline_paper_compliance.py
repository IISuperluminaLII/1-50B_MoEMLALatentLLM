"""
Integration tests for paper-compliant pipeline implementations.

This test module validates that the data preprocessing pipeline correctly
implements the algorithms and requirements specified in cited academic papers:

- Lee et al. (2022): MinHash LSH with Jaccard verification
- Joulin et al. (2016): FastText quality classification requirement
- Kim et al. (2024): KenLM perplexity filtering requirement
- Xie et al. (2023): DoReMi with reference loss requirement

These tests ensure the implementation doesn't silently fall back to
placeholder heuristics that diverge from the cited methodologies.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.deduplication import MinHashDeduplicator
from src.data.quality_filters import FastTextQualityClassifier, KenLMQualityFilter
from src.data.domain_mixing import DomainMixer
from src.data.pipeline import DataPipeline, PipelineConfig


class TestMinHashJaccardVerification:
    """
    Test MinHash LSH with Jaccard verification per Lee et al. (2022).

    Reference:
        Lee et al. (2022). "Deduplicating Training Data Makes Language Models Better."
        arXiv:2107.06499
    """

    def test_minhash_verifies_jaccard_before_filtering(self):
        """
        Test that MinHash LSH candidates are verified with exact Jaccard similarity.

        This test ensures unrelated documents that collide in LSH buckets
        are NOT incorrectly filtered as duplicates.
        """
        # Create deduplicator with threshold=0.8
        dedup = MinHashDeduplicator(
            num_perm=128,
            threshold=0.8,
            n_gram=13,
            seed=42
        )

        # Create test documents:
        # We need documents with controlled Jaccard similarities
        # Using longer repeated text to ensure high overlap

        base_text = "The quick brown fox jumps over the lazy dog. " * 10

        documents = [
            base_text,  # doc_1: Original
            base_text,  # doc_2: Exact duplicate (Jaccard = 1.0)
            base_text + "The quick brown fox jumps over the lazy dog. " * 2,  # doc_3: Very high similarity (Jaccard ~0.83)
            base_text + "A different sentence added here. " * 5,  # doc_4: Medium similarity (Jaccard ~0.67)
            "A completely different document about machine learning models. " * 10,  # doc_5: Low similarity
        ]

        doc_ids = [f"doc_{i}" for i in range(1, 6)]

        # Run deduplication
        unique_docs, unique_ids, _ = dedup.deduplicate(documents, doc_ids)

        # Verify Jaccard similarities (for debugging)
        jac_1_2 = dedup.compute_jaccard_similarity(documents[0], documents[1])
        jac_1_3 = dedup.compute_jaccard_similarity(documents[0], documents[2])
        jac_1_4 = dedup.compute_jaccard_similarity(documents[0], documents[3])

        # Assertions:
        # 1. doc_2 (exact duplicate) should be removed: Jaccard=1.0 >= 0.8
        assert "doc_2" not in unique_ids, f"Exact duplicate should be removed (Jaccard={jac_1_2:.3f})"

        # 2. doc_3 (high similarity) should be checked against threshold
        if jac_1_3 >= 0.8:
            assert "doc_3" not in unique_ids, f"High similarity document (Jaccard={jac_1_3:.3f} >= 0.8) should be removed"
        else:
            assert "doc_3" in unique_ids, f"doc_3 (Jaccard={jac_1_3:.3f} < 0.8) should survive"

        # 3. doc_4 (medium similarity) should survive if Jaccard < 0.8
        if jac_1_4 < 0.8:
            assert "doc_4" in unique_ids, f"Medium similarity document (Jaccard={jac_1_4:.3f} < 0.8) should survive"

        # 4. doc_5 (low similarity) should survive
        assert "doc_5" in unique_ids, "Low similarity document should survive"

        # 5. doc_1 (original) should be kept
        assert "doc_1" in unique_ids, "Original document should be kept"

        # Verify at least exact duplicate was removed
        assert len(unique_docs) < len(documents), "Should have removed at least one duplicate"
        assert "doc_2" not in unique_ids, "Exact duplicate must be removed"

    def test_minhash_computes_actual_jaccard_not_lsh_estimate(self):
        """
        Test that deduplication uses exact Jaccard computation, not LSH estimate.

        LSH provides candidates (approximate), but final decision must use
        exact Jaccard similarity per Broder (1997).
        """
        dedup = MinHashDeduplicator(
            num_perm=64,  # Lower num_perm increases LSH collisions
            threshold=0.7,
            n_gram=5,
            seed=42
        )

        # Create documents with controlled similarity
        # Using repeated patterns to ensure predictable Jaccard values
        base = "hello world " * 50
        doc1 = base
        doc2 = base + "extra text " * 25  # Adds ~33% more, Jaccard should be ~0.75
        doc3 = "completely different text " * 50  # Low overlap

        documents = [doc1, doc2, doc3]
        doc_ids = ["doc_1", "doc_2", "doc_3"]

        unique_docs, unique_ids, _ = dedup.deduplicate(documents, doc_ids)

        # Compute exact Jaccard similarities
        jaccard_1_2 = dedup.compute_jaccard_similarity(doc1, doc2)
        jaccard_1_3 = dedup.compute_jaccard_similarity(doc1, doc3)

        # Verify deduplication results match Jaccard thresholds
        # Key assertion: results should be determined by exact Jaccard, not LSH estimate
        if jaccard_1_2 >= 0.7:
            assert "doc_2" not in unique_ids, f"doc_2 (Jaccard={jaccard_1_2:.3f} >= 0.7) should be filtered"
        else:
            assert "doc_2" in unique_ids, f"doc_2 (Jaccard={jaccard_1_2:.3f} < 0.7) should survive"

        # doc_3 should always survive (very different text)
        assert "doc_3" in unique_ids, f"doc_3 (Jaccard={jaccard_1_3:.3f}) should survive (low similarity)"

        # doc_1 should always be kept (original)
        assert "doc_1" in unique_ids, "doc_1 (original) should be kept"

    def test_streaming_detects_cross_batch_duplicates(self):
        """
        Test that streaming deduplication detects duplicates across batch boundaries.

        Regression test for bug where deduplicate_streaming() would re-instantiate
        stored_texts for each batch, causing cross-batch near-duplicates to survive
        even though Lee et al. (2022) guarantees they should be removed.
        """
        pytest.importorskip("datasketch")

        # Create deduplicator with small batch size to force multiple batches
        dedup = MinHashDeduplicator(
            num_perm=128,
            threshold=0.8,
            n_gram=13,
            seed=42
        )

        # Create documents with controlled cross-batch duplicates
        base_text = "The quick brown fox jumps over the lazy dog. " * 10

        # Batch 1: doc_1, doc_2 (doc_2 is different)
        # Batch 2: doc_3 (near-duplicate of doc_1 from previous batch)
        documents = [
            ("doc_1", base_text),  # Batch 1
            ("doc_2", "Completely different text about machine learning and AI systems. " * 10),  # Batch 1
            ("doc_3", base_text + "The quick brown fox jumps over the lazy dog. " * 2),  # Batch 2: near-duplicate of doc_1
        ]

        # Compute expected Jaccard similarity
        jaccard_1_3 = dedup.compute_jaccard_similarity(documents[0][1], documents[2][1])

        # Process in streaming mode with batch_size=2 (forces doc_3 into second batch)
        unique_docs = list(dedup.deduplicate_streaming(iter(documents), batch_size=2))

        # Extract IDs
        unique_ids = [doc_id for doc_id, _ in unique_docs]

        # Assertions:
        # 1. doc_1 and doc_2 should be kept (different docs in batch 1)
        assert "doc_1" in unique_ids, "Original doc_1 should be kept"
        assert "doc_2" in unique_ids, "Different doc_2 should be kept"

        # 2. doc_3 should be removed if Jaccard >= 0.8 (cross-batch duplicate detection)
        if jaccard_1_3 >= 0.8:
            assert "doc_3" not in unique_ids, \
                f"doc_3 (Jaccard={jaccard_1_3:.3f} >= 0.8) should be removed even though it's in a different batch"
            assert len(unique_docs) == 2, "Should keep exactly 2 documents (doc_1, doc_2)"
        else:
            # If Jaccard < 0.8, doc_3 should survive
            assert "doc_3" in unique_ids, \
                f"doc_3 (Jaccard={jaccard_1_3:.3f} < 0.8) should survive"
            assert len(unique_docs) == 3, "Should keep all 3 documents"

    def test_pipeline_streaming_uses_jaccard_verification(self, temp_dir):
        """
        Test that streaming pipeline also uses Jaccard verification.

        Ensures the fix applies to both batch and streaming modes and that
        high-similarity documents (not just exact duplicates) are filtered
        when Jaccard >= threshold.
        """
        # Create input file with duplicates using longer text for reliable Jaccard
        input_file = temp_dir / "stream_input.jsonl"

        base_doc = "The quick brown fox jumps over the lazy dog. " * 10
        documents = [
            {"id": "doc_1", "text": base_doc},
            {"id": "doc_2", "text": base_doc},  # Exact duplicate (Jaccard = 1.0)
            {"id": "doc_3", "text": base_doc + "The quick brown fox jumps over the lazy dog. " * 2},  # High similarity
            {"id": "doc_4", "text": "A completely different document about AI systems and technology. " * 10},
        ]

        with open(input_file, 'w') as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")

        # Compute exact Jaccard similarity between doc_1 and doc_3
        dedup = MinHashDeduplicator(num_perm=128, threshold=0.8, n_gram=13, seed=42)
        jaccard_1_3 = dedup.compute_jaccard_similarity(documents[0]["text"], documents[2]["text"])

        # Create pipeline config with MinHash deduplication
        config = PipelineConfig(
            input_path=str(input_file),
            output_dir=str(temp_dir / "output"),
            enable_cleaning=False,
            enable_deduplication=True,
            enable_heuristic_filters=False,
            enable_quality_filters=False,
            enable_domain_mixing=False,
            dedup_config={
                "method": "minhash",
                "threshold": 0.8,
                "num_perm": 128,
            },
            show_progress=False,
        )

        pipeline = DataPipeline(config)
        stats = pipeline.process_and_save()

        # Load output to verify which documents survived
        output_file = Path(config.output_dir) / "final.jsonl"
        assert output_file.exists(), "Output file should exist"

        output_docs = []
        with open(output_file, 'r') as f:
            for line in f:
                output_docs.append(json.loads(line))

        output_ids = {doc["id"] for doc in output_docs}

        # Assertions based on Jaccard similarity
        if jaccard_1_3 >= 0.8:
            # High-Jaccard case: both doc_2 (exact) and doc_3 (high-similarity) should be removed
            assert "doc_2" not in output_ids, "Exact duplicate (doc_2) must be removed"
            assert "doc_3" not in output_ids, \
                f"High-similarity doc_3 (Jaccard={jaccard_1_3:.3f} >= 0.8) must be removed"

            # Both doc_2 and doc_3 should be filtered
            assert stats.documents_deduplicated >= 2, \
                f"Should remove both exact duplicate and high-similarity document, got {stats.documents_deduplicated}"

            # Only doc_1 and doc_4 should survive
            assert stats.total_output_documents == 2, \
                f"Should keep exactly 2 documents (doc_1, doc_4), got {stats.total_output_documents}"
            assert output_ids == {"doc_1", "doc_4"}, \
                f"Only doc_1 and doc_4 should survive, got {output_ids}"
        else:
            # Low-Jaccard case: only doc_2 (exact duplicate) should be removed, doc_3 survives
            assert "doc_2" not in output_ids, "Exact duplicate (doc_2) must be removed"
            assert "doc_3" in output_ids, \
                f"doc_3 (Jaccard={jaccard_1_3:.3f} < 0.8) should survive"

            assert stats.documents_deduplicated >= 1, "At least exact duplicate must be removed"
            assert stats.total_output_documents == 3, \
                f"Should keep 3 documents (doc_1, doc_3, doc_4), got {stats.total_output_documents}"
            assert output_ids == {"doc_1", "doc_3", "doc_4"}, \
                f"doc_1, doc_3, doc_4 should survive, got {output_ids}"

        # doc_1 and doc_4 should always be kept
        assert "doc_1" in output_ids, "Original doc_1 should always be kept"
        assert "doc_4" in output_ids, "Different doc_4 should always be kept"


class TestQualityFilterEnforcement:
    """
    Test quality filter model requirements per Joulin (2016) and Kim (2024).

    References:
        Joulin et al. (2016). "Bag of Tricks for Efficient Text Classification."
        Kim et al. (2024). "DataComp-LM: In Search of the Next Generation of
        Training Sets for Language Models."
    """

    def test_fasttext_requires_model_by_default(self):
        """
        Test that FastText classifier raises ValueError when no model is provided.

        This ensures compliance with Joulin et al. (2016) methodology and prevents
        silent fallback to heuristic scoring.
        """
        with pytest.raises(ValueError, match="FastTextQualityClassifier requires a trained model"):
            FastTextQualityClassifier(
                model_path=None,
                threshold=0.5,
                allow_fallback=False  # Strict mode
            )

    def test_fasttext_allows_fallback_when_explicitly_enabled(self):
        """
        Test that fallback is allowed when explicitly requested.

        Users can opt into heuristic fallback by setting allow_fallback=True,
        but they must do so explicitly.
        """
        # Should NOT raise when allow_fallback=True
        classifier = FastTextQualityClassifier(
            model_path=None,
            threshold=0.5,
            allow_fallback=True
        )

        # Should emit warning about using heuristic fallback
        assert classifier.model is None
        assert classifier.allow_fallback is True

    def test_kenlm_requires_model_by_default(self):
        """
        Test that KenLM filter raises ValueError when no model is provided.

        This ensures compliance with Kim et al. (2024) and Wenzek et al. (2019)
        perplexity filtering methodology.
        """
        with pytest.raises(ValueError, match="KenLMQualityFilter requires a trained language model"):
            KenLMQualityFilter(
                model_path=None,
                max_perplexity=1000.0,
                allow_fallback=False  # Strict mode
            )

    def test_kenlm_allows_fallback_when_explicitly_enabled(self):
        """
        Test that fallback is allowed when explicitly requested.
        """
        # Should NOT raise when allow_fallback=True
        filter = KenLMQualityFilter(
            model_path=None,
            max_perplexity=1000.0,
            allow_fallback=True
        )

        assert filter.model is None
        assert filter.allow_fallback is True

    def test_pipeline_fails_loudly_without_quality_models(self, temp_dir):
        """
        Test that pipeline initialization fails when quality filters are enabled
        but no models are provided.
        """
        config = PipelineConfig(
            input_path="dummy.jsonl",
            output_dir=str(temp_dir / "output"),
            enable_cleaning=False,
            enable_deduplication=False,
            enable_heuristic_filters=False,
            enable_quality_filters=True,  # Enable quality filtering
            enable_domain_mixing=False,
            quality_config={
                "use_fasttext": True,
                "fasttext_model_path": None,  # No model provided
                "allow_fallback": False,  # Strict mode (default)
            },
        )

        # Should raise ValueError during pipeline initialization
        with pytest.raises(ValueError, match="FastTextQualityClassifier requires a trained model"):
            pipeline = DataPipeline(config)

    def test_pipeline_accepts_model_path_without_error(self, temp_dir):
        """
        Test that providing a model path (even if file doesn't exist) doesn't raise
        the "model required" error.

        The actual model loading may fail (file not found), but that's a different
        error than the "model required" ValueError.
        """
        # Create a dummy model file path (doesn't need to exist for this test)
        model_path = temp_dir / "fasttext_model.bin"

        # This should NOT raise ValueError about "requires a trained model"
        # It may emit a warning about failing to load, but that's expected
        classifier = FastTextQualityClassifier(
            model_path=str(model_path),
            threshold=0.5,
        )

        # Classifier should have been created (even if model failed to load)
        assert classifier is not None
        assert classifier.model_path == str(model_path)


class TestDoReMiReferenceRequirement:
    """
    Test DoReMi reference loss requirement per Xie et al. (2023).

    Reference:
        Xie et al. (2023). "DoReMi: Optimizing Data Mixtures Speeds Up Language
        Model Pretraining." arXiv:2305.10429
    """

    def test_doremi_warns_without_reference_losses(self):
        """
        Test that DoReMi emits warning when reference_losses are not provided.

        DoReMi algorithm requires reference model losses for excess loss computation.
        Without them, the algorithm falls back to mean-centered losses which is
        NOT compliant with the paper.
        """
        mixer = DomainMixer(composition="doremi", random_seed=42)

        domain_losses = {
            "code": 4.5,
            "common_crawl": 2.0,
            "wikipedia": 3.0,
            "academic": 3.5,
            "books": 3.2,
            "news": 2.8,
            "social": 3.0,
        }

        # Should emit UserWarning about missing reference_losses
        with pytest.warns(UserWarning, match="DoReMi optimization called without reference_losses"):
            mixer.optimize_weights_doremi(
                domain_losses,
                reference_losses=None  # Missing reference losses
            )

    def test_doremi_succeeds_with_reference_losses(self):
        """
        Test that DoReMi runs without warnings when reference_losses are provided.
        """
        mixer = DomainMixer(composition="doremi", random_seed=42)

        # Reference model baseline (uniform)
        reference_losses = {
            "code": 3.0,
            "common_crawl": 3.0,
            "wikipedia": 3.0,
            "academic": 3.0,
            "books": 3.0,
            "news": 3.0,
            "social": 3.0,
        }

        # Proxy model losses
        domain_losses = {
            "code": 4.5,  # High excess loss (+1.5)
            "common_crawl": 2.0,  # Low excess loss (-1.0)
            "wikipedia": 3.0,  # Zero excess loss
            "academic": 3.2,
            "books": 3.1,
            "news": 3.3,
            "social": 3.4,
        }

        # Should NOT emit warning with reference losses
        import warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            mixer.optimize_weights_doremi(
                domain_losses,
                reference_losses=reference_losses
            )

        # Check that no DoReMi-related warnings were emitted
        doremi_warnings = [w for w in warning_list if "DoReMi" in str(w.message)]
        assert len(doremi_warnings) == 0, "Should not emit DoReMi warnings when reference_losses provided"

    def test_doremi_upweights_high_excess_loss_domains(self):
        """
        Test that DoReMi correctly upweights domains with high excess loss.

        Per Xie et al. (2023), domains where proxy_loss > reference_loss
        should be upweighted (more training emphasis needed).
        """
        mixer = DomainMixer(composition="doremi", random_seed=42)

        # Reference baseline
        reference_losses = {
            "code": 3.0,
            "common_crawl": 3.0,
            "wikipedia": 3.0,
            "academic": 3.0,
            "books": 3.0,
            "news": 3.0,
            "social": 3.0,
        }

        # Code has HIGH excess loss (+2.5), common_crawl has LOW excess loss (-1.5)
        domain_losses = {
            "code": 5.5,  # Excess: +2.5
            "common_crawl": 1.5,  # Excess: -1.5
            "wikipedia": 3.0,  # Excess: 0
            "academic": 3.0,
            "books": 3.0,
            "news": 3.0,
            "social": 3.0,
        }

        # Get initial weights (uniform)
        initial_code_weight = mixer.domain_weights.weights["code"]
        initial_cc_weight = mixer.domain_weights.weights["common_crawl"]

        # Run 5 iterations
        for _ in range(5):
            mixer.optimize_weights_doremi(
                domain_losses,
                reference_losses=reference_losses
            )

        # Get final weights
        final_code_weight = mixer.domain_weights.weights["code"]
        final_cc_weight = mixer.domain_weights.weights["common_crawl"]

        # Code (high excess loss) should have increased weight
        assert final_code_weight > initial_code_weight, \
            f"Code weight should increase (high excess loss). Initial: {initial_code_weight:.4f}, Final: {final_code_weight:.4f}"

        # Common crawl (low/negative excess loss) should have decreased weight
        assert final_cc_weight < initial_cc_weight, \
            f"Common crawl weight should decrease (low excess loss). Initial: {initial_cc_weight:.4f}, Final: {final_cc_weight:.4f}"

        # Code should have higher weight than common_crawl
        assert final_code_weight > final_cc_weight, \
            f"Code (excess +2.5) should have more weight than common_crawl (excess -1.5)"


class TestEndToEndPipelineCompliance:
    """
    End-to-end integration tests for paper-compliant pipeline execution.
    """

    def test_full_pipeline_with_strict_requirements(self, temp_dir):
        """
        Test that a full pipeline run enforces all paper requirements.

        This test verifies that:
        1. MinHash uses Jaccard verification
        2. Quality filters require models (or explicit fallback)
        3. Pipeline fails loudly when requirements aren't met
        """
        # Create input data
        input_data = [
            {"id": "doc_1", "text": "The quick brown fox jumps over the lazy dog"},
            {"id": "doc_2", "text": "The quick brown fox jumps over the lazy dog"},  # Duplicate
            {"id": "doc_3", "text": "A different document about machine learning"},
        ]

        # Test 1: Pipeline with deduplication (should work)
        config = PipelineConfig(
            input_path="dummy.jsonl",
            output_dir=str(temp_dir / "dedup_output"),
            enable_cleaning=False,
            enable_deduplication=True,
            enable_heuristic_filters=False,
            enable_quality_filters=False,
            enable_domain_mixing=False,
            dedup_config={"method": "minhash", "threshold": 0.8},
            show_progress=False,
        )

        pipeline = DataPipeline(config)
        stats = pipeline.process_and_save(input_data=input_data)

        # Should remove exact duplicate
        assert stats.documents_deduplicated == 1
        assert stats.total_output_documents == 2

        # Test 2: Pipeline with quality filters but no models (should fail)
        config_fail = PipelineConfig(
            input_path="dummy.jsonl",
            output_dir=str(temp_dir / "quality_output"),
            enable_cleaning=False,
            enable_deduplication=False,
            enable_heuristic_filters=False,
            enable_quality_filters=True,
            enable_domain_mixing=False,
            quality_config={
                "use_fasttext": True,
                "fasttext_model_path": None,  # No model
                "allow_fallback": False,  # Strict mode
            },
            show_progress=False,
        )

        with pytest.raises(ValueError, match="FastTextQualityClassifier requires a trained model"):
            pipeline_fail = DataPipeline(config_fail)

    def test_full_pipeline_with_doremi_and_reference_losses(self, temp_dir):
        """
        Test full pipeline with DoReMi domain mixing and reference losses.
        """
        # Create input with domain-labeled documents
        input_data = [
            {"id": f"code_{i}", "text": "def foo(): pass\nclass Bar: pass\nimport sys\n" * 20}
            for i in range(30)
        ] + [
            {"id": f"web_{i}", "text": "This is generic web content about various topics " * 20}
            for i in range(30)
        ] + [
            {"id": f"wiki_{i}", "text": "== Article Title ==\n[[Category:Test]]\nWikipedia content " * 20}
            for i in range(30)
        ]

        # Create loss feedback file
        loss_feedback_file = temp_dir / "loss_feedback.json"
        loss_feedback = {
            "domain_losses": {
                "code": 4.5,
                "common_crawl": 2.0,
                "wikipedia": 3.0,
                "academic": 3.0,
                "books": 3.0,
                "news": 3.0,
                "social": 3.0,
            },
            "reference_losses": {
                "code": 3.0,
                "common_crawl": 3.0,
                "wikipedia": 3.0,
                "academic": 3.0,
                "books": 3.0,
                "news": 3.0,
                "social": 3.0,
            },
            "reference_weights": {
                "code": 0.143,
                "common_crawl": 0.143,
                "wikipedia": 0.143,
                "academic": 0.143,
                "books": 0.143,
                "news": 0.143,
                "social": 0.142,
            }
        }

        with open(loss_feedback_file, 'w') as f:
            json.dump(loss_feedback, f)

        # Create pipeline with DoReMi
        config = PipelineConfig(
            input_path="dummy.jsonl",
            output_dir=str(temp_dir / "doremi_output"),
            enable_cleaning=False,
            enable_deduplication=False,
            enable_heuristic_filters=False,
            enable_quality_filters=False,
            enable_domain_mixing=True,
            domain_config={
                "composition": "doremi",
                "num_iterations": 3,
                "doremi_loss_feedback_path": str(loss_feedback_file),
                "target_size": 90,
                "random_seed": 42,
            },
            show_progress=False,
        )

        pipeline = DataPipeline(config)
        stats = pipeline.process_and_save(input_data=input_data)

        # Verify DoReMi ran with reference losses
        assert stats.domain_stats is not None
        assert stats.domain_stats["iteration"] == 3
        assert stats.domain_stats["reference_losses"] is not None

        # Code has high excess loss (+1.5), should be upweighted
        assert stats.domain_stats["target_weights"]["code"] > 0.143, \
            "Code domain should be upweighted (high excess loss)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
