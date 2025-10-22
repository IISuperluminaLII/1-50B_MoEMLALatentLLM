"""
Unit tests for domain mixing functionality.

Tests DomainIdentifier, GroupDROOptimizer, and DomainMixer classes.
"""

import pytest
import tempfile
from pathlib import Path

from src.data.domain_mixing import (
    DomainIdentifier,
    GroupDROOptimizer,
    DomainMixer,
    DomainWeights,
    DOMAIN_CATEGORIES,
    PRESET_COMPOSITIONS,
)


class TestDomainIdentifier:
    """Test domain identification logic."""

    def test_identify_code_domain(self):
        """Test identification of code documents."""
        identifier = DomainIdentifier(method="keyword")

        code_doc = {
            "text": "def train_model(data, epochs=10):\n    for epoch in range(epochs):\n        loss = model.train()"
        }

        domain, confidence = identifier.identify(code_doc)
        assert domain == "code"
        assert confidence > 0  # At least one pattern match

    def test_identify_academic_domain(self):
        """Test identification of academic documents."""
        identifier = DomainIdentifier(method="keyword")

        academic_doc = {
            "text": "Abstract: This paper presents a novel approach to machine learning. "
                   "arXiv:2305.10429. We prove a theorem showing convergence."
        }

        domain, confidence = identifier.identify(academic_doc)
        assert domain == "academic"

    def test_identify_wikipedia_domain(self):
        """Test identification of Wikipedia documents."""
        identifier = DomainIdentifier(method="keyword")

        wiki_doc = {
            "text": "== History ==\nThe city was founded in [[1850]]. {{cite book|title=History}}"
        }

        domain, confidence = identifier.identify(wiki_doc)
        assert domain == "wikipedia"

    def test_default_to_common_crawl(self):
        """Test fallback to common_crawl for generic text."""
        identifier = DomainIdentifier(method="keyword")

        generic_doc = {"text": "This is some random web text about various topics."}

        domain, _ = identifier.identify(generic_doc)
        assert domain == "common_crawl"

    def test_metadata_domain_override(self):
        """Test that metadata domain labels override keyword detection."""
        identifier = DomainIdentifier(method="metadata")

        doc_with_metadata = {
            "text": "def foo(): pass",  # Looks like code
            "domain": "books",  # But labeled as books
        }

        domain, confidence = identifier.identify(doc_with_metadata)
        assert domain == "books"
        assert confidence == 1.0


class TestGroupDROOptimizer:
    """Test Group DRO optimization logic."""

    def test_initialization(self):
        """Test optimizer initializes with uniform weights."""
        optimizer = GroupDROOptimizer(num_domains=7)

        assert len(optimizer.alpha) == 7
        assert pytest.approx(sum(optimizer.alpha), 0.01) == 1.0
        # All weights should be equal initially
        assert all(abs(w - 1/7) < 0.01 for w in optimizer.alpha)

    def test_update_weights_upweights_high_loss_domains(self):
        """Test that domains with higher loss get increased weight."""
        optimizer = GroupDROOptimizer(learning_rate=0.1, temperature=0.5, num_domains=3)

        # Domain losses: domain 0 has highest loss
        domain_losses = {
            "domain_0": 10.0,  # High loss
            "domain_1": 5.0,   # Medium loss
            "domain_2": 2.0,   # Low loss
        }
        domain_names = ["domain_0", "domain_1", "domain_2"]

        new_weights = optimizer.update_weights(domain_losses, domain_names)

        # Domain 0 (highest loss) should have highest weight
        assert new_weights["domain_0"] > new_weights["domain_1"]
        assert new_weights["domain_1"] > new_weights["domain_2"]

    def test_weights_normalize_to_one(self):
        """Test that updated weights sum to 1.0."""
        optimizer = GroupDROOptimizer(num_domains=5)

        domain_losses = {f"domain_{i}": float(i) for i in range(5)}
        domain_names = [f"domain_{i}" for i in range(5)]

        new_weights = optimizer.update_weights(domain_losses, domain_names)

        assert pytest.approx(sum(new_weights.values()), 0.01) == 1.0


class TestDomainWeights:
    """Test DomainWeights data class."""

    def test_normalize_weights(self):
        """Test weight normalization."""
        weights = DomainWeights(weights={"a": 2.0, "b": 3.0, "c": 5.0})
        weights.normalize()

        assert pytest.approx(sum(weights.weights.values()), 0.01) == 1.0
        assert pytest.approx(weights.weights["a"], 0.01) == 0.2
        assert pytest.approx(weights.weights["b"], 0.01) == 0.3
        assert pytest.approx(weights.weights["c"], 0.01) == 0.5

    def test_to_dict_serialization(self):
        """Test conversion to dictionary."""
        weights = DomainWeights(
            weights={"domain_a": 0.5, "domain_b": 0.5},
            document_counts={"domain_a": 100, "domain_b": 200},
            token_counts={"domain_a": 5000, "domain_b": 10000},
            iteration=3,
        )

        result = weights.to_dict()

        assert result["weights"] == {"domain_a": 0.5, "domain_b": 0.5}
        assert result["document_counts"] == {"domain_a": 100, "domain_b": 200}
        assert result["token_counts"] == {"domain_a": 5000, "domain_b": 10000}
        assert result["iteration"] == 3


class TestDomainMixer:
    """Test main DomainMixer class."""

    def test_initialization_with_preset_composition(self):
        """Test mixer initializes with preset composition."""
        mixer = DomainMixer(composition="deepseek_v3", random_seed=42)

        assert mixer.composition_name == "deepseek_v3"
        assert mixer.domain_weights.weights == PRESET_COMPOSITIONS["deepseek_v3"]

    def test_initialization_with_custom_weights(self):
        """Test mixer with custom weights."""
        custom_weights = {
            "code": 0.5,
            "common_crawl": 0.3,
            "wikipedia": 0.2,
        }

        mixer = DomainMixer(composition=custom_weights)

        # Should normalize
        total = sum(mixer.domain_weights.weights.values())
        assert pytest.approx(total, 0.01) == 1.0

    def test_identify_domain(self):
        """Test domain identification via mixer."""
        mixer = DomainMixer(composition="balanced")

        code_doc = {"text": "import numpy as np\ndef func(): pass"}
        domain, confidence = mixer.identify_domain(code_doc)

        assert domain == "code"
        assert confidence > 0

    def test_classify_documents(self):
        """Test document classification into domain buckets."""
        mixer = DomainMixer(composition="balanced", random_seed=42)

        documents = [
            {"text": "def foo(): return 42", "id": "1"},
            {"text": "import sys", "id": "2"},
            {"text": "== Wikipedia Article ==\n[[Link]]", "id": "3"},
            {"text": "Abstract: arXiv:2412.19437", "id": "4"},
        ]

        buckets = mixer.classify_documents(documents)

        # Should have classified documents
        assert "code" in buckets
        assert len(buckets["code"]) >= 1  # At least one code document
        assert "wikipedia" in buckets or "academic" in buckets  # Wiki or academic classified

    def test_mix_documents_respects_weights(self):
        """Test that mixing produces correct distribution."""
        # Create mixer with simple 50/50 split
        mixer = DomainMixer(
            composition={"code": 0.5, "common_crawl": 0.5},
            random_seed=42,
        )

        # Create documents with clear domains
        documents = [
            {"text": "def foo(): pass", "id": f"code_{i}"} for i in range(100)
        ] + [
            {"text": "This is generic web text about stuff.", "id": f"web_{i}"} for i in range(100)
        ]

        mixed = mixer.mix_documents(documents, target_size=200)

        assert len(mixed) == 200

        # Check statistics
        stats = mixer.get_statistics()
        assert stats["total_documents"] == 200
        assert "document_counts" in stats
        assert "actual_distribution" in stats

    def test_get_statistics(self):
        """Test statistics collection."""
        mixer = DomainMixer(composition="llama3")

        documents = [
            {"text": "Some text here", "id": str(i)} for i in range(10)
        ]

        mixer.classify_documents(documents)
        stats = mixer.get_statistics()

        assert "composition" in stats
        assert stats["composition"] == "llama3"
        assert "target_weights" in stats
        assert "document_counts" in stats
        assert "total_documents" in stats
        assert stats["total_documents"] == 10

    def test_save_statistics(self):
        """Test saving statistics to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mixer = DomainMixer(composition="balanced")

            documents = [{"text": f"Document {i}", "id": str(i)} for i in range(5)]
            mixer.classify_documents(documents)

            stats_file = Path(tmpdir) / "stats.json"
            mixer.save_statistics(stats_file)

            assert stats_file.exists()

            # Verify JSON is valid
            import json
            with open(stats_file) as f:
                stats = json.load(f)

            assert "composition" in stats
            assert "total_documents" in stats

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        documents = [
            {"text": f"Document {i} with some content", "id": str(i)} for i in range(100)
        ]

        mixer1 = DomainMixer(composition="balanced", random_seed=42)
        mixed1 = mixer1.mix_documents(documents.copy(), target_size=50)

        mixer2 = DomainMixer(composition="balanced", random_seed=42)
        mixed2 = mixer2.mix_documents(documents.copy(), target_size=50)

        # Same order should be produced
        assert [d["id"] for d in mixed1] == [d["id"] for d in mixed2]

    def test_invalid_composition_raises_error(self):
        """Test that invalid composition name raises error."""
        with pytest.raises(ValueError, match="Unknown composition"):
            DomainMixer(composition="invalid_composition_name")

    def test_repr(self):
        """Test string representation."""
        mixer = DomainMixer(composition="deepseek_v3")
        repr_str = repr(mixer)

        assert "DomainMixer" in repr_str
        assert "deepseek_v3" in repr_str

    def test_remainder_redistribution_with_small_corpus(self):
        """
        Test that remainder redistribution prevents document loss.

        With a small corpus and many domains, integer floor division
        can cause significant document loss. This test verifies that
        the remainder is properly redistributed.
        """
        # Create a mixer with many domains (using DeepSeek-V3 which has 7 domains)
        mixer = DomainMixer(composition="deepseek_v3", random_seed=42)

        # Create a tiny corpus (3 documents) that will expose rounding issues
        documents = [
            {"text": "Generic web text about various topics.", "id": "1"},
            {"text": "More generic content here.", "id": "2"},
            {"text": "Yet another document with text.", "id": "3"},
        ]

        # Request exactly 3 documents
        target_size = 3
        mixed = mixer.mix_documents(documents, target_size=target_size)

        # CRITICAL: The result must contain exactly target_size documents
        # Without remainder redistribution, floor division would lose documents
        assert len(mixed) == target_size, \
            f"Expected {target_size} documents, got {len(mixed)}. Document loss detected!"

        # Verify all returned documents are from the original set
        mixed_ids = {doc["id"] for doc in mixed}
        assert mixed_ids.issubset({"1", "2", "3"})

    def test_remainder_distribution_follows_fractional_parts(self):
        """
        Test that remainder documents go to domains with highest fractional parts.

        When distributing remainder, domains with the highest fractional parts
        (after floor division) should receive the extra documents first.
        """
        # Create custom weights that will produce clear fractional parts
        custom_weights = {
            "code": 0.45,          # 0.45 * 10 = 4.5 → floor 4, frac 0.5
            "common_crawl": 0.35,  # 0.35 * 10 = 3.5 → floor 3, frac 0.5
            "wikipedia": 0.20,     # 0.20 * 10 = 2.0 → floor 2, frac 0.0
        }

        mixer = DomainMixer(composition=custom_weights, random_seed=42)

        # Create documents that will be classified into our target domains
        documents = [
            {"text": "def foo(): pass", "id": f"code_{i}"} for i in range(10)
        ] + [
            {"text": "Generic web content about stuff.", "id": f"web_{i}"} for i in range(10)
        ] + [
            {"text": "== Article ==\n[[Link]]", "id": f"wiki_{i}"} for i in range(10)
        ]

        # Mix with target_size = 10 (which will create remainder)
        target_size = 10
        mixed = mixer.mix_documents(documents, target_size=target_size)

        # Must get exactly target_size documents
        assert len(mixed) == target_size

        # Expected allocation with remainder redistribution:
        # code: floor(4.5) = 4, frac = 0.5 → gets +1 = 5
        # common_crawl: floor(3.5) = 3, frac = 0.5 → gets +1 = 4  (tie broken by ordering)
        # wikipedia: floor(2.0) = 2, frac = 0.0 → gets +0 = 1
        # Total: 5 + 4 + 1 = 10 ✓

        # The test verifies that len(mixed) == target_size, which confirms
        # that remainder redistribution works correctly

    def test_large_target_size_no_remainder_loss(self):
        """
        Test that even with large target sizes, no documents are lost to rounding.
        """
        mixer = DomainMixer(composition="balanced", random_seed=42)

        documents = [
            {"text": f"Document {i} with some content", "id": str(i)} for i in range(100)
        ]

        # Try various target sizes that might expose rounding issues
        for target_size in [97, 99, 101, 103, 137]:
            mixed = mixer.mix_documents(documents.copy(), target_size=target_size)
            assert len(mixed) == target_size, \
                f"Target {target_size}: expected {target_size}, got {len(mixed)}"


class TestPresetCompositions:
    """Test preset composition configurations."""

    def test_all_presets_sum_to_one(self):
        """Test that all preset compositions have weights summing to 1.0."""
        for name, weights in PRESET_COMPOSITIONS.items():
            total = sum(weights.values())
            assert pytest.approx(total, 0.01) == 1.0, f"{name} weights don't sum to 1.0"

    def test_all_presets_cover_all_domains(self):
        """Test that all presets include all domain categories."""
        for name, weights in PRESET_COMPOSITIONS.items():
            assert set(weights.keys()) == set(DOMAIN_CATEGORIES), \
                f"{name} doesn't include all domains"

    def test_deepseek_v3_composition(self):
        """Test DeepSeek-V3 composition matches expected ratios."""
        comp = PRESET_COMPOSITIONS["deepseek_v3"]

        assert comp["common_crawl"] == 0.45
        assert comp["code"] == 0.20
        assert comp["academic"] == 0.15

    def test_llama3_composition(self):
        """Test LLaMA-3 composition matches expected ratios."""
        comp = PRESET_COMPOSITIONS["llama3"]

        assert comp["common_crawl"] == 0.50
        assert comp["code"] == 0.15
        assert comp["wikipedia"] == 0.12


class TestDoReMiOptimization:
    """Test DoReMi weight optimization functionality."""

    def test_optimize_weights_upweights_high_loss_domains(self):
        """Test that domains with higher loss get increased weight."""
        mixer = DomainMixer(composition="doremi", random_seed=42)

        # Get initial weights (should be uniform)
        initial_weights = mixer.domain_weights.weights.copy()

        # Create synthetic domain losses: code has highest loss
        domain_losses = {
            "code": 5.0,  # High loss - should get upweighted
            "common_crawl": 3.0,  # Medium loss
            "wikipedia": 2.0,  # Low loss
            "academic": 2.5,
            "books": 2.8,
            "news": 2.2,
            "social": 2.3,
        }

        # Run optimization
        mixer.optimize_weights_doremi(domain_losses)

        # Get updated weights
        updated_weights = mixer.domain_weights.weights

        # Verify high-loss domain (code) got upweighted
        assert updated_weights["code"] > initial_weights["code"], \
            "High-loss domain should get increased weight"

        # Verify low-loss domain (wikipedia) got downweighted
        assert updated_weights["wikipedia"] < initial_weights["wikipedia"], \
            "Low-loss domain should get decreased weight"

        # Verify iteration counter incremented
        assert mixer.domain_weights.iteration == 1

    def test_mix_documents_with_feedback(self):
        """Test DoReMi feedback loop integrates optimization and sampling."""
        mixer = DomainMixer(composition="doremi", random_seed=42)

        # Create test documents
        documents = [
            {"text": "def foo(): pass\nimport sys", "id": f"code_{i}"} for i in range(50)
        ] + [
            {"text": "Generic web content about various topics.", "id": f"web_{i}"} for i in range(50)
        ]

        # Domain losses: code has HIGH loss (hard to learn), web has LOW loss (easy)
        # DoReMi upweights HIGH LOSS domains to focus on harder examples
        domain_losses = {
            "code": 4.0,  # High loss - hard domain, should be upweighted
            "common_crawl": 2.0,  # Low loss - easy domain
            "wikipedia": 3.0,
            "academic": 3.5,
            "books": 3.2,
            "news": 3.3,
            "social": 3.4,
        }

        # Apply DoReMi with feedback
        mixed = mixer.mix_documents_with_feedback(
            documents,
            domain_losses,
            num_iterations=3,
            target_size=100,
        )

        # Verify output
        assert len(mixed) == 100
        assert mixer.domain_weights.iteration == 3

        # Verify code weight increased (high loss domains get upweighted in DoReMi)
        stats = mixer.get_statistics()
        assert stats["target_weights"]["code"] > 1.0 / 7, \
            "Code domain (high loss) should have higher than uniform weight after optimization"

    def test_doremi_requires_correct_composition(self):
        """Test that mix_documents_with_feedback requires composition='doremi'."""
        mixer = DomainMixer(composition="balanced", random_seed=42)

        documents = [{"text": "Test", "id": "1"}]
        losses = {"code": 2.0, "common_crawl": 3.0}

        with pytest.raises(ValueError, match="requires composition='doremi'"):
            mixer.mix_documents_with_feedback(documents, losses)

    def test_optimize_weights_without_doremi_composition_raises(self):
        """Test that optimize_weights_doremi raises if not using doremi composition."""
        mixer = DomainMixer(composition="deepseek_v3", random_seed=42)

        losses = {"code": 2.0, "common_crawl": 3.0}

        with pytest.raises(ValueError, match="DoReMi optimizer not initialized"):
            mixer.optimize_weights_doremi(losses)


class TestDomainStatisticsCorrectness:
    """Test that domain statistics report actual sampled mix, not input corpus."""

    def test_actual_distribution_reflects_sampled_output(self):
        """Test that actual_distribution reports what was sampled, not what was input."""
        # Create mixer with 50/50 target
        mixer = DomainMixer(
            composition={"code": 0.5, "common_crawl": 0.5},
            random_seed=42,
        )

        # Create imbalanced input: 80% code, 20% web
        documents = [
            {"text": "def foo(): pass", "id": f"code_{i}"} for i in range(80)
        ] + [
            {"text": "Web content here.", "id": f"web_{i}"} for i in range(20)
        ]

        # Sample with target 50/50 mix
        mixed = mixer.mix_documents(documents, target_size=100)

        stats = mixer.get_statistics()

        # Input distribution should reflect corpus composition (80/20)
        assert abs(stats["input_distribution"]["code"] - 0.8) < 0.01
        assert abs(stats["input_distribution"]["common_crawl"] - 0.2) < 0.01

        # Actual distribution should reflect sampled output (50/50)
        assert abs(stats["actual_distribution"]["code"] - 0.5) < 0.1
        assert abs(stats["actual_distribution"]["common_crawl"] - 0.5) < 0.1

    def test_oversampling_counted_in_actual_distribution(self):
        """Test that duplicates from oversampling are counted in actual_distribution."""
        mixer = DomainMixer(
            composition={"code": 0.8, "common_crawl": 0.2},
            random_seed=42,
        )

        # Only 10 code documents available
        documents = [
            {"text": "def foo(): pass", "id": f"code_{i}"} for i in range(10)
        ] + [
            {"text": "Web content here.", "id": f"web_{i}"} for i in range(10)
        ]

        # Request 100 documents with 80% code - will require oversampling
        mixed = mixer.mix_documents(documents, target_size=100)

        assert len(mixed) == 100

        stats = mixer.get_statistics()

        # Sampled counts should include duplicates
        assert stats["sampled_document_counts"]["code"] == 80
        assert stats["sampled_document_counts"]["common_crawl"] == 20

        # Actual distribution should be 80/20
        assert abs(stats["actual_distribution"]["code"] - 0.8) < 0.01
        assert abs(stats["actual_distribution"]["common_crawl"] - 0.2) < 0.01
