"""
Quick test runner for data sanitization modules.

Runs basic smoke tests to verify implementations are working correctly.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_preliminary_cleaning():
    """Test preliminary cleaning module."""
    print("Testing preliminary cleaning module...")

    from data.preliminary_cleaning import PreliminaryCleaner

    cleaner = PreliminaryCleaner()

    # Test 1: Unicode normalization
    text = "ﬁle"  # U+FB01
    result = cleaner.clean(text)
    assert result == "file", f"Expected 'file', got '{result}'"
    print("  ✓ Unicode normalization (ﬁ→fi)")

    # Test 2: HTML entities
    text = "AT&amp;T &lt;div&gt;"
    result = cleaner.clean(text)
    assert "AT&T" in result and "<div>" in result, f"HTML unescape failed: {result}"
    print("  ✓ HTML entity unescaping")

    # Test 3: Whitespace normalization
    text = "Hello    World"
    result = cleaner.clean(text)
    assert result == "Hello World", f"Expected 'Hello World', got '{result}'"
    print("  ✓ Whitespace normalization")

    # Test 4: Control characters
    text = "Hello\x00World"
    result = cleaner.clean(text)
    assert "\x00" not in result, "Control characters not removed"
    print("  ✓ Control character removal")

    print("✓ All preliminary cleaning tests passed!\n")


def test_deduplication():
    """Test deduplication module."""
    print("Testing deduplication module...")

    from data.deduplication import MinHashDeduplicator, ExactDeduplicator

    # Test 1: MinHash initialization
    dedup = MinHashDeduplicator(num_perm=128, threshold=0.8, n_gram=13, seed=42)
    assert dedup.num_perm == 128
    assert dedup.threshold == 0.8
    assert dedup.n_gram == 13
    print("  ✓ MinHash initialization")

    # Test 2: N-gram generation
    dedup_small = MinHashDeduplicator(n_gram=3)
    ngrams = dedup_small.generate_n_grams("hello")
    assert ngrams == ['hel', 'ell', 'llo'], f"Expected ['hel', 'ell', 'llo'], got {ngrams}"
    print("  ✓ N-gram generation")

    # Test 3: Exact duplicate detection
    docs = ["doc A", "doc A", "doc B"]
    unique_docs, unique_ids, _ = dedup.deduplicate(docs)
    assert len(unique_docs) == 2, f"Expected 2 unique docs, got {len(unique_docs)}"
    assert "doc A" in unique_docs
    assert "doc B" in unique_docs
    print("  ✓ Exact duplicate detection")

    # Test 4: Statistics
    stats = dedup.get_stats()
    assert stats is not None
    assert stats.total_documents == 3
    assert stats.unique_documents == 2
    assert stats.duplicate_documents == 1
    print("  ✓ Statistics tracking")

    # Test 5: ExactDeduplicator
    exact_dedup = ExactDeduplicator()
    docs = ["doc A", "doc A", "doc B"]
    unique_docs, unique_ids = exact_dedup.deduplicate(docs)
    assert len(unique_docs) == 2
    print("  ✓ Exact deduplicator")

    # Test 6: Jaccard similarity
    dedup_sim = MinHashDeduplicator(n_gram=3)
    sim = dedup_sim.compute_jaccard_similarity("hello", "hello")
    assert sim == 1.0, f"Expected 1.0, got {sim}"
    sim = dedup_sim.compute_jaccard_similarity("abc", "xyz")
    assert sim == 0.0, f"Expected 0.0, got {sim}"
    print("  ✓ Jaccard similarity computation")

    # Test 7: MinHash determinism
    dedup1 = MinHashDeduplicator(seed=42)
    dedup2 = MinHashDeduplicator(seed=42)
    text = "Test text for hashing"
    minhash1 = dedup1.compute_minhash(text)
    minhash2 = dedup2.compute_minhash(text)
    assert list(minhash1.hashvalues) == list(minhash2.hashvalues)
    print("  ✓ MinHash determinism")

    print("✓ All deduplication tests passed!\n")


def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")

    try:
        from data.preliminary_cleaning import PreliminaryCleaner
        print("  ✓ PreliminaryCleaner imported")
    except ImportError as e:
        print(f"  ✗ Failed to import PreliminaryCleaner: {e}")
        return False

    try:
        from data.deduplication import MinHashDeduplicator, ExactDeduplicator, DeduplicationStats
        print("  ✓ Deduplication modules imported")
    except ImportError as e:
        print(f"  ✗ Failed to import deduplication modules: {e}")
        return False

    print("✓ All imports successful!\n")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Data Sanitization Module Test Runner")
    print("=" * 60)
    print()

    try:
        # Test imports first
        if not test_imports():
            print("\n✗ Import tests failed. Please install required dependencies:")
            print("  pip install ftfy datasketch")
            sys.exit(1)

        # Run tests
        test_preliminary_cleaning()
        test_deduplication()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install ftfy datasketch")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
