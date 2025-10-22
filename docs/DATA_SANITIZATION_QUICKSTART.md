# Data Sanitization - Quick Start Guide

## Installation

```bash
# Install required dependencies
pip install ftfy>=6.1.0 datasketch>=1.6.0 regex>=2023.10.0
```

---

## 1. Preliminary Text Cleaning

### Basic Usage

```python
from src.data.preliminary_cleaning import PreliminaryCleaner

# Initialize with default settings (recommended)
cleaner = PreliminaryCleaner()

# Clean a single text
dirty_text = "ﬁle &amp; café  \x00  multiple    spaces\n\n\n\nnext"
clean_text = cleaner.clean(dirty_text)
# Result: "file & café multiple spaces\n\nnext"

# Get statistics
stats = cleaner.get_stats(dirty_text, clean_text)
print(f"Removed {stats['chars_removed']} characters")
print(f"Reduction: {stats['reduction_ratio']:.1%}")
```

### Custom Configuration

```python
# Customize cleaning steps
cleaner = PreliminaryCleaner(
    unicode_normalization="NFC",    # Options: NFC, NFD, NFKC, NFKD
    fix_encoding=True,              # Fix mojibake (encoding errors)
    unescape_html=True,             # Convert HTML entities
    remove_control_chars=True,      # Remove non-printable chars
    normalize_whitespace=True       # Normalize spaces/newlines
)
```

### Batch Processing

```python
documents = ["dirty text 1", "dirty text 2", ...]
cleaned_docs = [cleaner.clean(doc) for doc in documents]
```

---

## 2. Deduplication

### MinHash LSH Deduplication (Near-Duplicates)

```python
from src.data.deduplication import MinHashDeduplicator

# Initialize with recommended settings (per Lee et al. 2022)
dedup = MinHashDeduplicator(
    num_perm=128,        # Hash functions (10-450, higher = more accurate)
    threshold=0.8,       # Jaccard similarity threshold (0.0-1.0)
    n_gram=13,          # Character n-gram size (Gopher uses 13)
    seed=42             # For reproducibility
)

# Deduplicate a list of documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumps over the lazy cat",  # Similar
    "Something completely different"
]

unique_docs, unique_ids, dup_map = dedup.deduplicate(
    documents,
    doc_ids=None,              # Auto-generate IDs if None
    return_duplicates=True     # Get duplicate mapping
)

print(f"Original: {len(documents)} docs")
print(f"Unique: {len(unique_docs)} docs")

# Get statistics
stats = dedup.get_stats()
print(f"Deduplication ratio: {stats.deduplication_ratio:.1%}")
print(f"Processing time: {stats.processing_time_seconds:.2f}s")
```

### Streaming Deduplication (Large Datasets)

```python
# For datasets that don't fit in memory
def document_generator():
    """Yield (doc_id, doc_text) tuples from your source."""
    for i in range(1000000):
        doc_id = f"doc_{i}"
        doc_text = load_document_from_disk(i)
        yield doc_id, doc_text

dedup = MinHashDeduplicator()

# Process in batches
for doc_id, doc_text in dedup.deduplicate_streaming(
    document_generator(),
    batch_size=10000  # Adjust based on memory
):
    # Save unique documents
    save_to_output(doc_id, doc_text)
```

### Exact Deduplication (Faster, Exact Matches Only)

```python
from src.data.deduplication import ExactDeduplicator

# Initialize
exact_dedup = ExactDeduplicator(
    hash_algorithm='sha256'  # Options: md5, sha1, sha256, sha512
)

# Deduplicate
unique_docs, unique_ids = exact_dedup.deduplicate(documents)
```

### Advanced: Compute Similarities

```python
dedup = MinHashDeduplicator(n_gram=3)

# Exact Jaccard similarity (slow, accurate)
text1 = "hello world"
text2 = "hello there"
similarity = dedup.compute_jaccard_similarity(text1, text2)
print(f"Exact Jaccard: {similarity:.3f}")

# Estimated Jaccard with MinHash (fast, approximate)
minhash1 = dedup.compute_minhash(text1)
minhash2 = dedup.compute_minhash(text2)
similarity_est = dedup.estimate_jaccard_similarity(minhash1, minhash2)
print(f"Estimated Jaccard: {similarity_est:.3f}")
```

---

## 3. Complete Pipeline Example

```python
from src.data.preliminary_cleaning import PreliminaryCleaner
from src.data.deduplication import MinHashDeduplicator

def sanitize_corpus(documents, verbose=True):
    """
    Complete sanitization pipeline:
    1. Preliminary text cleaning
    2. MinHash deduplication
    """
    # Step 1: Clean all documents
    if verbose:
        print("Step 1/2: Cleaning text...")

    cleaner = PreliminaryCleaner()
    cleaned_docs = []
    total_removed = 0

    for doc in documents:
        clean_doc = cleaner.clean(doc)
        cleaned_docs.append(clean_doc)

        stats = cleaner.get_stats(doc, clean_doc)
        total_removed += stats['chars_removed']

    if verbose:
        print(f"  Removed {total_removed} characters total")

    # Step 2: Deduplicate
    if verbose:
        print("Step 2/2: Deduplicating...")

    dedup = MinHashDeduplicator(num_perm=128, threshold=0.8)
    unique_docs, unique_ids, _ = dedup.deduplicate(cleaned_docs)

    stats = dedup.get_stats()
    if verbose:
        print(f"  Original: {stats.total_documents} docs")
        print(f"  Unique: {stats.unique_documents} docs")
        print(f"  Removed: {stats.duplicate_documents} duplicates")
        print(f"  Time: {stats.processing_time_seconds:.2f}s")

    return unique_docs, unique_ids

# Usage
corpus = load_raw_corpus()
clean_corpus, doc_ids = sanitize_corpus(corpus)
save_corpus(clean_corpus, doc_ids)
```

---

## 4. Performance Tips

### Memory Optimization

```python
# Use streaming for large datasets
dedup = MinHashDeduplicator()

# Process in batches instead of loading all at once
for doc_id, doc_text in dedup.deduplicate_streaming(
    document_source,
    batch_size=5000  # Smaller batches = less memory
):
    process_document(doc_id, doc_text)
```

### Speed Optimization

```python
# Use fewer hash functions for faster processing
# (trade-off: less accurate duplicate detection)
fast_dedup = MinHashDeduplicator(
    num_perm=64,      # Default is 128, GPT-3 uses 10
    threshold=0.85    # Higher threshold = fewer false positives
)

# Use exact deduplication if you only need exact matches
# (much faster than MinHash)
exact_dedup = ExactDeduplicator()
```

### Parallel Processing

```python
from multiprocessing import Pool

def clean_batch(docs):
    cleaner = PreliminaryCleaner()
    return [cleaner.clean(doc) for doc in docs]

# Split corpus into batches
batch_size = 10000
batches = [corpus[i:i+batch_size]
           for i in range(0, len(corpus), batch_size)]

# Clean in parallel
with Pool() as pool:
    cleaned_batches = pool.map(clean_batch, batches)

# Flatten results
cleaned_corpus = [doc for batch in cleaned_batches for doc in batch]
```

---

## 5. Configuration Presets

### For Web Scraping (Noisy Data)

```python
# Aggressive cleaning for web data
cleaner = PreliminaryCleaner(
    unicode_normalization="NFKC",  # Compatibility normalization
    fix_encoding=True,             # Fix mojibake
    unescape_html=True,            # Convert HTML entities
    remove_control_chars=True,     # Remove control chars
    normalize_whitespace=True      # Normalize whitespace
)

# Stricter deduplication for web duplicates
dedup = MinHashDeduplicator(
    num_perm=256,      # Higher accuracy
    threshold=0.7,     # Lower threshold = more aggressive
    n_gram=13
)
```

### For Books/Articles (Clean Data)

```python
# Lighter cleaning for already-clean data
cleaner = PreliminaryCleaner(
    unicode_normalization="NFC",   # Canonical normalization
    fix_encoding=False,            # Assume correct encoding
    unescape_html=False,           # No HTML expected
    remove_control_chars=True,     # Still remove control chars
    normalize_whitespace=True      # Normalize whitespace
)

# Conservative deduplication
dedup = MinHashDeduplicator(
    num_perm=128,
    threshold=0.9,     # Higher threshold = only very similar docs
    n_gram=13
)
```

### For Code Repositories

```python
# Preserve intentional formatting
cleaner = PreliminaryCleaner(
    unicode_normalization="NFC",
    fix_encoding=True,
    unescape_html=False,
    remove_control_chars=True,
    normalize_whitespace=False     # Keep original whitespace!
)

# Higher threshold for code (minor changes = different code)
dedup = MinHashDeduplicator(
    num_perm=256,
    threshold=0.95,    # Very high threshold
    n_gram=13
)
```

---

## 6. Testing Your Setup

### Quick Smoke Test

```python
# Run quick test script
python scripts/run_data_tests.py
```

Expected output:
```
============================================================
Data Sanitization Module Test Runner
============================================================

Testing module imports...
  ✓ PreliminaryCleaner imported
  ✓ Deduplication modules imported
✓ All imports successful!

Testing preliminary cleaning module...
  ✓ Unicode normalization (ﬁ→fi)
  ✓ HTML entity unescaping
  ✓ Whitespace normalization
  ✓ Control character removal
✓ All preliminary cleaning tests passed!

Testing deduplication module...
  ✓ MinHash initialization
  ✓ N-gram generation
  ✓ Exact duplicate detection
  ✓ Statistics tracking
  ✓ Exact deduplicator
  ✓ Jaccard similarity computation
  ✓ MinHash determinism
✓ All deduplication tests passed!

============================================================
✓ ALL TESTS PASSED!
============================================================
```

### Full Test Suite

```bash
# Install pytest
pip install pytest pytest-cov

# Run all data pipeline tests with coverage
pytest tests/unit/test_pipeline.py tests/integration/test_dolma_integration.py -v --cov=src/data --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

---

## 7. Common Issues

### ImportError: datasketch

```bash
pip install datasketch>=1.6.0
```

### ImportError: ftfy

```bash
pip install ftfy>=6.1.0
```

### Memory Error with Large Corpus

Use streaming deduplication instead of batch:

```python
# Instead of:
unique_docs, _, _ = dedup.deduplicate(all_documents)  # OOM!

# Use:
for doc_id, doc in dedup.deduplicate_streaming(doc_generator()):
    save(doc_id, doc)
```

### Slow Performance

1. Reduce `num_perm` (128 → 64 or even 10)
2. Use exact deduplication if possible
3. Use multiprocessing for cleaning
4. Increase `batch_size` for streaming (if memory allows)

---

## 8. API Reference

### PreliminaryCleaner

```python
class PreliminaryCleaner:
    def __init__(
        self,
        unicode_normalization: str = "NFKC",
        fix_encoding: bool = True,
        unescape_html: bool = True,
        remove_control_chars: bool = True,
        normalize_whitespace: bool = True
    )

    def clean(self, text: str) -> str

    def get_stats(self, original: str, cleaned: str) -> dict
```

### MinHashDeduplicator

```python
class MinHashDeduplicator:
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        n_gram: int = 13,
        seed: int = 42
    )

    def generate_n_grams(self, text: str) -> List[str]

    def compute_minhash(self, text: str) -> MinHash

    def compute_jaccard_similarity(self, text1: str, text2: str) -> float

    def estimate_jaccard_similarity(
        self, minhash1: MinHash, minhash2: MinHash
    ) -> float

    def deduplicate(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        return_duplicates: bool = False
    ) -> Tuple[List[str], List[str], Optional[Dict]]

    def deduplicate_streaming(
        self,
        document_stream: Iterator[Tuple[str, str]],
        batch_size: int = 1000
    ) -> Iterator[Tuple[str, str]]

    def get_stats(self) -> Optional[DeduplicationStats]

    def clear(self)
```

### ExactDeduplicator

```python
class ExactDeduplicator:
    def __init__(self, hash_algorithm: str = 'sha256')

    def compute_hash(self, text: str) -> str

    def deduplicate(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]

    def clear(self)
```

---

## 9. References

### Research Papers

1. **Lee et al. (2022)** - "Deduplicating Training Data Makes Language Models Better"
   - arXiv:2107.06499
   - MinHash LSH methodology

2. **Zhou et al. (2025)** - "A Survey of LLM × DATA"
   - arXiv:2505.18458
   - Complete sanitization pipeline

For full citation list, see [DATA_SANITIZATION_STATUS.md](../DATA_SANITIZATION_STATUS.md)

---

## 10. Next Steps

After running preliminary cleaning and deduplication:

1. **Heuristic Filtering** (coming soon)
   - Document length filters
   - Repetition detection
   - Language detection

2. **Quality Filtering** (coming soon)
   - FastText classifier
   - KenLM perplexity scoring

3. **Domain Mixing** (coming soon)
   - Domain classification
   - Sample weighting

See [DATA_SANITIZATION_STATUS.md](../DATA_SANITIZATION_STATUS.md) for roadmap.

---

*Last Updated: 2025-10-16*
