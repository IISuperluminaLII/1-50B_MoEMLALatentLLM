# Deduplication Methods Guide

Complete guide to choosing and using the four deduplication methods available in this pipeline.

---

## Available Methods

### 1. MinHash LSH (Default) - CPU-based Near-Duplicate Detection

**Paper**: Lee et al. (2022) "Deduplicating Training Data Makes Language Models Better" ([arXiv:2107.06499](https://arxiv.org/abs/2107.06499))

**Best For**: General near-duplicate detection with high accuracy

**Requirements**: `datasketch>=1.6.0` (mandatory - no fallback)

**Usage**:
```python
from src.data.pipeline import DataPipeline, PipelineConfig

config = PipelineConfig(
    input_path="data.jsonl",
    output_dir="output",
    enable_deduplication=True,
    dedup_config={
        "method": "minhash",
        "num_perm": 128,      # Number of hash functions (10-450)
        "threshold": 0.8,     # Jaccard similarity threshold (0.0-1.0)
        "n_gram": 13,         # Character n-gram size
        "seed": 42            # For reproducibility
    }
)

pipeline = DataPipeline(config)
stats = pipeline.process_and_save()
```

**How It Works**:
1. **Phase 1 (LSH)**: Compute MinHash signatures and query LSH index for candidate duplicates (approximate, fast)
2. **Phase 2 (Verification)**: Verify candidates with exact Jaccard similarity computation (precise, slower)
3. Only documents with `Jaccard >= threshold` are marked as duplicates

**Parameters**:
- `num_perm`: More hash functions = higher accuracy but slower
  - GPT-3: 10 permutations
  - Gopher: 450 permutations
  - Default: 128 (good balance)
- `threshold`: Higher = fewer duplicates removed
  - 0.8 (default): Recommended by Lee et al.
  - 0.9: More conservative (keeps more documents)
  - 0.7: More aggressive (removes more near-duplicates)
- `n_gram`: Character n-gram size
  - 13 (default): Used by Gopher, good for English
  - 5: Better for languages with shorter words

**Performance**:
- Speed: ~10K documents/second (single-threaded, num_perm=128)
- Memory: O(num_docs × num_perm × 8 bytes)
- Accuracy: Jaccard estimation within 10% for num_perm >= 128

---

### 2. FED GPU - GPU-Accelerated MinHash

**Paper**: Son et al. (2025) "FED: Fast and Efficient Dataset Deduplication Framework with GPU Acceleration" ([arXiv:2501.01046](https://arxiv.org/abs/2501.01046))

**Best For**: Large-scale deduplication with GPU available (107× speedup)

**Requirements**:
- PyTorch with CUDA support
- `cupy-cuda12x>=13.0.0` (CUDA arrays)
- `faiss-gpu>=1.7.4` (GPU similarity search)
- CUDA-capable GPU

**Usage**:
```python
config = PipelineConfig(
    input_path="large_corpus.jsonl",
    output_dir="output",
    enable_deduplication=True,
    dedup_config={
        "method": "fed",
        "threshold": 0.8,
        "num_perm": 128,
        "batch_size": 10000,   # GPU batch size
        "device": "cuda:0"     # GPU device
    }
)

pipeline = DataPipeline(config)
stats = pipeline.process_and_save()

# Check GPU utilization
print(f"GPU time: {stats.gpu_time_seconds:.2f}s")
print(f"CPU time: {stats.cpu_time_seconds:.2f}s")
```

**How It Works**:
1. Batch documents for GPU processing
2. Compute MinHash signatures on GPU using CUDA kernels (parallel)
3. Query FAISS-GPU LSH index for candidates (GPU-accelerated)
4. Verify with Jaccard similarity (maintains Lee et al. compliance)

**Performance**:
- Speed: **107× faster** than CPU MinHash
- GPU Memory: ~2-4GB for 1M documents (batch_size=10000)
- Ideal for: >100K documents with GPU available

**Limitations**:
- Requires CUDA GPU (will raise RuntimeError if unavailable)
- Higher memory usage than LSHBloom
- GPU dependencies can be complex to install

---

### 3. LSHBloom - Memory-Efficient for Extreme Scale

**Paper**: Khan et al. (2024) "LSHBloom: Memory-efficient, Extreme-scale Document Deduplication" ([arXiv:2411.04257](https://arxiv.org/abs/2411.04257))

**Best For**: Deduplicating 100B+ documents with limited memory

**Requirements**: `pybloom-live>=4.0.0`

**Usage**:
```python
config = PipelineConfig(
    input_path="massive_corpus.jsonl",
    output_dir="output",
    enable_deduplication=True,
    dedup_config={
        "method": "lshbloom",
        "threshold": 0.8,
        "num_bands": 20,              # LSH bands
        "rows_per_band": 6,           # Rows per band
        "bloom_capacity": 100000,     # Expected items per Bloom filter
        "bloom_error_rate": 0.001,    # False positive rate (0.1%)
    }
)

pipeline = DataPipeline(config)
stats = pipeline.process_and_save()

# Check memory usage
print(f"Bloom filter memory: {stats.bloom_memory_mb:.2f} MB")
```

**How It Works**:
1. Partition MinHash signature into bands (LSH banding technique)
2. For each band, store signature in Bloom filter (probabilistic membership)
3. Query Bloom filters for candidates (constant memory per band)
4. Verify with exact Jaccard similarity

**Performance**:
- Memory: **10× reduction** vs standard MinHash
  - LSHBloom: O(num_bands × bloom_capacity) - independent of corpus size
  - MinHash: O(num_docs × num_perm)
- Speed: ~5-8K documents/second (single-threaded)
- Scalability: Can process 100B+ documents on modest hardware

**Parameters**:
- `num_bands × rows_per_band = total hash functions` (typically 120)
  - More bands = higher recall (fewer false negatives)
  - More rows per band = higher precision (fewer false positives)
  - Default: 20 bands × 6 rows = 120 hash functions
- `bloom_capacity`: Expected number of unique signatures per band
  - Higher = less frequent Bloom filter saturation
  - Lower = more memory efficient but higher false positive rate
  - Default: 100,000
- `bloom_error_rate`: Bloom filter false positive rate
  - 0.001 (0.1%, default): Good balance
  - 0.0001 (0.01%): More memory but fewer false positives

**Limitations**:
- Bloom filters are probabilistic (small chance of false positives)
- Cannot retrieve document IDs from Bloom filters (must verify all stored texts)
- Slightly slower than MinHash for small corpora (<100K documents)

---

### 4. Exact - Hash-Based Exact Matching

**Best For**: Fast exact duplicate detection only (no near-duplicates)

**Requirements**: None (stdlib only)

**Usage**:
```python
config = PipelineConfig(
    input_path="data.jsonl",
    output_dir="output",
    enable_deduplication=True,
    dedup_config={
        "method": "exact",
        "hash_algorithm": "sha256"  # Options: md5, sha1, sha256, sha512
    }
)

pipeline = DataPipeline(config)
stats = pipeline.process_and_save()
```

**How It Works**:
1. Compute hash of entire document text
2. Store hash in set
3. Skip documents with already-seen hash

**Performance**:
- Speed: **Fastest** method (~50K documents/second)
- Memory: O(num_unique_docs × 32 bytes) for SHA-256
- Accuracy: **100%** for exact duplicates, **0%** for near-duplicates

**Use Cases**:
- Pre-filtering before MinHash/FED/LSHBloom
- When only exact duplicates matter (e.g., synthetic data)
- Quick deduplication for testing

---

## Method Comparison

| Method      | Speed       | Memory      | Accuracy | Near-Duplicates | Requirements       |
|-------------|-------------|-------------|----------|------------------|--------------------|
| MinHash LSH | Fast        | Medium      | High     | ✅ Yes           | datasketch         |
| FED GPU     | **Fastest** | High (GPU)  | High     | ✅ Yes           | CUDA, cupy, faiss  |
| LSHBloom    | Medium      | **Lowest**  | High     | ✅ Yes           | pybloom-live       |
| Exact       | Very Fast   | Low         | Perfect  | ❌ No            | None               |

---

## Choosing a Method

### Decision Tree:

1. **Do you only need exact duplicate removal?**
   - ✅ Yes → Use **Exact** (fastest, simplest)
   - ❌ No → Continue to step 2

2. **Do you have a CUDA GPU available?**
   - ✅ Yes → Use **FED GPU** (107× speedup, best for >100K docs)
   - ❌ No → Continue to step 3

3. **How large is your corpus?**
   - **<100K documents** → Use **MinHash LSH** (best accuracy, standard method)
   - **100K-10M documents** → Use **MinHash LSH** or **LSHBloom**
   - **>10M documents** → Use **LSHBloom** (memory-efficient, scales to 100B+)

4. **Memory constrained?**
   - ✅ Yes → Use **LSHBloom** (10× memory reduction)
   - ❌ No → Use **MinHash LSH**

---

## Combined Deduplication (Recommended for Production)

For best results, use multiple methods in sequence:

```python
# Step 1: Remove exact duplicates first (fast pre-filter)
config_exact = PipelineConfig(
    input_path="raw_data.jsonl",
    output_dir="exact_dedup",
    enable_deduplication=True,
    dedup_config={"method": "exact"}
)

pipeline_exact = DataPipeline(config_exact)
pipeline_exact.process_and_save()

# Step 2: Remove near-duplicates from survivors
config_minhash = PipelineConfig(
    input_path="exact_dedup/final.jsonl",
    output_dir="final_dedup",
    enable_deduplication=True,
    dedup_config={
        "method": "fed" if torch.cuda.is_available() else "lshbloom",
        "threshold": 0.8
    }
)

pipeline_minhash = DataPipeline(config_minhash)
stats = pipeline_minhash.process_and_save()

print(f"Final corpus: {stats.total_output_documents:,} documents")
```

**Why This Works**:
- Exact dedup removes 20-40% of duplicates quickly (1-2 minutes for 1M docs)
- Near-duplicate methods only process survivors (faster, less memory)
- Total deduplication ratio: 40-60% for web-scraped data

---

## Factory Function (Advanced)

For programmatic method selection:

```python
from src.data.deduplication import create_deduplicator

# Create deduplicator based on runtime conditions
import torch

if torch.cuda.is_available():
    dedup = create_deduplicator("fed", threshold=0.8, batch_size=10000)
elif num_documents > 10_000_000:
    dedup = create_deduplicator("lshbloom", threshold=0.8)
else:
    dedup = create_deduplicator("minhash", threshold=0.8, num_perm=128)

# Use deduplicator
unique_docs, unique_ids, _ = dedup.deduplicate(documents, doc_ids)
```

---

## Troubleshooting

### MinHash: "ImportError: requires datasketch library"
```bash
pip install datasketch>=1.6.0
```
**Note**: No silent fallback - datasketch is mandatory for Lee et al. (2022) compliance

### FED: "RuntimeError: CUDA is not available"
- Verify GPU is installed: `nvidia-smi`
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Install CuPy: `pip install cupy-cuda12x`
- Install FAISS-GPU: `pip install faiss-gpu`

### LSHBloom: "ImportError: requires pybloom-live"
```bash
pip install pybloom-live>=4.0.0
```

### Memory Errors with MinHash/FED
- Reduce `num_perm` (e.g., 64 instead of 128)
- Use LSHBloom instead (10× memory reduction)
- Process in smaller batches

---

## Citations

If you use these methods in research, please cite the appropriate papers:

```bibtex
@article{lee2022deduplicating,
  title={Deduplicating Training Data Makes Language Models Better},
  author={Lee, Katherine and Ippolito, Daphne and Nystrom, Andrew and others},
  journal={arXiv preprint arXiv:2107.06499},
  year={2022}
}

@article{son2025fed,
  title={FED: Fast and Efficient Dataset Deduplication Framework with GPU Acceleration},
  author={Son, Youngjun and others},
  journal={arXiv preprint arXiv:2501.01046},
  year={2025}
}

@article{khan2024lshbloom,
  title={LSHBloom: Memory-efficient, Extreme-scale Document Deduplication},
  author={Khan, Arham and others},
  journal={arXiv preprint arXiv:2411.04257},
  year={2024}
}
```

---

*Last Updated: 2025-10-22*
