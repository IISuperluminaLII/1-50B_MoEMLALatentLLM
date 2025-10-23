"""
Performance benchmarks comparing all deduplication methods.

Compares speed, memory usage, and accuracy of:
- MinHash LSH (CPU)
- Exact hash deduplication
- LSHBloom (memory-efficient)
- FED GPU (if available)

Usage:
    python benchmarks/deduplication_comparison.py
"""

import time
import psutil
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.deduplication import MinHashDeduplicator, ExactDeduplicator, create_deduplicator


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def generate_test_documents(num_docs: int, duplicate_ratio: float = 0.3) -> Tuple[List[str], List[str]]:
    """
    Generate test documents with controlled duplicate ratio.

    Args:
        num_docs: Total number of documents
        duplicate_ratio: Fraction of documents that are duplicates

    Returns:
        (documents, doc_ids)
    """
    num_unique = int(num_docs * (1 - duplicate_ratio))
    num_duplicates = num_docs - num_unique

    documents = []
    doc_ids = []

    # Generate unique documents
    for i in range(num_unique):
        text = f"Document {i}: " + " ".join([
            f"word{j}" for j in range(100)
        ]) + f" unique content {i} " * 10
        documents.append(text)
        doc_ids.append(f"doc_{i}")

    # Generate duplicates (copies of existing documents)
    for i in range(num_duplicates):
        duplicate_idx = i % num_unique
        documents.append(documents[duplicate_idx])
        doc_ids.append(f"dup_{i}")

    return documents, doc_ids


def benchmark_method(
    method_name: str,
    create_fn,
    documents: List[str],
    doc_ids: List[str]
) -> Dict:
    """
    Benchmark a single deduplication method.

    Returns:
        Dictionary with timing, memory, and accuracy metrics
    """
    print(f"\nBenchmarking {method_name}...")

    try:
        # Create deduplicator
        mem_before = get_memory_mb()
        dedup = create_fn()
        mem_after_init = get_memory_mb()

        # Run deduplication
        start_time = time.time()
        unique_docs, unique_ids, _ = dedup.deduplicate(documents, doc_ids)
        end_time = time.time()
        mem_after_dedup = get_memory_mb()

        # Get stats
        stats = dedup.get_stats()

        elapsed_time = end_time - start_time
        docs_per_second = len(documents) / elapsed_time if elapsed_time > 0 else 0
        mem_used_init = mem_after_init - mem_before
        mem_used_dedup = mem_after_dedup - mem_after_init

        results = {
            "method": method_name,
            "success": True,
            "total_docs": len(documents),
            "unique_docs": len(unique_docs),
            "duplicates_removed": len(documents) - len(unique_docs),
            "dedup_ratio": stats.deduplication_ratio if stats else 0.0,
            "elapsed_seconds": elapsed_time,
            "docs_per_second": docs_per_second,
            "memory_init_mb": mem_used_init,
            "memory_dedup_mb": mem_used_dedup,
            "memory_total_mb": mem_used_init + mem_used_dedup,
        }

        print(f"  ✓ Processed {len(documents):,} docs in {elapsed_time:.2f}s")
        print(f"  ✓ Speed: {docs_per_second:,.0f} docs/sec")
        print(f"  ✓ Removed {results['duplicates_removed']:,} duplicates ({results['dedup_ratio']:.1%})")
        print(f"  ✓ Memory: {mem_used_dedup:.1f} MB")

        return results

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return {
            "method": method_name,
            "success": False,
            "error": str(e),
        }


def main():
    """Run benchmarks for all deduplication methods."""
    print("=" * 80)
    print("DEDUPLICATION METHOD COMPARISON BENCHMARK")
    print("=" * 80)

    # Test configurations
    test_sizes = [1000, 10000]  # Document counts
    duplicate_ratio = 0.3  # 30% duplicates

    all_results = []

    for num_docs in test_sizes:
        print(f"\n{'='*80}")
        print(f"TEST: {num_docs:,} documents ({duplicate_ratio:.0%} duplicates)")
        print(f"{'='*80}")

        # Generate test data
        documents, doc_ids = generate_test_documents(num_docs, duplicate_ratio)
        print(f"Generated {len(documents):,} documents ({len(set(documents[:int(num_docs*(1-duplicate_ratio))]))},) unique)")

        # Benchmark each method
        methods = [
            (
                "Exact (SHA-256)",
                lambda: ExactDeduplicator(hash_algorithm="sha256")
            ),
            (
                "MinHash LSH (128 perm)",
                lambda: MinHashDeduplicator(num_perm=128, threshold=0.8)
            ),
            (
                "MinHash LSH (64 perm)",
                lambda: MinHashDeduplicator(num_perm=64, threshold=0.8)
            ),
        ]

        # Add LSHBloom if available
        try:
            from src.data.deduplication_lshbloom import LSHBloomDeduplicator, LSHBloomConfig
            methods.append((
                "LSHBloom (20 bands)",
                lambda: create_deduplicator("lshbloom", threshold=0.8)
            ))
        except ImportError:
            print("\n⚠ LSHBloom skipped (pybloom-live not installed)")

        # Add FED GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                methods.append((
                    "FED GPU",
                    lambda: create_deduplicator("fed", threshold=0.8)
                ))
            else:
                print("\n⚠ FED GPU skipped (CUDA not available)")
        except ImportError:
            print("\n⚠ FED GPU skipped (torch not installed)")

        # Run benchmarks
        for method_name, create_fn in methods:
            result = benchmark_method(method_name, create_fn, documents, doc_ids)
            result["num_docs"] = num_docs
            all_results.append(result)

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")

    # Group by document count
    for num_docs in test_sizes:
        print(f"\n{num_docs:,} Documents:")
        print(f"{'-'*80}")
        print(f"{'Method':<25} {'Speed (docs/s)':<15} {'Memory (MB)':<15} {'Dedup %':<10}")
        print(f"{'-'*80}")

        for result in all_results:
            if result.get("num_docs") == num_docs and result.get("success"):
                print(f"{result['method']:<25} "
                      f"{result['docs_per_second']:>12,.0f}   "
                      f"{result['memory_dedup_mb']:>12.1f}   "
                      f"{result['dedup_ratio']:>8.1%}")

    # Save results to JSON
    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"={'='*80}\n")


if __name__ == "__main__":
    main()
