# PDF Citations Directory

This directory contains PDF copies of key research papers referenced throughout the codebase.

## Directory Structure

- **01_Architecture** - Core model architecture papers (DeepSeek-V3, etc.)
- **02_Scaling_Laws** - Scaling laws and compute-optimal training papers
- **03_Data_Sources** - Large-scale dataset papers (Dolma, C4, The Pile, etc.)
- **04_Deduplication** - Deduplication methodology papers (including MinHash)
- **05_Quality_Filtering** - Quality filtering and data curation papers
- **06_Domain_Mixing** - Domain mixing and data composition papers (DoReMi)
- **07_Data_Practices** - General LLM data practices and survey papers

## Complete Paper List

All arXiv IDs are listed in [paper_ids.txt](paper_ids.txt).

### Papers Currently Available

#### 01_Architecture
- ✅ DeepSeek-AI et al. (2024) - "DeepSeek-V3 Technical Report" (arXiv:2412.19437)
- ✅ Fedus et al. (2021) - "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (arXiv:2101.03961)
- ✅ Su et al. (2021) - "RoFormer: Enhanced Transformer with Rotary Position Embedding" (arXiv:2104.09864)
- ✅ Barbero et al. (2024) - "Round and Round We Go! What makes Rotary Positional Encodings useful?" (arXiv:2410.06205)

#### 02_Scaling_Laws
- ✅ Rae et al. (2021) - "Scaling Language Models: Methods, Analysis & Insights from Training Gopher" (arXiv:2112.11446)
- ✅ Kaplan et al. (2020) - "Scaling Laws for Neural Language Models" (arXiv:2001.08361)
- ✅ Hoffmann et al. (2022) - "Training Compute-Optimal Large Language Models" (Chinchilla) (arXiv:2203.15556)

#### 03_Data_Sources
- ✅ Raffel et al. (2020) - "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5/C4) (arXiv:1910.10683)
- ✅ Gao et al. (2021) - "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" (arXiv:2101.00027)
- ✅ Soldaini et al. (2024) - "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research" (arXiv:2402.00159)

#### 04_Deduplication
- ✅ Lee et al. (2022) - "Deduplicating Training Data Makes Language Models Better" (arXiv:2107.06499)
- ✅ Khan et al. (2024) - "LSHBloom: Memory-efficient, Extreme-scale Document Deduplication" (arXiv:2411.04257)
- ✅ Son et al. (2025) - "FED: Fast and Efficient Dataset Deduplication Framework with GPU Acceleration" (arXiv:2501.01046)
- ✅ Broder (1997) - "On the resemblance and containment of documents" (DOI:10.1109/SEQUEN.1997.666900)

#### 05_Quality_Filtering
- ✅ Penedo et al. (2024) - "The RefinedWeb Dataset for Falcon LLM" (arXiv:2306.01116)
- ✅ Penedo et al. (2024) - "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale" (arXiv:2406.17557)
- ✅ Li et al. (2024) - "DataComp-LM: In search of the next generation of training sets for language models" (arXiv:2406.11794)
- ✅ Nait Saada et al. (2024) - "The Data-Quality Illusion" (arXiv:2510.00866)
- ✅ Kim et al. (2024) - "Rethinking KenLM: Good and Bad Model Ensembles for Efficient Text Quality Filtering" (arXiv:2409.09613)
- ✅ Joulin et al. (2016) - "Bag of Tricks for Efficient Text Classification" (FastText) (arXiv:1607.01759)
- ✅ Wenzek et al. (2019) - "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data" (arXiv:1911.00359)

#### 06_Domain_Mixing
- ✅ Xie et al. (2023) - "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining" (arXiv:2305.10429)

#### 07_Data_Practices
- ✅ Zhou et al. (2025) - "A Survey of LLM × DATA" (arXiv:2505.18458)
- ✅ Gloeckle et al. (2024) - "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737)
- ✅ Mehra et al. (2025) - "On multi-token prediction for efficient LLM inference" (arXiv:2502.09419)
- ✅ Cai et al. (2025) - "FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction" (arXiv:2509.18362)

### Citation Status Summary

**Total Papers:** 26 PDFs across 7 categories (25 arXiv + 1 non-arXiv)
- ✅ All critical SOTA pipeline citations are now present
- ✅ All code-referenced papers have been downloaded and organized
- ✅ Non-arXiv papers tracked in `paper_metadata.txt`
- ✅ Citation verification tests implemented in `tests/unit/test_citations.py`

### Recently Added (2025-10-21)

1. ✅ **Xie et al. (2023) - "DoReMi"** (arXiv:2305.10429) → `06_Domain_Mixing/`
   - Fixed arXiv ID discrepancy (was incorrectly listed as 2403.06640)

2. ✅ **Zhou et al. (2025) - "Data × LLM"** (arXiv:2505.18458) → `07_Data_Practices/`
   - Critical paper referenced throughout pipeline code
   - Note: First author is Xuanhe Zhou; cite as "Zhou et al." not "Li et al."

3. ✅ **Joulin et al. (2016) - "FastText"** (arXiv:1607.01759) → `05_Quality_Filtering/`
   - Foundation paper for text classification quality filters

4. ✅ **Broder (1997) - "MinHash"** (DOI:10.1109/SEQUEN.1997.666900) → `04_Deduplication/`
   - Seminal paper for document resemblance and containment

5. ✅ **Wenzek et al. (2019) - "CCNet"** (arXiv:1911.00359) → `05_Quality_Filtering/`
   - High-quality monolingual dataset extraction from web crawls
   - Referenced in DeepSeek configs for quality filtering

6. ✅ **Gloeckle et al. (2024) - "Better & Faster LLMs via Multi-token Prediction"** (arXiv:2404.19737) → `07_Data_Practices/`
   - Multi-token prediction (MTP) methodology paper
   - Referenced in `tests/unit/test_mtp.py`

7. ✅ **Mehra et al. (2025) - "On multi-token prediction for efficient LLM inference"** (arXiv:2502.09419) → `07_Data_Practices/`
   - MTP efficiency analysis for inference

8. ✅ **Cai et al. (2025) - "FastMTP"** (arXiv:2509.18362) → `07_Data_Practices/`
   - Accelerating LLM inference with enhanced MTP

9. ✅ **Su et al. (2021) - "RoFormer"** (arXiv:2104.09864) → `01_Architecture/`
   - Rotary Position Embedding (RoPE) foundation paper
   - Referenced in `tests/unit/test_rope.py`

10. ✅ **Barbero et al. (2024) - "Round and Round We Go!"** (arXiv:2410.06205) → `01_Architecture/`
    - Analysis of what makes Rotary Positional Encodings useful

11. ✅ **Fedus et al. (2021) - "Switch Transformers"** (arXiv:2101.03961) → `01_Architecture/`
    - Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
    - Referenced in MoE load balancing loss implementation (`src/model/moe.py`)

### Notes

- **FineWeb-Edu**: Confirmed to be part of the main FineWeb paper (arXiv:2406.17557), not a separate citation
- **DCLM vs KenLM**: The KenLM filtering paper (arXiv:2409.09613) is present and addresses similar concerns as DCLM
- **DoReMi**: The correct arXiv ID is 2305.10429 (published 2023). Previously incorrectly referenced as 2403.06640 in some documents.

## How to Update Citations

1. Ensure the `arxiv` package is installed: `pip install arxiv>=2.0.0` (or install from `requirements.txt`)
2. Add arXiv IDs to `paper_ids.txt`
3. Run `python download_papers.py paper_ids.txt` to download PDFs
4. Manually organize PDFs into appropriate subdirectories
5. Update `arxiv_to_pdf_mapping.json` with new entries
6. Update this README

## Paper ID Format

The `paper_ids.txt` file contains arXiv identifiers (without the "arXiv:" prefix):
```
2412.19437
2203.15556
...
```

## Citation Coverage Status

All tracked arXiv IDs in `paper_ids.txt` have been downloaded and mapped:

- ✅ All 25 arXiv papers downloaded and organized by category
- ✅ All papers referenced in code have corresponding PDFs
- ✅ Mapping file (`arxiv_to_pdf_mapping.json`) is complete and up-to-date
- ✅ No missing or outstanding citation downloads

To add new citations, update `paper_ids.txt` and run `python download_papers.py`

## Verification

To verify all cited papers have matching PDFs, run:
```bash
pytest tests/unit/test_citations.py
```

The citation validation test (`test_documentation_arxiv_ids_are_valid`) scans the following for arXiv references:
- Root-level markdown files (`*.md`)
- All documentation files (`docs/**/*.md`)
- Source code files (`src/**/*.py`)
- Scripts (`scripts/*.py`)
- Configuration files (`configs/**/*.{json,yaml,yml,md}`)
- This README file

Any arXiv ID found in these files must exist in `paper_ids.txt` or `paper_metadata.txt`.

## References in Code

The following modules reference papers that should be in this directory:

- `src/data/pipeline.py` - References Zhou et al. (2025) arXiv:2505.18458
- `src/data/deduplication.py` - References Lee et al. (2022) arXiv:2107.06499
- `src/data/deduplication_fed.py` - References Son et al. (2025) arXiv:2501.01046 (GPU-accelerated FED)
- `src/data/deduplication_lshbloom.py` - References Khan et al. (2024) arXiv:2411.04257 (Memory-efficient LSHBloom)
- `src/data/quality_filters.py` - References FastText, KenLM papers
- `src/data/heuristic_filters.py` - References Zhou et al., DCLM, The Pile
- `docs/DATA_SANITIZATION_STATUS.md` - Lists all major references

---

*Last Updated: 2025-10-22*
