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

#### 05_Quality_Filtering
- ✅ Penedo et al. (2024) - "The RefinedWeb Dataset for Falcon LLM" (arXiv:2306.01116)
- ✅ Penedo et al. (2024) - "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale" (arXiv:2406.17557)
- ✅ Li et al. (2024) - "DataComp-LM: In search of the next generation of training sets for language models" (arXiv:2406.11794)
- ✅ Zhou et al. (2024) - "A Survey of LLM × DATA" (arXiv:2510.00866)
- ✅ Saada et al. (2024) - "The Data-Quality Illusion: Rethinking Classifier-Based Quality Filtering for LLM Pretraining"
- ✅ Kim et al. (2024) - "Rethinking KenLM: Good and Bad Model Ensembles for Efficient Text Quality Filtering"

### Citation Status Summary

**Total Papers:** 19 PDFs across 7 categories
- ✅ All critical SOTA pipeline citations are now present
- ✅ All code-referenced papers have been downloaded and organized
- ✅ Non-arXiv papers tracked in `paper_metadata.txt`
- ✅ Citation verification tests implemented in `tests/unit/test_citations.py`

### Recently Added (2025-10-21)

1. ✅ **Xie et al. (2023) - "DoReMi"** (arXiv:2305.10429) → `06_Domain_Mixing/`
   - Fixed arXiv ID discrepancy (was incorrectly listed as 2403.06640)

2. ✅ **Zhou et al. (2025) - "Data × LLM"** (arXiv:2505.18458) → `07_Data_Practices/`
   - Critical paper referenced throughout pipeline code

3. ✅ **Joulin et al. (2016) - "FastText"** (arXiv:1607.01759) → `05_Quality_Filtering/`
   - Foundation paper for text classification quality filters

4. ✅ **Broder (1997) - "MinHash"** (DOI:10.1109/SEQUEN.1997.666900) → `04_Deduplication/`
   - Seminal paper for document resemblance and containment

### Notes

- **FineWeb-Edu**: Confirmed to be part of the main FineWeb paper (arXiv:2406.17557), not a separate citation
- **DCLM vs KenLM**: The KenLM filtering paper (arXiv:2409.09613) is present and addresses similar concerns as DCLM
- **Removed**: Control systems paper (arXiv:2403.06640) - unrelated to LLM data mixing (see `paper_metadata.txt` for audit trail)

## How to Update Citations

1. Add arXiv IDs to `paper_ids.txt`
2. Run `python download_papers.py` to download PDFs
3. Manually organize PDFs into appropriate subdirectories
4. Update this README

## Paper ID Format

The `paper_ids.txt` file contains arXiv identifiers (without the "arXiv:" prefix):
```
2412.19437
2203.15556
...
```

## Missing ArXiv IDs Needed

To complete the citation library, please add these arXiv IDs to `paper_ids.txt`:

- [ ] 2505.18458 (Data × LLM - may not exist yet, check availability)
- [ ] FineWeb-Edu paper ID (or confirm it's part of 2406.17557)
- [ ] Confirm 2403.06640 (DoReMi) is correct and download PDF
- [ ] Confirm 2409.09613 (DCLM) is correct and download PDF

## Verification

To verify all cited papers have matching PDFs, run:
```bash
pytest tests/unit/test_citations.py
```

## References in Code

The following modules reference papers that should be in this directory:

- `src/data/pipeline.py` - References Li et al. (2025) arXiv:2505.18458
- `src/data/deduplication.py` - References Lee et al. (2022), FED, LSHBloom
- `src/data/quality_filters.py` - References FastText, KenLM papers
- `src/data/heuristic_filters.py` - References Li et al., DCLM, The Pile
- `docs/DATA_SANITIZATION_STATUS.md` - Lists all major references

---

*Last Updated: 2025-10-21*
