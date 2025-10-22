# PDF Citations Directory

This directory contains PDF copies of key research papers referenced throughout the codebase.

## Directory Structure

- **01_Architecture** - Core model architecture papers (DeepSeek-V3, etc.)
- **02_Scaling_Laws** - Scaling laws and compute-optimal training papers
- **03_Data_Sources** - Large-scale dataset papers (Dolma, C4, The Pile, etc.)
- **04_Deduplication** - Deduplication methodology papers
- **05_Quality_Filtering** - Quality filtering and data curation papers
- **06_Domain_Mixing** - Domain mixing and data composition papers

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

### Papers Missing or Needs Attention

#### Missing Papers (referenced in code but not in pdf_citations)

1. **Li et al. (2025) - "Data × LLM: From Principles to Practices"**
   - arXiv:2505.18458
   - **Status**: ❌ Missing - This is a critical paper referenced throughout the pipeline code
   - **Location needed**: Could fit in 05_Quality_Filtering or a new 07_Data_Practices folder
   - **Note**: This appears to be a future paper (2025) - may not be publicly available yet

2. **Lozhkov et al. (2024) - "FineWeb-Edu"**
   - **Status**: ❌ Missing - No arXiv ID in paper_ids.txt
   - **Location needed**: 05_Quality_Filtering
   - **Alternative**: May be part of the FineWeb paper already present

3. **Xie et al. (2023/2024) - "DoReMi: Domain Reweighting with Minimax"**
   - arXiv:2403.06640 (listed in paper_ids.txt)
   - **Status**: ❌ PDF not downloaded yet
   - **Location needed**: 06_Domain_Mixing

4. **Nguyen et al. (2024) - "DCLM: DataComp for Language Models"**
   - arXiv:2409.09613 (listed in paper_ids.txt)
   - **Status**: ❌ PDF not downloaded yet
   - **Location needed**: 05_Quality_Filtering

#### Papers in Wrong Location

5. **Wang et al. - "Passive iFIR Filters for Data-Driven Control"**
   - **Status**: ⚠️ **Unrelated paper** - This appears to be about control systems, not LLM data mixing
   - **Current location**: 06_Domain_Mixing
   - **Action needed**: Remove or relocate if somehow relevant

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
