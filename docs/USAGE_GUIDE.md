# Data Preprocessing Usage Guide

## Quick Start

### For Testing (Recommended)

Use HuggingFace's `wikitext` dataset for quick testing:

```bash
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input wikitext \
    --output ./test_output
```

This will:
- Load first 1000 samples from wikitext dataset
- Apply all preprocessing stages
- Save results to `./test_output/`

### For Your Own Data

```bash
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input /path/to/your/data.jsonl \
    --output /mnt/shared/my_preprocessed_data
```

**Input Format**: JSONL file with `text` field per line:
```jsonl
{"text": "Your document text here..."}
{"text": "Another document..."}
```

## Usage Examples

### 1. Test with HuggingFace Dataset

```bash
# Test with wikitext (small, well-known dataset)
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input wikitext \
    --output ./test_output

# Test with c4 dataset
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input c4 \
    --output ./c4_preprocessed
```

### 2. Preprocess Your Data

```bash
# Basic usage
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input ./my_data/train.jsonl \
    --output ./preprocessed_data

# Custom output location
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input ./raw_data.jsonl \
    --output /mnt/shared/preprocessed_data
```

### 3. Disable Specific Stages

```bash
# Skip deduplication (faster)
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input ./data.jsonl \
    --no-deduplication \
    --output ./preprocessed

# Minimal preprocessing (cleaning only)
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input ./data.jsonl \
    --no-deduplication \
    --no-heuristic-filters \
    --no-domain-mixing \
    --output ./cleaned_only
```

### 4. Use Different Compositions

```bash
# DeepSeek-V3 style (default): 92% text, 5% code, 3% math
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input ./data.jsonl \
    --composition deepseek_v3 \
    --output ./preprocessed

# Llama 3 style: Balanced technical content
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input ./data.jsonl \
    --composition llama3 \
    --output ./preprocessed

# Balanced: Equal weight across domains
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input ./data.jsonl \
    --composition balanced \
    --output ./preprocessed
```

### 5. Integrated with Training

```bash
# Preprocess then train
python src/training/train.py \
    --config configs/deepseek_v3_small.yaml \
    --preprocess-data \
    --data-config configs/data_preprocessing.yaml \
    --raw-data-path ./raw_data/train.jsonl

# Or use already preprocessed data
python src/training/train.py \
    --config configs/deepseek_v3_small.yaml \
    --preprocessed-data-path ./preprocessed_data/final.jsonl
```

## Input Data Formats

### JSONL (JSON Lines)

One JSON object per line:
```jsonl
{"text": "Document 1 content..."}
{"text": "Document 2 content..."}
{"text": "Document 3 content..."}
```

**Optional fields**:
```jsonl
{"id": "doc_1", "text": "Content...", "domain": "web_general"}
{"id": "doc_2", "text": "Code...", "domain": "code"}
```

### HuggingFace Datasets (for testing)

Available datasets:
- `wikitext` - Wikipedia articles (recommended for testing)
- `c4` - Colossal Clean Crawled Corpus
- Any HuggingFace dataset with a `text` field

The pipeline automatically:
- Loads first 1000 samples for testing
- Handles different field names (`text`, `content`, `article`)
- Converts to standard format

## Output Structure

After preprocessing:

```
output_dir/
├── final.jsonl                # Final preprocessed dataset
├── pipeline_stats.json        # Complete statistics
└── intermediate/              # Intermediate outputs (if enabled)
    ├── 01_cleaned.jsonl      # After preliminary cleaning
    ├── 02_deduplicated.jsonl # After deduplication
    └── 03_mixed.jsonl        # After domain mixing
```

### Final Output Format

JSONL file:
```jsonl
{"id": "doc_0", "text": "Cleaned and filtered text...", "domain": "web_general"}
{"id": "doc_1", "text": "Another document...", "domain": "code"}
```

### Statistics File

`pipeline_stats.json`:
```json
{
  "total_input_documents": 1000,
  "total_output_documents": 850,
  "total_processing_time_seconds": 45.2,
  "documents_after_cleaning": 1000,
  "documents_after_dedup": 920,
  "documents_after_heuristic": 870,
  "documents_after_quality": 870,
  "documents_after_mixing": 850
}
```

## Command-Line Arguments

### Required

- `--config`: Path to YAML configuration file
- `--input`: Input data path (JSONL file or HF dataset name) **REQUIRED**

### Optional Overrides

- `--output`: Output directory (default from config)
- `--output-format`: Output format (`jsonl`, `parquet`, `hf_dataset`)
- `--batch-size`: Batch size for processing
- `--no-progress`: Disable progress bars
- `--no-intermediate`: Don't save intermediate outputs

### Stage Toggles

- `--no-cleaning`: Skip preliminary cleaning
- `--no-deduplication`: Skip deduplication
- `--no-heuristic-filters`: Skip heuristic filtering
- `--no-quality-filters`: Skip quality filtering
- `--no-domain-mixing`: Skip domain mixing

### Domain Mixing

- `--composition`: Domain composition (`deepseek_v3`, `llama3`, `balanced`)

### Debugging

- `--verbose`: Enable verbose logging
- `--dry-run`: Print configuration and exit

## Error Handling

### Missing Input

If you forget `--input`:
```bash
ERROR: Input data path is required!

Provide input data via --input argument:

  1. Test with HuggingFace dataset (recommended for testing):
     python scripts/preprocess_data.py \
       --config configs/data_preprocessing.yaml \
       --input wikitext \
       --output ./test_output

  2. Use your own JSONL file:
     python scripts/preprocess_data.py \
       --config configs/data_preprocessing.yaml \
       --input /path/to/your/data.jsonl \
       --output /mnt/shared/preprocessed
```

### File Not Found

If input file doesn't exist:
```
Error: Input file not found: /path/to/data.jsonl
Please provide a valid file path or HuggingFace dataset name.
```

### Invalid Configuration

If config file is invalid:
```
Error loading configuration: [error message]
```

## Performance Tips

### 1. Start with Small Sample

Test on small dataset first:
```bash
# Take first 1000 lines
head -n 1000 large_file.jsonl > sample.jsonl

python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input sample.jsonl \
    --output ./test_output
```

### 2. Disable Slow Stages for Testing

```bash
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input ./data.jsonl \
    --no-deduplication \
    --no-quality-filters \
    --output ./quick_test
```

### 3. Increase Batch Size

For faster processing:
```bash
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input ./data.jsonl \
    --batch-size 50000 \
    --output ./preprocessed
```

### 4. Monitor Progress

Enable progress bars (default):
```bash
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input ./data.jsonl \
    --output ./preprocessed

# Output:
# [1/5] Preliminary cleaning and heuristic filtering...
#   Processed 10,000/100,000 documents
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'datasets'"

Install HuggingFace datasets:
```bash
pip install datasets
```

### Issue: "ModuleNotFoundError: No module named 'datasketch'"

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Out of Memory

Reduce batch size:
```bash
python scripts/preprocess_data.py \
    --config configs/data_preprocessing.yaml \
    --input ./data.jsonl \
    --batch-size 1000 \
    --output ./preprocessed
```

### Issue: Too Many Documents Filtered

Relax filtering thresholds in `configs/data_preprocessing.yaml`:
```yaml
heuristic_filters:
  min_doc_length: 100     # Lower from 200
  min_word_count: 25      # Lower from 50
  min_alpha_ratio: 0.3    # Lower from 0.5
```

## Next Steps

1. **Test the pipeline**:
   ```bash
   python scripts/preprocess_data.py \
       --config configs/data_preprocessing.yaml \
       --input wikitext \
       --output ./test_output
   ```

2. **Preprocess your data**:
   ```bash
   python scripts/preprocess_data.py \
       --config configs/data_preprocessing.yaml \
       --input /path/to/your/data.jsonl \
       --output /mnt/shared/preprocessed
   ```

3. **Use in training**:
   ```bash
   python src/training/train.py \
       --config configs/deepseek_v3_small.yaml \
       --preprocessed-data-path /mnt/shared/preprocessed/final.jsonl
   ```

## Documentation

- [DATA_PREPROCESSING.md](DATA_PREPROCESSING.md) - Complete preprocessing guide
- [CONFIGURATIONS.md](CONFIGURATIONS.md) - Model configurations
- [TESTING.md](TESTING.md) - Test suite documentation
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide

## Support

All preprocessing follows SOTA methods from research papers (2022-2025):
- Zhou et al. (2025) - Data × LLM survey and preliminary cleaning
- Lee et al. (2022) - MinHash deduplication
- DeepSeek-V3 (2024) - Domain composition
- Gopher, C4, RefinedWeb, FineWeb - Heuristic filters
