# Experiment Notebooks for Paper Revision

## Overview

| Notebook | Purpose | GPU Required | Est. Time |
|----------|---------|:---:|-----------|
| `R1_multiseed.ipynb` | Multi-seed evaluation (3 seeds × 2 models) | Yes | 10-14h |
| `R2_ablations.ipynb` | Shuffled-rationale + Plain-continuation ablations | Yes | 5-7h |
| `R3_R6_R8_analysis.ipynb` | Bootstrap test, parse failures, rationale quality | No (CPU) | <5 min |

## Setup on Google Colab Pro

### Step 1: Upload data to Google Drive

Create this folder structure on Google Drive:

```
MyDrive/
  ViTHSD/
    src/
      config.py          ← from research/src/
      data_preparation.py ← from research/src/
      evaluation.py       ← from research/src/
      models.py           ← from research/src/
    data/
      train.xlsx          ← from dataset/raw/
      test.xlsx           ← from dataset/raw/
      dataset_rationale.json ← from dataset/processed/
```

### Step 2: Upload notebooks

Upload each `.ipynb` file to Colab:
- Go to https://colab.research.google.com
- File → Upload notebook → select the `.ipynb` file

### Step 3: Configure Colab runtime

- Runtime → Change runtime type → **GPU** (A100 preferred, V100 or T4 also ok)
- For R1 & R2: make sure you have Colab Pro for long runtime

### Step 4: Run order

```
1. R1_multiseed.ipynb     → produces results_multiseed.json + all_predictions.json
2. R2_ablations.ipynb     → produces results_ablations.json
3. R3_R6_R8_analysis.ipynb → produces final analysis (uses outputs from NB1 & NB2)
```

## What to send back

After running all notebooks, download these files from `MyDrive/ViTHSD/outputs/`:

1. **`results_multiseed.json`** — Mean ± std for RSQwen and Vanilla across 3 seeds
2. **`all_predictions.json`** — Raw predictions (needed for bootstrap test in NB3)
3. **`results_ablations.json`** — Ablation results (shuffled-rationale + plain-continuation)
4. **`analysis_results.json`** — Bootstrap p-values, parse failure stats, rationale quality sample

Send me all 4 JSON files and I'll integrate the results into the paper.

## Estimated Colab Pro GPU Hours

| Run | Time (A100) | Time (V100) | Time (T4) |
|-----|-------------|-------------|-----------|
| RSQwen seed 42 | ~1.5h | ~2.5h | ~4h |
| RSQwen seed 123 | ~1.5h | ~2.5h | ~4h |
| RSQwen seed 456 | ~1.5h | ~2.5h | ~4h |
| Vanilla seed 42 | ~1h | ~2h | ~3h |
| Vanilla seed 123 | ~1h | ~2h | ~3h |
| Vanilla seed 456 | ~1h | ~2h | ~3h |
| **NB1 Total** | **~7.5h** | **~14h** | **~21h** |
| Shuffled-rationale | ~1.5h | ~2.5h | ~4h |
| Plain-continuation | ~1.5h | ~2.5h | ~4h |
| **NB2 Total** | **~3h** | **~5h** | **~8h** |

> If on T4, NB1 may exceed 24h. In that case, modify the `SEEDS` list to run 2 seeds first, save, then rerun with the remaining seed.
