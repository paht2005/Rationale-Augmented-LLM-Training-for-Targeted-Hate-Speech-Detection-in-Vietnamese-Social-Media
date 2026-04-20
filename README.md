<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)" />
  </a>
</p>

<h1 align="center">HARE: Rationale-Augmented Training for Targeted Hate Speech Detection in Vietnamese Social Media</h1>

<p align="center">
  <img src="Thumbnail.png" alt="HARE project thumbnail" width="420" />
</p>

<p align="center">
  IE403.Q11 Course Project · Social Media Mining · UIT, VNU-HCM
</p>

HARE is an explainable hate speech analysis framework for Vietnamese social media text. The system combines multi-label classification with rationale extraction and implied-statement inference via a two-stage QLoRA pipeline on Qwen2.5-3B-Instruct, then maps evidence spans back to source text for transparent moderation review.

## Table of Contents

- [Abstract](#abstract)
- [Research Objectives](#research-objectives)
- [Core Contributions](#core-contributions)
- [Repository Structure](#repository-structure)
- [Experimental Summary](#experimental-summary)
- [Methodology](#methodology)
- [Setup and Reproducibility](#setup-and-reproducibility)
- [Backend Preview API](#backend-preview-api)
- [Project Artifacts](#project-artifacts)
- [Team](#team)
- [Citation](#citation)

---

## Abstract

This repository presents HARE, developed for the IE403.Q11 course project. The framework addresses two central requirements in hate speech detection: classification accuracy and interpretability. A two-stage QLoRA pipeline fine-tunes Qwen2.5-3B-Instruct on the ViTHSD dataset (10,001 comments, 11 multi-labels): Stage 1 trains a multi-label classifier on 7,540 oversampled examples; Stage 2 continues on 1,221 CoT-rationale tuples annotated with human-verified implied statements. In addition to predicting labels, HARE extracts rationales supporting traceable moderation decisions.

Multi-seed evaluation (seeds 42, 123, 456) reveals a key finding: HARE exhibits a **boundary redistribution** effect — gaining an average +8.35 F1 on Offensive-level labels (Individuals#Offensive +20.38, Groups#Offensive +12.67) while Hate-level labels decline by −2.64 on average, consistent across all three seeds.

> [!IMPORTANT]
> The repository is structured for academic reproducibility and code inspection.
> Full runnable demo assets (model weights, LoRA adapters) are distributed externally due to storage constraints.

---

## Research Objectives

- Design a Vietnamese hate speech detection pipeline with explicit explainability outputs.
- Assess whether rationale-aware supervision reshapes the Offensive–Hate decision boundary on ViTHSD.
- Characterize per-label boundary redistribution through rigorous multi-seed evaluation.
- Provide an application-ready prototype for real-time moderation analysis.

---

## Core Contributions

- **Two-stage QLoRA training** on Qwen2.5-3B-Instruct: Stage 1 (label-only) + Stage 2 (CoT rationale continuation).
- **1,221 rationale tuples** generated via Gemini 2.0 Flash with three-iteration prompt refinement and human verification.
- **Multi-label categorization** across five targets: Individuals, Groups, Religion, Race/Ethnicity, and Politics — each with Offensive/Hate severity levels (11 classes total).
- **Boundary redistribution finding**: rationale training consistently shifts predictions toward Offensive-level labels, documented across three independent seeds.
- **Matched-budget ablation controls** (Shuffled-Rationale, Plain-Continuation) providing directional evidence that rationale semantic structure drives the redistribution.
- **Rationale-to-span highlighting** for interpretable evidence localization in a FastAPI backend.

---

## Repository Structure

```text
.
|-- app-preview/
|   |-- backend-logic/          # FastAPI routes, model wrapper, highlighting, YouTube integration
|   |-- frontend-snippet/       # HTML entry/template snippet
|   `-- sample-outputs/         # Sample inference outputs (JSON)
|-- dataset/
|   |-- raw/                    # Original ViTHSD source files (train/dev/test .xlsx)
|   `-- processed/              # Annotated training data (dataset_rationale.json)
|-- docs/
|   `-- IE403.Q11-Nhom2_slide.pdf   # Course presentation slides
|-- models/
|   `-- visobert_datasetA_rationale_attn_state_dict.pt  # ViSoBERT baseline checkpoint
|-- results/
|   `-- figures/
|       `-- live-demo.gif       # Animated demo of the moderation UI
|-- research/
|   |-- notebooks/              # Baseline and fine-tuning notebooks (Kaggle-compatible)
|   |   |-- base-flant5.ipynb
|   |   |-- base-phobert.ipynb
|   |   |-- base-qwen.ipynb
|   |   |-- qwen_rationale.ipynb
|   |   |-- test_prompts.ipynb
|   |   |-- experiments/        # Paper revision experiments (multi-seed, ablations, analysis)
|   |   |   |-- R1_multiseed.ipynb
|   |   |   |-- R1_seed42.ipynb
|   |   |   |-- R1_seed123.ipynb
|   |   |   |-- R1_seed456.ipynb
|   |   |   |-- R2_ablations.ipynb
|   |   |   |-- R3_R6_R8_analysis.ipynb
|   |   |   |-- README_EXPERIMENTS.md
|   |   |   `-- outputs/        # JSON/CSV outputs from experiment runs
|   |   `-- outputs/            # Inference outputs from baseline notebooks
|   |-- prompts/                # Prompt iterations (v1_initial, v2_refined, v3_final)
|   `-- src/                    # Reusable Python modules: config, data_preparation, models, evaluation
|-- requirements.txt
`-- README.md
```

---

## Experimental Summary

Multi-seed evaluation (seeds 42, 123, 456) on the ViTHSD test set (1,800 samples). HARE and Vanilla results are mean ± std over three seeds; baselines are single-run under the same 7,540-sample oversampled training protocol.

| Model | Precision (Micro) | Recall (Micro) | F1-Micro | F1-Macro |
|---|---:|---:|---:|---:|
| BiGRU-LSTM-CNN | 0.4198 | **0.6798** | 0.5191 | 0.3190 |
| PhoBERT-base | 0.5620 | 0.5310 | 0.5412 | 0.2586 |
| Flan-T5-base | 0.4810 | 0.4520 | 0.4684 | 0.1311 |
| Qwen2.5 Vanilla (Stage 1) | **0.6102 ± 0.017** | 0.5843 ± 0.022 | **0.5967 ± 0.013** | 0.3295 ± 0.016 |
| **HARE (Stage 1 + 2, proposed)** | 0.5091 ± 0.027 | 0.6111 ± 0.011 | 0.5551 ± 0.016 | **0.3570 ± 0.044** |

> [!NOTE]
> HARE trades aggregate F1-Micro for higher Recall-Micro and F1-Macro. The primary finding is a per-label **boundary redistribution**: HARE gains +8.35 avg F1 on Offensive-level labels while Hate-level labels decline by −2.64 avg F1 — a pattern consistent across all three seeds and opposite to preliminary single-run observations.

---

## Methodology

### Two-stage QLoRA fine-tuning

1. **Stage 1 (Classification):** Fine-tune Qwen2.5-3B-Instruct on 7,540 oversampled ViTHSD training examples (label-only supervision). QLoRA with 4-bit NF4 quantization, LoRA rank r=8, α=16, targeting all attention and feed-forward projections.
2. **Stage 2 (Rationale-Augmented Continuation):** Continue from the Stage 1 checkpoint on 1,221 CoT-rationale tuples. Each tuple contains the comment, implied statement, step-by-step evidence rationale, and label — teaching the model to associate implicit expressions with their semantic intent. Rationale supervision is applied **only at training time**; inference produces label predictions only.

### Rationale generation pipeline

- 7,528 hate/offensive instances annotated with human-verified implied statements.
- Gemini 2.0 Flash generates constrained four-step CoT rationales (Target → Implied → Evidence → Verdict) in structured JSON.
- Three prompt iterations (v1 → v2 → v3) refined via human evaluation (human scores: 40 → 67 → 89/100).
- Automatic consistency filter: each generated rationale is re-verified by Gemini; mismatches discarded. Final acceptance rate ≈ 16% (1,221 of 7,528).

### Explainability pipeline

- Unicode-aware span mapping projects rationale evidence tokens back to original user text for moderation highlighting.
- Rule-based hate keyword matching augments model predictions in the frontend moderator interface.

---

## Setup and Reproducibility

### 1. Environment setup

```bash
git clone https://github.com/paht2005/IE403.Q11_Hate-Speech-Detection-and-Highlighting-for-Vietnamese-Project.git
cd IE403.Q11_Hate-Speech-Detection-and-Highlighting-for-Vietnamese-Project

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Reproduce notebook experiments

```bash
cd research/notebooks
jupyter notebook
```

> [!NOTE]
> Notebooks were originally authored for Kaggle/Google Colab paths.
> When running locally, update file paths to match the repository layout (e.g., `../../dataset/processed/dataset_rationale.json`).
> For paper revision experiments (multi-seed, ablations), see `research/notebooks/experiments/README_EXPERIMENTS.md`.

### 3. Experiment outputs

Pre-computed JSON results from all paper experiments are in `research/notebooks/experiments/outputs/`:
- `results_seed_{42,123,456}.json` — per-seed evaluation results
- `predictions_seed_{42,123,456}.json` — raw model predictions
- `results_ablations.json` — matched-budget ablation controls
- `analysis_results.json` — bootstrap significance tests and parse-failure analysis

---

## Backend Preview API

The backend preview module is in `app-preview/backend-logic/`.

Available routes:

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/v1/analyze` | Single comment inference |
| `POST` | `/v1/analyze/batch` | Batch inference |
| `GET` | `/v1/youtube/comments` | Fetch and analyze YouTube comments |

> [!CAUTION]
> End-to-end inference requires local model assets (LoRA adapter checkpoints, tokenizer, hate keywords file).
> These are part of the external demo package — see Project Artifacts below.

---

## Project Artifacts

| Artifact | Location |
|---|---|
| Live demo GIF | `results/figures/live-demo.gif` |
| Sample inference output | `app-preview/sample-outputs/results_datasetA_qwen_stage2.json` |
| Experiment results | `research/notebooks/experiments/outputs/` |
| Course slides | `docs/IE403.Q11-Nhom2_slide.pdf` |
| Full demo package (weights + runnable app) | [OneDrive](https://nklod-my.sharepoint.com/:f:/g/personal/phatxinhchao_nklod_onmicrosoft_com/IgAYGfiHj2ZsTpr2aebNbSfrAVG0YJ0LkziTmToc1uIn1oY?e=nnp26c) |

<p align="center">
  <img src="results/figures/live-demo.gif" alt="HARE live demo" width="900" />
</p>

---

## Team

| No. | Student ID | Full Name | Role | GitHub |
|---:|:---:|---|---|---|
| 1 | 23521143 | Phat Nguyen Cong | Leader | [paht2005](https://github.com/paht2005) |
| 2 | 23520032 | An Truong Hoang Thanh | Member | [Awnpz](https://github.com/Awnpz) |
| 3 | 23520023 | An Nguyen Xuan | Member | [annx-uit](https://github.com/annx-uit) |
| 4 | 23520158 | Binh Mai Thai | Member | [maibinhkznk209](https://github.com/maibinhkznk209/) |
| 5 | 21520255 | Huong Nguyen Le Quynh | Member | [tracycute](https://github.com/tracycute) |

---

## Citation

If this repository supports your research or coursework, please cite the project repository and the associated IE403 report/paper artifacts.

---

## License

This project is for academic use in the course **IE403.Q11 - Social Media Mining** at the University of Information Technology (UIT – VNU-HCM).

Licensed under the MIT License — see [LICENSE](./LICENSE) for details.
