<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)" />
  </a>
</p>

<h1 align="center">HARE: Hate Speech Detection and Highlighting for Vietnamese</h1>

<p align="center">
  <img src="Thumbnail.png" alt="HARE project thumbnail" width="420" />
</p>

<p align="center">
  IE403.Q11 Course Project - Social Media Mining - UIT, VNU-HCM
</p>

HARE is an explainable hate speech analysis framework for Vietnamese social media text. The system combines multi-label classification with rationale extraction and implied-statement inference, then maps evidence spans back to source text for transparent review.

## Table of Contents

- [Abstract](#abstract)
- [Research Objectives](#research-objectives)
- [Core Contributions](#core-contributions)
- [Updated Repository Structure](#updated-repository-structure)
- [Experimental Summary](#experimental-summary)
- [Methodology](#methodology)
- [Setup and Reproducibility](#setup-and-reproducibility)
- [Backend Preview API](#backend-preview-api)
- [Project Artifacts](#project-artifacts)
- [Team](#team)
- [Citation](#citation)

---

## Abstract

This repository presents the implementation of HARE, developed for the IE403.Q11 course project. The framework addresses two central requirements in hate speech detection: classification accuracy and interpretability. In addition to predicting labels, HARE extracts rationales and inferred implied statements, supporting traceable and explainable model decisions in Vietnamese social media contexts.

> [!IMPORTANT]
> The repository is structured for academic reproducibility and code inspection.
> Full runnable demo assets (including large model files) are distributed externally due to storage constraints.

---

## Research Objectives

- Design a Vietnamese hate speech detection pipeline with explicit explainability outputs.
- Assess whether rationale-aware supervision improves challenging and implicit hate categories.
- Provide an application-ready prototype for real-time moderation analysis.

---

## Core Contributions

- Two-stage semantic alignment training on Qwen2.5-3B (QLoRA-based).
- Multi-label categorization across five targets: individuals, groups, religion, race/ethnicity, and politics.
- Rationale-to-span highlighting for interpretable evidence localization.
- FastAPI backend preview supporting single inference, batch inference, and YouTube comment analysis.

---

## Updated Repository Structure

```text
.
|-- app-preview/
|   |-- backend-logic/          # FastAPI logic: API routes, model wrapper, highlighting, YouTube integration
|   |-- frontend-snippet/       # Frontend entry/template snippet
|   `-- sample-outputs/         # Sample inference outputs and model metadata
|-- dataset/
|   |-- raw/                    # Original ViTHSD source files
|   `-- processed/              # Processed training data (dataset_rationale.json)
|-- docs/
|   |-- IE403.Q11-Nhom2_report.pdf
|   `-- IE403.Q11-Nhom2_slide.pdf
|-- results/
|   |-- figures/
|   |   `-- live-demo.gif
|   `-- videos/
|       `-- demo_video.mp4
|-- research/
|   |-- notebooks/              # Baseline, prompt, and fine-tuning notebooks
|   |-- prompts/                # Prompt versions (v1, v2, v3)
|   `-- src/                    # Data prep, configuration, modeling, evaluation
|-- MAPR2026/                   # LaTeX manuscript sources and submission materials
|-- requirements.txt
`-- README.md
```

---

## Experimental Summary

HARE achieves the highest overall micro F1 on the ViTHSD test split among evaluated baselines.

| Model | Precision (Micro) | Recall (Micro) | F1-score (Micro) |
|---|---:|---:|---:|
| PhoBERT-base | 0.5620 | 0.5310 | 0.5412 |
| Flan-T5-base | 0.4810 | 0.4520 | 0.4684 |
| Qwen2.5 (Vanilla, Stage 1) | 0.6100 | 0.5600 | 0.5900 |
| **HARE (Qwen2.5 + Rationales, Stage 1 + 2)** | **0.6347** | **0.5735** | **0.6026** |

The strongest gains are observed in context-heavy categories, especially political hate speech, where rationale-guided learning improves implicit intent recognition.

---

## Methodology

### Two-stage semantic alignment

1. Stage 1: Fine-tune Qwen2.5 on label-only supervision to establish class boundaries.
2. Stage 2: Continue training with rationale and implied-statement supervision to align semantic reasoning with final labels.

### Explainability pipeline

- Rationale extraction identifies evidence tokens/spans for moderation decisions.
- Unicode-aware span mapping projects rationale evidence back to original user text.
- Implied-statement generation helps interpret indirect or metaphorical hostility.

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
> Several notebooks were originally authored for Kaggle paths.
> When running locally, update file paths to the repository layout (for example `dataset/processed/dataset_rationale.json`).

---

## Backend Preview API

The backend preview module is located in `app-preview/backend-logic`.

Available routes:
- `GET /health`
- `POST /v1/analyze`
- `POST /v1/analyze/batch`
- `GET /v1/youtube/comments`

> [!CAUTION]
> End-to-end backend inference requires local model assets (keywords file, adapter checkpoints, tokenizer files).
> These files are part of the external demo package, not fully embedded in this repository.

---

## Project Artifacts

- Demo GIF: `results/figures/live-demo.gif`
- Demo video: `results/videos/demo_video.mp4`
- Sample outputs: `app-preview/sample-outputs/results_datasetA_qwen_stage2.json`
- Full demo package (external storage):
  - https://nklod-my.sharepoint.com/:f:/g/personal/phatxinhchao_nklod_onmicrosoft_com/IgAYGfiHj2ZsTpr2aebNbSfrAVG0YJ0LkziTmToc1uIn1oY?e=nnp26c

<p align="center">
  <img src="results/figures/live-demo.gif" alt="HARE live demo" width="900" />
</p>

- Project video: `demo_video.mp4`
- External full demo package (weights + runnable application):
  - https://nklod-my.sharepoint.com/:f:/g/personal/phatxinhchao_nklod_onmicrosoft_com/IgAYGfiHj2ZsTpr2aebNbSfrAVG0YJ0LkziTmToc1uIn1oY?e=nnp26c

---

## Team

| No. | Student ID | Full Name | Role | GitHub |
|---:|:---:|---|---|---|
| 1 | 23521143 | Phat Nguyen Cong | Leader | [paht2005](https://github.com/paht2005) |
| 2 | 23520032 | An Truong Hoang Thanh | Member | [Awnpz](https://github.com/Awnpz) |
| 3 | 23520023 |An Nguyen Xuan | Member | [annx-uit](https://github.com/annx-uit) |
| 4 | 23520158 | Binh Mai Thai | Member | [maibinhkznk209](https://github.com/maibinhkznk209/) |
| 5 | 21520255 | Huong Nguyen Le Quynh | Member | [tracycute](https://github.com/tracycute) |

---

## Citation

If this repository supports your research or coursework, please cite the project repository and the associated IE403 report/paper artifacts.

---

## **License**
This project is for academic use in the course **IE403.Q11 - Social Media Mining** at the University of Information Technology (UIT – VNU-HCM).

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
