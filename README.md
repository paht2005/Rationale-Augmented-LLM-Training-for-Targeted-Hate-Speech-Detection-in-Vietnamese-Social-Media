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

This repository presents HARE, an explainable framework for Vietnamese hate speech detection. The system integrates multi-label classification, rationale extraction, and implied statement inference to support transparent moderation decisions on social media text.


## Table of Contents

- [Abstract](#abstract)
- [Research Objectives](#research-objectives)
- [Core Contributions](#core-contributions)
- [Repository Structure](#repository-structure)
- [Experimental Results](#experimental-results)
- [Methodology](#methodology)
- [Setup and Reproducibility](#setup-and-reproducibility)
- [Backend Preview API](#backend-preview-api)
- [Demo Resources](#demo-resources)
- [Team](#team)
- [Citation](#citation)
- [License](#license)

---
## Abstract

HARE is developed within the IE403.Q11 course to address two requirements in hate speech analysis: predictive performance and interpretability. Beyond assigning labels, the framework identifies textual evidence (rationales) and transforms implicit hostile expressions into explicit implied statements. This improves traceability of model decisions and supports downstream moderation workflows.

> [!IMPORTANT]
> This repository is organized for academic reproducibility and source inspection.
> The complete runnable demo (including large model artifacts) is distributed externally due to storage constraints.

---
## Research Objectives

- Build a Vietnamese hate speech detection pipeline with explainable outputs.
- Evaluate whether rationale-augmented supervision improves difficult label groups.
- Provide a practical full-stack prototype for interactive analysis.

---
## Core Contributions

- A two-stage fine-tuning strategy on Qwen2.5-3B with QLoRA.
- Multi-label classification across five targets: individuals, groups, religion, race/ethnicity, and politics.
- Rationale-aligned highlighting that maps model evidence back to original text spans.
- A FastAPI-based backend preview with single-text, batch, and YouTube-comment analysis routes.

---
## Repository Structure

```text
.
|-- app-preview/
|   |-- backend-logic/        # FastAPI preview code (API, model wrapper, highlighting, YouTube integration)
|   `-- frontend-snippet/     # Frontend snippet (index template)
|-- dataset/
|   |-- raw/                  # Original ViTHSD files
|   `-- processed/            # Processed rationale dataset
|-- research/
|   |-- notebooks/            # Baseline and fine-tuning experiments
|   |-- prompts/              # Prompt evolution (v1 -> v3)
|   `-- src/                  # Data preparation, model config, evaluation
|-- MAPR2026/                 # LaTeX sources for paper/report
|-- requirements.txt
`-- README.md
```

---
## Experimental Results

The proposed HARE configuration provides the best overall micro F1 on ViTHSD test data among compared baselines.

| Model | Precision (Micro) | Recall (Micro) | F1-score (Micro) |
|---|---:|---:|---:|
| PhoBERT-base | 0.5620 | 0.5310 | 0.5412 |
| Flan-T5-base | 0.4810 | 0.4520 | 0.4684 |
| Qwen2.5 (Vanilla, Stage 1) | 0.6100 | 0.5600 | 0.5900 |
| **HARE (Qwen2.5 + Rationales, Stage 1 + 2)** | **0.6347** | **0.5735** | **0.6026** |

The most notable gains are observed in politically sensitive and context-dependent cases, where rationale-guided training improves interpretation of implicit hostility.

---
## Methodology

### Two-stage semantic alignment

1. Stage 1: Fine-tune Qwen2.5 on ViTHSD labels to establish class boundaries.
2. Stage 2: Continue training with rationale and implied-statement supervision to align latent reasoning with final labels.

### Explainability layer

- Rationale extraction identifies toxic evidence in free-form text.
- Span mapping links rationale evidence to character-level highlights in the original sentence.
- Implied statement generation makes indirect hostility more explicit for human review.

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
> Notebook paths were initially configured for Kaggle environments and may require local path updates.
> If needed, re-point dataset references to `dataset/processed/dataset_rationale.json`.

---
## Backend Preview API

The `app-preview/backend-logic` module contains the main backend logic used by the complete demo package.

Available routes:
- `GET /health`
- `POST /v1/analyze`
- `POST /v1/analyze/batch`
- `GET /v1/youtube/comments`

> [!CAUTION]
> The preview backend expects local assets (for example `data/hate_keywords.json`, adapter checkpoints, and tokenizer files).
> End-to-end execution requires the external demo package or equivalent local model assets.

---
## Demo Resources

- Demo animation: `live-demo.gif`

<p align="center">
  <img src="live-demo.gif" alt="Hate Speech Detection and Highlighting for Vietnamese with Rationale Extraction Demo" width="900">
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
