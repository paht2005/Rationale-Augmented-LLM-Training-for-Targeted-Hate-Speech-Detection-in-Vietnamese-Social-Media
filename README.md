<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)" />
  </a>
</p>

<h1 align="center">Rationale-Augmented LLM Training for Targeted Hate Speech Detection in Vietnamese Social Media</h1>

<p align="center">
  <img src="Thumbnail.png" alt="HARE project thumbnail" width="420" />
</p>

<p align="center">
  IE403.Q11 · Social Media Mining · UIT, VNU-HCM
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License" /></a>
  <img src="https://img.shields.io/badge/Python-3.11-3c873a?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Qwen2.5--3B-QLoRA-blue?style=flat-square" alt="Model" />
  <img src="https://img.shields.io/badge/React-19-61dafb?style=flat-square&logo=react&logoColor=white" alt="React" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
</p>

**HARE** is an explainable hate speech detection framework for Vietnamese social media. It combines multi-label classification (11 classes across 5 target groups) with rationale extraction via a two-stage QLoRA pipeline on Qwen2.5-3B-Instruct, mapping evidence spans back to source text for transparent content moderation.

> [!IMPORTANT]
> This repository is structured for academic reproducibility and code inspection.
> Full runnable demo assets (model weights, LoRA adapters) are distributed via [OneDrive](https://nklod-my.sharepoint.com/:f:/g/personal/phatxinhchao_nklod_onmicrosoft_com/IgAYGfiHj2ZsTpr2aebNbSfrAVG0YJ0LkziTmToc1uIn1oY?e=nnp26c) due to storage constraints.

## Table of Contents

- [Highlights](#highlights)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Demo Application](#demo-application)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Prompt Engineering](#prompt-engineering)
- [Project Artifacts](#project-artifacts)
- [Team](#team)
- [Citation](#citation)

---

## Highlights

| | |
|---|---|
|  **Two-stage QLoRA fine-tuning** | Stage 1 trains multi-label classification on 7,540 oversampled examples; Stage 2 continues with 1,221 CoT-rationale tuples for implicit hate pattern learning. |
|  **1,221 human-verified rationale tuples** | Generated via Gemini 2.0 Flash with three prompt iterations (quality: 40 → 67 → 89/100) and automatic consistency filtering (16% acceptance rate). |
|  **Boundary redistribution finding** | Rationale training consistently shifts predictions toward Offensive-level labels (+8.35 avg F1) while Hate-level declines (−2.64 avg F1), verified across 3 seeds with ablation controls. |
|  **Rationale-to-span highlighting** | Unicode-aware span mapping projects model evidence back to original text for interpretable moderation. |
|  **Real-time demo** | FastAPI backend + React frontend with YouTube comment stream integration. |

---

## Methodology

### Overview

We adapt the **HARE framework** (Yang et al., EMNLP 2023) to the Vietnamese context. The pipeline combines chain-of-thought (CoT) prompting with multi-label hate speech detection across the [ViTHSD](https://arxiv.org/abs/2404.19252) dataset (10,001 comments, 5 target domains).

### Two-Stage Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: Classification                       │
│  Qwen2.5-3B-Instruct + QLoRA (4-bit)                               │
│  Input: Vietnamese social media comment                             │
│  Output: Multi-label prediction (11 labels)                         │
│  Training data: 7,540 oversampled examples                          │
│  Optimizer: AdamW, lr=2×10⁻⁵, batch=16                             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: Rationale-Augmented                       │
│  Continue QLoRA from Stage 1 checkpoint                              │
│  Input: Comment + implied statement                                  │
│  Output: Label + 4-step chain-of-thought rationale                   │
│  Training data: 1,221 filtered CoT tuples                            │
│  Filtering: Double-check mechanism (only retain when prediction      │
│             matches gold label → 16% acceptance rate)                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Label Schema

The model classifies into **11 multi-labels** across **5 target groups**, each with a 3-level severity scale:

| Target Group | Clean | Offensive | Hate |
|---|:---:|:---:|:---:|
| Individuals | ✓ | `individuals#offensive` | `individuals#hate` |
| Groups | ✓ | `groups#offensive` | `groups#hate` |
| Religion/Creed | ✓ | `religion#offensive` | `religion#hate` |
| Race/Ethnicity | ✓ | `race#offensive` | `race#hate` |
| Politics | ✓ | `politics#offensive` | `politics#hate` |

> Each target can be labeled at only **one** severity level per comment (Offensive **OR** Hate, not both).

### Rationale Filtering Mechanism

```
         ┌──────────┐       ┌──────────────────┐
Comment → │  Model   │ ───── │ Predicted label C│
         └──────────┘       └────────┬─────────┘
                                     │
                            ┌────────▼─────────┐
                            │ C == Gold Label?  │
                            └────────┬─────────┘
                          Yes ┘      └ No
                    ┌─────────┐    ┌───────────┐
                    │Keep C + R│    │Keep C only│
                    │(rationale)│   │(discard R)│
                    └──────────┘   └───────────┘
```

---

## Dataset

### Source

**ViTHSD** (Vo et al., 2024) — 10,001 Vietnamese social media comments with multi-target, multi-level hate speech annotations.

- **Split:** 80% train / 10% dev / 10% test (stratified sampling)
- **Annotation:** 5 target domains × 3 levels (Clean / Offensive / Hate)
- **Inter-annotator agreement:** κ = 0.45

### Rationale Augmentation

We enrich ViTHSD with two additional elements per hate/offensive comment:

| Field | Description | Example |
|---|---|---|
| `implied_statement` | A short noun phrase (3–8 words) expressing the underlying stereotype | "admin ngu dốt" (admin is stupid) |
| `rationale` | A 4-step CoT explanation grounding the label decision | See below |

**Example entry from `dataset/processed/dataset_rationale.json`:**

```json
{
  "id": 2535,
  "content": "ăn bom trên nhà thằng adm trốc tôm thôi",
  "dataset": "train",
  "labels": ["individuals#hate"],
  "implied_statement": "admin ngu dốt",
  "rationale": [
    "Cụm từ 'thằng adm' trực tiếp chỉ định cá nhân với tông khinh miệt",
    "Cách dùng 'trốc tôm' nhằm hạ thấp trí tuệ người quản trị",
    "Cách diễn đạt ngắn gọn, cắt nghĩa trực tiếp thể hiện ý định xúc phạm",
    "Ngôn từ tổng thể phản ánh định kiến cá nhân ở mức độ căm ghét"
  ]
}
```

### Data Generation Pipeline

```
ViTHSD comments ──→ Gemini 2.0 Flash ──→ Implied statements (human-verified)
                                              │
                                              ▼
                           GPT o3-mini ──→ 4-step rationales
                                              │
                                              ▼
                           Consistency filter ──→ 1,221 accepted tuples (16% rate)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for model inference)
- Node.js 18+ (for the frontend demo)

### Installation

```bash
git clone https://github.com/paht2005/Rationale-Augmented-LLM-Training-for-Targeted-Hate-Speech-Detection-in-Vietnamese-Social-Media.git
cd Rationale-Augmented-LLM-Training-for-Targeted-Hate-Speech-Detection-in-Vietnamese-Social-Media

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Run Experiment Notebooks

```bash
cd research/notebooks
jupyter notebook
```

> [!NOTE]
> Notebooks were originally authored for Kaggle/Google Colab paths.
> When running locally, update file paths to match the repository layout (e.g., `../../dataset/processed/dataset_rationale.json`).

### Run the Demo (requires model weights)

```bash
# Backend
cd demo/backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (in a new terminal)
cd demo/frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

> [!CAUTION]
> End-to-end inference requires model assets (LoRA adapters, tokenizer, hate keywords).
> Download from [OneDrive](https://nklod-my.sharepoint.com/:f:/g/personal/phatxinhchao_nklod_onmicrosoft_com/IgAYGfiHj2ZsTpr2aebNbSfrAVG0YJ0LkziTmToc1uIn1oY?e=nnp26c) and place in `demo/backend/checkpoints/`.

---

## Demo Application

<p align="center">
  <img src="results/figures/live-demo.gif" alt="HARE live demo" width="900" />
</p>

### Architecture

```
┌──────────────────┐     HTTP      ┌────────────────────────────────────────┐
│   React Frontend │◄─────────────►│          FastAPI Backend               │
│   (Vite + React  │               │                                        │
│    19, port 5173)│               │  ┌──────────┐  ┌────────────────────┐  │
└──────────────────┘               │  │ YouTube  │  │ Qwen2.5-3B-Instruct│  │
                                   │  │ API      │  │ + QLoRA (4-bit)    │  │
                                   │  └──────────┘  └────────────────────┘  │
                                   │  ┌──────────────────────────────────┐  │
                                   │  │ Lexicon-based Highlight Engine   │  │
                                   │  │ (hate_keywords.json, 300+ terms) │  │
                                   │  └──────────────────────────────────┘  │
                                   └────────────────────────────────────────┘
```

### Features

| Tab | Description |
|---|---|
| **Manual Input** | Paste a Vietnamese comment → get classification + highlighted spans |
| **YouTube** | Enter a YouTube video URL → stream & analyze all comments in real-time |
| **Group Info** | View team and project information |

### API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Health check with keywords count |
| `POST` | `/v1/analyze` | Single comment analysis with label, score, and highlight spans |
| `POST` | `/v1/analyze/batch` | Batch inference (1–200 texts) |
| `GET` | `/v1/youtube/comments` | Fetch and analyze YouTube video comments |

### Request/Response Example

**Request:**
```json
POST /v1/analyze
{
  "text": "Bọn này mà làm lãnh đạo thì đất nước chỉ có tụt hậu thôi!",
  "lang": "vi",
  "options": {
    "return_spans": true,
    "threshold": 0.5,
    "highlight_policy": "lexicon"
  }
}
```

**Response:**
```json
{
  "request_id": "a1b2c3d4-...",
  "label": "HATE",
  "score": 0.87,
  "labels": [
    {"name": "groups#hate", "score": 0.87},
    {"name": "politics#offensive", "score": 0.42}
  ],
  "highlights": [
    {"start": 0, "end": 7, "text": "Bọn này", "type": "negative", "source": "lexicon", "confidence": 1.0}
  ],
  "metadata": {"latency_ms": 234}
}
```

### Docker Deployment

```bash
cd demo/backend
docker build -t hare-backend .
docker run -p 8000:8000 hare-backend
```

---

## Repository Structure

```text
.
├── .gitignore                          # Git ignore rules (see ignored dirs below)
├── LICENSE                             # MIT License
├── README.md                           # This file
├── Thumbnail.png                       # Project thumbnail image
├── requirements.txt                    # Python dependencies (research environment)
│
├── app-preview/                        # Lightweight code preview (no weights required)
│   ├── backend-logic/                  #   FastAPI routes, model wrapper, highlighting, YouTube API
│   │   ├── __init__.py
│   │   ├── config.py                   #     App settings & environment variables
│   │   ├── highlight.py                #     Lexicon-based span highlighting engine
│   │   ├── main.py                     #     FastAPI app & route definitions
│   │   ├── model.py                    #     Qwen2.5 model loading & inference wrapper
│   │   ├── schemas.py                  #     Pydantic request/response models
│   │   └── youtube.py                  #     YouTube Data API v3 comment fetcher
│   ├── frontend-snippet/               #   HTML entry template
│   │   └── index.html
│   └── sample-outputs/                 #   Sample inference JSONs
│       ├── __huggingface_repos__.json
│       └── results_datasetA_qwen_stage2.json
│
├── dataset/
│   ├── raw/                            #   Original ViTHSD source files (.xlsx)
│   └── processed/
│       └── dataset_rationale.json      #   1,221 rationale-augmented training samples
│
├── demo/                               # ⚠️  IGNORED by .gitignore (distributed via OneDrive)
│   ├── backend/                        #   Full FastAPI server (Dockerfile included)
│   │   ├── Dockerfile                  #     Container deployment config
│   │   ├── requirements.txt            #     Backend-specific dependencies (PyTorch+CUDA)
│   │   ├── app/                        #     Application code
│   │   │   ├── __init__.py
│   │   │   ├── config.py               #       Pydantic settings (model paths, CORS, API keys)
│   │   │   ├── highlight.py            #       Unicode-aware keyword span matching
│   │   │   ├── main.py                 #       FastAPI routes (/analyze, /batch, /youtube)
│   │   │   ├── model.py                #       Qwen2.5-3B + QLoRA lazy-loading & inference
│   │   │   ├── schemas.py              #       Request/Response Pydantic schemas
│   │   │   └── youtube.py              #       Async YouTube comment fetcher (httpx)
│   │   ├── checkpoints/                #     LoRA adapters & tokenizer (download separately)
│   │   │   ├── offload/
│   │   │   ├── qwen_datasetA_stage2_lora_adapters/
│   │   │   └── qwen_datasetA_stage2_tokenizer/
│   │   └── data/
│   │       └── hate_keywords.json      #     300+ Vietnamese hate speech keywords
│   └── frontend/                       #   React 19 + Vite 7 frontend
│       ├── package.json
│       ├── vite.config.js
│       ├── index.html
│       └── src/
│           ├── App.jsx                 #     Router: /, /youtube, /group
│           ├── App.css
│           ├── main.jsx
│           └── components/
│               ├── Sidebar.jsx         #       Navigation sidebar
│               ├── InputComment.jsx    #       Manual text input & analysis
│               ├── YoutubeComments.jsx #       YouTube comment stream analyzer
│               └── GroupInfo.jsx       #       Team & project info page
│
├── research/
│   ├── notebooks/                      #   Baseline & fine-tuning notebooks (Kaggle/Colab)
│   │   ├── base-flant5.ipynb           #     Flan-T5-base baseline
│   │   ├── base-phobert.ipynb          #     PhoBERT-base baseline
│   │   ├── base-qwen.ipynb             #     Qwen2.5-3B Stage 1 (vanilla)
│   │   ├── qwen_rationale.ipynb        #     Qwen2.5-3B Stage 2 (rationale-augmented)
│   │   ├── test_prompts.ipynb          #     Prompt testing & evaluation
│   │   └── experiments/                #   Paper revision experiments (MAPR2026)
│   │       ├── R1_multiseed.ipynb      #     Multi-seed orchestrator
│   │       ├── R1_seed42.ipynb         #     Seed 42 evaluation
│   │       ├── R1_seed123.ipynb        #     Seed 123 evaluation
│   │       ├── R1_seed456.ipynb        #     Seed 456 evaluation
│   │       ├── R2_ablations.ipynb      #     Shuffled-rationale & plain-continuation
│   │       ├── R3_R6_R8_analysis.ipynb #     Bootstrap tests, parse failures, quality
│   │       ├── R4_constrained_decoding.ipynb        #  Constrained decoding main
│   │       ├── R4_seed42_constrained_decoding.ipynb  # Constrained decoding seed 42
│   │       ├── R4_seed123_constrained_decoding.ipynb # Constrained decoding seed 123
│   │       ├── R4_seed456_constrained_decoding.ipynb # Constrained decoding seed 456
│   │       ├── R5_multiseed_ablations.ipynb          # Multi-seed ablation orchestrator
│   │       ├── R5_seed42_ablations.ipynb             # Ablation seed 42
│   │       ├── R5_seed123_ablations.ipynb            # Ablation seed 123
│   │       ├── R5_seed456_ablations.ipynb            # Ablation seed 456
│   │       └── outputs/                #     Experiment result JSONs & checkpoints
│   ├── prompts/                        #   Prompt engineering iterations
│   │   ├── v1_initial/                 #     ⚠️ IGNORED (only v4_final tracked)
│   │   ├── v2_refined/                 #     ⚠️ IGNORED
│   │   ├── v3_pre-final/               #     ⚠️ IGNORED
│   │   └── v4_final/                   #     Final templates (tracked)
│   │       ├── prompt_implied_statement.txt  # Implied statement extraction prompt
│   │       └── prompt_rationale.txt          # 4-step CoT rationale generation prompt
│   └── src/                            #   Reusable Python modules
│       ├── config.py                   #     Paths, label schema, model hyperparams
│       ├── data_preparation.py         #     Data loading, preprocessing, oversampling
│       ├── models.py                   #     Model wrappers (PhoBERT, FlanT5, Qwen QLoRA)
│       └── evaluation.py              #     Metrics: F1, precision, recall, hamming loss
│
├── models/                             # ⚠️  IGNORED by .gitignore (large model weights)
│   └── visobert_datasetA_rationale_attn_state_dict.pt
│
├── results/
│   ├── figures/
│   │   └── live-demo.gif              #   Animated demo of the web UI
│   └── videos/                         # ⚠️  *.mp4 IGNORED (demo_video.mp4 on OneDrive)
│
├── docs/
│   ├── IE403.Q11-Nhom2_slide.pdf       #   Course presentation slides (tracked)
│   └── Rationale_Augmented_LLM_MAPR2026.pdf  # Published paper PDF
│   # Other docs/* files are IGNORED by .gitignore
│
└── MAPR2026/                           # ⚠️  IGNORED (LaTeX sources, only .gitkeep tracked)
    └── Rationale_Augmented_LLM_MAPR2026.pdf  # Paper PDF (whitelisted in .gitignore)
```

### Ignored Directories & Files (`.gitignore`)

| Path | Reason |
|---|---|
| `demo/` | Full demo app with model weights — distributed via OneDrive |
| `output/` | Training output artifacts (large) |
| `models/` | Model weight files (`*.bin`, `*.safetensors`, `*.pt`) |
| `MAPR2026/*` | LaTeX source files (only `Rationale_Augmented_LLM_MAPR2026.pdf` tracked) |
| `docs/*` | Only `IE403.Q11-Nhom2_slide.pdf` and `Rationale_Augmented_LLM_MAPR2026.pdf` tracked |
| `research/prompts/v1-v3` | Superseded prompt iterations (only `v4_final/` tracked) |
| `results/videos/*.mp4` | Large video files |
| `.venv/`, `venv/`, `__pycache__/` | Python environment & cache |
| `node_modules/` | Frontend dependencies |
| `.ipynb_checkpoints/` | Jupyter auto-saves |
| `*.bin`, `*.safetensors` | ML model weight files |

---

## Results

### Multi-Seed Evaluation

Evaluated on ViTHSD test set (1,800 samples) with seeds 42, 123, 456. HARE and Vanilla report mean ± std; baselines are single-run.

| Model | Precision | Recall | F1-Micro | F1-Macro |
|---|---:|---:|---:|---:|
| BiGRU-LSTM-CNN | 0.4198 | **0.6798** | 0.5191 | 0.3190 |
| PhoBERT-base | 0.5620 | 0.5310 | 0.5412 | 0.2586 |
| Flan-T5-base | 0.4810 | 0.4520 | 0.4684 | 0.1311 |
| Qwen2.5 Vanilla (Stage 1) | **0.6102 ± 0.017** | 0.5843 ± 0.022 | **0.5967 ± 0.013** | 0.3295 ± 0.016 |
| **HARE (proposed)** | 0.5091 ± 0.027 | 0.6111 ± 0.011 | 0.5551 ± 0.016 | **0.3570 ± 0.044** |

> [!NOTE]
> HARE trades F1-Micro for higher Recall and F1-Macro. The primary finding is a per-label **boundary redistribution**: +8.35 avg F1 on Offensive-level labels, −2.64 on Hate-level — consistent across all three seeds.

### Key Findings

1. **Boundary Redistribution Effect** — Rationale training shifts the decision boundary: Offensive-level labels gain +6.84 F1 while Hate-level gains +2.77 F1 (paper revision numbers with constrained decoding).
2. **Rationale Quality Impact** — Only 16% of generated rationales pass the double-check filter, but those high-quality tuples are sufficient to induce meaningful behavioral change.
3. **Parse-Failure Analysis** — Identified that parse failures in CoT output correlate with model uncertainty at label boundaries, serving as a confidence calibration signal.

### Ablation Studies

| Condition | F1-Macro | Δ vs. HARE |
|---|---:|---:|
| HARE (full) | 0.3570 | — |
| Shuffled rationales | 0.3312 | −0.0258 |
| Plain continuation (no CoT) | 0.3295 | −0.0275 |
| Stage 1 only (Vanilla) | 0.3295 | −0.0275 |

---

## Prompt Engineering

The rationale generation pipeline evolved through **4 prompt versions**, evaluated by 4 annotators on a 0–100 scale:

| Version | Avg Score | Key Changes |
|---|---:|---|
| v1 (initial) | 40/100 | Minimal instructions, basic JSON output |
| v2 (refined) | 67/100 | Added label definitions, target types, format constraints |
| v3 (pre-final) | 78/100 | Strict 4-step requirement, implied statement grounding |
| **v4 (final)** | **89/100** | Error handling, teencode/no-diacritics support, full format safeguards |

### Final Prompt Design (`research/prompts/v4_final/`)

Two prompts are used in sequence:

1. **`prompt_implied_statement.txt`** — Extracts a 3–8 word noun phrase capturing the underlying stereotype (handles teencode, emoji, mixed Vietnamese spelling)
2. **`prompt_rationale.txt`** — Generates a 4-point CoT rationale:
   - Point 1: Identify specific hateful words/phrases
   - Point 2: Explain how language conveys prejudice/dehumanization
   - Point 3: Connect to the implied statement
   - Point 4: Justify severity level (HATE vs. OFFENSIVE)

---

## Experiments

### Baseline Models

| Model | Architecture | Training |
|---|---|---|
| BiGRU-LSTM-CNN | Bi-GRU + LSTM + CNN ensemble | ViTHSD original baseline |
| PhoBERT-base | Vietnamese pre-trained BERT | Fine-tuned, max_len=256 |
| Flan-T5-base | Encoder-decoder, instruction-tuned | CoT generation, max_output=64 |
| ViSoBERT | Vietnamese social BERT | + attention layer |
| **Qwen2.5-3B** | Decoder-only LLM | **QLoRA 4-bit, two-stage** |

### Experiment Matrix (Paper Revision)

| ID | Experiment | Seeds | Description |
|---|---|---|---|
| R1 | Multi-seed | 42, 123, 456 | Reproducibility across seeds |
| R2 | Ablations | 42 | Shuffled-rationale, plain-continuation |
| R3 | Statistical tests | — | Bootstrap significance testing |
| R4 | Constrained decoding | 42, 123, 456 | Force valid label format in output |
| R5 | Per-seed ablations | 42, 123, 456 | Full ablation per seed |

### Training Configuration

```python
# Stage 1 (Classification)
model = "Qwen/Qwen2.5-3B-Instruct"
quantization = "4-bit (bitsandbytes)"
lora_r = 8
lora_alpha = 16
learning_rate = 2e-5
batch_size = 16
max_epochs = 10
early_stopping = "validation F1"

# Stage 2 (Rationale-Augmented)
# Continues from Stage 1 checkpoint
training_data = 1221  # filtered rationale tuples
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Qwen2.5-3B-Instruct (QLoRA 4-bit via PEFT + bitsandbytes) |
| Training | PyTorch 2.0+, Transformers, TRL, Accelerate |
| Rationale Gen | Gemini 2.0 Flash (implied statements), GPT o3-mini (rationales) |
| Backend | FastAPI 0.115, Uvicorn, Pydantic v2 |
| Frontend | React 19, Vite 7, React Router 7 |
| Deployment | Docker (Python 3.11-slim base) |
| Data | Pandas, scikit-learn, HuggingFace Datasets |
| Notebooks | Kaggle/Google Colab (Tesla P100 / T4 GPU) |

---

## Project Artifacts

| Artifact | Location |
|---|---|
| **Paper (PDF)** | `docs/Rationale_Augmented_LLM_MAPR2026.pdf` |
| Live demo GIF | `results/figures/live-demo.gif` |
| Demo video | `results/videos/demo_video.mp4` |
| Sample outputs | `app-preview/sample-outputs/results_datasetA_qwen_stage2.json` |
| Experiment results | `research/notebooks/experiments/outputs/` |
| Course slides | `docs/IE403.Q11-Nhom2_slide.pdf` |
| Paper LaTeX source | `MAPR2026/` (ignored, available on request) |
| Full demo package | [OneDrive](https://nklod-my.sharepoint.com/:f:/g/personal/phatxinhchao_nklod_onmicrosoft_com/IgAYGfiHj2ZsTpr2aebNbSfrAVG0YJ0LkziTmToc1uIn1oY?e=nnp26c) |

---

## Team

| No. | Student ID | Full Name | Role | GitHub |
|---:|:---:|---|---|---|
| 1 | 23521143 | Phat Nguyen Cong | Leader | [paht2005](https://github.com/paht2005) |
| 2 | 23520032 | An Truong Hoang Thanh | Member | [Awnpz](https://github.com/Awnpz) |
| 3 | 23520023 | An Nguyen Xuan | Member | [annx-uit](https://github.com/annx-uit) |
| 4 | 23520158 | Binh Mai Thai | Member | [maibinhkznk209](https://github.com/maibinhkznk209/) |
| 5 | 21520255 | Huong Nguyen Le Quynh | Member | [tracycute](https://github.com/tracycute) |

**Advisor:** Tin Van Huynh — University of Information Technology, VNU-HCM

---

## Citation

If this repository supports your research, please cite:

```bibtex
@misc{hare-vietnamese-2026,
  title   = {Rationale-Augmented LLM Training for Targeted Hate Speech Detection in Vietnamese Social Media},
  author  = {Truong, An Hoang-Thanh and Nguyen, Phat Cong and Nguyen, An Xuan and Mai, Binh Thai and Nguyen, Huong Le-Quynh and Huynh, Tin Van},
  year    = {2026},
  url     = {https://github.com/paht2005/Rationale-Augmented-LLM-Training-for-Targeted-Hate-Speech-Detection-in-Vietnamese-Social-Media}
}
```

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

<p align="center">
  <sub>Made with passion at University of Information Technology, VNU-HCM</sub>
</p>
