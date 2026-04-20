"""
Configuration Module
Configuration for Multi-Label Classification Pipeline
"""

from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path("/kaggle/working")
DATA_DIR = Path("/kaggle/input/vithsd-experiment/data")
OUTPUT_DIR = Path("/kaggle/working/outputs")

# ==================== TARGET COLUMNS ====================
# Target label columns in the CSV file
TARGET_COLUMNS = [
    'individual',
    'groups', 
    'religion/creed',
    'race/ethnicity',
    'politics'
]

TARGET_NAMES = [
    'individuals',
    'groups',
    'religion',
    'race',
    'politics'
]

# Text column
TEXT_COLUMN = 'content'

# ==================== MULTI-LABEL CONFIG ====================
# 11 labels for multi-label classification
# Logic: normal if all levels <=1, otherwise specify target#level
FINAL_LABELS = [
    "normal",
    "individuals#offensive",
    "individuals#hate",
    "groups#offensive", 
    "groups#hate",
    "religion#offensive",
    "religion#hate",
    "race#offensive",
    "race#hate",
    "politics#offensive",
    "politics#hate"
]

# ==================== CATEGORY CONSTRAINT ====================
# IMPORTANT: Each category can only have ONE level (offensive OR hate, NOT both)
# This is enforced in both training and inference
CATEGORIES = ['individuals', 'groups', 'religion', 'race', 'politics']

# Mapping: category -> (offensive_label, hate_label)
CATEGORY_LABELS = {
    'individuals': ('individuals#offensive', 'individuals#hate'),
    'groups': ('groups#offensive', 'groups#hate'),
    'religion': ('religion#offensive', 'religion#hate'),
    'race': ('race#offensive', 'race#hate'),
    'politics': ('politics#offensive', 'politics#hate'),
}

# Mapping: category -> (offensive_idx, hate_idx) in FINAL_LABELS
CATEGORY_INDICES = {
    'individuals': (1, 2),
    'groups': (3, 4),
    'religion': (5, 6),
    'race': (7, 8),
    'politics': (9, 10),
}

NUM_LABELS = len(FINAL_LABELS)  # 11

# ==================== PROMPTS ====================
# Prompt template for multi-label classification
CLASSIFICATION_PROMPT = """Classify hate speech levels in the following Vietnamese text.

Text: {text}

Possible labels (select one or more):
- normal: Normal text, no hate speech
- individuals#offensive: Offensive towards individuals
- individuals#hate: Hate speech towards individuals
- groups#offensive: Offensive towards groups
- groups#hate: Hate speech towards groups
- religion#offensive: Offensive towards religion
- religion#hate: Hate speech towards religion
- race#offensive: Offensive towards race/ethnicity
- race#hate: Hate speech towards race/ethnicity
- politics#offensive: Offensive towards politics
- politics#hate: Hate speech towards politics

Return list of applicable labels, separated by comma:"""

# Simple prompt for FlanT5
SIMPLE_PROMPT = "Classify multi-label hate speech: {text}"

# ==================== MODEL CONFIGS ====================
MODEL_CONFIGS = {
    "logistic": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "C": 1.0
    },
    "phobert": {
        "model_name": "vinai/phobert-base",
        "batch_size": 16,
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "max_length": 256
    },
    "flant5": {
        "model_name": "google/flan-t5-base",
        "batch_size": 8,
        "num_epochs": 3,
        "learning_rate": 3e-5,
        "max_input_length": 256,
        "max_output_length": 64
    },
    "phogpt": {
        "model_name": "vinai/PhoGPT-4B-Chat",
        "batch_size": 4,
        "num_epochs": 2,
        "learning_rate": 2e-4,
        "max_length": 512,
        "lora_r": 8,
        "lora_alpha": 16
    }
}


if __name__ == "__main__":
    print("Multi-Label Classification Config")
    print("=" * 50)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"\nLabels ({NUM_LABELS}):")
    for i, label in enumerate(FINAL_LABELS):
        print(f"  {i}: {label}")
