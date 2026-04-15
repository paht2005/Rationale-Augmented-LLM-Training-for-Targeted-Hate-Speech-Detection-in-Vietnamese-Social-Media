"""
Data Preparation Module
Read and prepare ViHSD data for Multi-Label Classification
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import re

from config import DATA_DIR, FINAL_LABELS, TARGET_COLUMNS, TARGET_NAMES, TEXT_COLUMN


@dataclass
class DataSample:
    """A single data sample"""
    text: str
    original_labels: Dict[str, int] = field(default_factory=dict)  # {target: level}
    multi_labels: List[str] = field(default_factory=list)  # ["normal"] or ["individuals#hate", "groups#offensive", ...]


class TextPreprocessor:
    """Basic text preprocessing"""
    
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess(self, text: str) -> str:
        """Preprocessing pipeline"""
        text = self.clean_text(text)
        return text


def get_multi_labels(row: pd.Series) -> List[str]:
    """
    Convert original labels to multi-label format
    
    Logic:
    - If all targets have level 0 or 1 -> ["normal"]
    - If any target has level >= 2 -> specify target#level_name
    
    Level mapping:
    - 0: Clean
    - 1: Clean (with slight indicators)
    - 2: Offensive
    - 3: Hate
    """
    labels = []
    level_names = {2: "offensive", 3: "hate"}
    
    for col, target_name in zip(TARGET_COLUMNS, TARGET_NAMES):
        if col in row:
            level = row[col]
            if level >= 2 and level in level_names:
                label = f"{target_name}#{level_names[level]}"
                labels.append(label)
    
    if not labels:
        labels = ["normal"]
    
    return labels


class ViHSDLoader:
    """Loader for ViHSD dataset (Multi-Label)"""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.preprocessor = TextPreprocessor()
        self._cache: Dict[str, List[DataSample]] = {}
    
    def load_split(self, split: str = 'train') -> List[DataSample]:
        """Load a split of the dataset
        
        Args:
            split: 'train', 'dev', 'test'
        
        Returns:
            List of DataSample
        """
        if split in self._cache:
            return self._cache[split]
        
        # Try to read file (could be actual .xlsx or CSV renamed to .xlsx)
        file_path_xlsx = self.data_dir / f'{split}.xlsx'
        file_path_csv = self.data_dir / f'{split}.csv'
        
        if file_path_xlsx.exists():
            # Try to read as Excel first
            try:
                df = pd.read_excel(file_path_xlsx, engine='openpyxl')
            except Exception as e:
                # If fail, it might be CSV with .xlsx extension
                try:
                    df = pd.read_csv(file_path_xlsx)
                except Exception:
                    raise ValueError(f"Cannot read {file_path_xlsx} as Excel or CSV: {e}")
        elif file_path_csv.exists():
            df = pd.read_csv(file_path_csv)
        else:
            raise FileNotFoundError(f"File not found: {split}.xlsx or {split}.csv in {self.data_dir}")
        
        samples = []
        for idx, row in df.iterrows():
            # Get text
            text = str(row.get(TEXT_COLUMN, ''))
            text = self.preprocessor.preprocess(text)
            
            if not text:
                continue
            
            # Get original labels
            original_labels = {}
            for col, target_name in zip(TARGET_COLUMNS, TARGET_NAMES):
                if col in row:
                    original_labels[target_name] = int(row[col])
            
            # Get multi-labels
            multi_labels = get_multi_labels(row)
            
            sample = DataSample(
                text=text,
                original_labels=original_labels,
                multi_labels=multi_labels
            )
            samples.append(sample)
        
        self._cache[split] = samples
        print(f"[ViHSDLoader] Loaded {len(samples)} samples from {split}")
        
        return samples
    
    def prepare_dataset_multilabel(self, split: str = 'train') -> Tuple[List[str], List[List[str]]]:
        """
        Prepare dataset for multi-label classification
        
        Returns:
            Tuple[List[str], List[List[str]]]: (texts, list_of_label_lists)
            Example: (["text1", "text2"], [["normal"], ["individuals#hate", "groups#offensive"]])
        """
        samples = self.load_split(split)
        texts = [s.text for s in samples]
        labels = [s.multi_labels for s in samples]
        return texts, labels
    
    def prepare_dataset_multilabel_binary(self, split: str = 'train') -> Tuple[List[str], List[List[int]]]:
        """
        Prepare dataset for multi-label with binary encoding
        
        Returns:
            Tuple[List[str], List[List[int]]]: (texts, binary_label_matrix)
            Each row is a binary vector for 11 labels
        """
        texts, labels_list = self.prepare_dataset_multilabel(split)
        
        # Convert to binary
        binary_labels = []
        for labels in labels_list:
            row = [0] * len(FINAL_LABELS)
            for label in labels:
                if label in FINAL_LABELS:
                    idx = FINAL_LABELS.index(label)
                    row[idx] = 1
            binary_labels.append(row)
        
        return texts, binary_labels
    
    def prepare_dataset_generative(
        self, 
        split: str = 'train',
        prompt_template: str = "Classify multi-label hate speech: {text}"
    ) -> Tuple[List[str], List[str]]:
        """
        Prepare dataset for generative models (FlanT5, PhoGPT)
        
        Returns:
            Tuple[List[str], List[str]]: (prompts, target_outputs)
            target_outputs are comma-separated labels
        """
        texts, labels_list = self.prepare_dataset_multilabel(split)
        
        prompts = [prompt_template.format(text=text) for text in texts]
        targets = [", ".join(sorted(labels)) for labels in labels_list]
        
        return prompts, targets
    
    def get_label_distribution(self, split: str = 'train') -> Dict[str, int]:
        """Get distribution of labels"""
        _, labels_list = self.prepare_dataset_multilabel(split)
        
        distribution = {label: 0 for label in FINAL_LABELS}
        for labels in labels_list:
            for label in labels:
                if label in distribution:
                    distribution[label] += 1
        
        return distribution
    
    def print_statistics(self, split: str = 'train'):
        """Print dataset statistics"""
        samples = self.load_split(split)
        distribution = self.get_label_distribution(split)
        
        print(f"\n Statistics for {split} split:")
        print(f"   Total samples: {len(samples)}")
        print(f"\n   Label distribution:")
        
        for label in FINAL_LABELS:
            count = distribution[label]
            pct = count / len(samples) * 100
            print(f"      {label:25s}: {count:5d} ({pct:5.1f}%)")


def load_dataset_B_json(
    json_path: Optional[str] = None,
    include_rationale: bool = False
) -> Tuple:
    """
    Load Dataset B from JSON file
    Format:
    {
        "id": 0,
        "content": "text...",
        "labels": ["individuals#hate", "race#hate"],
        "rationale": ["reason1", "reason2"],
        "implied_statement": "..."
    }
    
    Args:
        json_path: Path to JSON file
        include_rationale: If True, also return rationale and implied_statement
    
    Returns:
        If include_rationale=False:
            Tuple[List[str], List[List[str]]]: (texts, labels_list)
        If include_rationale=True:
            Tuple[List[str], List[List[str]], List[List[str]], List[str]]:
            (texts, labels_list, rationale_list, implied_list)
    """
    if json_path is None:
        json_path = str(Path(__file__).parent.parent / "dataset" / "dataset.json")
    
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset B not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    labels_list = []
    rationale_list = []
    implied_list = []
    
    for item in data:
        text = item.get('content', '')
        labels = item.get('labels', ['normal'])
        rationale = item.get('rationale', [])
        implied = item.get('implied_statement', '')
        
        # Ensure labels is a list
        if not isinstance(labels, list):
            labels = ['normal']
        
        # If no labels, default to normal
        if not labels:
            labels = ['normal']
        
        # Ensure rationale is a list
        if not isinstance(rationale, list):
            rationale = [rationale] if rationale else []
        
        # Ensure implied is a string
        if not isinstance(implied, str):
            implied = str(implied) if implied else ''
        
        texts.append(text)
        labels_list.append(labels)
        rationale_list.append(rationale)
        implied_list.append(implied)
    
    print(f"[Dataset B] Loaded {len(texts)} samples from {path}")
    
    if include_rationale:
        print(f"  - With rationale: {sum(1 for r in rationale_list if r)} samples have rationale")
        print(f"  - With implied: {sum(1 for i in implied_list if i)} samples have implied_statement")
        return texts, labels_list, rationale_list, implied_list
    
    return texts, labels_list


def load_dataset_A(split: str = 'train', data_dir: Path = DATA_DIR) -> Tuple[List[str], List[List[str]]]:
    """
    Load Dataset A (original ViTHSD, only content + labels)
    
    Args:
        split: 'train', 'dev', 'test'
        data_dir: Path to data directory
    
    Returns:
        Tuple[List[str], List[List[str]]]: (texts, labels_list)
    """
    loader = ViHSDLoader(data_dir)
    return loader.prepare_dataset_multilabel(split)


def load_dataset_B(
    json_path: Optional[str] = None,
    include_rationale: bool = True
) -> Tuple:
    """
    Load Dataset B (ViTHSD + rationale + implied_statement)
    
    Args:
        json_path: Path to JSON file
        include_rationale: If True, return rationale and implied (default True)
    
    Returns:
        If include_rationale=True (default):
            Tuple[List[str], List[List[str]], List[List[str]], List[str]]:
            (texts, labels_list, rationale_list, implied_list)
        If include_rationale=False:
            Tuple[List[str], List[List[str]]]: (texts, labels_list)
    """
    return load_dataset_B_json(json_path, include_rationale)


def prepare_train_test_dataset_A(
    data_dir: Path = DATA_DIR
) -> Dict:
    """
    Prepare train/dev/test for Dataset A
    
    Returns:
        Dict with keys: 'train', 'dev', 'test'
        Each value is (texts, labels_list)
    """
    loader = ViHSDLoader(data_dir)
    
    return {
        'train': loader.prepare_dataset_multilabel('train'),
        'dev': loader.prepare_dataset_multilabel('dev'),
        'test': loader.prepare_dataset_multilabel('test'),
    }


def prepare_train_test_dataset_B(
    json_path: Optional[str] = None,
    test_ratio: float = 0.1,
    dev_ratio: float = 0.1,
    random_state: int = 42
) -> Dict:
    """
    Prepare train/dev/test for Dataset B
    
    Args:
        json_path: Path to JSON file
        test_ratio: Ratio for test set
        dev_ratio: Ratio for dev set
        random_state: Random seed
    
    Returns:
        Dict with keys: 'train', 'dev', 'test'
        Each value is (texts, labels_list, rationale_list, implied_list)
    """
    from sklearn.model_selection import train_test_split
    
    texts, labels, rationale, implied = load_dataset_B(json_path, include_rationale=True)
    
    # First split: train+dev vs test
    train_dev_texts, test_texts, train_dev_labels, test_labels, \
    train_dev_rationale, test_rationale, train_dev_implied, test_implied = train_test_split(
        texts, labels, rationale, implied,
        test_size=test_ratio,
        random_state=random_state
    )
    
    # Second split: train vs dev
    dev_size = dev_ratio / (1 - test_ratio)
    train_texts, dev_texts, train_labels, dev_labels, \
    train_rationale, dev_rationale, train_implied, dev_implied = train_test_split(
        train_dev_texts, train_dev_labels, train_dev_rationale, train_dev_implied,
        test_size=dev_size,
        random_state=random_state
    )
    
    print(f"[Dataset B Split] Train: {len(train_texts)}, Dev: {len(dev_texts)}, Test: {len(test_texts)}")
    
    return {
        'train': (train_texts, train_labels, train_rationale, train_implied),
        'dev': (dev_texts, dev_labels, dev_rationale, dev_implied),
        'test': (test_texts, test_labels, test_rationale, test_implied),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Data Preparation Module")
    print("=" * 70)
    
    # Test Dataset A
    print("\n Test Dataset A (original ViTHSD):")
    try:
        loader = ViHSDLoader()
        texts, labels = loader.prepare_dataset_multilabel('train')
        print(f"   Loaded {len(texts)} train samples")
        
        # Show example
        for i in range(min(3, len(texts))):
            if labels[i] != ["normal"]:
                print(f"   Example: '{texts[i][:50]}...' -> {labels[i]}")
                break
    except FileNotFoundError as e:
        print(f"   Dataset A not found: {e}")
    
    # Test Dataset B
    print("\n Test Dataset B (with rationale + implied):")
    try:
        texts, labels, rationale, implied = load_dataset_B(include_rationale=True)
        print(f"   Loaded {len(texts)} samples")
        
        # Show example with rationale
        for i in range(min(5, len(texts))):
            if rationale[i] or implied[i]:
                print(f"\n   Example {i+1}:")
                print(f"     Text: '{texts[i][:50]}...'")
                print(f"     Labels: {labels[i]}")
                print(f"     Rationale: {rationale[i][:2] if len(rationale[i]) > 2 else rationale[i]}")
                print(f"     Implied: '{implied[i][:50]}...' " if implied[i] else "     Implied: None")
                break
    except FileNotFoundError as e:
        print(f"   Dataset B not found: {e}")
    
    print("\n Data preparation module ready!")
