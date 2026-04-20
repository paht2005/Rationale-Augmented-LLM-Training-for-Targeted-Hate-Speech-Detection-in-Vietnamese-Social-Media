"""
Models Module - ViTHSD Multi-Label Classification

Supports 2 Datasets:
- Dataset A: Original ViTHSD (only content + labels)
- Dataset B: ViTHSD + Rationale + Implied Statement

3 Models:
1. PhoBERT: Knowledge Distillation (Teacher has rationale, Student does not)
2. FlanT5: Chain-of-Thought (generate rationale -> implied -> labels)
3. Qwen: Chain-of-Thought with QLoRA 4-bit

Author: ViTHSD Team
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import FINAL_LABELS, NUM_LABELS, CATEGORIES, CATEGORY_INDICES, CATEGORY_LABELS


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def resolve_label_conflicts(binary: np.ndarray, prefer_hate: bool = True) -> np.ndarray:
    """
    Resolve conflicts: Each category can only have ONE level (offensive OR hate, NOT both)
    """
    binary = np.array(binary).copy()
    is_1d = binary.ndim == 1
    
    if is_1d:
        binary = binary.reshape(1, -1)
    
    for category, (off_idx, hate_idx) in CATEGORY_INDICES.items():
        both_active = (binary[:, off_idx] == 1) & (binary[:, hate_idx] == 1)
        if prefer_hate:
            binary[both_active, off_idx] = 0
        else:
            binary[both_active, hate_idx] = 0
    
    if is_1d:
        binary = binary.squeeze(0)
    
    return binary


def resolve_labels_list(labels: List[str], prefer_hate: bool = True) -> List[str]:
    """Resolve conflicts in a list of label strings."""
    resolved = list(labels)
    
    for category, (off_label, hate_label) in CATEGORY_LABELS.items():
        if off_label in resolved and hate_label in resolved:
            if prefer_hate:
                resolved.remove(off_label)
            else:
                resolved.remove(hate_label)
    
    return resolved


def convert_to_binary(y_labels: List[List[str]], labels: List[str] = FINAL_LABELS) -> np.ndarray:
    """Convert label lists to binary matrix."""
    binary = []
    for label_list in y_labels:
        row = [0] * len(labels)
        for label in label_list:
            if label in labels:
                row[labels.index(label)] = 1
        binary.append(row)
    return np.array(binary)


def convert_from_binary(binary: np.ndarray, labels: List[str] = FINAL_LABELS) -> List[List[str]]:
    """Convert binary matrix to label lists."""
    results = []
    for row in binary:
        label_list = []
        for i, val in enumerate(row):
            if val >= 0.5:
                label_list.append(labels[i])
        if not label_list:
            label_list = ["normal"]
        results.append(label_list)
    return results


def oversample_minority_labels(
    X: List[str], 
    y: List[List[str]], 
    labels: List[str], 
    min_samples: int = 100,
    rationale: List[List[str]] = None,
    implied: List[str] = None
) -> Tuple:
    """Oversample minority labels."""
    import random
    
    label_indices = {label: [] for label in labels}
    for i, label_list in enumerate(y):
        for label in label_list:
            if label in label_indices:
                label_indices[label].append(i)
    
    X_new, y_new = list(X), list(y)
    rationale_new = list(rationale) if rationale else None
    implied_new = list(implied) if implied else None
    added_count = 0
    
    for label in labels:
        indices = label_indices[label]
        count = len(indices)
        
        if count > 0 and count < min_samples:
            need = min_samples - count
            sampled_indices = random.choices(indices, k=need)
            for idx in sampled_indices:
                X_new.append(X[idx])
                y_new.append(y[idx])
                if rationale_new is not None:
                    rationale_new.append(rationale[idx])
                if implied_new is not None:
                    implied_new.append(implied[idx])
                added_count += 1
    
    print(f"  [Oversampling] Added {added_count} samples for minority labels")
    
    if rationale_new is not None and implied_new is not None:
        return X_new, y_new, rationale_new, implied_new
    return X_new, y_new


# =============================================================================
# BASE CLASS
# =============================================================================

class ModelWrapper(ABC):
    """Base class for models"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train: List[str], y_train: List[List[str]], **kwargs):
        """Train model"""
        pass
    
    @abstractmethod
    def predict(self, X_test: List[str]) -> Tuple[List[List[str]], Any]:
        """Predict"""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# PHOBERT - KNOWLEDGE DISTILLATION
# =============================================================================

class PhoBERTWrapper(ModelWrapper):
    """
    PhoBERT Multi-Label Classification with Knowledge Distillation
    
    Dataset A (no rationale): Train directly
    Dataset B (with rationale): 
        - Teacher model: Train with rationale-augmented input
        - Student model: Distill from Teacher, inference does not need rationale
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = NUM_LABELS,
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        max_length: int = 256,
        device: str = None,
        # Knowledge Distillation params
        use_knowledge_distillation: bool = False,
        kd_alpha: float = 0.5,
        kd_temperature: float = 3.0
    ):
        super().__init__("PhoBERT_MultiLabel")
        self.model_name = model_name
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.labels = FINAL_LABELS
        
        # Knowledge Distillation
        self.use_knowledge_distillation = use_knowledge_distillation
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        
        import torch
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.tokenizer = None
        self.teacher_model = None
    
    def _augment_with_rationale(self, content: str, rationale: List[str], implied: str = "") -> str:
        """Augment content with rationale and implied statement for Teacher"""
        parts = [content]
        if implied:
            parts.append(f"[IMPLIED] {implied}")
        if rationale:
            parts.append(f"[RATIONALE] {' | '.join(rationale)}")
        return " ".join(parts)
    
    def _train_single_model(
        self, 
        X_train: List[str], 
        y_train: List[List[str]],
        teacher_logits: np.ndarray = None
    ):
        """Train a single model (Teacher or Student)"""
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import get_linear_schedule_with_warmup
        from torch.optim import AdamW
        from tqdm import tqdm
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )
        self.model.to(self.device)
        
        # Convert labels
        y_binary = convert_to_binary(y_train, self.labels)
        
        # Class weights
        class_weights = []
        for i in range(len(self.labels)):
            label_counts = y_binary[:, i].sum()
            weight = len(y_binary) / (2.0 * label_counts) if label_counts > 0 else 1.0
            class_weights.append(weight)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        # Tokenize
        print(f"  Tokenizing {len(X_train)} samples...")
        
        # Cap max_length to model's max_position_embeddings
        try:
            max_pos = self.model.config.max_position_embeddings
            safe_max_length = min(self.max_length, max_pos)
        except Exception:
            safe_max_length = self.max_length
        
        encodings = self.tokenizer(
            X_train,
            truncation=True,
            padding=True,
            max_length=safe_max_length,
            return_tensors='pt'
        )
        
        # Create dataset
        if teacher_logits is not None:
            dataset = TensorDataset(
                encodings['input_ids'],
                encodings['attention_mask'],
                torch.tensor(y_binary, dtype=torch.float),
                torch.tensor(teacher_logits, dtype=torch.float)
            )
        else:
            dataset = TensorDataset(
                encodings['input_ids'],
                encodings['attention_mask'],
                torch.tensor(y_binary, dtype=torch.float)
            )
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
        
        # Training
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch in progress:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Loss calculation
                if teacher_logits is not None and len(batch) > 3:
                    # Knowledge Distillation loss
                    teacher_soft = batch[3].to(self.device)
                    
                    # Distillation loss (KL divergence)
                    teacher_probs = F.softmax(teacher_soft / self.kd_temperature, dim=-1)
                    student_log_probs = F.log_softmax(logits / self.kd_temperature, dim=-1)
                    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                    kd_loss = kd_loss * (self.kd_temperature ** 2)
                    
                    # True label loss
                    true_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=class_weights)
                    
                    # Combined loss
                    loss = self.kd_alpha * kd_loss + (1 - self.kd_alpha) * true_loss
                else:
                    # Standard loss
                    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
                    loss = loss_fn(logits, labels)
                
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                progress.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
    
    def _get_logits(self, X: List[str]) -> np.ndarray:
        """Get model logits for Knowledge Distillation"""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        encodings = self.tokenizer(
            X,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        self.model.eval()
        all_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu().numpy())
        
        return np.vstack(all_logits)
    
    def train(
        self, 
        X_train: List[str], 
        y_train: List[List[str]],
        rationale: List[List[str]] = None,
        implied: List[str] = None
    ):
        """
        Train PhoBERT model
        
        Dataset A: rationale=None, implied=None → Standard training
        Dataset B: rationale provided → Knowledge Distillation
        """
        import torch
        import gc
        
        print(f"[{self.name}] Training...")
        print(f"  Device: {self.device}")
        print(f"  Original samples: {len(X_train)}")
        
        has_rationale = rationale is not None and implied is not None
        
        if has_rationale and self.use_knowledge_distillation:
            # ===== DATASET B: Knowledge Distillation =====
            print(f"  Mode: Knowledge Distillation (Dataset B)")
            
            # Step 1: Oversample with rationale
            X_train, y_train, rationale, implied = oversample_minority_labels(
                X_train, y_train, self.labels, min_samples=200,
                rationale=rationale, implied=implied
            )
            
            # Step 2: Train Teacher with augmented input
            print(f"\n  [Step 1/3] Training Teacher model with rationale...")
            X_augmented = [
                self._augment_with_rationale(x, r, i) 
                for x, r, i in zip(X_train, rationale, implied)
            ]
            self._train_single_model(X_augmented, y_train)
            
            # Step 3: Get Teacher soft predictions
            print(f"\n  [Step 2/3] Getting Teacher soft predictions...")
            teacher_logits = self._get_logits(X_augmented)
            
            # Save teacher model reference
            self.teacher_model = self.model
            self.model = None
            
            # Step 4: Train Student with plain input + Teacher soft labels
            print(f"\n  [Step 3/3] Training Student model with KD...")
            self._train_single_model(X_train, y_train, teacher_logits=teacher_logits)
            
            # Cleanup teacher
            del self.teacher_model
            torch.cuda.empty_cache()
            gc.collect()
            
        else:
            # ===== DATASET A: Standard Training =====
            print(f"  Mode: Standard Training (Dataset A)")
            
            X_train, y_train = oversample_minority_labels(X_train, y_train, self.labels, min_samples=200)
            self._train_single_model(X_train, y_train)
        
        self.is_trained = True
        print(f"  Training completed")
    
    def _extract_hate_words_attention(
        self, 
        text: str, 
        input_ids: 'torch.Tensor',
        attention_mask: 'torch.Tensor',
        top_k: int = 5
    ) -> List[str]:
        """
        Extract hate words using attention weights.
        Uses attention scores from the last layer to identify important tokens.
        """
        import torch
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # Get attention from the last layer, last head
            # Shape: (batch, num_heads, seq_len, seq_len)
            attentions = outputs.attentions[-1]  # Last layer
            
            # Average across all heads
            # Shape: (batch, seq_len, seq_len)
            avg_attention = attentions.mean(dim=1)
            
            # Get attention from CLS token (index 0) to other tokens
            # Shape: (batch, seq_len)
            cls_attention = avg_attention[0, 0, :]
            
            # Mask padding tokens
            cls_attention = cls_attention * attention_mask[0].float()
            
            # Get top-k token indices (skip CLS and SEP tokens)
            # CLS = index 0, SEP is usually at the end
            cls_attention[0] = 0  # Skip CLS
            
            # Find SEP token and zero it
            seq_len = attention_mask[0].sum().item()
            if seq_len > 1:
                cls_attention[int(seq_len) - 1] = 0  # Skip SEP
            
            # Get top-k indices
            top_indices = torch.topk(cls_attention, min(top_k, int(seq_len) - 2)).indices.tolist()
            
            # Decode tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            
            # Extract words (handle subword tokens)
            hate_words = []
            for idx in top_indices:
                if idx < len(tokens):
                    token = tokens[idx]
                    # Skip special tokens and padding
                    if token in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]']:
                        continue
                    # Handle subword tokens (PhoBERT uses @@ or ▁)
                    word = token.replace('@@', '').replace('▁', '').strip()
                    if word and len(word) > 1:  # Skip single chars
                        hate_words.append(word)
            
            return hate_words
    
    def _find_original_words(self, text: str, highlighted_tokens: List[str]) -> List[str]:
        """
        Find original words in the source text from highlighted tokens
        """
        import re
        
        original_words = []
        text_lower = text.lower()
        
        for token in highlighted_tokens:
            token_lower = token.lower()
            
            # Find the word containing the token in the original text
            # Pattern: word boundary or token appears in text
            for word in text.split():
                word_clean = re.sub(r'[^\w]', '', word)
                if token_lower in word_clean.lower():
                    if word not in original_words:
                        original_words.append(word)
                    break
        
        return original_words
    
    def predict(self, X_test: List[str], threshold: float = 0.5, extract_hate_words: bool = False) -> Tuple[List[List[str]], np.ndarray, Optional[List[List[str]]]]:
        """
        Predict (always uses Student model, no rationale needed)
        
        Args:
            X_test: List of texts to classify
            threshold: Classification threshold
            extract_hate_words: If True, also extract hate words using attention (for inference only)
        
        Returns:
            y_labels: Predicted labels
            y_binary: Binary predictions
            hate_words: List of hate words per sample (only if extract_hate_words=True)
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm
        
        if not self.is_trained:
            raise ValueError("Model has not been trained")
        
        # Cap max_length
        try:
            max_pos = self.model.config.max_position_embeddings
            safe_max_length = min(self.max_length, max_pos)
        except Exception:
            safe_max_length = self.max_length
        
        encodings = self.tokenizer(
            X_test,
            truncation=True,
            padding=True,
            max_length=safe_max_length,
            return_tensors='pt'
        )
        
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits)
                all_probs.append(probs.cpu().numpy())
        
        all_probs = np.vstack(all_probs)
        
        # Apply threshold and resolve conflicts
        y_binary = np.zeros_like(all_probs, dtype=int)
        
        for i in range(len(all_probs)):
            probs = all_probs[i]
            binary = (probs >= threshold).astype(int)
            
            # Resolve conflicts by probability
            for category, (off_idx, hate_idx) in CATEGORY_INDICES.items():
                if binary[off_idx] == 1 and binary[hate_idx] == 1:
                    if probs[off_idx] > probs[hate_idx]:
                        binary[hate_idx] = 0
                    else:
                        binary[off_idx] = 0
            
            y_binary[i] = binary
        
        y_labels = convert_from_binary(y_binary, self.labels)
        
        # Extract hate words if requested (inference only)
        all_hate_words = None
        if extract_hate_words:
            all_hate_words = []
            print("  Extracting hate words using attention...")
            for i, text in enumerate(tqdm(X_test, desc="Extracting hate words")):
                # Check if this sample has hate/offensive labels
                if y_binary[i].sum() > 0 and 'normal' not in y_labels[i]:
                    # Re-encode single sample for attention extraction
                    single_encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding=True,
                        max_length=safe_max_length,
                        return_tensors='pt'
                    )
                    input_ids = single_encoding['input_ids'].to(self.device)
                    attention_mask = single_encoding['attention_mask'].to(self.device)
                    
                    # Extract highlighted tokens
                    highlighted = self._extract_hate_words_attention(
                        text, input_ids, attention_mask, top_k=5
                    )
                    
                    # Find original words in text
                    original_words = self._find_original_words(text, highlighted)
                    all_hate_words.append(original_words)
                else:
                    all_hate_words.append([])
        
        if extract_hate_words:
            return y_labels, y_binary, all_hate_words
        else:
            return y_labels, y_binary


# =============================================================================
# FLANT5 - CHAIN OF THOUGHT
# =============================================================================

class FlanT5Wrapper(ModelWrapper):
    """
    FlanT5 Multi-Label Classification with Chain-of-Thought
    
    Dataset A: Generate labels only
    Dataset B: Generate implied_statement → rationale → labels (CoT)
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 3e-5,
        max_input_length: int = 256,
        max_output_length: int = 64,
        device: str = None,
        use_chain_of_thought: bool = False
    ):
        super().__init__("FlanT5_MultiLabel")
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.labels = FINAL_LABELS
        self.use_chain_of_thought = use_chain_of_thought
        
        import torch
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.tokenizer = None
    
    def _format_input_standard(self, text: str) -> str:
        """Format input for Dataset A - SIMPLIFIED prompt (like experiment version)"""
        labels_str = ", ".join(self.labels)
        return f"""You are a Vietnamese hate speech classification system. Classify the text into one or more labels.
Valid labels: {labels_str}
Return only the label names, separated by commas.

Text to classify: {text}"""
    
    def _format_input_cot(self, text: str) -> str:
        """Format input for Dataset B (Chain-of-Thought)"""
        labels_str = ", ".join(self.labels)
        return f"""Analyze the Vietnamese text for hate speech using step-by-step reasoning.

RULES:
1. First identify the implied meaning (if any)
2. Then explain your reasoning
3. Finally, provide the labels
4. Each category can only have ONE level (offensive OR hate, not both)

Valid labels: {labels_str}

Text: {text}

Analysis:"""
    
    def _format_output_standard(self, labels_list: List[str]) -> str:
        """Format output for Dataset A"""
        resolved = resolve_labels_list(labels_list)
        return ", ".join(sorted(resolved))
    
    def _format_output_cot(self, labels_list: List[str], rationale: List[str], implied: str) -> str:
        """Format output for Dataset B (Chain-of-Thought)"""
        resolved = resolve_labels_list(labels_list)
        
        implied_text = implied if implied else "None"
        rationale_text = "; ".join(rationale) if rationale else "No specific reasoning needed"
        labels_text = ", ".join(sorted(resolved))
        
        return f"""Implied Statement: {implied_text}
Rationale: {rationale_text}
Labels: {labels_text}"""
    
    def _parse_output_standard(self, output: str) -> List[str]:
        """Parse output for Dataset A"""
        import re
        
        output = output.lower().strip()
        output = output.replace(";", ",").replace(" and ", ",")
        
        parts = [p.strip() for p in output.replace("\n", ",").split(",") if p.strip()]
        
        labels = []
        for part in parts:
            for valid_label in self.labels:
                if valid_label.lower() == part or valid_label.lower() in part:
                    if valid_label not in labels:
                        labels.append(valid_label)
                    break
        
        if not labels:
            labels = ["normal"]
        
        return resolve_labels_list(labels)
    
    def _parse_output_cot(self, output: str) -> Tuple[str, List[str], List[str]]:
        """Parse output for Dataset B (Chain-of-Thought)"""
        import re
        
        # Extract implied statement
        implied_match = re.search(r'Implied Statement:\s*(.+?)(?=\nRationale:|$)', output, re.DOTALL)
        implied = implied_match.group(1).strip() if implied_match else ""
        if implied.lower() in ["none", "no implication", ""]:
            implied = ""
        
        # Extract rationale
        rationale_match = re.search(r'Rationale:\s*(.+?)(?=\nLabels:|$)', output, re.DOTALL)
        rationale_text = rationale_match.group(1).strip() if rationale_match else ""
        
        if rationale_text.lower() == "no specific reasoning needed":
            rationale = []
        else:
            rationale = [r.strip() for r in re.split(r'[;.]', rationale_text) if r.strip()]
        
        # Extract labels
        labels_match = re.search(r'Labels:\s*(.+?)$', output, re.DOTALL)
        labels_text = labels_match.group(1).strip() if labels_match else output
        
        labels = []
        for part in re.split(r'[,;]', labels_text):
            part = part.strip().lower()
            for valid_label in self.labels:
                if valid_label.lower() == part or valid_label.lower() in part:
                    if valid_label not in labels:
                        labels.append(valid_label)
                    break
        
        if not labels:
            labels = ["normal"]
        
        return implied, rationale, resolve_labels_list(labels)
    
    def train(
        self, 
        X_train: List[str], 
        y_train: List[List[str]],
        rationale: List[List[str]] = None,
        implied: List[str] = None
    ):
        """Train FlanT5 model"""
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        from transformers import get_linear_schedule_with_warmup
        from torch.optim import AdamW
        from tqdm import tqdm
        
        print(f"[{self.name}] Training...")
        print(f"  Device: {self.device}")
        print(f"  Original samples: {len(X_train)}")
        
        has_rationale = rationale is not None and implied is not None
        use_cot = has_rationale and self.use_chain_of_thought
        
        if use_cot:
            print(f"  Mode: Chain-of-Thought (Dataset B)")
            self.max_output_length = max(self.max_output_length, 256)  # CoT needs longer output
            
            X_train, y_train, rationale, implied = oversample_minority_labels(
                X_train, y_train, self.labels, min_samples=200,
                rationale=rationale, implied=implied
            )
            
            inputs = [self._format_input_cot(x) for x in X_train]
            outputs = [
                self._format_output_cot(y, r, i) 
                for y, r, i in zip(y_train, rationale, implied)
            ]
        else:
            print(f"  Mode: Standard (Dataset A)")
            
            X_train, y_train = oversample_minority_labels(X_train, y_train, self.labels, min_samples=200)
            
            inputs = [self._format_input_standard(x) for x in X_train]
            outputs = [self._format_output_standard(y) for y in y_train]
        
        # Load model
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Tokenize
        print(f"  Tokenizing...")
        input_encodings = self.tokenizer(
            inputs, truncation=True, padding=True,
            max_length=self.max_input_length, return_tensors='pt'
        )
        output_encodings = self.tokenizer(
            outputs, truncation=True, padding=True,
            max_length=self.max_output_length, return_tensors='pt'
        )
        
        # Dataset
        class T5Dataset(Dataset):
            def __init__(self, input_enc, output_enc):
                self.input_ids = input_enc['input_ids']
                self.attention_mask = input_enc['attention_mask']
                self.labels = output_enc['input_ids']
            
            def __len__(self):
                return len(self.input_ids)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'labels': self.labels[idx]
                }
        
        dataset = T5Dataset(input_encodings, output_encodings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
        
        # Training
        print(f"  Training for {self.num_epochs} epochs...")
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch in progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                progress.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
        
        self.is_trained = True
        print(f"  Training completed")
    
    def _format_input_inference(self, text: str) -> str:
        """Format input for INFERENCE - SIMPLIFIED prompt (like experiment version)"""
        labels_str = ", ".join(self.labels)
        return f"""You are a Vietnamese hate speech classification system. Classify the text into one or more labels.
Valid labels: {labels_str}
Return only the label names, separated by commas.

Text to classify: {text}"""
    
    def _format_input_inference_with_hate_words(self, text: str) -> str:
        """Format input for INFERENCE with hate word extraction"""
        labels_str = ", ".join(self.labels)
        return f"""Classify the Vietnamese text into hate speech labels and extract hate words.

RULES:
1. Return labels first, separated by commas
2. Each category can only have ONE level (offensive OR hate, not both)
3. If no hate speech: normal
4. Extract EXACT hate words from the original text (do NOT modify them)
5. Hate words are words that cause the text to be offensive/hateful
6. Copy hate words EXACTLY as they appear in the original text

OUTPUT FORMAT:
Labels: <labels>
Hate Words: <word1>, <word2>, ... (exact words from text, or "None" if normal)

Valid labels: {labels_str}

Text: {text}

Output:"""
    
    def _parse_output_with_hate_words(self, output: str) -> Tuple[List[str], List[str]]:
        """Parse output containing hate words"""
        import re
        
        # Extract labels
        labels_match = re.search(r'Labels?:\s*(.+?)(?=\nHate|$)', output, re.IGNORECASE | re.DOTALL)
        labels_text = labels_match.group(1).strip() if labels_match else output
        
        labels_text = labels_text.lower().replace(";", ",").replace(" and ", ",")
        parts = [p.strip() for p in labels_text.split(",") if p.strip()]
        
        labels = []
        for part in parts:
            for valid_label in self.labels:
                if valid_label.lower() == part or valid_label.lower() in part:
                    if valid_label not in labels:
                        labels.append(valid_label)
                    break
        
        if not labels:
            labels = ["normal"]
        
        labels = resolve_labels_list(labels)
        
        # Extract hate words
        hate_words = []
        hate_match = re.search(r'Hate Words?:\s*(.+?)$', output, re.IGNORECASE | re.DOTALL)
        if hate_match:
            hate_text = hate_match.group(1).strip()
            if hate_text.lower() not in ["none", "không", "không có"]:
                hate_words = [w.strip() for w in hate_text.split(",") if w.strip()]
        
        return labels, hate_words
    
    def predict(self, X_test: List[str], extract_hate_words: bool = False) -> Tuple[List[List[str]], List[str], Optional[List[List[str]]]]:
        """
        Predict - ALWAYS uses fast inference (labels only, no CoT output)
        Model learned from CoT during training, but inference is fast!
        
        Args:
            X_test: List of texts to classify
            extract_hate_words: If True, also extract hate words from text (inference only)
        
        Returns:
            predictions_labels: Predicted labels
            predictions_raw: Raw model outputs
            hate_words: List of hate words per sample (only if extract_hate_words=True)
        """
        import torch
        from tqdm import tqdm
        
        if not self.is_trained:
            raise ValueError("Model has not been trained")
        
        # Choose prompt based on whether we need hate words
        if extract_hate_words:
            inputs = [self._format_input_inference_with_hate_words(x) for x in X_test]
            max_new_tokens = 100  # Longer for hate words
        else:
            inputs = [self._format_input_inference(x) for x in X_test]
            max_new_tokens = 50
        
        self.model.eval()
        predictions_labels = []
        predictions_raw = []
        all_hate_words = [] if extract_hate_words else None
        
        with torch.no_grad():
            for i in tqdm(range(0, len(inputs), self.batch_size), desc="Predicting"):
                batch_inputs = inputs[i:i+self.batch_size]
                
                encodings = self.tokenizer(
                    batch_inputs, truncation=True, padding=True,
                    max_length=self.max_input_length, return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,  # Greedy decoding (faster)
                    do_sample=False,
                    early_stopping=True
                )
                
                for output in outputs:
                    decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                    predictions_raw.append(decoded)
                    
                    if extract_hate_words:
                        labels, hate_words = self._parse_output_with_hate_words(decoded)
                        predictions_labels.append(labels)
                        all_hate_words.append(hate_words)
                    else:
                        labels = self._parse_output_standard(decoded)
                        predictions_labels.append(labels)
        
        if extract_hate_words:
            return predictions_labels, predictions_raw, all_hate_words
        else:
            return predictions_labels, predictions_raw


# =============================================================================
# QWEN - CHAIN OF THOUGHT WITH QLORA
# =============================================================================

class QwenWrapper(ModelWrapper):
    """
    Qwen2.5-3B Multi-Label Classification with QLoRA 4-bit and Chain-of-Thought
    
    Dataset A: Generate labels only
    Dataset B: Generate implied_statement → rationale → labels (CoT)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        batch_size: int = 4,
        num_epochs: int = 2,
        learning_rate: float = 2e-4,
        max_length: int = 512,
        lora_r: int = 8,
        lora_alpha: int = 16,
        device: str = None,
        use_chain_of_thought: bool = False
    ):
        super().__init__("Qwen2.5-3B_MultiLabel")
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.labels = FINAL_LABELS
        self.use_chain_of_thought = use_chain_of_thought
        
        import torch
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.tokenizer = None
    
    def _format_input_standard(self, text: str) -> str:
        """Format input for Dataset A - SIMPLIFIED prompt (like experiment version)"""
        labels_str = ", ".join(self.labels)
        return f"""<|im_start|>system
You are a Vietnamese hate speech classification system. Classify the text into one or more labels.
Valid labels: {labels_str}
Return only the label names, separated by commas.<|im_end|>
<|im_start|>user
Text to classify: {text}<|im_end|>
<|im_start|>assistant
"""
    
    def _format_input_cot(self, text: str) -> str:
        """Format input for Dataset B (Chain-of-Thought) - same as FlanT5 but wrapped in chat format"""
        labels_str = ", ".join(self.labels)
        return f"""<|im_start|>system
Analyze the Vietnamese text for hate speech using step-by-step reasoning.

RULES:
1. First identify the implied meaning (if any)
2. Then explain your reasoning
3. Finally, provide the labels
4. Each category can only have ONE level (offensive OR hate, not both)

Valid labels: {labels_str}<|im_end|>
<|im_start|>user
Text: {text}<|im_end|>
<|im_start|>assistant
Analysis:"""
    
    def _format_output_standard(self, labels_list: List[str]) -> str:
        """Format output for Dataset A"""
        resolved = resolve_labels_list(labels_list)
        return ", ".join(sorted(resolved)) + "<|im_end|>"
    
    def _format_output_cot(self, labels_list: List[str], rationale: List[str], implied: str) -> str:
        """Format output for Dataset B - same as FlanT5"""
        resolved = resolve_labels_list(labels_list)
        
        implied_text = implied if implied else "None"
        rationale_text = "; ".join(rationale) if rationale else "No specific reasoning needed"
        labels_text = ", ".join(sorted(resolved))
        
        return f"""Implied Statement: {implied_text}
Rationale: {rationale_text}
Labels: {labels_text}<|im_end|>"""
    
    def _parse_output(self, output: str) -> List[str]:
        """Parse output from Qwen - IMPROVED to not depend on 'Labels:' prefix"""
        import re
        
        # Clean output
        output = output.replace("<|im_end|>", "").strip()
        
        # Try to extract from "Labels:" section if present (optional)
        labels_match = re.search(r'labels?:\s*(.+?)$', output, re.IGNORECASE | re.DOTALL)
        if labels_match:
            output = labels_match.group(1).strip()
        
        # Parse labels (case-insensitive matching)
        output_lower = output.lower()
        output_clean = output_lower.replace(";", ",").replace(" and ", ",")
        parts = [p.strip() for p in re.split(r'[,\n]', output_clean) if p.strip()]
        
        labels = []
        for part in parts:
            for valid_label in self.labels:
                if valid_label.lower() == part or valid_label.lower() in part:
                    if valid_label not in labels:
                        labels.append(valid_label)
                    break
        
        if not labels:
            labels = ["normal"]
        
        # Resolve conflicts: Keep only ONE level per category (offensive OR hate)
        # This prevents model from predicting both "individuals#offensive" AND "individuals#hate"
        # Ground truth never has conflicts, but model may hallucinate them
        return resolve_labels_list(labels)
    
    def train(
        self, 
        X_train: List[str], 
        y_train: List[List[str]],
        rationale: List[List[str]] = None,
        implied: List[str] = None
    ):
        """Train Qwen with QLoRA"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from torch.utils.data import DataLoader, Dataset
        from tqdm import tqdm
        
        print(f"[{self.name}] Training with QLoRA (4-bit)...")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Original samples: {len(X_train)}")
        
        has_rationale = rationale is not None and implied is not None
        use_cot = has_rationale and self.use_chain_of_thought
        
        if use_cot:
            print(f"  Mode: Chain-of-Thought (Dataset B)")
            
            X_train, y_train, rationale, implied = oversample_minority_labels(
                X_train, y_train, self.labels, min_samples=200,
                rationale=rationale, implied=implied
            )
            
            training_texts = [
                self._format_input_cot(x) + self._format_output_cot(y, r, i)
                for x, y, r, i in zip(X_train, y_train, rationale, implied)
            ]
        else:
            print(f"  Mode: Standard (Dataset A)")
            
            X_train, y_train = oversample_minority_labels(X_train, y_train, self.labels, min_samples=200)
            
            training_texts = [
                self._format_input_standard(x) + self._format_output_standard(y)
                for x, y in zip(X_train, y_train)
            ]
        
        print(f"  LoRA config: r={self.lora_r}, alpha={self.lora_alpha}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 4-bit config with better error handling
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        except Exception as e:
            print(f"  Warning: BitsAndBytes config failed: {e}")
            print(f"   Trying to reinstall bitsandbytes...")
            import subprocess
            try:
                subprocess.run(["pip", "install", "--upgrade", "--force-reinstall", "bitsandbytes"], 
                             check=True, capture_output=True)
                print(f"   Reinstalled bitsandbytes, retrying...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            except Exception as e2:
                print(f"   Failed to fix bitsandbytes: {e2}")
                raise RuntimeError(
                    "BitsAndBytes not properly installed. Please run: "
                    "pip install --upgrade --force-reinstall bitsandbytes"
                ) from e
        
        # Load model with error handling
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        except ImportError as e:
            if "bitsandbytes" in str(e):
                print(f"  BitsAndBytes version issue: {e}")
                print(f"   Attempting to upgrade bitsandbytes...")
                import subprocess
                try:
                    subprocess.run(
                        ["pip", "install", "-U", "bitsandbytes"], 
                        check=True, 
                        capture_output=True,
                        text=True
                    )
                    print(f"   Upgraded bitsandbytes successfully")
                    print(f"   Please RESTART the kernel and run again")
                    raise RuntimeError(
                        "BitsAndBytes has been upgraded. "
                        "Please RESTART the kernel and run the notebook again."
                    ) from e
                except subprocess.CalledProcessError as e2:
                    print(f"   Failed to upgrade: {e2}")
                    raise RuntimeError(
                        "Cannot upgrade bitsandbytes. Please manually run:\n"
                        "  !pip install -U bitsandbytes\n"
                        "Then RESTART the kernel."
                    ) from e
            else:
                raise
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Dataset
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length):
                self.encodings = tokenizer(
                    texts, truncation=True, padding=True,
                    max_length=max_length, return_tensors='pt'
                )
            
            def __len__(self):
                return len(self.encodings['input_ids'])
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx]
                }
        
        dataset = TextDataset(training_texts, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Training
        print(f"  Training for {self.num_epochs} epochs...")
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch in progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                progress.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
        
        self.is_trained = True
        print(f"  Training completed")
    
    def train_stage2_alignment(
        self,
        pairs: List[Dict[str, str]],
        rationale_list: List[List[str]] = None,
        labels_list: List[List[str]] = None,
        num_epochs: int = 1,
        learning_rate: float = 5e-5
    ):
        """
        Stage 2: Rationale-augmented continuation training.
        Continues LoRA training on CoT-formatted data to align model
        with rationale semantics.

        Args:
            pairs: List of dicts with 'content' and 'implied' keys.
            rationale_list: Per-sample rationale lists. If None, empty rationales used.
            labels_list: Per-sample label lists. If None, defaults to ['normal'].
            num_epochs: Training epochs for Stage 2 (default 1).
            learning_rate: Lower LR to avoid catastrophic forgetting.
        """
        import torch
        from torch.utils.data import DataLoader, Dataset
        from tqdm import tqdm

        if self.model is None or self.tokenizer is None:
            raise ValueError("Stage 1 must be completed before Stage 2. Call train() first.")

        print(f"[{self.name}] Stage 2: Semantic alignment")
        print(f"  Pairs: {len(pairs)}, Epochs: {num_epochs}, LR: {learning_rate}")

        training_texts = []
        for i, pair in enumerate(pairs):
            content = pair['content']
            implied = pair['implied']
            rat = rationale_list[i] if rationale_list and i < len(rationale_list) else []
            lab = labels_list[i] if labels_list and i < len(labels_list) else ['normal']
            input_text = self._format_input_cot(content)
            output_text = self._format_output_cot(lab, rat, implied)
            training_texts.append(input_text + output_text)

        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length):
                self.encodings = tokenizer(
                    texts, truncation=True, padding=True,
                    max_length=max_length, return_tensors='pt'
                )
            def __len__(self):
                return len(self.encodings['input_ids'])
            def __getitem__(self, idx):
                return {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx]
                }

        dataset = TextDataset(training_texts, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            progress = tqdm(dataloader, desc=f"  Stage2 Epoch {epoch+1}/{num_epochs}")
            for batch in progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                progress.set_postfix({'loss': f'{loss.item():.4f}'})
            avg_loss = total_loss / len(dataloader)
            print(f"  Stage2 Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

        print(f"  Stage 2 alignment completed")
    
    def _format_input_inference(self, text: str) -> str:
        """Format input for INFERENCE - SIMPLIFIED prompt (like experiment version)"""
        labels_str = ", ".join(self.labels)
        return f"""<|im_start|>system
You are a Vietnamese hate speech classification system. Classify the text into one or more labels.
Valid labels: {labels_str}
Return only the label names, separated by commas.<|im_end|>
<|im_start|>user
Text to classify: {text}<|im_end|>
<|im_start|>assistant
"""
    
    def _format_input_inference_with_hate_words(self, text: str) -> str:
        """Format input for INFERENCE with hate word extraction"""
        labels_str = ", ".join(self.labels)
        # Core prompt same as FlanT5, only wrapped in chat format
        return f"""<|im_start|>system
Classify the Vietnamese text into hate speech labels and extract hate words.

RULES:
1. Return labels first, separated by commas
2. Each category can only have ONE level (offensive OR hate, not both)
3. If no hate speech: normal
4. Extract EXACT hate words from the original text (do NOT modify them)
5. Hate words are words that cause the text to be offensive/hateful
6. Copy hate words EXACTLY as they appear in the original text

OUTPUT FORMAT:
Labels: <labels>
Hate Words: <word1>, <word2>, ... (exact words from text, or "None" if normal)

Valid labels: {labels_str}<|im_end|>
<|im_start|>user
Text: {text}<|im_end|>
<|im_start|>assistant
Output:"""
    
    def _parse_output_with_hate_words(self, output: str) -> Tuple[List[str], List[str]]:
        """Parse output containing hate words"""
        import re
        
        # Split by newline or "Hate Words:"
        parts = re.split(r'\n|Hate Words?:', output, maxsplit=1)
        
        # Parse labels from first part
        labels_text = parts[0].replace("<|im_end|>", "").strip().lower()
        labels_text = labels_text.replace(";", ",").replace(" and ", ",")
        
        labels = []
        for part in labels_text.split(","):
            part = part.strip()
            for valid_label in self.labels:
                if valid_label.lower() == part or valid_label.lower() in part:
                    if valid_label not in labels:
                        labels.append(valid_label)
                    break
        
        if not labels:
            labels = ["normal"]
        
        labels = resolve_labels_list(labels)
        
        # Parse hate words from second part
        hate_words = []
        if len(parts) > 1:
            hate_text = parts[1].replace("<|im_end|>", "").strip()
            if hate_text.lower() not in ["none", "không", "không có", ""]:
                hate_words = [w.strip() for w in hate_text.split(",") if w.strip()]
        
        return labels, hate_words
    
    def predict(self, X_test: List[str], extract_hate_words: bool = False) -> Tuple[List[List[str]], List[str], Optional[List[List[str]]]]:
        """
        Predict - ALWAYS uses fast inference (labels only, no CoT output)
        Model learned from CoT during training, but inference is fast!
        
        Args:
            X_test: List of texts to classify
            extract_hate_words: If True, also extract hate words from text (inference only)
        
        Returns:
            predictions_labels: Predicted labels
            predictions_raw: Raw model outputs
            hate_words: List of hate words per sample (only if extract_hate_words=True)
        """
        import torch
        from tqdm import tqdm
        
        if not self.is_trained:
            raise ValueError("Model has not been trained")
        
        self.model.eval()
        predictions_labels = []
        predictions_raw = []
        all_hate_words = [] if extract_hate_words else None
        
        with torch.no_grad():
            for text in tqdm(X_test, desc="Predicting"):
                # Choose prompt based on whether we need hate words
                if extract_hate_words:
                    input_text = self._format_input_inference_with_hate_words(text)
                    max_new_tokens = 100  # Longer for hate words
                else:
                    input_text = self._format_input_inference(text)
                    max_new_tokens = 50
                
                inputs = self.tokenizer(
                    input_text, return_tensors='pt', truncation=True, max_length=self.max_length
                )
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.encode("<|im_end|>")[0] if "<|im_end|>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id
                )
                
                new_tokens = outputs[0][len(input_ids[0]):]
                decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                output = decoded.replace("<|im_end|>", "").strip()
                
                predictions_raw.append(output)
                
                if extract_hate_words:
                    labels, hate_words = self._parse_output_with_hate_words(output)
                    predictions_labels.append(labels)
                    all_hate_words.append(hate_words)
                else:
                    predictions_labels.append(self._parse_output(output))
        
        if extract_hate_words:
            return predictions_labels, predictions_raw, all_hate_words
        else:
            return predictions_labels, predictions_raw


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_model(
    model_type: str,
    dataset_type: str = "A",
    **kwargs
) -> ModelWrapper:
    """
    Factory function to create model
    
    Args:
        model_type: "phobert", "flant5", "qwen"
        dataset_type: "A" (standard) or "B" (with rationale/implied)
        **kwargs: Additional model parameters
    
    Returns:
        Model instance
    
    Example:
        # Dataset A (standard)
        model = create_model("phobert", dataset_type="A")
        model.train(texts, labels)
        
        # Dataset B (with rationale)
        model = create_model("flant5", dataset_type="B")
        model.train(texts, labels, rationale=rationale, implied=implied)
    """
    model_type = model_type.lower()
    use_advanced = dataset_type.upper() == "B"
    
    if model_type == "phobert":
        return PhoBERTWrapper(
            use_knowledge_distillation=use_advanced,
            **kwargs
        )
    elif model_type == "flant5":
        return FlanT5Wrapper(
            use_chain_of_thought=use_advanced,
            **kwargs
        )
    elif model_type == "qwen":
        return QwenWrapper(
            use_chain_of_thought=use_advanced,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: phobert, flant5, qwen")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Models Module")
    print("=" * 70)
    
    # Test data
    X_train = [
        "Đây là văn bản bình thường",
        "Tôi ghét nhóm người này, họ thật tệ",
        "Cá nhân này thật đáng khinh",
    ]
    y_train = [
        ["normal"],
        ["groups#hate"],
        ["individuals#offensive"],
    ]
    rationale = [
        [],
        ["Sử dụng từ ngữ thù địch", "Nhắm vào nhóm người cụ thể"],
        ["Ngôn ngữ xúc phạm cá nhân"],
    ]
    implied = [
        "",
        "Nhóm người này không đáng được tôn trọng",
        "Cá nhân này thấp kém",
    ]
    
    print("\n Test 1: PhoBERT Dataset A (Standard)")
    model_a = create_model("phobert", dataset_type="A", num_epochs=1)
    print(f"   Model: {model_a}")
    print(f"   use_knowledge_distillation: {model_a.use_knowledge_distillation}")
    
    print("\n Test 2: PhoBERT Dataset B (Knowledge Distillation)")
    model_b = create_model("phobert", dataset_type="B", num_epochs=1)
    print(f"   Model: {model_b}")
    print(f"   use_knowledge_distillation: {model_b.use_knowledge_distillation}")
    
    print("\n Test 3: FlanT5 Dataset A (Standard)")
    model_c = create_model("flant5", dataset_type="A", num_epochs=1)
    print(f"   Model: {model_c}")
    print(f"   use_chain_of_thought: {model_c.use_chain_of_thought}")
    
    print("\n Test 4: FlanT5 Dataset B (Chain-of-Thought)")
    model_d = create_model("flant5", dataset_type="B", num_epochs=1)
    print(f"   Model: {model_d}")
    print(f"   use_chain_of_thought: {model_d.use_chain_of_thought}")
    
    print("\n All tests passed!")
