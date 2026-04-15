"""
Evaluation Module
Evaluate and compare models for Multi-Label Classification
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    hamming_loss, multilabel_confusion_matrix
)

from config import FINAL_LABELS, OUTPUT_DIR


class Evaluator:
    """Evaluate and compare model results (Multi-label)"""
    
    def __init__(self, labels: List[str] = FINAL_LABELS):
        self.labels = labels
        self.results: List[Dict[str, Any]] = []
    
    def _convert_to_binary(self, y: List[List[str]]) -> np.ndarray:
        """Convert list of label strings to binary matrix"""
        binary = []
        for labels_list in y:
            row = [0] * len(self.labels)
            for label in labels_list:
                if label in self.labels:
                    idx = self.labels.index(label)
                    row[idx] = 1
            binary.append(row)
        return np.array(binary)
    
    def evaluate(self, y_true, y_pred, model_name: str, dataset_version: str) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset version (Multi-label)
        
        Args:
            y_true: Ground truth labels (list of label strings or binary matrix)
            y_pred: Predicted labels (list of label strings or binary matrix)
        """
        # Convert to binary matrix if needed
        if len(y_true) > 0 and isinstance(y_true[0], list) and len(y_true[0]) > 0 and isinstance(y_true[0][0], str):
            y_true_binary = self._convert_to_binary(y_true)
        else:
            y_true_binary = np.array(y_true)
            
        if len(y_pred) > 0 and isinstance(y_pred[0], list) and len(y_pred[0]) > 0 and isinstance(y_pred[0][0], str):
            y_pred_binary = self._convert_to_binary(y_pred)
        else:
            y_pred_binary = np.array(y_pred)
        
        # Metrics
        subset_accuracy = accuracy_score(y_true_binary, y_pred_binary)
        h_loss = hamming_loss(y_true_binary, y_pred_binary)
        f1_samples = f1_score(y_true_binary, y_pred_binary, average='samples', zero_division=0)
        precision_samples = precision_score(y_true_binary, y_pred_binary, average='samples', zero_division=0)
        recall_samples = recall_score(y_true_binary, y_pred_binary, average='samples', zero_division=0)
        f1_macro = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
        f1_micro = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
        precision_macro = precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
        precision_micro = precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
        recall_macro = recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
        recall_micro = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
        
        # Per-label F1, Precision, Recall
        f1_per_label = f1_score(y_true_binary, y_pred_binary, average=None, zero_division=0)
        precision_per_label = precision_score(y_true_binary, y_pred_binary, average=None, zero_division=0)
        recall_per_label = recall_score(y_true_binary, y_pred_binary, average=None, zero_division=0)
        
        f1_per_label_dict = {}
        precision_per_label_dict = {}
        recall_per_label_dict = {}
        
        for i, label in enumerate(self.labels):
            f1_per_label_dict[label] = float(f1_per_label[i])
            precision_per_label_dict[label] = float(precision_per_label[i])
            recall_per_label_dict[label] = float(recall_per_label[i])
        
        result = {
            'model': model_name,
            'dataset': dataset_version,
            'subset_accuracy': float(subset_accuracy),
            'hamming_loss': float(h_loss),
            'f1_samples': float(f1_samples),
            'f1_macro': float(f1_macro),
            'f1_micro': float(f1_micro),
            'precision_samples': float(precision_samples),
            'precision_macro': float(precision_macro),
            'precision_micro': float(precision_micro),
            'recall_samples': float(recall_samples),
            'recall_macro': float(recall_macro),
            'recall_micro': float(recall_micro),
            'f1_per_label': f1_per_label_dict,
            'precision_per_label': precision_per_label_dict,
            'recall_per_label': recall_per_label_dict,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_true),
        }
        
        self.results.append(result)
        return result
    
    def print_result(self, result: Dict[str, Any]):
        """Print formatted result"""
        print(f"\n{'='*70}")
        print(f" {result['model']} on Dataset {result['dataset']} (Multi-Label)")
        print(f"{'='*70}")
        print(f"  Subset Accuracy (Exact Match): {result['subset_accuracy']:.4f}")
        print(f"  Hamming Loss:                  {result['hamming_loss']:.4f}")
        print(f"  ")
        print(f"  F1 Samples:                    {result['f1_samples']:.4f}")
        print(f"  F1 Macro:                      {result['f1_macro']:.4f}")
        print(f"  F1 Micro:                      {result['f1_micro']:.4f}")
        print(f"  ")
        print(f"  Precision (Samples):           {result['precision_samples']:.4f}")
        print(f"  Precision (Macro):             {result['precision_macro']:.4f}")
        print(f"  Precision (Micro):             {result['precision_micro']:.4f}")
        print(f"  ")
        print(f"  Recall (Samples):              {result['recall_samples']:.4f}")
        print(f"  Recall (Macro):                {result['recall_macro']:.4f}")
        print(f"  Recall (Micro):                {result['recall_micro']:.4f}")
        
        print(f"\n  F1 per Label:")
        for label, f1 in result['f1_per_label'].items():
            if f1 > 0:
                print(f"    {label:25s}: {f1:.4f}")
    
    def get_comparison_matrix(self) -> pd.DataFrame:
        """Create comparison matrix"""
        if not self.results:
            return pd.DataFrame()
        df = pd.DataFrame(self.results)
        matrix = df.pivot(index='model', columns='dataset', values='f1_macro')
        return matrix
    
    def get_full_comparison(self) -> pd.DataFrame:
        """Create full comparison table"""
        if not self.results:
            return pd.DataFrame()
        rows = []
        for r in self.results:
            rows.append({
                'Model': r['model'],
                'Dataset': r['dataset'],
                'Subset Acc': f"{r['subset_accuracy']:.4f}",
                'Hamming': f"{r['hamming_loss']:.4f}",
                'F1 Samples': f"{r['f1_samples']:.4f}",
                'F1 Macro': f"{r['f1_macro']:.4f}",
                'F1 Micro': f"{r['f1_micro']:.4f}",
            })
        return pd.DataFrame(rows)
    
    def save_results(self, output_dir: Path = OUTPUT_DIR):
        """Save results to file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "experiment_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {json_path}")
        
        matrix = self.get_comparison_matrix()
        if not matrix.empty:
            csv_path = output_dir / "comparison_matrix.csv"
            matrix.to_csv(csv_path)
            print(f"Saved comparison matrix to {csv_path}")
        
        full_df = self.get_full_comparison()
        if not full_df.empty:
            csv_path = output_dir / "full_comparison.csv"
            full_df.to_csv(csv_path, index=False)
            print(f"Saved full comparison to {csv_path}")
    
    def generate_report(self) -> str:
        """Generate text report"""
        report = []
        report.append("=" * 80)
        report.append("MULTI-LABEL EXPERIMENT RESULTS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        report.append("\n COMPARISON MATRIX (F1 Macro)")
        report.append("-" * 60)
        matrix = self.get_comparison_matrix()
        if not matrix.empty:
            report.append(matrix.to_string())
        
        report.append("\n\n FULL COMPARISON")
        report.append("-" * 60)
        full_df = self.get_full_comparison()
        if not full_df.empty:
            report.append(full_df.to_string(index=False))
        
        report.append("\n\n DETAILED RESULTS")
        report.append("-" * 60)
        for r in self.results:
            report.append(f"\n{r['model']} - Dataset {r['dataset']}:")
            report.append(f"  Subset Accuracy: {r['subset_accuracy']:.4f}")
            report.append(f"  Hamming Loss: {r['hamming_loss']:.4f}")
            report.append(f"  F1 Macro: {r['f1_macro']:.4f}")
        
        if self.results:
            best = max(self.results, key=lambda x: x['f1_macro'])
            report.append(f"\n\n ANALYSIS")
            report.append(f"Best Model: {best['model']} on Dataset {best['dataset']} (F1: {best['f1_macro']:.4f})")
        
        return "\n".join(report)


if __name__ == "__main__":
    print("Testing Evaluator (Multi-Label)...")
    
    evaluator = Evaluator()
    y_true = [["normal"], ["individuals#hate"], ["groups#offensive", "individuals#hate"], ["normal"]]
    y_pred = [["normal"], ["individuals#hate"], ["individuals#hate"], ["individuals#offensive"]]
    
    result = evaluator.evaluate(y_true, y_pred, "TestModel", "A")
    evaluator.print_result(result)
