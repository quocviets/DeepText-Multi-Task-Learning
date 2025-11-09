# -*- coding: utf-8 -*-
"""
Model Evaluation Module
=======================

This module provides comprehensive evaluation utilities for the DeepText Multi-Task Learning model.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(self, 
                 model,
                 data: Dict[str, Any],
                 class_names: Dict[str, List[str]] = None,
                 output_dir: str = "reports"):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained Keras model
            data: Processed data dictionary
            class_names: Class names for each task
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.data = data
        self.class_names = class_names or data.get('class_names', {
            'emotion': ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo'],
            'hate': ['hate', 'offensive', 'neutral'],
            'violence': ['sex_viol', 'phys_viol', 'no_viol']
        })
        self.output_dir = output_dir
        self.evaluation_results = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/confusion_matrices", exist_ok=True)
        os.makedirs(f"{output_dir}/roc_curves", exist_ok=True)
        os.makedirs(f"{output_dir}/precision_recall_curves", exist_ok=True)
    
    def evaluate_model(self, 
                      test_data: Optional[Tuple] = None,
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            test_data: Optional test data tuple (X_test, y_test)
            save_results: Whether to save evaluation results
            
        Returns:
            Comprehensive evaluation results
        """
        if test_data is None:
            X_test = self.data['X_test']
            y_test = {
                'emotion_output': self.data['y_emotion_test'],
                'hate_output': self.data['y_hate_test'],
                'violence_output': self.data['y_violence_test']
            }
        else:
            X_test, y_test = test_data
        
        print("Evaluating model...")
        
        # Get predictions
        predictions = self.model.predict(X_test, verbose=1)
        
        # Evaluate each task
        results = {}
        task_names = ['emotion', 'hate', 'violence']
        
        for i, task in enumerate(task_names):
            print(f"\nEvaluating {task} task...")
            
            y_true = y_test[f'{task}_output']
            y_pred = predictions[i]
            
            # Convert to class indices
            y_true_classes = np.argmax(y_true, axis=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            task_results = self._evaluate_task(
                y_true, y_pred, y_true_classes, y_pred_classes, task
            )
            
            results[task] = task_results
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(results)
        results['overall'] = overall_metrics
        
        self.evaluation_results = results
        
        # Save results
        if save_results:
            self._save_evaluation_results(results)
        
        return results
    
    def _evaluate_task(self, 
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_true_classes: np.ndarray,
                      y_pred_classes: np.ndarray,
                      task: str) -> Dict[str, Any]:
        """
        Evaluate a single task.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            y_true_classes: True class indices
            y_pred_classes: Predicted class indices
            task: Task name
            
        Returns:
            Task evaluation results
        """
        class_names = self.class_names[task]
        
        # Basic metrics
        accuracy = np.mean(y_pred_classes == y_true_classes)
        
        # Precision, Recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average=None, zero_division=0
        )
        
        # Macro averages
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average='weighted', zero_division=0
        )
        
        # Classification report
        report = classification_report(
            y_true_classes, y_pred_classes,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # ROC AUC (for multi-class)
        try:
            if len(class_names) > 2:
                # Multi-class ROC AUC
                y_true_bin = label_binarize(y_true_classes, classes=range(len(class_names)))
                roc_auc = roc_auc_score(y_true_bin, y_pred, multi_class='ovr', average='macro')
            else:
                # Binary ROC AUC
                roc_auc = roc_auc_score(y_true_classes, y_pred[:, 1])
        except:
            roc_auc = 0.0
        
        # Per-class metrics
        per_class_metrics = []
        for i, class_name in enumerate(class_names):
            per_class_metrics.append({
                'class': class_name,
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            })
        
        return {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'per_class_metrics': per_class_metrics,
            'predictions': y_pred_classes.tolist(),
            'true_labels': y_true_classes.tolist(),
            'probabilities': y_pred.tolist()
        }
    
    def _calculate_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall metrics across all tasks.
        
        Args:
            results: Task-specific results
            
        Returns:
            Overall metrics
        """
        # Calculate macro averages across tasks
        macro_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
        overall_metrics = {}
        
        for metric in macro_metrics:
            values = [results[task][metric] for task in ['emotion', 'hate', 'violence'] 
                     if metric in results[task]]
            overall_metrics[f'mean_{metric}'] = float(np.mean(values))
            overall_metrics[f'std_{metric}'] = float(np.std(values))
        
        # Calculate overall accuracy (average across tasks)
        overall_metrics['overall_accuracy'] = overall_metrics['mean_accuracy']
        
        return overall_metrics
    
    def plot_confusion_matrices(self, 
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot confusion matrices for all tasks.
        
        Args:
            save_path: Path to save plot
            figsize: Figure size
        """
        if self.evaluation_results is None:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Confusion Matrices', fontsize=16)
        
        tasks = ['emotion', 'hate', 'violence']
        
        for i, task in enumerate(tasks):
            cm = np.array(self.evaluation_results[task]['confusion_matrix'])
            class_names = self.class_names[task]
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm_normalized, 
                       annot=True, 
                       fmt='.2f',
                       cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names,
                       ax=axes[i])
            
            axes[i].set_title(f'{task.title()} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, 
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot ROC curves for all tasks.
        
        Args:
            save_path: Path to save plot
            figsize: Figure size
        """
        if self.evaluation_results is None:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('ROC Curves', fontsize=16)
        
        tasks = ['emotion', 'hate', 'violence']
        
        for i, task in enumerate(tasks):
            y_true = np.array(self.data[f'y_{task}_test'])
            y_pred = np.array(self.evaluation_results[task]['probabilities'])
            class_names = self.class_names[task]
            
            # Calculate ROC curve for each class
            if len(class_names) > 2:
                # Multi-class ROC
                y_true_bin = label_binarize(
                    np.argmax(y_true, axis=1), 
                    classes=range(len(class_names))
                )
                
                for j, class_name in enumerate(class_names):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, j], y_pred[:, j])
                    auc = roc_auc_score(y_true_bin[:, j], y_pred[:, j])
                    axes[i].plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.2f})')
            else:
                # Binary ROC
                fpr, tpr, _ = roc_curve(y_true[:, 1], y_pred[:, 1])
                auc = roc_auc_score(y_true[:, 1], y_pred[:, 1])
                axes[i].plot(fpr, tpr, label=f'AUC = {auc:.2f}')
            
            axes[i].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{task.title()} ROC Curve')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, 
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot precision-recall curves for all tasks.
        
        Args:
            save_path: Path to save plot
            figsize: Figure size
        """
        if self.evaluation_results is None:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Precision-Recall Curves', fontsize=16)
        
        tasks = ['emotion', 'hate', 'violence']
        
        for i, task in enumerate(tasks):
            y_true = np.array(self.data[f'y_{task}_test'])
            y_pred = np.array(self.evaluation_results[task]['probabilities'])
            class_names = self.class_names[task]
            
            # Calculate PR curve for each class
            if len(class_names) > 2:
                # Multi-class PR
                y_true_bin = label_binarize(
                    np.argmax(y_true, axis=1), 
                    classes=range(len(class_names))
                )
                
                for j, class_name in enumerate(class_names):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, j], y_pred[:, j])
                    ap = average_precision_score(y_true_bin[:, j], y_pred[:, j])
                    axes[i].plot(recall, precision, label=f'{class_name} (AP = {ap:.2f})')
            else:
                # Binary PR
                precision, recall, _ = precision_recall_curve(y_true[:, 1], y_pred[:, 1])
                ap = average_precision_score(y_true[:, 1], y_pred[:, 1])
                axes[i].plot(recall, precision, label=f'AP = {ap:.2f}')
            
            axes[i].set_xlabel('Recall')
            axes[i].set_ylabel('Precision')
            axes[i].set_title(f'{task.title()} Precision-Recall Curve')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curves saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, 
                                 save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Path where report was saved
        """
        if self.evaluation_results is None:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.output_dir}/evaluation_report_{timestamp}.txt"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("DeepText Multi-Task Learning - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-" * 20 + "\n")
            overall = self.evaluation_results['overall']
            f.write(f"Overall Accuracy: {overall['overall_accuracy']:.4f}\n")
            f.write(f"Mean F1-Score: {overall['mean_f1_macro']:.4f} ± {overall['std_f1_macro']:.4f}\n")
            f.write(f"Mean Precision: {overall['mean_precision_macro']:.4f} ± {overall['std_precision_macro']:.4f}\n")
            f.write(f"Mean Recall: {overall['mean_recall_macro']:.4f} ± {overall['std_recall_macro']:.4f}\n")
            f.write(f"Mean ROC AUC: {overall['mean_roc_auc']:.4f} ± {overall['std_roc_auc']:.4f}\n\n")
            
            # Task-specific metrics
            for task in ['emotion', 'hate', 'violence']:
                f.write(f"{task.upper()} TASK METRICS\n")
                f.write("-" * 20 + "\n")
                
                task_results = self.evaluation_results[task]
                f.write(f"Accuracy: {task_results['accuracy']:.4f}\n")
                f.write(f"F1-Score (Macro): {task_results['f1_macro']:.4f}\n")
                f.write(f"F1-Score (Weighted): {task_results['f1_weighted']:.4f}\n")
                f.write(f"Precision (Macro): {task_results['precision_macro']:.4f}\n")
                f.write(f"Recall (Macro): {task_results['recall_macro']:.4f}\n")
                f.write(f"ROC AUC: {task_results['roc_auc']:.4f}\n\n")
                
                # Per-class metrics
                f.write("Per-Class Metrics:\n")
                f.write("-" * 15 + "\n")
                for class_metric in task_results['per_class_metrics']:
                    f.write(f"{class_metric['class']}:\n")
                    f.write(f"  Precision: {class_metric['precision']:.4f}\n")
                    f.write(f"  Recall: {class_metric['recall']:.4f}\n")
                    f.write(f"  F1-Score: {class_metric['f1_score']:.4f}\n")
                    f.write(f"  Support: {class_metric['support']}\n")
                f.write("\n")
        
        print(f"Evaluation report saved to {save_path}")
        return save_path
    
    def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.output_dir}/evaluation_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for task, task_results in results.items():
            if task == 'overall':
                serializable_results[task] = task_results
            else:
                serializable_results[task] = {
                    k: v for k, v in task_results.items() 
                    if k not in ['probabilities']  # Skip large arrays
                }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"Evaluation results saved to {results_file}")


def quick_evaluate(model_path: str,
                  data_path: str,
                  output_dir: str = "reports") -> ModelEvaluator:
    """
    Quick evaluation function.
    
    Args:
        model_path: Path to saved model
        data_path: Path to processed data
        output_dir: Output directory
        
    Returns:
        ModelEvaluator instance
    """
    import tensorflow as tf
    from ..data_preprocessing.preprocess_text import DataProcessor
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load data
    processor = DataProcessor(None)
    data = processor.load_processed_data(data_path)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, data, output_dir=output_dir)
    
    # Run evaluation
    evaluator.evaluate_model()
    
    # Generate plots and reports
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_precision_recall_curves()
    evaluator.generate_evaluation_report()
    
    return evaluator














