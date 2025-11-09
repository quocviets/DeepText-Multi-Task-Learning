# -*- coding: utf-8 -*-
"""
Metrics Utilities Module
========================

This module provides custom metrics and evaluation functions for the DeepText Multi-Task Learning model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """
    Comprehensive metrics calculator for multi-task learning.
    """
    
    def __init__(self, class_names: Dict[str, List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: Class names for each task
        """
        self.class_names = class_names or {
            'emotion': ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo'],
            'hate': ['hate', 'offensive', 'neutral'],
            'violence': ['sex_viol', 'phys_viol', 'no_viol']
        }
    
    def calculate_basic_metrics(self, 
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               task_name: str = "task") -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            task_name: Name of the task
            
        Returns:
            Dictionary of basic metrics
        """
        # Convert to class indices
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Basic metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision_macro = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        
        return {
            f'{task_name}_accuracy': accuracy,
            f'{task_name}_precision_macro': precision_macro,
            f'{task_name}_recall_macro': recall_macro,
            f'{task_name}_f1_macro': f1_macro,
            f'{task_name}_precision_weighted': precision_weighted,
            f'{task_name}_recall_weighted': recall_weighted,
            f'{task_name}_f1_weighted': f1_weighted
        }
    
    def calculate_roc_auc(self, 
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         task_name: str = "task") -> Dict[str, float]:
        """
        Calculate ROC AUC metrics.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            task_name: Name of the task
            
        Returns:
            Dictionary of ROC AUC metrics
        """
        y_true_classes = np.argmax(y_true, axis=1)
        class_names = self.class_names.get(task_name, [])
        
        try:
            if len(class_names) > 2:
                # Multi-class ROC AUC
                y_true_bin = label_binarize(y_true_classes, classes=range(len(class_names)))
                roc_auc_macro = roc_auc_score(y_true_bin, y_pred, multi_class='ovr', average='macro')
                roc_auc_weighted = roc_auc_score(y_true_bin, y_pred, multi_class='ovr', average='weighted')
                
                # Per-class ROC AUC
                roc_auc_per_class = []
                for i in range(len(class_names)):
                    try:
                        auc = roc_auc_score(y_true_bin[:, i], y_pred[:, i])
                        roc_auc_per_class.append(auc)
                    except:
                        roc_auc_per_class.append(0.0)
                
                return {
                    f'{task_name}_roc_auc_macro': roc_auc_macro,
                    f'{task_name}_roc_auc_weighted': roc_auc_weighted,
                    f'{task_name}_roc_auc_per_class': roc_auc_per_class
                }
            else:
                # Binary ROC AUC
                roc_auc = roc_auc_score(y_true_classes, y_pred[:, 1])
                return {f'{task_name}_roc_auc': roc_auc}
        except:
            return {f'{task_name}_roc_auc': 0.0}
    
    def calculate_precision_recall_auc(self, 
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     task_name: str = "task") -> Dict[str, float]:
        """
        Calculate Precision-Recall AUC metrics.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            task_name: Name of the task
            
        Returns:
            Dictionary of PR AUC metrics
        """
        y_true_classes = np.argmax(y_true, axis=1)
        class_names = self.class_names.get(task_name, [])
        
        try:
            if len(class_names) > 2:
                # Multi-class PR AUC
                y_true_bin = label_binarize(y_true_classes, classes=range(len(class_names)))
                pr_auc_macro = average_precision_score(y_true_bin, y_pred, average='macro')
                pr_auc_weighted = average_precision_score(y_true_bin, y_pred, average='weighted')
                
                # Per-class PR AUC
                pr_auc_per_class = []
                for i in range(len(class_names)):
                    try:
                        auc = average_precision_score(y_true_bin[:, i], y_pred[:, i])
                        pr_auc_per_class.append(auc)
                    except:
                        pr_auc_per_class.append(0.0)
                
                return {
                    f'{task_name}_pr_auc_macro': pr_auc_macro,
                    f'{task_name}_pr_auc_weighted': pr_auc_weighted,
                    f'{task_name}_pr_auc_per_class': pr_auc_per_class
                }
            else:
                # Binary PR AUC
                pr_auc = average_precision_score(y_true_classes, y_pred[:, 1])
                return {f'{task_name}_pr_auc': pr_auc}
        except:
            return {f'{task_name}_pr_auc': 0.0}
    
    def calculate_per_class_metrics(self, 
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   task_name: str = "task") -> Dict[str, Any]:
        """
        Calculate per-class metrics.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            task_name: Name of the task
            
        Returns:
            Dictionary of per-class metrics
        """
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        class_names = self.class_names.get(task_name, [])
        
        # Per-class precision, recall, f1
        precision_per_class = precision_score(y_true_classes, y_pred_classes, average=None, zero_division=0)
        recall_per_class = recall_score(y_true_classes, y_pred_classes, average=None, zero_division=0)
        f1_per_class = f1_score(y_true_classes, y_pred_classes, average=None, zero_division=0)
        
        # Support (number of true instances for each class)
        support = np.bincount(y_true_classes, minlength=len(class_names))
        
        # Create per-class metrics dictionary
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[f'{task_name}_{class_name}'] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support[i])
            }
        
        return per_class_metrics
    
    def calculate_confusion_matrix_metrics(self, 
                                         y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         task_name: str = "task") -> Dict[str, Any]:
        """
        Calculate confusion matrix and related metrics.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            task_name: Name of the task
            
        Returns:
            Dictionary of confusion matrix metrics
        """
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        class_names = self.class_names.get(task_name, [])
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        return {
            f'{task_name}_confusion_matrix': cm.tolist(),
            f'{task_name}_confusion_matrix_normalized': cm_normalized.tolist(),
            f'{task_name}_per_class_accuracy': per_class_accuracy.tolist(),
            f'{task_name}_class_names': class_names
        }
    
    def calculate_multi_task_metrics(self, 
                                   results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate overall multi-task metrics.
        
        Args:
            results: Results dictionary from individual tasks
            
        Returns:
            Dictionary of multi-task metrics
        """
        tasks = ['emotion', 'hate', 'violence']
        
        # Extract metrics for each task
        metrics = {}
        for task in tasks:
            if task in results:
                task_results = results[task]
                metrics[f'{task}_accuracy'] = task_results.get('accuracy', 0.0)
                metrics[f'{task}_f1_macro'] = task_results.get('f1_macro', 0.0)
                metrics[f'{task}_f1_weighted'] = task_results.get('f1_weighted', 0.0)
                metrics[f'{task}_roc_auc'] = task_results.get('roc_auc', 0.0)
        
        # Calculate overall metrics
        overall_metrics = {}
        
        # Average accuracy across tasks
        accuracies = [metrics.get(f'{task}_accuracy', 0.0) for task in tasks]
        overall_metrics['overall_accuracy'] = np.mean(accuracies)
        overall_metrics['overall_accuracy_std'] = np.std(accuracies)
        
        # Average F1 macro across tasks
        f1_macros = [metrics.get(f'{task}_f1_macro', 0.0) for task in tasks]
        overall_metrics['overall_f1_macro'] = np.mean(f1_macros)
        overall_metrics['overall_f1_macro_std'] = np.std(f1_macros)
        
        # Average F1 weighted across tasks
        f1_weighted = [metrics.get(f'{task}_f1_weighted', 0.0) for task in tasks]
        overall_metrics['overall_f1_weighted'] = np.mean(f1_weighted)
        overall_metrics['overall_f1_weighted_std'] = np.std(f1_weighted)
        
        # Average ROC AUC across tasks
        roc_aucs = [metrics.get(f'{task}_roc_auc', 0.0) for task in tasks]
        overall_metrics['overall_roc_auc'] = np.mean(roc_aucs)
        overall_metrics['overall_roc_auc_std'] = np.std(roc_aucs)
        
        return overall_metrics
    
    def generate_classification_report(self, 
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     task_name: str = "task") -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            task_name: Name of the task
            
        Returns:
            Classification report string
        """
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        class_names = self.class_names.get(task_name, [])
        
        return classification_report(
            y_true_classes, y_pred_classes,
            target_names=class_names,
            zero_division=0
        )
    
    def calculate_all_metrics(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             task_name: str = "task") -> Dict[str, Any]:
        """
        Calculate all available metrics for a task.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            task_name: Name of the task
            
        Returns:
            Dictionary of all metrics
        """
        all_metrics = {}
        
        # Basic metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred, task_name)
        all_metrics.update(basic_metrics)
        
        # ROC AUC metrics
        roc_auc_metrics = self.calculate_roc_auc(y_true, y_pred, task_name)
        all_metrics.update(roc_auc_metrics)
        
        # PR AUC metrics
        pr_auc_metrics = self.calculate_precision_recall_auc(y_true, y_pred, task_name)
        all_metrics.update(pr_auc_metrics)
        
        # Per-class metrics
        per_class_metrics = self.calculate_per_class_metrics(y_true, y_pred, task_name)
        all_metrics.update(per_class_metrics)
        
        # Confusion matrix metrics
        cm_metrics = self.calculate_confusion_matrix_metrics(y_true, y_pred, task_name)
        all_metrics.update(cm_metrics)
        
        # Classification report
        all_metrics[f'{task_name}_classification_report'] = self.generate_classification_report(
            y_true, y_pred, task_name
        )
        
        return all_metrics


def calculate_ensemble_metrics(predictions_list: List[np.ndarray],
                             y_true: np.ndarray,
                             method: str = 'average') -> Dict[str, Any]:
    """
    Calculate metrics for ensemble predictions.
    
    Args:
        predictions_list: List of prediction arrays from different models
        y_true: True labels (one-hot encoded)
        method: Ensemble method ('average', 'voting', 'weighted')
        
    Returns:
        Dictionary of ensemble metrics
    """
    if method == 'average':
        # Average predictions
        ensemble_pred = np.mean(predictions_list, axis=0)
    elif method == 'voting':
        # Majority voting
        pred_classes = [np.argmax(pred, axis=1) for pred in predictions_list]
        ensemble_pred_classes = []
        for i in range(len(pred_classes[0])):
            votes = [pred[i] for pred in pred_classes]
            ensemble_pred_classes.append(max(set(votes), key=votes.count))
        ensemble_pred_classes = np.array(ensemble_pred_classes)
        
        # Convert back to one-hot
        num_classes = y_true.shape[1]
        ensemble_pred = np.eye(num_classes)[ensemble_pred_classes]
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(y_true, ensemble_pred, "ensemble")
    
    return metrics


def calculate_confidence_metrics(y_pred: np.ndarray,
                               y_true: np.ndarray,
                               confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Calculate confidence-based metrics.
    
    Args:
        y_pred: Predicted probabilities
        y_true: True labels (one-hot encoded)
        confidence_threshold: Confidence threshold for predictions
        
    Returns:
        Dictionary of confidence metrics
    """
    # Calculate confidence (max probability)
    confidence = np.max(y_pred, axis=1)
    
    # Filter by confidence threshold
    high_conf_mask = confidence >= confidence_threshold
    low_conf_mask = confidence < confidence_threshold
    
    # Calculate accuracy for high and low confidence predictions
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    high_conf_accuracy = accuracy_score(
        y_true_classes[high_conf_mask], 
        y_pred_classes[high_conf_mask]
    ) if np.any(high_conf_mask) else 0.0
    
    low_conf_accuracy = accuracy_score(
        y_true_classes[low_conf_mask], 
        y_pred_classes[low_conf_mask]
    ) if np.any(low_conf_mask) else 0.0
    
    return {
        'high_confidence_accuracy': high_conf_accuracy,
        'low_confidence_accuracy': low_conf_accuracy,
        'high_confidence_ratio': np.mean(high_conf_mask),
        'low_confidence_ratio': np.mean(low_conf_mask),
        'mean_confidence': np.mean(confidence),
        'confidence_std': np.std(confidence)
    }














