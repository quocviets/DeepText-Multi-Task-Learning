# -*- coding: utf-8 -*-
"""
Plotting Utilities Module
=========================

This module provides advanced plotting utilities for the DeepText Multi-Task Learning model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PlottingUtils:
    """
    Advanced plotting utilities for multi-task learning visualization.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', palette: str = 'husl'):
        """
        Initialize plotting utilities.
        
        Args:
            style: Matplotlib style
            palette: Seaborn color palette
        """
        plt.style.use(style)
        sns.set_palette(palette)
        self.colors = sns.color_palette(palette)
    
    def plot_confusion_matrix_heatmap(self, 
                                    cm: np.ndarray,
                                    class_names: List[str],
                                    title: str = "Confusion Matrix",
                                    figsize: Tuple[int, int] = (8, 6),
                                    save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f',
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax,
                   cbar_kws={'label': 'Normalized Count'})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves_multi_class(self, 
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   class_names: List[str],
                                   title: str = "ROC Curves",
                                   figsize: Tuple[int, int] = (10, 8),
                                   save_path: Optional[str] = None) -> None:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            class_names: List of class names
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Binarize labels
        y_true_bin = label_binarize(np.argmax(y_true, axis=1), classes=range(len(class_names)))
        
        # Calculate ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i, class_name in enumerate(class_names):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            ax.plot(fpr[i], tpr[i], 
                   label=f'{class_name} (AUC = {roc_auc[i]:.2f})',
                   linewidth=2)
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.6)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves_multi_class(self, 
                                               y_true: np.ndarray,
                                               y_pred: np.ndarray,
                                               class_names: List[str],
                                               title: str = "Precision-Recall Curves",
                                               figsize: Tuple[int, int] = (10, 8),
                                               save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curves for multi-class classification.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            class_names: List of class names
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Binarize labels
        y_true_bin = label_binarize(np.argmax(y_true, axis=1), classes=range(len(class_names)))
        
        # Calculate PR curve for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        for i, class_name in enumerate(class_names):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred[:, i])
            average_precision[i] = average_precision_score(y_true_bin[:, i], y_pred[:, i])
            
            ax.plot(recall[i], precision[i], 
                   label=f'{class_name} (AP = {average_precision[i]:.2f})',
                   linewidth=2)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curves saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, 
                              metrics_dict: Dict[str, Dict[str, float]],
                              metric_name: str = "f1_macro",
                              title: str = "Metrics Comparison",
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = None) -> None:
        """
        Plot metrics comparison across tasks or models.
        
        Args:
            metrics_dict: Dictionary of metrics for different tasks/models
            metric_name: Name of metric to plot
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract data
        tasks = list(metrics_dict.keys())
        values = [metrics_dict[task].get(metric_name, 0.0) for task in tasks]
        
        # Bar plot
        bars = ax1.bar(tasks, values, color=self.colors[:len(tasks)], alpha=0.7)
        ax1.set_title(f'{metric_name.title()} by Task', fontsize=14, fontweight='bold')
        ax1.set_ylabel(metric_name.title())
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(values, labels=tasks, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'{metric_name.title()} Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")
        
        plt.show()
    
    def plot_training_progress(self, 
                             history: Dict[str, List[float]],
                             metrics: List[str] = None,
                             title: str = "Training Progress",
                             figsize: Tuple[int, int] = (15, 10),
                             save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive training progress.
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
        """
        if metrics is None:
            metrics = ['loss', 'accuracy']
        
        # Determine number of subplots
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Flatten axes if needed
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot training and validation metrics
            if metric in history:
                ax.plot(history[metric], label=f'Training {metric.title()}', linewidth=2)
            if f'val_{metric}' in history:
                ax.plot(history[f'val_{metric}'], label=f'Validation {metric.title()}', linewidth=2)
            
            ax.set_title(f'{metric.title()} Progress', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress saved to {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, 
                              y_data: np.ndarray,
                              class_names: List[str],
                              title: str = "Class Distribution",
                              figsize: Tuple[int, int] = (10, 6),
                              save_path: Optional[str] = None) -> None:
        """
        Plot class distribution.
        
        Args:
            y_data: Labels (one-hot encoded)
            class_names: List of class names
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Convert to class indices
        y_classes = np.argmax(y_data, axis=1)
        
        # Count classes
        class_counts = np.bincount(y_classes, minlength=len(class_names))
        
        # Bar plot
        bars = ax1.bar(range(len(class_counts)), class_counts, 
                      color=self.colors[:len(class_names)], alpha=0.7)
        ax1.set_title('Class Counts', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(class_names)))
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(class_counts, labels=class_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves_smooth(self, 
                                   history: Dict[str, List[float]],
                                   window_size: int = 5,
                                   title: str = "Smoothed Learning Curves",
                                   figsize: Tuple[int, int] = (15, 10),
                                   save_path: Optional[str] = None) -> None:
        """
        Plot smoothed learning curves.
        
        Args:
            history: Training history dictionary
            window_size: Window size for moving average
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Smooth function
        def smooth_curve(values, window_size):
            if len(values) < window_size:
                return values
            return pd.Series(values).rolling(window=window_size, center=True).mean().fillna(values)
        
        # Plot loss
        axes[0, 0].plot(history['loss'], alpha=0.3, label='Raw Training Loss')
        axes[0, 0].plot(smooth_curve(history['loss'], window_size), 
                       label='Smoothed Training Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], alpha=0.3, label='Raw Validation Loss')
        axes[0, 0].plot(smooth_curve(history['val_loss'], window_size), 
                       label='Smoothed Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot accuracy for each task
        tasks = ['emotion', 'hate', 'violence']
        for i, task in enumerate(tasks):
            row = (i + 1) // 2
            col = (i + 1) % 2
            
            if f'{task}_output_accuracy' in history:
                axes[row, col].plot(history[f'{task}_output_accuracy'], alpha=0.3, 
                                  label=f'Raw Training {task.title()}')
                axes[row, col].plot(smooth_curve(history[f'{task}_output_accuracy'], window_size), 
                                  label=f'Smoothed Training {task.title()}', linewidth=2)
                axes[row, col].plot(history[f'val_{task}_output_accuracy'], alpha=0.3, 
                                  label=f'Raw Validation {task.title()}')
                axes[row, col].plot(smooth_curve(history[f'val_{task}_output_accuracy'], window_size), 
                                  label=f'Smoothed Validation {task.title()}', linewidth=2)
                axes[row, col].set_title(f'{task.title()} Accuracy', fontsize=14, fontweight='bold')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Accuracy')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Smoothed learning curves saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, 
                            model_results: Dict[str, Dict[str, float]],
                            metric_name: str = "f1_macro",
                            title: str = "Model Comparison",
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None) -> None:
        """
        Plot model comparison.
        
        Args:
            model_results: Dictionary of results for different models
            metric_name: Name of metric to compare
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract data
        models = list(model_results.keys())
        tasks = ['emotion', 'hate', 'violence']
        
        # Prepare data for plotting
        data = []
        for model in models:
            for task in tasks:
                if task in model_results[model]:
                    value = model_results[model][task].get(metric_name, 0.0)
                    data.append({'Model': model, 'Task': task, 'Value': value})
        
        df = pd.DataFrame(data)
        
        # Bar plot
        sns.barplot(data=df, x='Task', y='Value', hue='Model', ax=ax1)
        ax1.set_title(f'{metric_name.title()} by Model and Task', fontsize=14, fontweight='bold')
        ax1.set_ylabel(metric_name.title())
        ax1.legend(title='Model')
        
        # Heatmap
        pivot_df = df.pivot(index='Model', columns='Task', values='Value')
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
        ax2.set_title(f'{metric_name.title()} Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison saved to {save_path}")
        
        plt.show()
    
    def create_dashboard(self, 
                        history: Dict[str, List[float]],
                        data: Dict[str, Any],
                        evaluation_results: Dict[str, Any],
                        save_path: Optional[str] = None) -> None:
        """
        Create comprehensive dashboard.
        
        Args:
            history: Training history
            data: Processed data
            evaluation_results: Evaluation results
            save_path: Path to save dashboard
        """
        fig = plt.figure(figsize=(20, 24))
        fig.suptitle('DeepText Multi-Task Learning - Comprehensive Dashboard', 
                    fontsize=20, fontweight='bold')
        
        # Training history
        ax1 = plt.subplot(4, 4, 1)
        plt.plot(history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Total Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Task accuracies
        tasks = ['emotion', 'hate', 'violence']
        for i, task in enumerate(tasks):
            ax = plt.subplot(4, 4, i + 2)
            if f'{task}_output_accuracy' in history:
                plt.plot(history[f'{task}_output_accuracy'], 
                        label=f'Training {task.title()}', linewidth=2)
                plt.plot(history[f'val_{task}_output_accuracy'], 
                        label=f'Validation {task.title()}', linewidth=2)
                plt.title(f'{task.title()} Accuracy', fontsize=12, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        # Data distribution
        for i, task in enumerate(tasks):
            ax = plt.subplot(4, 4, i + 5)
            y_train = data[f'y_{task}_train']
            y_train_classes = np.argmax(y_train, axis=1)
            class_counts = np.bincount(y_train_classes)
            class_names = data['class_names'][task]
            
            plt.bar(range(len(class_counts)), class_counts, alpha=0.7)
            plt.title(f'{task.title()} Distribution', fontsize=12, fontweight='bold')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(range(len(class_names)), class_names, rotation=45)
        
        # Evaluation metrics
        ax_metrics = plt.subplot(4, 4, 8)
        ax_metrics.axis('off')
        
        metrics_text = "Final Metrics:\n"
        for task in tasks:
            if task in evaluation_results:
                acc = evaluation_results[task].get('accuracy', 0.0)
                f1 = evaluation_results[task].get('f1_macro', 0.0)
                metrics_text += f"{task.title()}: Acc={acc:.3f}, F1={f1:.3f}\n"
        
        ax_metrics.text(0.1, 0.9, metrics_text, transform=ax_metrics.transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Confusion matrices
        for i, task in enumerate(tasks):
            ax = plt.subplot(4, 4, i + 9)
            if task in evaluation_results:
                cm = np.array(evaluation_results[task]['confusion_matrix'])
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax)
                ax.set_title(f'{task.title()} Confusion Matrix', fontsize=12, fontweight='bold')
        
        # Learning curves
        for i, task in enumerate(tasks):
            ax = plt.subplot(4, 4, i + 12)
            if f'{task}_output_accuracy' in history:
                # Smooth curves
                window = max(1, len(history[f'{task}_output_accuracy']) // 10)
                train_acc = pd.Series(history[f'{task}_output_accuracy']).rolling(window=window).mean()
                val_acc = pd.Series(history[f'val_{task}_output_accuracy']).rolling(window=window).mean()
                
                plt.plot(train_acc, label=f'Training {task.title()}', linewidth=2)
                plt.plot(val_acc, label=f'Validation {task.title()}', linewidth=2)
                plt.title(f'{task.title()} Learning Curve', fontsize=12, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        # Overall summary
        ax_summary = plt.subplot(4, 4, 16)
        ax_summary.axis('off')
        
        summary_text = f"""
        Dataset Summary:
        • Training samples: {len(data['X_train']):,}
        • Validation samples: {len(data['X_val']):,}
        • Test samples: {len(data['X_test']):,}
        • Vocabulary size: {data['vocab_size']:,}
        • Max sequence length: {data['max_length']}
        """
        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        plt.show()


def create_publication_plots(history: Dict[str, List[float]],
                           evaluation_results: Dict[str, Any],
                           output_dir: str = "reports") -> None:
    """
    Create publication-ready plots.
    
    Args:
        history: Training history
        evaluation_results: Evaluation results
        output_dir: Output directory
    """
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("Set2")
    
    plotter = PlottingUtils()
    
    # Create plots
    plotter.plot_training_progress(history, save_path=f"{output_dir}/training_progress.png")
    plotter.plot_metrics_comparison(evaluation_results, save_path=f"{output_dir}/metrics_comparison.png")
    plotter.plot_learning_curves_smooth(history, save_path=f"{output_dir}/learning_curves.png")
    
    print("Publication-ready plots created!")











