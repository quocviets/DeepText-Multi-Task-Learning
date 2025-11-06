# -*- coding: utf-8 -*-
"""
Training Visualization Module
============================

This module provides visualization utilities for training progress and model analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrainingVisualizer:
    """
    Training visualization utilities.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_history(self, 
                             history: Dict[str, List[float]],
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot comprehensive training history.
        
        Args:
            history: Training history dictionary
            save_path: Path to save plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Plot total loss
        axes[0, 0].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot accuracy for each task
        tasks = ['emotion', 'hate', 'violence']
        for i, task in enumerate(tasks):
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            if f'{task}_output_accuracy' in history:
                axes[row, col].plot(history[f'{task}_output_accuracy'], 
                                  label=f'Training {task.title()} Accuracy', 
                                  linewidth=2)
                axes[row, col].plot(history[f'val_{task}_output_accuracy'], 
                                  label=f'Validation {task.title()} Accuracy', 
                                  linewidth=2)
                axes[row, col].set_title(f'{task.title()} Accuracy', 
                                       fontsize=14, fontweight='bold')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Accuracy')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_loss_curves(self, 
                        history: Dict[str, List[float]],
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot detailed loss curves for each task.
        
        Args:
            history: Training history dictionary
            save_path: Path to save plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Loss Curves by Task', fontsize=16, fontweight='bold')
        
        # Total loss
        axes[0, 0].plot(history['loss'], label='Training', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Task-specific losses
        tasks = ['emotion', 'hate', 'violence']
        for i, task in enumerate(tasks):
            row = (i + 1) // 2
            col = (i + 1) % 2
            
            if f'{task}_output_loss' in history:
                axes[row, col].plot(history[f'{task}_output_loss'], 
                                  label=f'Training {task.title()}', 
                                  linewidth=2)
                axes[row, col].plot(history[f'val_{task}_output_loss'], 
                                  label=f'Validation {task.title()}', 
                                  linewidth=2)
                axes[row, col].set_title(f'{task.title()} Loss', 
                                       fontsize=14, fontweight='bold')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Loss')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, 
                            history: Dict[str, List[float]],
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot learning curves showing convergence.
        
        Args:
            history: Training history dictionary
            save_path: Path to save plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Learning Curves', fontsize=16, fontweight='bold')
        
        tasks = ['emotion', 'hate', 'violence']
        
        for i, task in enumerate(tasks):
            if f'{task}_output_accuracy' in history:
                # Calculate moving average
                window = max(1, len(history[f'{task}_output_accuracy']) // 10)
                train_acc = pd.Series(history[f'{task}_output_accuracy']).rolling(window=window).mean()
                val_acc = pd.Series(history[f'val_{task}_output_accuracy']).rolling(window=window).mean()
                
                axes[i].plot(train_acc, label=f'Training {task.title()}', linewidth=2)
                axes[i].plot(val_acc, label=f'Validation {task.title()}', linewidth=2)
                axes[i].set_title(f'{task.title()} Learning Curve', 
                                fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Accuracy')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, 
                              history: Dict[str, List[float]],
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot metrics comparison across tasks.
        
        Args:
            history: Training history dictionary
            save_path: Path to save plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Metrics Comparison Across Tasks', fontsize=16, fontweight='bold')
        
        tasks = ['emotion', 'hate', 'violence']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Final accuracies
        final_accuracies = []
        for task in tasks:
            if f'{task}_output_accuracy' in history:
                final_acc = history[f'{task}_output_accuracy'][-1]
                final_accuracies.append(final_acc)
            else:
                final_accuracies.append(0)
        
        axes[0, 0].bar(tasks, final_accuracies, color=colors, alpha=0.7)
        axes[0, 0].set_title('Final Training Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Final validation accuracies
        final_val_accuracies = []
        for task in tasks:
            if f'val_{task}_output_accuracy' in history:
                final_val_acc = history[f'val_{task}_output_accuracy'][-1]
                final_val_accuracies.append(final_val_acc)
            else:
                final_val_accuracies.append(0)
        
        axes[0, 1].bar(tasks, final_val_accuracies, color=colors, alpha=0.7)
        axes[0, 1].set_title('Final Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        
        # Training vs Validation comparison
        x = np.arange(len(tasks))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, final_accuracies, width, label='Training', alpha=0.7)
        axes[1, 0].bar(x + width/2, final_val_accuracies, width, label='Validation', alpha=0.7)
        axes[1, 0].set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(tasks)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
        
        # Overfitting analysis
        overfitting = np.array(final_accuracies) - np.array(final_val_accuracies)
        axes[1, 1].bar(tasks, overfitting, color=colors, alpha=0.7)
        axes[1, 1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Training - Validation Accuracy')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_data_distribution(self, 
                              data: Dict[str, Any],
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot data distribution across tasks.
        
        Args:
            data: Processed data dictionary
            save_path: Path to save plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        tasks = ['emotion', 'hate', 'violence']
        class_names = data.get('class_names', {
            'emotion': ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo'],
            'hate': ['hate', 'offensive', 'neutral'],
            'violence': ['sex_viol', 'phys_viol', 'no_viol']
        })
        
        for i, task in enumerate(tasks):
            # Training data distribution
            y_train = data[f'y_{task}_train']
            y_train_classes = np.argmax(y_train, axis=1)
            class_counts = np.bincount(y_train_classes)
            class_labels = class_names[task]
            
            axes[0, i].bar(range(len(class_counts)), class_counts, alpha=0.7)
            axes[0, i].set_title(f'{task.title()} Training Distribution', 
                               fontsize=14, fontweight='bold')
            axes[0, i].set_xlabel('Class')
            axes[0, i].set_ylabel('Count')
            axes[0, i].set_xticks(range(len(class_labels)))
            axes[0, i].set_xticklabels(class_labels, rotation=45)
            
            # Validation data distribution
            y_val = data[f'y_{task}_val']
            y_val_classes = np.argmax(y_val, axis=1)
            val_class_counts = np.bincount(y_val_classes)
            
            axes[1, i].bar(range(len(val_class_counts)), val_class_counts, alpha=0.7)
            axes[1, i].set_title(f'{task.title()} Validation Distribution', 
                               fontsize=14, fontweight='bold')
            axes[1, i].set_xlabel('Class')
            axes[1, i].set_ylabel('Count')
            axes[1, i].set_xticks(range(len(class_labels)))
            axes[1, i].set_xticklabels(class_labels, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Data distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_architecture(self, 
                               model,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (20, 15)) -> None:
        """
        Plot model architecture diagram.
        
        Args:
            model: Keras model
            save_path: Path to save plot
            figsize: Figure size
        """
        try:
            from tensorflow.keras.utils import plot_model
            
            plot_model(
                model,
                to_file=save_path or f"{self.output_dir}/model_architecture.png",
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=300
            )
            print(f"Model architecture saved to {save_path or f'{self.output_dir}/model_architecture.png'}")
        except ImportError:
            print("Graphviz not available. Install with: pip install graphviz")
        except Exception as e:
            print(f"Error plotting model architecture: {e}")
    
    def create_training_dashboard(self, 
                                 history: Dict[str, List[float]],
                                 data: Dict[str, Any],
                                 model,
                                 save_path: Optional[str] = None) -> None:
        """
        Create comprehensive training dashboard.
        
        Args:
            history: Training history dictionary
            data: Processed data dictionary
            model: Trained model
            save_path: Path to save dashboard
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.output_dir}/training_dashboard_{timestamp}.png"
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        fig.suptitle('DeepText Multi-Task Learning - Training Dashboard', 
                    fontsize=20, fontweight='bold')
        
        # Training history
        ax1 = plt.subplot(4, 3, 1)
        plt.plot(history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Total Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Task accuracies
        tasks = ['emotion', 'hate', 'violence']
        for i, task in enumerate(tasks):
            ax = plt.subplot(4, 3, i + 2)
            if f'{task}_output_accuracy' in history:
                plt.plot(history[f'{task}_output_accuracy'], 
                        label=f'Training {task.title()}', linewidth=2)
                plt.plot(history[f'val_{task}_output_accuracy'], 
                        label=f'Validation {task.title()}', linewidth=2)
                plt.title(f'{task.title()} Accuracy', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        # Data distribution
        class_names = data.get('class_names', {
            'emotion': ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo'],
            'hate': ['hate', 'offensive', 'neutral'],
            'violence': ['sex_viol', 'phys_viol', 'no_viol']
        })
        
        for i, task in enumerate(tasks):
            ax = plt.subplot(4, 3, i + 5)
            y_train = data[f'y_{task}_train']
            y_train_classes = np.argmax(y_train, axis=1)
            class_counts = np.bincount(y_train_classes)
            class_labels = class_names[task]
            
            plt.bar(range(len(class_counts)), class_counts, alpha=0.7)
            plt.title(f'{task.title()} Training Distribution', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(range(len(class_labels)), class_labels, rotation=45)
        
        # Model summary
        ax_summary = plt.subplot(4, 3, 8)
        ax_summary.axis('off')
        summary_text = f"""
        Model Summary:
        • Total Parameters: {model.count_params():,}
        • Trainable Parameters: {sum([np.prod(w.shape) for w in model.trainable_weights]):,}
        • Non-trainable Parameters: {model.count_params() - sum([np.prod(w.shape) for w in model.trainable_weights]):,}
        • Input Shape: {model.input_shape}
        • Output Shapes: {[output.shape for output in model.outputs]}
        """
        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        # Final metrics
        ax_metrics = plt.subplot(4, 3, 9)
        ax_metrics.axis('off')
        
        final_metrics = []
        for task in tasks:
            if f'{task}_output_accuracy' in history:
                train_acc = history[f'{task}_output_accuracy'][-1]
                val_acc = history[f'val_{task}_output_accuracy'][-1]
                final_metrics.append(f"{task.title()}: {train_acc:.3f} / {val_acc:.3f}")
        
        metrics_text = "Final Accuracies (Train/Val):\n" + "\n".join(final_metrics)
        ax_metrics.text(0.1, 0.9, metrics_text, transform=ax_metrics.transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        # Learning curves
        for i, task in enumerate(tasks):
            ax = plt.subplot(4, 3, i + 10)
            if f'{task}_output_accuracy' in history:
                # Calculate moving average
                window = max(1, len(history[f'{task}_output_accuracy']) // 10)
                train_acc = pd.Series(history[f'{task}_output_accuracy']).rolling(window=window).mean()
                val_acc = pd.Series(history[f'val_{task}_output_accuracy']).rolling(window=window).mean()
                
                plt.plot(train_acc, label=f'Training {task.title()}', linewidth=2)
                plt.plot(val_acc, label=f'Validation {task.title()}', linewidth=2)
                plt.title(f'{task.title()} Learning Curve', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training dashboard saved to {save_path}")
        plt.show()


def create_visualization_report(history_path: str,
                              data_path: str,
                              model_path: str,
                              output_dir: str = "reports") -> None:
    """
    Create comprehensive visualization report.
    
    Args:
        history_path: Path to training history JSON
        data_path: Path to processed data
        model_path: Path to trained model
        output_dir: Output directory
    """
    import tensorflow as tf
    from .evaluate import ModelEvaluator
    
    # Load data
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load data
    from ..data_preprocessing.preprocess_text import DataProcessor
    processor = DataProcessor(None)
    data = processor.load_processed_data(data_path)
    
    # Create visualizer
    visualizer = TrainingVisualizer(output_dir)
    
    # Generate all visualizations
    visualizer.plot_training_history(history)
    visualizer.plot_loss_curves(history)
    visualizer.plot_learning_curves(history)
    visualizer.plot_metrics_comparison(history)
    visualizer.plot_data_distribution(data)
    visualizer.plot_model_architecture(model)
    visualizer.create_training_dashboard(history, data, model)
    
    print("Visualization report completed!")











