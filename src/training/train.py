# -*- coding: utf-8 -*-
"""
Training Pipeline Module
========================

This module provides training utilities for the DeepText Multi-Task Learning model.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class TrainingPipeline:
    """
    Complete training pipeline for DeepText Multi-Task Learning model.
    """
    
    def __init__(self, 
                 model,
                 data: Dict[str, Any],
                 output_dir: str = "checkpoints",
                 model_name: str = "deeptext_multitask"):
        """
        Initialize training pipeline.
        
        Args:
            model: Compiled Keras model
            data: Processed data dictionary
            output_dir: Directory to save checkpoints and results
            model_name: Name for saved model files
        """
        self.model = model
        self.data = data
        self.output_dir = output_dir
        self.model_name = model_name
        self.history = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        
    def calculate_class_weights(self) -> Dict[str, Dict[int, float]]:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            Dictionary of class weights for each task
        """
        class_weights = {}
        
        # Emotion task
        y_emotion = self.data['y_emotion_train']
        emotion_classes = np.argmax(y_emotion, axis=1)
        emotion_weights = compute_class_weight(
            'balanced',
            classes=np.unique(emotion_classes),
            y=emotion_classes
        )
        class_weights['emotion'] = dict(zip(np.unique(emotion_classes), emotion_weights))
        
        # Hate task
        y_hate = self.data['y_hate_train']
        hate_classes = np.argmax(y_hate, axis=1)
        hate_weights = compute_class_weight(
            'balanced',
            classes=np.unique(hate_classes),
            y=hate_classes
        )
        class_weights['hate'] = dict(zip(np.unique(hate_classes), hate_weights))
        
        # Violence task
        y_violence = self.data['y_violence_train']
        violence_classes = np.argmax(y_violence, axis=1)
        violence_weights = compute_class_weight(
            'balanced',
            classes=np.unique(violence_classes),
            y=violence_classes
        )
        class_weights['violence'] = dict(zip(np.unique(violence_classes), violence_weights))
        
        print("Class weights calculated:")
        for task, weights in class_weights.items():
            print(f"  {task}: {weights}")
        
        return class_weights
    
    def create_callbacks(self, 
                        monitor: str = 'val_loss',
                        patience: int = 10,
                        min_delta: float = 0.001,
                        factor: float = 0.5,
                        min_lr: float = 1e-7) -> List:
        """
        Create training callbacks.
        
        Args:
            monitor: Metric to monitor
            patience: Early stopping patience
            min_delta: Minimum change to qualify as improvement
            factor: Learning rate reduction factor
            min_lr: Minimum learning rate
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_scheduler = ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience//2,
            min_lr=min_lr,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Model checkpoint
        checkpoint_path = f"{self.output_dir}/{self.model_name}_best_{{epoch:02d}}_{{val_loss:.3f}}.h5"
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        return callbacks
    
    def train(self,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.0,
              class_weights: Optional[Dict[str, Dict[int, float]]] = None,
              loss_weights: Optional[Dict[str, float]] = None,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            class_weights: Class weights for imbalanced data
            loss_weights: Loss weights for different tasks
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        # Prepare training data
        X_train = self.data['X_train']
        y_train = {
            'emotion_output': self.data['y_emotion_train'],
            'hate_output': self.data['y_hate_train'],
            'violence_output': self.data['y_violence_train']
        }
        
        # Prepare validation data
        X_val = self.data['X_val']
        y_val = {
            'emotion_output': self.data['y_emotion_val'],
            'hate_output': self.data['y_hate_val'],
            'violence_output': self.data['y_violence_val']
        }
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        print(f"Starting training for {epochs} epochs...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=verbose
        )
        
        print("Training completed!")
        return self.history.history
    
    def evaluate(self, 
                 test_data: Optional[Tuple] = None,
                 save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Optional test data tuple (X_test, y_test)
            save_results: Whether to save evaluation results
            
        Returns:
            Evaluation results dictionary
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
        
        # Evaluate model
        print("Evaluating model on test data...")
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Get predictions
        predictions = self.model.predict(X_test, verbose=1)
        
        # Calculate metrics for each task
        evaluation_results = {}
        task_names = ['emotion', 'hate', 'violence']
        
        for i, task in enumerate(task_names):
            y_true = y_test[f'{task}_output']
            y_pred = predictions[i]
            
            # Convert predictions to class indices
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_true, axis=1)
            
            # Calculate metrics
            accuracy = np.mean(y_pred_classes == y_true_classes)
            
            # Classification report
            class_names = self.data['class_names'][task]
            report = classification_report(
                y_true_classes, y_pred_classes,
                target_names=class_names,
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            
            evaluation_results[task] = {
                'accuracy': accuracy,
                'loss': results[i + 1],  # Skip total loss
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'predictions': y_pred_classes.tolist(),
                'true_labels': y_true_classes.tolist()
            }
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"{self.output_dir}/evaluation_results_{timestamp}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for task, results in evaluation_results.items():
                serializable_results[task] = {
                    'accuracy': float(results['accuracy']),
                    'loss': float(results['loss']),
                    'classification_report': results['classification_report'],
                    'confusion_matrix': results['confusion_matrix'],
                    'predictions': results['predictions'],
                    'true_labels': results['true_labels']
                }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            print(f"Evaluation results saved to {results_file}")
        
        return evaluation_results
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save model
            
        Returns:
            Path where model was saved
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.output_dir}/{self.model_name}_{timestamp}.h5"
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def save_training_history(self, filepath: Optional[str] = None) -> str:
        """
        Save training history.
        
        Args:
            filepath: Path to save history
            
        Returns:
            Path where history was saved
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.output_dir}/training_history_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history.history, f, ensure_ascii=False, indent=2)
        
        print(f"Training history saved to {filepath}")
        return filepath
    
    def plot_training_history(self, 
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save plot
            figsize: Figure size
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        history = self.history.history
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Training History', fontsize=16)
        
        # Plot loss
        axes[0, 0].plot(history['loss'], label='Training Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot accuracy for each task
        tasks = ['emotion', 'hate', 'violence']
        for i, task in enumerate(tasks):
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            if f'{task}_output_accuracy' in history:
                axes[row, col].plot(history[f'{task}_output_accuracy'], 
                                  label=f'Training {task.title()} Accuracy')
                axes[row, col].plot(history[f'val_{task}_output_accuracy'], 
                                  label=f'Validation {task.title()} Accuracy')
                axes[row, col].set_title(f'{task.title()} Accuracy')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Accuracy')
                axes[row, col].legend()
                axes[row, col].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def generate_model_summary(self, filepath: Optional[str] = None) -> str:
        """
        Generate and save model summary.
        
        Args:
            filepath: Path to save summary
            
        Returns:
            Path where summary was saved
        """
        if filepath is None:
            filepath = f"{self.output_dir}/model_summary.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("DeepText Multi-Task Learning Model Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Model architecture
            f.write("Model Architecture:\n")
            f.write("-" * 20 + "\n")
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))
            
            # Data information
            f.write(f"\nData Information:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Training samples: {len(self.data['X_train'])}\n")
            f.write(f"Validation samples: {len(self.data['X_val'])}\n")
            f.write(f"Test samples: {len(self.data['X_test'])}\n")
            f.write(f"Vocabulary size: {self.data['vocab_size']}\n")
            f.write(f"Max sequence length: {self.data['max_length']}\n")
            
            # Class information
            f.write(f"\nClass Information:\n")
            f.write("-" * 20 + "\n")
            for task, classes in self.data['class_names'].items():
                f.write(f"{task.title()}: {classes}\n")
        
        print(f"Model summary saved to {filepath}")
        return filepath


def quick_train(csv_path: str,
                model_class,
                output_dir: str = "checkpoints",
                epochs: int = 50,
                batch_size: int = 32,
                max_length: int = 100,
                vocab_size: int = 10000,
                **model_kwargs) -> TrainingPipeline:
    """
    Quick training function.
    
    Args:
        csv_path: Path to CSV file
        model_class: Model class to use
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        max_length: Maximum sequence length
        vocab_size: Vocabulary size
        **model_kwargs: Additional model parameters
        
    Returns:
        Trained TrainingPipeline instance
    """
    from ..data_preprocessing.preprocess_text import quick_process_data
    
    # Process data
    print("Processing data...")
    data, preprocessor, processor = quick_process_data(
        csv_path, max_length=max_length, vocab_size=vocab_size
    )
    
    # Create model
    print("Creating model...")
    model = model_class(
        vocab_size=data['vocab_size'],
        max_length=data['max_length'],
        **model_kwargs
    )
    model.build_model()
    model.compile_model()
    
    # Create training pipeline
    pipeline = TrainingPipeline(model.model, data, output_dir)
    
    # Calculate class weights
    class_weights = pipeline.calculate_class_weights()
    
    # Train model
    print("Training model...")
    pipeline.train(epochs=epochs, batch_size=batch_size, class_weights=class_weights)
    
    # Evaluate model
    print("Evaluating model...")
    pipeline.evaluate()
    
    # Save everything
    pipeline.save_model()
    pipeline.save_training_history()
    pipeline.plot_training_history()
    pipeline.generate_model_summary()
    
    return pipeline











