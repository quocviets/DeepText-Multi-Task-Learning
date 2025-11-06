# -*- coding: utf-8 -*-
"""
Configuration Module
====================

This module provides configuration settings and constants for the DeepText Multi-Task Learning model.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    # Model architecture
    vocab_size: int = 10000
    max_length: int = 100
    embedding_dim: int = 128
    lstm_units: int = 64
    dropout_rate: float = 0.3
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.1
    test_split: float = 0.2
    
    # Optimization
    use_attention: bool = True
    use_batch_norm: bool = True
    use_pretrained_embedding: bool = False
    
    # Class names
    emotion_classes: List[str] = None
    hate_classes: List[str] = None
    violence_classes: List[str] = None
    
    def __post_init__(self):
        """Set default class names if not provided."""
        if self.emotion_classes is None:
            self.emotion_classes = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
        if self.hate_classes is None:
            self.hate_classes = ['hate', 'offensive', 'neutral']
        if self.violence_classes is None:
            self.violence_classes = ['sex_viol', 'phys_viol', 'no_viol']


@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # Text preprocessing
    min_word_count: int = 2
    remove_punctuation: bool = True
    lowercase: bool = True
    
    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    
    # Column names
    text_column: str = "text"
    label_columns: List[str] = None
    
    def __post_init__(self):
        """Set default label columns if not provided."""
        if self.label_columns is None:
            self.label_columns = ["emotion", "hate", "violence"]


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Output directories
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    reports_dir: str = "reports"
    
    # Callbacks
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    reduce_lr_min_lr: float = 1e-7
    
    # Monitoring
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    
    # Logging
    verbose: int = 1
    save_freq: str = "epoch"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    
    # Output directories
    confusion_matrices_dir: str = "reports/confusion_matrices"
    roc_curves_dir: str = "reports/roc_curves"
    precision_recall_curves_dir: str = "reports/precision_recall_curves"
    
    # Metrics
    confidence_threshold: float = 0.5
    save_predictions: bool = True
    save_probabilities: bool = False  # Large files, only if needed
    
    # Visualization
    figsize: tuple = (10, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8"
    palette: str = "husl"


class Config:
    """Main configuration class."""
    
    def __init__(self, 
                 model_config: ModelConfig = None,
                 data_config: DataConfig = None,
                 training_config: TrainingConfig = None,
                 evaluation_config: EvaluationConfig = None):
        """
        Initialize configuration.
        
        Args:
            model_config: Model configuration
            data_config: Data configuration
            training_config: Training configuration
            evaluation_config: Evaluation configuration
        """
        self.model = model_config or ModelConfig()
        self.data = data_config or DataConfig()
        self.training = training_config or TrainingConfig()
        self.evaluation = evaluation_config or EvaluationConfig()
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.training.checkpoints_dir,
            self.training.logs_dir,
            self.training.reports_dir,
            self.evaluation.confusion_matrices_dir,
            self.evaluation.roc_curves_dir,
            self.evaluation.precision_recall_curves_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'vocab_size': self.model.vocab_size,
            'max_length': self.model.max_length,
            'embedding_dim': self.model.embedding_dim,
            'lstm_units': self.model.lstm_units,
            'dropout_rate': self.model.dropout_rate,
            'use_attention': self.model.use_attention,
            'use_batch_norm': self.model.use_batch_norm,
            'use_pretrained_embedding': self.model.use_pretrained_embedding
        }
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training parameters."""
        return {
            'epochs': self.model.epochs,
            'batch_size': self.model.batch_size,
            'learning_rate': self.model.learning_rate,
            'validation_split': self.model.validation_split,
            'test_split': self.model.test_split
        }
    
    def get_class_names(self) -> Dict[str, List[str]]:
        """Get class names for all tasks."""
        return {
            'emotion': self.model.emotion_classes,
            'hate': self.model.hate_classes,
            'violence': self.model.violence_classes
        }
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights for different tasks."""
        return {
            'emotion_output': 1.0,
            'hate_output': 1.0,
            'violence_output': 1.0
        }
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.data, key):
                setattr(self.data, key, value)
            elif hasattr(self.training, key):
                setattr(self.training, key, value)
            elif hasattr(self.evaluation, key):
                setattr(self.evaluation, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}'")
    
    def save_config(self, filepath: str):
        """Save configuration to file."""
        import json
        
        config_dict = {
            'model': {
                'vocab_size': self.model.vocab_size,
                'max_length': self.model.max_length,
                'embedding_dim': self.model.embedding_dim,
                'lstm_units': self.model.lstm_units,
                'dropout_rate': self.model.dropout_rate,
                'use_attention': self.model.use_attention,
                'use_batch_norm': self.model.use_batch_norm,
                'use_pretrained_embedding': self.model.use_pretrained_embedding,
                'emotion_classes': self.model.emotion_classes,
                'hate_classes': self.model.hate_classes,
                'violence_classes': self.model.violence_classes
            },
            'data': {
                'min_word_count': self.data.min_word_count,
                'remove_punctuation': self.data.remove_punctuation,
                'lowercase': self.data.lowercase,
                'raw_data_dir': self.data.raw_data_dir,
                'processed_data_dir': self.data.processed_data_dir,
                'text_column': self.data.text_column,
                'label_columns': self.data.label_columns
            },
            'training': {
                'checkpoints_dir': self.training.checkpoints_dir,
                'logs_dir': self.training.logs_dir,
                'reports_dir': self.training.reports_dir,
                'early_stopping_patience': self.training.early_stopping_patience,
                'early_stopping_min_delta': self.training.early_stopping_min_delta,
                'reduce_lr_patience': self.training.reduce_lr_patience,
                'reduce_lr_factor': self.training.reduce_lr_factor,
                'reduce_lr_min_lr': self.training.reduce_lr_min_lr,
                'monitor_metric': self.training.monitor_metric,
                'monitor_mode': self.training.monitor_mode,
                'verbose': self.training.verbose,
                'save_freq': self.training.save_freq
            },
            'evaluation': {
                'confusion_matrices_dir': self.evaluation.confusion_matrices_dir,
                'roc_curves_dir': self.evaluation.roc_curves_dir,
                'precision_recall_curves_dir': self.evaluation.precision_recall_curves_dir,
                'confidence_threshold': self.evaluation.confidence_threshold,
                'save_predictions': self.evaluation.save_predictions,
                'save_probabilities': self.evaluation.save_probabilities,
                'figsize': self.evaluation.figsize,
                'dpi': self.evaluation.dpi,
                'style': self.evaluation.style,
                'palette': self.evaluation.palette
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from file."""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Create config objects
        model_config = ModelConfig(**config_dict['model'])
        data_config = DataConfig(**config_dict['data'])
        training_config = TrainingConfig(**config_dict['training'])
        evaluation_config = EvaluationConfig(**config_dict['evaluation'])
        
        return cls(model_config, data_config, training_config, evaluation_config)


# Default configuration
DEFAULT_CONFIG = Config()

# Common configurations
SMALL_MODEL_CONFIG = Config(
    model_config=ModelConfig(
        vocab_size=5000,
        max_length=50,
        embedding_dim=64,
        lstm_units=32,
        dropout_rate=0.2
    )
)

LARGE_MODEL_CONFIG = Config(
    model_config=ModelConfig(
        vocab_size=50000,
        max_length=200,
        embedding_dim=256,
        lstm_units=128,
        dropout_rate=0.4
    )
)

FAST_TRAINING_CONFIG = Config(
    training_config=TrainingConfig(
        early_stopping_patience=5,
        reduce_lr_patience=3
    )
)

HIGH_QUALITY_CONFIG = Config(
    training_config=TrainingConfig(
        early_stopping_patience=20,
        reduce_lr_patience=10
    )
)











