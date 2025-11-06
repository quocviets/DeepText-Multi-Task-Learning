# -*- coding: utf-8 -*-
"""
DeepText Multi-Task Classifier - OPTIMIZED VERSION
Architecture: Shared Embedding + Shared BiLSTM + 3 Task-Specific Heads
Tasks: Emotion Classification, Hate Speech Detection, Violence Detection

OPTIMIZATIONS APPLIED:
- BatchNormalization for stable training
- Sigmoid activation for Hate/Violence (multi-label support)
- Pretrained embedding support
- Improved checkpoint naming
- Attention mechanism
- Class weight balancing
- Enhanced evaluation with confusion matrix
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import json
import os
from datetime import datetime

class DeepTextMultiTaskClassifierOptimized:
    """
    Optimized DeepText Multi-Task Classifier với các cải tiến:
    - BatchNormalization
    - Sigmoid cho Hate/Violence (multi-label)
    - Pretrained embedding support
    - Attention mechanism
    - Class weight balancing
    """
    
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=64, 
                 max_length=100, dropout_rate=0.3, learning_rate=0.001,
                 use_attention=True, use_pretrained_embedding=False,
                 pretrained_embedding_matrix=None):
        """
        Khởi tạo mô hình tối ưu
        
        Args:
            vocab_size: Kích thước từ vựng
            embedding_dim: Chiều embedding
            lstm_units: Số units trong LSTM
            max_length: Độ dài tối đa của câu
            dropout_rate: Tỷ lệ dropout
            learning_rate: Tốc độ học
            use_attention: Sử dụng attention mechanism
            use_pretrained_embedding: Sử dụng pretrained embedding
            pretrained_embedding_matrix: Ma trận embedding pretrained
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_attention = use_attention
        self.use_pretrained_embedding = use_pretrained_embedding
        self.pretrained_embedding_matrix = pretrained_embedding_matrix
        
        # Task configurations
        self.emotion_classes = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
        self.hate_classes = ['hate', 'offensive', 'neutral']
        self.violence_classes = ['sex_viol', 'phys_viol', 'no_viol']
        
        self.model = None
        self.history = None
        self.class_weights = None
        
    def build_model(self):
        """
        Xây dựng kiến trúc mô hình tối ưu
        """
        print("Building Optimized DeepText Multi-Task Classifier...")
        
        # Input layer
        input_layer = layers.Input(shape=(self.max_length,), name='text_input')
        
        # Shared Embedding Layer
        if self.use_pretrained_embedding and self.pretrained_embedding_matrix is not None:
            embedding = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                mask_zero=True,
                weights=[self.pretrained_embedding_matrix],
                trainable=False,  # Freeze pretrained embedding initially
                name='shared_embedding'
            )(input_layer)
        else:
            embedding = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                mask_zero=True,
                name='shared_embedding'
            )(input_layer)
        
        # Shared BiLSTM Layer
        shared_lstm = layers.Bidirectional(
            layers.LSTM(
                units=self.lstm_units,
                return_sequences=True,  # Always return sequences for attention
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name='shared_lstm'
            ),
            name='shared_bilstm'
        )(embedding)
        
        # Attention Layer (optional)
        if self.use_attention:
            # Self-attention mechanism
            attention = layers.MultiHeadAttention(
                num_heads=8,
                key_dim=64,
                name='self_attention'
            )(shared_lstm, shared_lstm)
            
            # Global max pooling
            attention_pooled = layers.GlobalMaxPooling1D(name='attention_pooling')(attention)
        else:
            # Global max pooling if no attention
            attention_pooled = layers.GlobalMaxPooling1D(name='global_pooling')(shared_lstm)
        
        # Shared Dense Layer with BatchNormalization
        shared_dense = layers.Dense(
            units=128,
            activation='relu',
            name='shared_dense'
        )(attention_pooled)
        
        # BatchNormalization for stable training
        shared_bn = layers.BatchNormalization(name='shared_batch_norm')(shared_dense)
        
        shared_dropout = layers.Dropout(
            rate=self.dropout_rate,
            name='shared_dropout'
        )(shared_bn)
        
        # Task-Specific Heads
        # 1. Emotion Classification Head (Softmax - single label)
        emotion_head = layers.Dense(
            units=64,
            activation='relu',
            name='emotion_dense1'
        )(shared_dropout)
        
        emotion_dropout = layers.Dropout(
            rate=self.dropout_rate,
            name='emotion_dropout'
        )(emotion_head)
        
        emotion_output = layers.Dense(
            units=len(self.emotion_classes),
            activation='softmax',  # Softmax for single-label classification
            name='emotion_output'
        )(emotion_dropout)
        
        # 2. Hate Speech Detection Head (Sigmoid - multi-label)
        hate_head = layers.Dense(
            units=32,
            activation='relu',
            name='hate_dense1'
        )(shared_dropout)
        
        hate_dropout = layers.Dropout(
            rate=self.dropout_rate,
            name='hate_dropout'
        )(hate_head)
        
        hate_output = layers.Dense(
            units=len(self.hate_classes),
            activation='sigmoid',  # Sigmoid for multi-label classification
            name='hate_output'
        )(hate_dropout)
        
        # 3. Violence Detection Head (Sigmoid - multi-label)
        violence_head = layers.Dense(
            units=32,
            activation='relu',
            name='violence_dense1'
        )(shared_dropout)
        
        violence_dropout = layers.Dropout(
            rate=self.dropout_rate,
            name='violence_dropout'
        )(violence_head)
        
        violence_output = layers.Dense(
            units=len(self.violence_classes),
            activation='sigmoid',  # Sigmoid for multi-label classification
            name='violence_output'
        )(violence_dropout)
        
        # Create model
        self.model = Model(
            inputs=input_layer,
            outputs=[emotion_output, hate_output, violence_output],
            name='DeepTextMultiTaskClassifierOptimized'
        )
        
        print("Optimized model architecture built successfully!")
        return self.model
    
    def compile_model(self, emotion_weight=1.0, hate_weight=1.0, violence_weight=1.0):
        """
        Compile mô hình với multi-loss và multi-weight tối ưu
        """
        print("Compiling optimized model with multi-task losses...")
        
        # Define losses for each task
        losses = {
            'emotion_output': 'categorical_crossentropy',  # Softmax + categorical
            'hate_output': 'binary_crossentropy',          # Sigmoid + binary
            'violence_output': 'binary_crossentropy'       # Sigmoid + binary
        }
        
        # Define loss weights
        loss_weights = {
            'emotion_output': emotion_weight,
            'hate_output': hate_weight,
            'violence_output': violence_weight
        }
        
        # Define metrics for each task
        metrics = {
            'emotion_output': ['accuracy', 'top_k_categorical_accuracy'],
            'hate_output': ['binary_accuracy'],
            'violence_output': ['binary_accuracy']
        }
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        print(f"Optimized model compiled with loss weights:")
        print(f"  Emotion: {emotion_weight} (categorical_crossentropy)")
        print(f"  Hate Speech: {hate_weight} (binary_crossentropy)")
        print(f"  Violence: {violence_weight} (binary_crossentropy)")
        
        return self.model
    
    def calculate_class_weights(self, y_train):
        """
        Tính toán class weights để cân bằng dữ liệu
        """
        print("Calculating class weights for balanced training...")
        
        self.class_weights = {}
        
        # Emotion task (categorical)
        emotion_labels = y_train[0]
        emotion_weights = {}
        for i, class_name in enumerate(self.emotion_classes):
            class_count = np.sum(emotion_labels[:, i])
            if class_count > 0:
                total_samples = len(emotion_labels)
                weight = total_samples / (len(self.emotion_classes) * class_count)
                emotion_weights[i] = weight
            else:
                emotion_weights[i] = 1.0
        
        self.class_weights['emotion'] = emotion_weights
        
        # Hate Speech task (binary)
        hate_labels = y_train[1]
        hate_weights = {}
        for i, class_name in enumerate(self.hate_classes):
            class_count = np.sum(hate_labels[:, i])
            if class_count > 0:
                total_samples = len(hate_labels)
                weight = total_samples / (len(self.hate_classes) * class_count)
                hate_weights[i] = weight
            else:
                hate_weights[i] = 1.0
        
        self.class_weights['hate'] = hate_weights
        
        # Violence task (binary)
        violence_labels = y_train[2]
        violence_weights = {}
        for i, class_name in enumerate(self.violence_classes):
            class_count = np.sum(violence_labels[:, i])
            if class_count > 0:
                total_samples = len(violence_labels)
                weight = total_samples / (len(self.violence_classes) * class_count)
                violence_weights[i] = weight
            else:
                violence_weights[i] = 1.0
        
        self.class_weights['violence'] = violence_weights
        
        print("Class weights calculated successfully!")
        return self.class_weights
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, verbose=1):
        """
        Training mô hình với class weights
        """
        print("Starting optimized training...")
        
        # Calculate class weights
        self.calculate_class_weights(y_train)
        
        # Create checkpoint directory
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'{checkpoint_dir}/multitask_best_{{epoch:02d}}_{{val_loss:.3f}}.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Training with class weights
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=verbose
        )
        
        print("Optimized training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test, plot_confusion_matrix=True):
        """
        Đánh giá mô hình với confusion matrix
        """
        print("Evaluating optimized model...")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Evaluate each task
        results = {}
        task_names = ['Emotion', 'Hate Speech', 'Violence']
        
        for i, (pred, true, task_name) in enumerate(zip(predictions, y_test, task_names)):
            print(f"\n{task_name} Task Results:")
            
            if i == 0:  # Emotion task (softmax)
                # Convert to class predictions
                pred_classes = np.argmax(pred, axis=1)
                true_classes = np.argmax(true, axis=1)
                
                # Calculate metrics
                accuracy = np.mean(pred_classes == true_classes)
                print(f"  Accuracy: {accuracy:.4f}")
                
                # Classification report
                class_names = self.emotion_classes
                report = classification_report(true_classes, pred_classes, 
                                            target_names=class_names, 
                                            zero_division=0)
                print(f"  Classification Report:\n{report}")
                
                # Confusion matrix
                if plot_confusion_matrix:
                    self.plot_confusion_matrix(true_classes, pred_classes, 
                                             class_names, f"{task_name} Confusion Matrix")
                
            else:  # Hate Speech and Violence tasks (sigmoid)
                # Convert to binary predictions (threshold = 0.5)
                pred_binary = (pred > 0.5).astype(int)
                true_binary = true.astype(int)
                
                # Calculate metrics
                accuracy = np.mean(pred_binary == true_binary)
                print(f"  Binary Accuracy: {accuracy:.4f}")
                
                # Per-class metrics
                class_names = self.hate_classes if i == 1 else self.violence_classes
                for j, class_name in enumerate(class_names):
                    true_pos = np.sum((true_binary[:, j] == 1) & (pred_binary[:, j] == 1))
                    false_pos = np.sum((true_binary[:, j] == 0) & (pred_binary[:, j] == 1))
                    false_neg = np.sum((true_binary[:, j] == 1) & (pred_binary[:, j] == 0))
                    
                    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    print(f"    {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
                
                # Confusion matrix for each class
                if plot_confusion_matrix:
                    for j, class_name in enumerate(class_names):
                        self.plot_confusion_matrix(true_binary[:, j], pred_binary[:, j], 
                                                 [f'Not {class_name}', class_name], 
                                                 f"{task_name} - {class_name}")
            
            results[task_name] = {
                'predictions': pred,
                'true_labels': true
            }
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, title):
        """
        Vẽ confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """
        Vẽ biểu đồ training history
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Emotion Accuracy
        axes[0, 1].plot(self.history.history['emotion_output_accuracy'], label='Training')
        axes[0, 1].plot(self.history.history['val_emotion_output_accuracy'], label='Validation')
        axes[0, 1].set_title('Emotion Task Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Hate Speech Accuracy
        axes[1, 0].plot(self.history.history['hate_output_binary_accuracy'], label='Training')
        axes[1, 0].plot(self.history.history['val_hate_output_binary_accuracy'], label='Validation')
        axes[1, 0].set_title('Hate Speech Task Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        
        # Violence Accuracy
        axes[1, 1].plot(self.history.history['violence_output_binary_accuracy'], label='Training')
        axes[1, 1].plot(self.history.history['val_violence_output_binary_accuracy'], label='Validation')
        axes[1, 1].set_title('Violence Task Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('optimized_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_card(self, filepath='model_card.json'):
        """
        Lưu model card với thông tin siêu tham số
        """
        model_card = {
            'model_name': 'DeepTextMultiTaskClassifierOptimized',
            'timestamp': datetime.now().isoformat(),
            'architecture': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'lstm_units': self.lstm_units,
                'max_length': self.max_length,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'use_attention': self.use_attention,
                'use_pretrained_embedding': self.use_pretrained_embedding
            },
            'tasks': {
                'emotion_classes': self.emotion_classes,
                'hate_classes': self.hate_classes,
                'violence_classes': self.violence_classes
            },
            'loss_functions': {
                'emotion_output': 'categorical_crossentropy',
                'hate_output': 'binary_crossentropy',
                'violence_output': 'binary_crossentropy'
            },
            'optimizations': [
                'BatchNormalization for stable training',
                'Sigmoid activation for multi-label tasks',
                'Attention mechanism for better context',
                'Class weight balancing',
                'Improved checkpoint naming'
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_card, f, indent=2, ensure_ascii=False)
        
        print(f"Model card saved to {filepath}")
    
    def get_model_summary(self):
        """
        In ra summary của mô hình tối ưu
        """
        if self.model is None:
            print("Model chưa được build!")
            return
        
        print("\n" + "="*80)
        print("OPTIMIZED DEEPTEXT MULTI-TASK CLASSIFIER SUMMARY")
        print("="*80)
        
        # Model summary
        self.model.summary()
        
        # Layer information
        print(f"\nOptimized Architecture:")
        print(f"  Input Shape: (batch_size, {self.max_length})")
        print(f"  Embedding: {self.vocab_size} -> {self.embedding_dim}")
        print(f"  Shared BiLSTM: {self.lstm_units} units")
        print(f"  Attention: {'Yes' if self.use_attention else 'No'}")
        print(f"  BatchNormalization: Yes")
        print(f"  Shared Dense: 128 units")
        
        print(f"\nTask-Specific Heads:")
        print(f"  Emotion: 64 -> {len(self.emotion_classes)} classes (softmax)")
        print(f"  Hate Speech: 32 -> {len(self.hate_classes)} classes (sigmoid)")
        print(f"  Violence: 32 -> {len(self.violence_classes)} classes (sigmoid)")
        
        # Count parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Non-trainable: {total_params - trainable_params:,}")
        
        print(f"\nOptimizations Applied:")
        print(f"  [OK] BatchNormalization for stable training")
        print(f"  [OK] Sigmoid activation for multi-label tasks")
        print(f"  [OK] Attention mechanism: {'Yes' if self.use_attention else 'No'}")
        print(f"  [OK] Class weight balancing")
        print(f"  [OK] Improved checkpoint naming")
        print(f"  [OK] Pretrained embedding: {'Yes' if self.use_pretrained_embedding else 'No'}")
        
        print("="*80)
    
    def save_model(self, filepath):
        """
        Lưu mô hình
        """
        if self.model is None:
            print("Model chưa được build!")
            return
        
        self.model.save(filepath)
        print(f"Optimized model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load mô hình
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Optimized model loaded from {filepath}")


def create_optimized_architecture_diagram():
    """
    Tạo sơ đồ kiến trúc mô hình tối ưu
    """
    print("Creating optimized model architecture diagram...")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define colors
    colors = {
        'input': '#E3F2FD',
        'embedding': '#BBDEFB',
        'shared': '#90CAF9',
        'attention': '#FFE0B2',
        'emotion': '#FFCDD2',
        'hate': '#F8BBD9',
        'violence': '#C8E6C9',
        'optimization': '#FFF3E0'
    }
    
    # Draw architecture
    y_pos = 0.9
    
    # Input
    ax.add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.8, 0.1, 
                              facecolor=colors['input'], edgecolor='black', linewidth=2))
    ax.text(0.5, y_pos, 'Input Text\n(seq_len, vocab_size)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    y_pos -= 0.2
    
    # Embedding
    ax.add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.8, 0.1, 
                              facecolor=colors['embedding'], edgecolor='black', linewidth=2))
    ax.text(0.5, y_pos, 'Shared Embedding Layer\n(vocab_size → embedding_dim)\n[Pretrained Support]', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    y_pos -= 0.2
    
    # Shared BiLSTM
    ax.add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.8, 0.1, 
                              facecolor=colors['shared'], edgecolor='black', linewidth=2))
    ax.text(0.5, y_pos, 'Shared BiLSTM Layer\n(lstm_units)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    y_pos -= 0.2
    
    # Attention
    ax.add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.8, 0.1, 
                              facecolor=colors['attention'], edgecolor='black', linewidth=2))
    ax.text(0.5, y_pos, 'Multi-Head Attention\n+ Global Max Pooling', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    y_pos -= 0.2
    
    # Shared Dense + BatchNorm
    ax.add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.8, 0.1, 
                              facecolor=colors['shared'], edgecolor='black', linewidth=2))
    ax.text(0.5, y_pos, 'Shared Dense + BatchNorm\n(128 units + Dropout)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    y_pos -= 0.3
    
    # Task-specific heads
    head_width = 0.25
    head_spacing = 0.1
    
    # Emotion Head
    ax.add_patch(plt.Rectangle((0.05, y_pos-0.05), head_width, 0.1, 
                              facecolor=colors['emotion'], edgecolor='black', linewidth=2))
    ax.text(0.175, y_pos, 'Emotion Head\n(7 classes)\nSoftmax', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Hate Speech Head
    ax.add_patch(plt.Rectangle((0.4, y_pos-0.05), head_width, 0.1, 
                              facecolor=colors['hate'], edgecolor='black', linewidth=2))
    ax.text(0.525, y_pos, 'Hate Speech Head\n(3 classes)\nSigmoid', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Violence Head
    ax.add_patch(plt.Rectangle((0.75, y_pos-0.05), head_width, 0.1, 
                              facecolor=colors['violence'], edgecolor='black', linewidth=2))
    ax.text(0.875, y_pos, 'Violence Head\n(3 classes)\nSigmoid', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Optimization features
    y_pos -= 0.2
    ax.add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.8, 0.1, 
                              facecolor=colors['optimization'], edgecolor='black', linewidth=2))
    ax.text(0.5, y_pos, 'OPTIMIZATIONS: BatchNorm + Class Weights + Attention + Multi-label Support', 
            ha='center', va='center', fontsize=10, weight='bold', style='italic')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Input to Embedding
    ax.annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.85), arrowprops=arrow_props)
    
    # Embedding to BiLSTM
    ax.annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.65), arrowprops=arrow_props)
    
    # BiLSTM to Attention
    ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.45), arrowprops=arrow_props)
    
    # Attention to Dense
    ax.annotate('', xy=(0.5, 0.15), xytext=(0.5, 0.25), arrowprops=arrow_props)
    
    # Dense to Heads
    ax.annotate('', xy=(0.175, 0.05), xytext=(0.5, 0.15), arrowprops=arrow_props)
    ax.annotate('', xy=(0.525, 0.05), xytext=(0.5, 0.15), arrowprops=arrow_props)
    ax.annotate('', xy=(0.875, 0.05), xytext=(0.5, 0.15), arrowprops=arrow_props)
    
    # Title
    ax.text(0.5, 0.95, 'DeepText Multi-Task Classifier - OPTIMIZED VERSION', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('optimized_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Optimized architecture diagram saved as 'optimized_model_architecture.png'")


if __name__ == "__main__":
    # Example usage
    print("DeepText Multi-Task Classifier - OPTIMIZED VERSION")
    print("="*60)
    
    # Create optimized model
    model = DeepTextMultiTaskClassifierOptimized(
        vocab_size=10000,
        embedding_dim=128,
        lstm_units=64,
        max_length=100,
        dropout_rate=0.3,
        learning_rate=0.001,
        use_attention=True,
        use_pretrained_embedding=False
    )
    
    # Build model
    model.build_model()
    
    # Compile model
    model.compile_model(
        emotion_weight=1.0,
        hate_weight=1.0,
        violence_weight=1.0
    )
    
    # Print summary
    model.get_model_summary()
    
    # Create architecture diagram
    create_optimized_architecture_diagram()
    
    # Save model card
    model.save_model_card()
    
    print("\nOptimized model ready for training!")
