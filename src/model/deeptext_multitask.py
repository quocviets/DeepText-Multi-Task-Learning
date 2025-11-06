# -*- coding: utf-8 -*-
"""
DeepText Multi-Task Classifier
Architecture: Shared Embedding + Shared BiLSTM + 3 Task-Specific Heads
Tasks: Emotion Classification, Hate Speech Detection, Violence Detection
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class DeepTextMultiTaskClassifier:
    """
    DeepText Multi-Task Classifier với kiến trúc:
    Input -> Embedding -> Shared BiLSTM -> 3 Task-Specific Heads
    """
    
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=64, 
                 max_length=100, dropout_rate=0.3, learning_rate=0.001):
        """
        Khởi tạo mô hình
        
        Args:
            vocab_size: Kích thước từ vựng
            embedding_dim: Chiều embedding
            lstm_units: Số units trong LSTM
            max_length: Độ dài tối đa của câu
            dropout_rate: Tỷ lệ dropout
            learning_rate: Tốc độ học
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Task configurations
        self.emotion_classes = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
        self.hate_classes = ['hate', 'offensive', 'neutral']
        self.violence_classes = ['sex_viol', 'phys_viol', 'no_viol']
        
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Xây dựng kiến trúc mô hình
        """
        print("Building DeepText Multi-Task Classifier...")
        
        # Input layer
        input_layer = layers.Input(shape=(self.max_length,), name='text_input')
        
        # Shared Embedding Layer
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
                return_sequences=False,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name='shared_lstm'
            ),
            name='shared_bilstm'
        )(embedding)
        
        # Shared Dense Layer
        shared_dense = layers.Dense(
            units=128,
            activation='relu',
            name='shared_dense'
        )(shared_lstm)
        
        shared_dropout = layers.Dropout(
            rate=self.dropout_rate,
            name='shared_dropout'
        )(shared_dense)
        
        # Task-Specific Heads
        # 1. Emotion Classification Head
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
            activation='softmax',
            name='emotion_output'
        )(emotion_dropout)
        
        # 2. Hate Speech Detection Head
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
            activation='softmax',
            name='hate_output'
        )(hate_dropout)
        
        # 3. Violence Detection Head
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
            activation='softmax',
            name='violence_output'
        )(violence_dropout)
        
        # Create model
        self.model = Model(
            inputs=input_layer,
            outputs=[emotion_output, hate_output, violence_output],
            name='DeepTextMultiTaskClassifier'
        )
        
        print("Model architecture built successfully!")
        return self.model
    
    def compile_model(self, emotion_weight=1.0, hate_weight=1.0, violence_weight=1.0):
        """
        Compile mô hình với multi-loss và multi-weight
        
        Args:
            emotion_weight: Trọng số cho emotion task
            hate_weight: Trọng số cho hate speech task
            violence_weight: Trọng số cho violence task
        """
        print("Compiling model with multi-task losses...")
        
        # Define losses for each task
        losses = {
            'emotion_output': 'categorical_crossentropy',
            'hate_output': 'categorical_crossentropy',
            'violence_output': 'categorical_crossentropy'
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
            'hate_output': ['accuracy', 'top_k_categorical_accuracy'],
            'violence_output': ['accuracy', 'top_k_categorical_accuracy']
        }
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        print(f"Model compiled with loss weights:")
        print(f"  Emotion: {emotion_weight}")
        print(f"  Hate Speech: {hate_weight}")
        print(f"  Violence: {violence_weight}")
        
        return self.model
    
    def get_model_summary(self):
        """
        In ra summary của mô hình
        """
        if self.model is None:
            print("Model chưa được build!")
            return
        
        print("\n" + "="*80)
        print("DEEPTEXT MULTI-TASK CLASSIFIER SUMMARY")
        print("="*80)
        
        # Model summary
        self.model.summary()
        
        # Layer information
        print(f"\nModel Architecture:")
        print(f"  Input Shape: (batch_size, {self.max_length})")
        print(f"  Embedding: {self.vocab_size} -> {self.embedding_dim}")
        print(f"  Shared BiLSTM: {self.lstm_units} units")
        print(f"  Shared Dense: 128 units")
        
        print(f"\nTask-Specific Heads:")
        print(f"  Emotion: 64 -> {len(self.emotion_classes)} classes")
        print(f"  Hate Speech: 32 -> {len(self.hate_classes)} classes")
        print(f"  Violence: 32 -> {len(self.violence_classes)} classes")
        
        # Count parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Non-trainable: {total_params - trainable_params:,}")
        
        print("="*80)
    
    def check_output_dimensions(self, sample_input):
        """
        Kiểm tra dimensions của output
        
        Args:
            sample_input: Input mẫu để test
        """
        if self.model is None:
            print("Model chưa được build!")
            return
        
        print("Checking output dimensions...")
        
        # Get model outputs
        outputs = self.model(sample_input)
        
        print(f"Input shape: {sample_input.shape}")
        print(f"Number of outputs: {len(outputs)}")
        
        for i, (output, task_name) in enumerate(zip(outputs, ['Emotion', 'Hate Speech', 'Violence'])):
            print(f"\n{task_name} Output:")
            print(f"  Shape: {output.shape}")
            print(f"  Expected classes: {output.shape[-1]}")
            
            # Check if probabilities sum to 1
            prob_sums = tf.reduce_sum(output, axis=1)
            print(f"  Probability sums (should be ~1.0): min={tf.reduce_min(prob_sums):.4f}, max={tf.reduce_max(prob_sums):.4f}")
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, verbose=1):
        """
        Training mô hình
        
        Args:
            X_train: Training input
            y_train: Training labels (list of 3 arrays)
            X_val: Validation input
            y_val: Validation labels (list of 3 arrays)
            epochs: Số epochs
            batch_size: Batch size
            verbose: Verbose level
        """
        print("Starting training...")
        
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
                filepath='best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Training
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Đánh giá mô hình
        
        Args:
            X_test: Test input
            y_test: Test labels (list of 3 arrays)
        """
        print("Evaluating model...")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Evaluate each task
        results = {}
        task_names = ['Emotion', 'Hate Speech', 'Violence']
        
        for i, (pred, true, task_name) in enumerate(zip(predictions, y_test, task_names)):
            print(f"\n{task_name} Task Results:")
            
            # Convert to class predictions
            pred_classes = np.argmax(pred, axis=1)
            true_classes = np.argmax(true, axis=1)
            
            # Calculate metrics
            accuracy = np.mean(pred_classes == true_classes)
            print(f"  Accuracy: {accuracy:.4f}")
            
            # Classification report
            class_names = [self.emotion_classes, self.hate_classes, self.violence_classes][i]
            report = classification_report(true_classes, pred_classes, 
                                        target_names=class_names, 
                                        zero_division=0)
            print(f"  Classification Report:\n{report}")
            
            results[task_name] = {
                'accuracy': accuracy,
                'predictions': pred_classes,
                'true_labels': true_classes
            }
        
        return results
    
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
        axes[1, 0].plot(self.history.history['hate_output_accuracy'], label='Training')
        axes[1, 0].plot(self.history.history['val_hate_output_accuracy'], label='Validation')
        axes[1, 0].set_title('Hate Speech Task Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        
        # Violence Accuracy
        axes[1, 1].plot(self.history.history['violence_output_accuracy'], label='Training')
        axes[1, 1].plot(self.history.history['val_violence_output_accuracy'], label='Validation')
        axes[1, 1].set_title('Violence Task Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """
        Lưu mô hình
        """
        if self.model is None:
            print("Model chưa được build!")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load mô hình
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def create_model_architecture_diagram():
    """
    Tạo sơ đồ kiến trúc mô hình
    """
    print("Creating model architecture diagram...")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define colors
    colors = {
        'input': '#E3F2FD',
        'embedding': '#BBDEFB',
        'shared': '#90CAF9',
        'emotion': '#FFCDD2',
        'hate': '#F8BBD9',
        'violence': '#C8E6C9'
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
    ax.text(0.5, y_pos, 'Shared Embedding Layer\n(vocab_size → embedding_dim)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    y_pos -= 0.2
    
    # Shared BiLSTM
    ax.add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.8, 0.1, 
                              facecolor=colors['shared'], edgecolor='black', linewidth=2))
    ax.text(0.5, y_pos, 'Shared BiLSTM Layer\n(lstm_units)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    y_pos -= 0.2
    
    # Shared Dense
    ax.add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.8, 0.1, 
                              facecolor=colors['shared'], edgecolor='black', linewidth=2))
    ax.text(0.5, y_pos, 'Shared Dense Layer\n(128 units)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    y_pos -= 0.3
    
    # Task-specific heads
    head_width = 0.25
    head_spacing = 0.1
    
    # Emotion Head
    ax.add_patch(plt.Rectangle((0.05, y_pos-0.05), head_width, 0.1, 
                              facecolor=colors['emotion'], edgecolor='black', linewidth=2))
    ax.text(0.175, y_pos, 'Emotion Head\n(7 classes)', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Hate Speech Head
    ax.add_patch(plt.Rectangle((0.4, y_pos-0.05), head_width, 0.1, 
                              facecolor=colors['hate'], edgecolor='black', linewidth=2))
    ax.text(0.525, y_pos, 'Hate Speech Head\n(3 classes)', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Violence Head
    ax.add_patch(plt.Rectangle((0.75, y_pos-0.05), head_width, 0.1, 
                              facecolor=colors['violence'], edgecolor='black', linewidth=2))
    ax.text(0.875, y_pos, 'Violence Head\n(3 classes)', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Input to Embedding
    ax.annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.85), arrowprops=arrow_props)
    
    # Embedding to BiLSTM
    ax.annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.65), arrowprops=arrow_props)
    
    # BiLSTM to Dense
    ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.45), arrowprops=arrow_props)
    
    # Dense to Heads
    ax.annotate('', xy=(0.175, 0.25), xytext=(0.5, 0.35), arrowprops=arrow_props)
    ax.annotate('', xy=(0.525, 0.25), xytext=(0.5, 0.35), arrowprops=arrow_props)
    ax.annotate('', xy=(0.875, 0.25), xytext=(0.5, 0.35), arrowprops=arrow_props)
    
    # Title
    ax.text(0.5, 0.95, 'DeepText Multi-Task Classifier Architecture', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Architecture diagram saved as 'model_architecture.png'")


if __name__ == "__main__":
    # Example usage
    print("DeepText Multi-Task Classifier")
    print("="*50)
    
    # Create model
    model = DeepTextMultiTaskClassifier(
        vocab_size=10000,
        embedding_dim=128,
        lstm_units=64,
        max_length=100,
        dropout_rate=0.3,
        learning_rate=0.001
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
    create_model_architecture_diagram()
    
    print("\nModel ready for training!")
