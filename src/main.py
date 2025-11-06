# -*- coding: utf-8 -*-
"""
Main Entry Point
================

This is the main entry point for the DeepText Multi-Task Learning system.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing.preprocess_text import quick_process_data
from model.multi_task_model_optimized import DeepTextMultiTaskClassifierOptimized
from training.train import TrainingPipeline
from training.evaluate import ModelEvaluator
from training.visualize import TrainingVisualizer
from utils.config import Config, DEFAULT_CONFIG


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='DeepText Multi-Task Learning System')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict', 'full_pipeline'], 
                       default='full_pipeline', help='Mode to run')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--config_path', type=str, help='Path to configuration file')
    parser.add_argument('--model_path', type=str, help='Path to saved model (for evaluate/predict)')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_path and os.path.exists(args.config_path):
        config = Config.load_config(args.config_path)
    else:
        config = DEFAULT_CONFIG
        # Update with command line arguments
        config.update_config(
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            vocab_size=args.vocab_size
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config.save_config(f"{args.output_dir}/config.json")
    
    print("=" * 60)
    print("DeepText Multi-Task Learning System")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Vocab size: {args.vocab_size}")
    print("=" * 60)
    
    if args.mode == 'train' or args.mode == 'full_pipeline':
        train_model(args.data_path, config, args.output_dir)
    
    if args.mode == 'evaluate' or args.mode == 'full_pipeline':
        if args.model_path:
            evaluate_model(args.model_path, args.data_path, config, args.output_dir)
        else:
            print("Warning: No model path provided for evaluation")
    
    if args.mode == 'predict':
        if args.model_path:
            predict_model(args.model_path, args.data_path, config, args.output_dir)
        else:
            print("Error: Model path required for prediction")
            return
    
    print("=" * 60)
    print("Process completed successfully!")
    print("=" * 60)


def train_model(data_path: str, config: Config, output_dir: str):
    """Train the model."""
    print("\n🚀 Starting training...")
    
    # Process data
    print("📊 Processing data...")
    data, preprocessor, processor = quick_process_data(
        data_path,
        max_length=config.model.max_length,
        vocab_size=config.model.vocab_size,
        test_size=config.model.test_split,
        val_size=config.model.validation_split
    )
    
    # Save processed data
    processor.save_processed_data(data, f"{output_dir}/processed_data.pkl")
    preprocessor.save_tokenizer(f"{output_dir}/tokenizer.pkl")
    
    # Create model
    print("🏗️ Creating model...")
    model = DeepTextMultiTaskClassifierOptimized(
        vocab_size=data['vocab_size'],
        max_length=data['max_length'],
        embedding_dim=config.model.embedding_dim,
        lstm_units=config.model.lstm_units,
        dropout_rate=config.model.dropout_rate,
        use_attention=config.model.use_attention,
        use_batch_norm=config.model.use_batch_norm,
        use_pretrained_embedding=config.model.use_pretrained_embedding
    )
    
    model.build_model()
    model.compile_model(learning_rate=config.model.learning_rate)
    
    # Create training pipeline
    pipeline = TrainingPipeline(model.model, data, output_dir)
    
    # Calculate class weights
    class_weights = pipeline.calculate_class_weights()
    
    # Train model
    print("🎯 Training model...")
    history = pipeline.train(
        epochs=config.model.epochs,
        batch_size=config.model.batch_size,
        class_weights=class_weights
    )
    
    # Save model
    model_path = pipeline.save_model()
    
    # Save training history
    history_path = pipeline.save_training_history()
    
    # Generate visualizations
    print("📈 Generating visualizations...")
    pipeline.plot_training_history()
    pipeline.generate_model_summary()
    
    # Evaluate model
    print("📊 Evaluating model...")
    evaluation_results = pipeline.evaluate()
    
    # Create visualizer
    visualizer = TrainingVisualizer(output_dir)
    visualizer.create_dashboard(history, data, evaluation_results)
    
    print(f"✅ Training completed! Model saved to: {model_path}")
    return model_path, history_path


def evaluate_model(model_path: str, data_path: str, config: Config, output_dir: str):
    """Evaluate the model."""
    print("\n📊 Starting evaluation...")
    
    # Load processed data
    processed_data_path = f"{output_dir}/processed_data.pkl"
    if os.path.exists(processed_data_path):
        from data_preprocessing.preprocess_text import DataProcessor
        processor = DataProcessor(None)
        data = processor.load_processed_data(processed_data_path)
    else:
        # Process data if not available
        data, preprocessor, processor = quick_process_data(
            data_path,
            max_length=config.model.max_length,
            vocab_size=config.model.vocab_size
        )
    
    # Load model
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, data, output_dir=output_dir)
    
    # Evaluate model
    evaluation_results = evaluator.evaluate_model()
    
    # Generate plots
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_precision_recall_curves()
    
    # Generate report
    report_path = evaluator.generate_evaluation_report()
    
    print(f"✅ Evaluation completed! Report saved to: {report_path}")
    return evaluation_results


def predict_model(model_path: str, data_path: str, config: Config, output_dir: str):
    """Make predictions with the model."""
    print("\n🔮 Starting prediction...")
    
    # Load data
    import pandas as pd
    df = pd.read_csv(data_path)
    
    # Load tokenizer
    tokenizer_path = f"{output_dir}/tokenizer.pkl"
    if os.path.exists(tokenizer_path):
        from data_preprocessing.preprocess_text import TextPreprocessor
        preprocessor = TextPreprocessor()
        preprocessor.load_tokenizer(tokenizer_path)
    else:
        print("Error: Tokenizer not found. Please train the model first.")
        return
    
    # Load model
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess texts
    texts = df[config.data.text_column].tolist()
    texts = [preprocessor.clean_text(text) for text in texts]
    sequences = preprocessor.texts_to_sequences(texts)
    
    # Make predictions
    predictions = model.predict(sequences, verbose=1)
    
    # Process predictions
    class_names = config.get_class_names()
    results = []
    
    for i, text in enumerate(texts):
        result = {
            'text': text,
            'emotion': class_names['emotion'][np.argmax(predictions[0][i])],
            'hate': class_names['hate'][np.argmax(predictions[1][i])],
            'violence': class_names['violence'][np.argmax(predictions[2][i])],
            'emotion_confidence': float(np.max(predictions[0][i])),
            'hate_confidence': float(np.max(predictions[1][i])),
            'violence_confidence': float(np.max(predictions[2][i]))
        }
        results.append(result)
    
    # Save predictions
    results_df = pd.DataFrame(results)
    predictions_path = f"{output_dir}/predictions.csv"
    results_df.to_csv(predictions_path, index=False)
    
    print(f"✅ Prediction completed! Results saved to: {predictions_path}")
    return results


if __name__ == "__main__":
    main()











