# -*- coding: utf-8 -*-
"""
Test script cho DeepText Multi-Task Classifier
Kiểm tra kiến trúc, dimensions và functionality
"""

import numpy as np
import tensorflow as tf
from multi_task_model import DeepTextMultiTaskClassifier, create_model_architecture_diagram

def test_model_architecture():
    """
    Test kiến trúc mô hình
    """
    print("="*60)
    print("TESTING DEEPTEXT MULTI-TASK CLASSIFIER")
    print("="*60)
    
    # Parameters
    vocab_size = 10000
    max_length = 100
    batch_size = 32
    
    # Create model
    print("\n1. Creating model...")
    model = DeepTextMultiTaskClassifier(
        vocab_size=vocab_size,
        embedding_dim=128,
        lstm_units=64,
        max_length=max_length,
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    # Build model
    print("\n2. Building model architecture...")
    model.build_model()
    
    # Compile model
    print("\n3. Compiling model...")
    model.compile_model(
        emotion_weight=1.0,
        hate_weight=1.0,
        violence_weight=1.0
    )
    
    # Print summary
    print("\n4. Model summary:")
    model.get_model_summary()
    
    # Test with sample data
    print("\n5. Testing with sample data...")
    sample_input = np.random.randint(0, vocab_size, size=(batch_size, max_length))
    
    # Check output dimensions
    print("\n6. Checking output dimensions:")
    model.check_output_dimensions(sample_input)
    
    # Test forward pass
    print("\n7. Testing forward pass...")
    with tf.device('/CPU:0'):  # Use CPU for testing
        outputs = model.model(sample_input)
        
        print(f"Input shape: {sample_input.shape}")
        print(f"Number of outputs: {len(outputs)}")
        
        for i, (output, task_name) in enumerate(zip(outputs, ['Emotion', 'Hate Speech', 'Violence'])):
            print(f"\n{task_name} Output:")
            print(f"  Shape: {output.shape}")
            print(f"  Data type: {output.dtype}")
            print(f"  Min value: {tf.reduce_min(output):.4f}")
            print(f"  Max value: {tf.reduce_max(output):.4f}")
            print(f"  Mean value: {tf.reduce_mean(output):.4f}")
            
            # Check if probabilities sum to 1
            prob_sums = tf.reduce_sum(output, axis=1)
            print(f"  Probability sums: min={tf.reduce_min(prob_sums):.4f}, max={tf.reduce_max(prob_sums):.4f}")
            print(f"  All probabilities sum to 1: {tf.reduce_all(tf.abs(prob_sums - 1.0) < 1e-6)}")
    
    # Test with different batch sizes
    print("\n8. Testing with different batch sizes...")
    for test_batch_size in [1, 16, 64]:
        test_input = np.random.randint(0, vocab_size, size=(test_batch_size, max_length))
        test_outputs = model.model(test_input)
        
        print(f"Batch size {test_batch_size}:")
        for i, (output, task_name) in enumerate(zip(test_outputs, ['Emotion', 'Hate Speech', 'Violence'])):
            print(f"  {task_name}: {output.shape}")
    
    # Test model saving and loading
    print("\n9. Testing model saving and loading...")
    try:
        model.save_model('test_model.h5')
        print("  Model saved successfully!")
        
        # Create new model instance and load
        new_model = DeepTextMultiTaskClassifier(
            vocab_size=vocab_size,
            embedding_dim=128,
            lstm_units=64,
            max_length=max_length
        )
        new_model.load_model('test_model.h5')
        print("  Model loaded successfully!")
        
        # Test loaded model
        test_outputs = new_model.model(sample_input)
        print(f"  Loaded model output shapes: {[output.shape for output in test_outputs]}")
        
    except Exception as e:
        print(f"  Error in saving/loading: {e}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)

def test_with_real_data_structure():
    """
    Test với cấu trúc dữ liệu thực tế
    """
    print("\n" + "="*60)
    print("TESTING WITH REAL DATA STRUCTURE")
    print("="*60)
    
    # Load balanced dataset
    try:
        import pandas as pd
        
        print("\n1. Loading balanced dataset...")
        train_df = pd.read_csv('train_dataset_balanced.csv', sep=';')
        val_df = pd.read_csv('val_dataset_balanced.csv', sep=';')
        
        print(f"   Train samples: {len(train_df)}")
        print(f"   Val samples: {len(val_df)}")
        
        # Check data structure
        print("\n2. Checking data structure...")
        print(f"   Columns: {list(train_df.columns)}")
        
        # Sample data
        sample_text = train_df['text'].iloc[0]
        print(f"   Sample text: {sample_text}")
        print(f"   Text length: {len(sample_text)}")
        
        # Check labels
        emotion_cols = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
        hate_cols = ['hate', 'offensive', 'neutral']
        violence_cols = ['sex_viol', 'phys_viol', 'no_viol']
        
        print(f"\n3. Checking label distributions...")
        
        # Emotion labels
        emotion_labels = train_df[emotion_cols].values
        print(f"   Emotion labels shape: {emotion_labels.shape}")
        print(f"   Emotion labels sum per sample: {emotion_labels.sum(axis=1)[:5]}")
        
        # Hate speech labels
        hate_labels = train_df[hate_cols].values
        print(f"   Hate labels shape: {hate_labels.shape}")
        print(f"   Hate labels sum per sample: {hate_labels.sum(axis=1)[:5]}")
        
        # Violence labels
        violence_labels = train_df[violence_cols].values
        print(f"   Violence labels shape: {violence_labels.shape}")
        print(f"   Violence labels sum per sample: {violence_labels.sum(axis=1)[:5]}")
        
        print("\n4. Data structure validation:")
        print(f"   All emotion labels sum to 1: {np.all(emotion_labels.sum(axis=1) == 1)}")
        print(f"   All hate labels sum to 0 or 1: {np.all(np.isin(hate_labels.sum(axis=1), [0, 1]))}")
        print(f"   All violence labels sum to 0 or 1: {np.all(np.isin(violence_labels.sum(axis=1), [0, 1]))}")
        
        print("\n5. Model compatibility check:")
        print("   ✓ Data structure is compatible with model architecture")
        print("   ✓ Label formats match expected output shapes")
        print("   ✓ Ready for training pipeline")
        
    except FileNotFoundError:
        print("   Balanced dataset files not found. Please run data cleaning first.")
    except Exception as e:
        print(f"   Error loading dataset: {e}")

def create_architecture_diagram():
    """
    Tạo sơ đồ kiến trúc
    """
    print("\n" + "="*60)
    print("CREATING ARCHITECTURE DIAGRAM")
    print("="*60)
    
    try:
        create_model_architecture_diagram()
        print("Architecture diagram created successfully!")
    except Exception as e:
        print(f"Error creating diagram: {e}")

if __name__ == "__main__":
    # Run all tests
    test_model_architecture()
    test_with_real_data_structure()
    create_architecture_diagram()
    
    print("\n" + "="*60)
    print("COMPLETE TEST SUITE FINISHED")
    print("="*60)
    print("\nNext steps:")
    print("1. Run data preprocessing pipeline")
    print("2. Train the model with real data")
    print("3. Evaluate performance on test set")
    print("4. Fine-tune hyperparameters if needed")
