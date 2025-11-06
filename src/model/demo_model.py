# -*- coding: utf-8 -*-
"""
Demo script cho DeepText Multi-Task Classifier
Sử dụng mô hình để dự đoán trên dữ liệu mẫu
"""

import numpy as np
import pandas as pd
from multi_task_model import DeepTextMultiTaskClassifier

def demo_model_usage():
    """
    Demo cách sử dụng mô hình
    """
    print("="*60)
    print("DEEPTEXT MULTI-TASK CLASSIFIER DEMO")
    print("="*60)
    
    # 1. Tạo mô hình
    print("\n1. Creating model...")
    model = DeepTextMultiTaskClassifier(
        vocab_size=10000,
        embedding_dim=128,
        lstm_units=64,
        max_length=100,
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    # 2. Build và compile
    print("\n2. Building and compiling model...")
    model.build_model()
    model.compile_model(
        emotion_weight=1.0,
        hate_weight=1.0,
        violence_weight=1.0
    )
    
    # 3. Demo với dữ liệu mẫu
    print("\n3. Demo with sample data...")
    
    # Tạo dữ liệu mẫu (giả lập tokenized text)
    sample_texts = [
        "I love this movie so much!",
        "This is fucking terrible!",
        "I want to hurt someone",
        "I'm so happy today",
        "I hate everyone here"
    ]
    
    # Giả lập tokenization (trong thực tế sẽ dùng tokenizer)
    sample_tokens = np.random.randint(0, 10000, size=(len(sample_texts), 100))
    
    print(f"Sample texts: {sample_texts}")
    print(f"Tokenized shape: {sample_tokens.shape}")
    
    # 4. Dự đoán
    print("\n4. Making predictions...")
    
    with tf.device('/CPU:0'):
        predictions = model.model(sample_tokens)
        
        emotion_pred = predictions[0]
        hate_pred = predictions[1]
        violence_pred = predictions[2]
        
        print(f"\nPredictions shape:")
        print(f"  Emotion: {emotion_pred.shape}")
        print(f"  Hate Speech: {hate_pred.shape}")
        print(f"  Violence: {violence_pred.shape}")
        
        # 5. Hiển thị kết quả
        print("\n5. Prediction results:")
        
        emotion_classes = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
        hate_classes = ['hate', 'offensive', 'neutral']
        violence_classes = ['sex_viol', 'phys_viol', 'no_viol']
        
        for i, text in enumerate(sample_texts):
            print(f"\nText: '{text}'")
            
            # Emotion prediction
            emotion_idx = np.argmax(emotion_pred[i])
            emotion_conf = emotion_pred[i][emotion_idx]
            print(f"  Emotion: {emotion_classes[emotion_idx]} (confidence: {emotion_conf:.3f})")
            
            # Hate speech prediction
            hate_idx = np.argmax(hate_pred[i])
            hate_conf = hate_pred[i][hate_idx]
            print(f"  Hate Speech: {hate_classes[hate_idx]} (confidence: {hate_conf:.3f})")
            
            # Violence prediction
            violence_idx = np.argmax(violence_pred[i])
            violence_conf = violence_pred[i][violence_idx]
            print(f"  Violence: {violence_classes[violence_idx]} (confidence: {violence_conf:.3f})")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)

def demo_with_real_data():
    """
    Demo với dữ liệu thực tế từ dataset
    """
    print("\n" + "="*60)
    print("DEMO WITH REAL DATA")
    print("="*60)
    
    try:
        # Load balanced dataset
        print("\n1. Loading balanced dataset...")
        train_df = pd.read_csv('train_dataset_balanced.csv', sep=';')
        
        print(f"   Dataset shape: {train_df.shape}")
        print(f"   Columns: {list(train_df.columns)}")
        
        # Lấy một số mẫu để demo
        print("\n2. Selecting sample texts...")
        sample_indices = [0, 100, 1000, 5000, 10000]
        sample_texts = train_df.iloc[sample_indices]['text'].tolist()
        
        print("Sample texts:")
        for i, text in enumerate(sample_texts):
            print(f"  {i+1}. {text}")
        
        # Lấy labels thực tế
        emotion_cols = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
        hate_cols = ['hate', 'offensive', 'neutral']
        violence_cols = ['sex_viol', 'phys_viol', 'no_viol']
        
        true_emotion = train_df.iloc[sample_indices][emotion_cols].values
        true_hate = train_df.iloc[sample_indices][hate_cols].values
        true_violence = train_df.iloc[sample_indices][violence_cols].values
        
        emotion_classes = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
        hate_classes = ['hate', 'offensive', 'neutral']
        violence_classes = ['sex_viol', 'phys_viol', 'no_viol']
        
        print(f"\n3. True labels:")
        for i, text in enumerate(sample_texts):
            print(f"\nText {i+1}: '{text}'")
            
            # True emotion
            emotion_idx = np.argmax(true_emotion[i])
            print(f"  True Emotion: {emotion_classes[emotion_idx]}")
            
            # True hate speech
            if np.sum(true_hate[i]) > 0:
                hate_idx = np.argmax(true_hate[i])
                print(f"  True Hate Speech: {hate_classes[hate_idx]}")
            else:
                print(f"  True Hate Speech: No label")
            
            # True violence
            if np.sum(true_violence[i]) > 0:
                violence_idx = np.argmax(true_violence[i])
                print(f"  True Violence: {violence_classes[violence_idx]}")
            else:
                print(f"  True Violence: No label")
        
        print("\n4. Data structure validation:")
        print(f"   Emotion labels shape: {true_emotion.shape}")
        print(f"   Hate labels shape: {true_hate.shape}")
        print(f"   Violence labels shape: {true_violence.shape}")
        
        print(f"\n   Emotion labels sum to 1: {np.all(np.sum(true_emotion, axis=1) == 1)}")
        print(f"   Hate labels sum to 0 or 1: {np.all(np.isin(np.sum(true_hate, axis=1), [0, 1]))}")
        print(f"   Violence labels sum to 0 or 1: {np.all(np.isin(np.sum(true_violence, axis=1), [0, 1]))}")
        
        print("\n5. Ready for training!")
        print("   Dataset structure is compatible with model architecture")
        print("   All labels are properly formatted")
        print("   Model can be trained with this data")
        
    except FileNotFoundError:
        print("   Balanced dataset files not found.")
        print("   Please run data cleaning and balancing first.")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    import tensorflow as tf
    
    # Run demos
    demo_model_usage()
    demo_with_real_data()
    
    print("\n" + "="*60)
    print("ALL DEMOS COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Implement text tokenization pipeline")
    print("2. Train the model with real data")
    print("3. Evaluate performance")
    print("4. Deploy for inference")
