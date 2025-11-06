"""
Example: Sử dụng ModelService programmatically
Ví dụ này cho thấy cách sử dụng model service mà không cần UI
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_app.model_service import ModelService, get_model_service


def example_single_prediction():
    """Ví dụ predict một text"""
    print("=" * 60)
    print("Example: Single Text Prediction")
    print("=" * 60)
    
    # Initialize model service
    model_path = "DeepText-MTL/checkpoints/models/best_model_20251027_085402.h5"
    config_path = "DeepText-MTL/config_default.json"
    train_data_path = "DeepText-MTL/checkpoints/train_clean.csv"
    
    print("\n1. Loading model...")
    service = ModelService(model_path, config_path)
    service.load_config()
    service.load_model()
    service.load_tokenizer(train_data_path)
    
    # Test text
    test_text = "Tôi cảm thấy rất vui vẻ và hạnh phúc hôm nay!"
    
    print(f"\n2. Predicting text: '{test_text}'")
    prediction = service.predict(test_text)
    
    print("\n3. Results:")
    print(f"   Emotion: {prediction['emotion']['label']} ({prediction['emotion']['confidence']:.2%})")
    print(f"   Hate: {prediction['hate']['labels']}")
    print(f"   Violence: {prediction['violence']['labels']}")
    
    print("\n4. Detailed probabilities:")
    print("   Emotion probabilities:")
    for emo, prob in prediction['emotion']['probabilities'].items():
        print(f"      {emo}: {prob:.2%}")
    
    print("\n   Hate probabilities:")
    for hate, prob in prediction['hate']['probabilities'].items():
        print(f"      {hate}: {prob:.2%}")
    
    print("\n   Violence probabilities:")
    for viol, prob in prediction['violence']['probabilities'].items():
        print(f"      {viol}: {prob:.2%}")
    
    print("\n" + "=" * 60)


def example_batch_prediction():
    """Ví dụ predict nhiều texts"""
    print("=" * 60)
    print("Example: Batch Prediction")
    print("=" * 60)
    
    # Initialize model service
    model_path = "DeepText-MTL/checkpoints/models/best_model_20251027_085402.h5"
    train_data_path = "DeepText-MTL/checkpoints/train_clean.csv"
    
    print("\n1. Loading model...")
    service = get_model_service(
        model_path=model_path,
        train_data_path=train_data_path
    )
    
    # Test texts
    test_texts = [
        "Tôi cảm thấy rất vui vẻ!",
        "Đây là một tin nhắn tức giận và thù địch",
        "Tôi yêu bạn rất nhiều"
    ]
    
    print(f"\n2. Predicting {len(test_texts)} texts...")
    results = service.predict_batch(test_texts)
    
    print("\n3. Results:")
    for i, result in enumerate(results, 1):
        print(f"\n   Text {i}: {result['text']}")
        print(f"   Emotion: {result['emotion']['label']} ({result['emotion']['confidence']:.2%})")
        print(f"   Hate: {result['hate']['labels']}")
        print(f"   Violence: {result['violence']['labels']}")
    
    print("\n" + "=" * 60)


def example_model_info():
    """Ví dụ lấy thông tin model"""
    print("=" * 60)
    print("Example: Model Information")
    print("=" * 60)
    
    model_path = "DeepText-MTL/checkpoints/models/best_model_20251027_085402.h5"
    
    print("\n1. Loading model...")
    service = ModelService(model_path)
    service.load_model()
    
    print("\n2. Model Information:")
    info = service.get_model_info()
    
    print(f"   Status: {info['status']}")
    print(f"   Model Path: {info['model_path']}")
    print(f"   Input Shape: {info['input_shape']}")
    print(f"   Outputs: {info['outputs']}")
    print(f"   Total Parameters: {info['total_params']:,}")
    print(f"   Max Length: {info['max_length']}")
    print(f"   Vocab Size: {info['vocab_size']}")
    
    print("\n   Emotion Classes:")
    for cls in info['emotion_classes']:
        print(f"      - {cls}")
    
    print("\n   Hate Classes:")
    for cls in info['hate_classes']:
        print(f"      - {cls}")
    
    print("\n   Violence Classes:")
    for cls in info['violence_classes']:
        print(f"      - {cls}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Service Examples")
    parser.add_argument(
        "--example",
        choices=["single", "batch", "info", "all"],
        default="all",
        help="Example to run"
    )
    
    args = parser.parse_args()
    
    if args.example == "single" or args.example == "all":
        example_single_prediction()
    
    if args.example == "batch" or args.example == "all":
        example_batch_prediction()
    
    if args.example == "info" or args.example == "all":
        example_model_info()

