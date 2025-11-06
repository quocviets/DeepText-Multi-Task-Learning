"""
Model Service Layer - Tích hợp với checkpoint models
Load models từ checkpoint và xử lý predictions
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Dict, List, Tuple, Optional
import json


@tf.keras.utils.register_keras_serializable()
class Cast(tf.keras.layers.Layer):
    """Custom layer để load model"""
    def __init__(self, dtype='float32', **kwargs):
        super(Cast, self).__init__(**kwargs)
        self.target_dtype = dtype
    
    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)
    
    def get_config(self):
        config = super(Cast, self).get_config()
        config.update({'dtype': self.target_dtype})
        return config


class ModelService:
    """Service để load và sử dụng model từ checkpoint"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Khởi tạo ModelService
        
        Args:
            model_path: Đường dẫn đến file model (.h5)
            config_path: Đường dẫn đến file config (optional)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.tokenizer = None
        self.config = None
        
        # Task configurations
        self.emotion_classes = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
        self.hate_classes = ['hate', 'offensive', 'neutral']
        self.violence_classes = ['sex_viol', 'phys_viol', 'no_viol']
        
        # Model parameters (mặc định từ code)
        self.max_words = 10000
        self.max_len = 100
        
        # Thresholds
        self.hate_threshold = 0.5
        self.violence_threshold = 0.5
        
    def load_config(self):
        """Load config từ file JSON"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                
            # Update parameters từ config
            if 'model' in self.config:
                model_config = self.config['model']
                self.max_words = model_config.get('vocab_size', self.max_words)
                self.max_len = model_config.get('max_length', self.max_len)
                
                if 'emotion_classes' in model_config:
                    self.emotion_classes = model_config['emotion_classes']
                if 'hate_classes' in model_config:
                    self.hate_classes = model_config['hate_classes']
                if 'violence_classes' in model_config:
                    self.violence_classes = model_config['violence_classes']
    
    def load_model(self):
        """Load model từ checkpoint"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        custom_objects = {'Cast': Cast}
        
        try:
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects=custom_objects,
                compile=False
            )
            print("✅ Model loaded successfully")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Outputs: {len(self.model.outputs)} outputs")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def load_tokenizer(self, train_data_path: Optional[str] = None):
        """
        Load hoặc tạo tokenizer
        
        Args:
            train_data_path: Đường dẫn đến training data để fit tokenizer
                           (nếu None, sẽ tạo tokenizer mới)
        """
        print("Loading tokenizer...")
        
        if train_data_path and os.path.exists(train_data_path):
            try:
                # Thử đọc CSV với nhiều separator khác nhau
                train_df = None
                for sep in [';', ',', '\t']:
                    try:
                        train_df = pd.read_csv(train_data_path, sep=sep, encoding='utf-8')
                        if 'text' in train_df.columns:
                            break
                    except:
                        continue
                
                # Nếu không tìm thấy, thử đọc không có separator
                if train_df is None or 'text' not in train_df.columns:
                    train_df = pd.read_csv(train_data_path, encoding='utf-8')
                
                # Kiểm tra cột 'text'
                if 'text' not in train_df.columns:
                    # Tìm cột có thể chứa text (cột đầu tiên hoặc có tên tương tự)
                    possible_cols = [col for col in train_df.columns 
                                    if 'text' in col.lower() or 'content' in col.lower()]
                    if possible_cols:
                        text_col = possible_cols[0]
                        all_texts = train_df[text_col].astype(str).values
                        print(f"⚠️  Using column '{text_col}' instead of 'text'")
                    else:
                        # Sử dụng cột đầu tiên
                        text_col = train_df.columns[0]
                        all_texts = train_df[text_col].astype(str).values
                        print(f"⚠️  Using first column '{text_col}' as text")
                else:
                    all_texts = train_df['text'].astype(str).values
                
                # Fit tokenizer
                self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
                self.tokenizer.fit_on_texts(all_texts)
                print(f"✅ Tokenizer fitted from training data")
                print(f"   File: {train_data_path}")
                print(f"   Vocabulary size: {len(self.tokenizer.word_index)}")
                print(f"   Number of texts: {len(all_texts)}")
                
            except Exception as e:
                raise Exception(f"Error loading tokenizer from {train_data_path}: {str(e)}")
        else:
            # Tạo tokenizer mới (chỉ dùng cho demo, không khuyến khích)
            self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
            print("⚠️  Warning: Using new tokenizer without training data")
            print("⚠️  This may cause prediction errors due to vocabulary mismatch!")
    
    def preprocess_text(self, text: str) -> np.ndarray:
        """
        Preprocess text thành sequence
        
        Args:
            text: Input text
            
        Returns:
            Padded sequence array
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer chưa được load. Hãy gọi load_tokenizer() trước.")
        
        # Convert text thành sequence
        sequences = self.tokenizer.texts_to_sequences([text])
        
        # Pad sequence
        padded = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding='post',
            truncating='post'
        )
        
        return padded
    
    def predict(self, text: str) -> Dict:
        """
        Predict từ text input
        
        Args:
            text: Input text
            
        Returns:
            Dictionary chứa predictions cho 3 tasks
        """
        if self.model is None:
            raise ValueError("Model chưa được load. Hãy gọi load_model() trước.")
        
        # Preprocess
        X = self.preprocess_text(text)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        
        # Process predictions
        emotion_probs = predictions[0][0]  # (7,)
        hate_probs = predictions[1][0]     # (3,)
        violence_probs = predictions[2][0] # (3,)
        
        # Emotion: argmax (single class)
        emotion_idx = np.argmax(emotion_probs)
        emotion_label = self.emotion_classes[emotion_idx]
        emotion_confidence = float(emotion_probs[emotion_idx])
        
        # Hate: threshold-based (multi-label)
        hate_labels = []
        hate_confidences = {}
        for i, class_name in enumerate(self.hate_classes):
            prob = float(hate_probs[i])
            if prob > self.hate_threshold:
                hate_labels.append(class_name)
            hate_confidences[class_name] = prob
        
        # Violence: threshold-based (multi-label)
        violence_labels = []
        violence_confidences = {}
        for i, class_name in enumerate(self.violence_classes):
            prob = float(violence_probs[i])
            if prob > self.violence_threshold:
                violence_labels.append(class_name)
            violence_confidences[class_name] = prob
        
        return {
            'emotion': {
                'label': emotion_label,
                'confidence': emotion_confidence,
                'probabilities': {self.emotion_classes[i]: float(emotion_probs[i]) 
                                 for i in range(len(self.emotion_classes))}
            },
            'hate': {
                'labels': hate_labels,
                'confidences': hate_confidences,
                'probabilities': hate_confidences
            },
            'violence': {
                'labels': violence_labels,
                'confidences': violence_confidences,
                'probabilities': violence_confidences
            },
            'raw_predictions': {
                'emotion': emotion_probs.tolist(),
                'hate': hate_probs.tolist(),
                'violence': violence_probs.tolist()
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("Model chưa được load. Hãy gọi load_model() trước.")
        
        # Preprocess all texts
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        # Predict
        predictions = self.model.predict(X, verbose=0, batch_size=32)
        
        # Process each prediction
        results = []
        for i in range(len(texts)):
            emotion_probs = predictions[0][i]
            hate_probs = predictions[1][i]
            violence_probs = predictions[2][i]
            
            # Emotion
            emotion_idx = np.argmax(emotion_probs)
            emotion_label = self.emotion_classes[emotion_idx]
            
            # Hate
            hate_labels = [self.hate_classes[j] for j in range(len(self.hate_classes))
                          if hate_probs[j] > self.hate_threshold]
            
            # Violence
            violence_labels = [self.violence_classes[j] for j in range(len(self.violence_classes))
                             if violence_probs[j] > self.violence_threshold]
            
            results.append({
                'text': texts[i],
                'emotion': {
                    'label': emotion_label,
                    'confidence': float(emotion_probs[emotion_idx]),
                    'probabilities': {self.emotion_classes[j]: float(emotion_probs[j])
                                    for j in range(len(self.emotion_classes))}
                },
                'hate': {
                    'labels': hate_labels,
                    'confidences': {self.hate_classes[j]: float(hate_probs[j])
                                  for j in range(len(self.hate_classes))}
                },
                'violence': {
                    'labels': violence_labels,
                    'confidences': {self.violence_classes[j]: float(violence_probs[j])
                                  for j in range(len(self.violence_classes))}
                }
            })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Lấy thông tin về model"""
        if self.model is None:
            return {"status": "Model chưa được load"}
        
        return {
            "status": "Model đã được load",
            "model_path": self.model_path,
            "input_shape": str(self.model.input_shape),
            "outputs": len(self.model.outputs),
            "total_params": self.model.count_params(),
            "emotion_classes": self.emotion_classes,
            "hate_classes": self.hate_classes,
            "violence_classes": self.violence_classes,
            "max_length": self.max_len,
            "vocab_size": self.max_words
        }


# Singleton instance
_model_service_instance = None

def get_model_service(model_path: str = None, 
                     config_path: str = None,
                     train_data_path: str = None) -> ModelService:
    """
    Get hoặc tạo ModelService instance (singleton pattern)
    
    Args:
        model_path: Đường dẫn đến model file
        config_path: Đường dẫn đến config file
        train_data_path: Đường dẫn đến training data để fit tokenizer
        
    Returns:
        ModelService instance
    """
    global _model_service_instance
    
    if _model_service_instance is None:
        if model_path is None:
            raise ValueError("model_path is required for first initialization")
        
        _model_service_instance = ModelService(model_path, config_path)
        _model_service_instance.load_config()
        _model_service_instance.load_model()
        
        # Load tokenizer
        tokenizer_loaded = False
        
        if train_data_path and os.path.exists(train_data_path):
            _model_service_instance.load_tokenizer(train_data_path)
            tokenizer_loaded = True
        else:
            # Tìm training data trong project với nhiều đường dẫn khác nhau
            possible_paths = [
                "DeepText-MTL/checkpoints/train_clean.csv",
                "../DeepText-MTL/checkpoints/train_clean.csv",
                "../../DeepText-MTL/checkpoints/train_clean.csv",
                "checkpoints/train_clean.csv",
                "../checkpoints/train_clean.csv",
                "train_clean.csv",
                os.path.join(os.path.dirname(os.path.dirname(model_path)), "train_clean.csv"),
                os.path.join(os.path.dirname(model_path), "train_clean.csv"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        _model_service_instance.load_tokenizer(path)
                        tokenizer_loaded = True
                        print(f"✅ Tokenizer loaded from: {path}")
                        break
                    except Exception as e:
                        print(f"⚠️  Failed to load tokenizer from {path}: {str(e)}")
                        continue
        
        # Kiểm tra xem tokenizer đã được load chưa
        if not tokenizer_loaded or _model_service_instance.tokenizer is None:
            raise ValueError(
                "Tokenizer chưa được load! Vui lòng cung cấp đường dẫn đến training data.\n"
                f"Đã tìm trong các đường dẫn sau nhưng không tìm thấy:\n"
                f"- DeepText-MTL/checkpoints/train_clean.csv\n"
                f"- checkpoints/train_clean.csv\n"
                f"- train_clean.csv\n"
                f"\nVui lòng nhập đường dẫn đúng trong sidebar."
            )
    
    return _model_service_instance


def reset_model_service():
    """Reset singleton instance để cho phép load model mới"""
    global _model_service_instance
    _model_service_instance = None

