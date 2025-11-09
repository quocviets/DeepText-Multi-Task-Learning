# -*- coding: utf-8 -*-
"""
Text Preprocessing Module
========================

This module provides text preprocessing and tokenization utilities
for the DeepText Multi-Task Learning model.
"""

import pandas as pd
import numpy as np
import pickle
import re
import string
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')


class TextPreprocessor:
    """
    Text preprocessing class for cleaning and preparing text data.
    """
    
    def __init__(self, 
                 max_length: int = 100,
                 vocab_size: int = 10000,
                 min_word_count: int = 2,
                 remove_punctuation: bool = True,
                 lowercase: bool = True):
        """
        Initialize text preprocessor.
        
        Args:
            max_length: Maximum sequence length
            vocab_size: Maximum vocabulary size
            min_word_count: Minimum word count to include in vocabulary
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert to lowercase
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.min_word_count = min_word_count
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.tokenizer = None
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and normalizing.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text == '':
            return ''
            
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            # Keep Vietnamese characters and basic punctuation
            text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ.,!?]', '', text)
        
        return text.strip()
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           text_column: str = 'text',
                           label_columns: List[str] = None) -> pd.DataFrame:
        """
        Preprocess entire dataframe.
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            label_columns: List of label column names
            
        Returns:
            Preprocessed dataframe
        """
        if label_columns is None:
            label_columns = ['emotion', 'hate', 'violence']
            
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Clean text column
        df_processed[text_column] = df_processed[text_column].apply(self.clean_text)
        
        # Remove rows with empty text
        df_processed = df_processed[df_processed[text_column].str.len() > 0]
        
        # Remove rows with all NaN labels
        if label_columns:
            df_processed = df_processed.dropna(subset=label_columns, how='all')
        
        print(f"Preprocessed {len(df_processed)} samples (removed {len(df) - len(df_processed)} empty samples)")
        
        return df_processed
    
    def fit_tokenizer(self, texts: List[str]) -> None:
        """
        Fit tokenizer on text data.
        
        Args:
            texts: List of text strings
        """
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token='<OOV>',
            filters='',
            lower=self.lowercase
        )
        
        # Filter out empty texts
        texts = [text for text in texts if text and text.strip()]
        
        self.tokenizer.fit_on_texts(texts)
        
        # Remove words with count < min_word_count
        if self.min_word_count > 1:
            word_counts = self.tokenizer.word_counts
            words_to_remove = [word for word, count in word_counts.items() 
                             if count < self.min_word_count]
            
            for word in words_to_remove:
                del self.tokenizer.word_index[word]
                del self.tokenizer.word_counts[word]
        
        print(f"Tokenizer fitted on {len(texts)} texts")
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
    
    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to sequences.
        
        Args:
            texts: List of text strings
            
        Returns:
            Padded sequences array
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call fit_tokenizer() first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_length, 
            padding='post', 
            truncating='post'
        )
        
        return padded_sequences
    
    def encode_labels(self, labels: List[str], 
                     label_type: str = 'emotion') -> np.ndarray:
        """
        Encode labels to categorical format.
        
        Args:
            labels: List of label strings
            label_type: Type of labels ('emotion', 'hate', 'violence')
            
        Returns:
            Encoded labels array
        """
        # Define label mappings
        label_mappings = {
            'emotion': ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo'],
            'hate': ['hate', 'offensive', 'neutral'],
            'violence': ['sex_viol', 'phys_viol', 'no_viol']
        }
        
        if label_type not in label_mappings:
            raise ValueError(f"Unknown label_type: {label_type}")
        
        class_names = label_mappings[label_type]
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        
        # Convert labels to indices
        label_indices = []
        for label in labels:
            if pd.isna(label) or label == '':
                # Use 'neutral' or 'no_emo' as default
                if label_type == 'emotion':
                    label_indices.append(label_to_idx['no_emo'])
                elif label_type == 'hate':
                    label_indices.append(label_to_idx['neutral'])
                elif label_type == 'violence':
                    label_indices.append(label_to_idx['no_viol'])
            else:
                if label in label_to_idx:
                    label_indices.append(label_to_idx[label])
                else:
                    # Handle unknown labels
                    if label_type == 'emotion':
                        label_indices.append(label_to_idx['no_emo'])
                    elif label_type == 'hate':
                        label_indices.append(label_to_idx['neutral'])
                    elif label_type == 'violence':
                        label_indices.append(label_to_idx['no_viol'])
        
        # Convert to categorical
        return to_categorical(label_indices, num_classes=len(class_names))
    
    def save_tokenizer(self, filepath: str) -> None:
        """
        Save tokenizer to file.
        
        Args:
            filepath: Path to save tokenizer
        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save. Call fit_tokenizer() first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Tokenizer saved to {filepath}")
    
    def load_tokenizer(self, filepath: str) -> None:
        """
        Load tokenizer from file.
        
        Args:
            filepath: Path to load tokenizer from
        """
        with open(filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from {filepath}")


class DataProcessor:
    """
    Complete data processing pipeline.
    """
    
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        
    def process_dataset(self, 
                       df: pd.DataFrame,
                       text_column: str = 'text',
                       label_columns: List[str] = None,
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42) -> Dict[str, Any]:
        """
        Complete dataset processing pipeline.
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            label_columns: List of label column names
            test_size: Test set size ratio
            val_size: Validation set size ratio
            random_state: Random seed
            
        Returns:
            Dictionary containing processed data
        """
        if label_columns is None:
            label_columns = ['emotion', 'hate', 'violence']
        
        # Preprocess dataframe
        df_processed = self.preprocessor.preprocess_dataframe(
            df, text_column, label_columns
        )
        
        # Fit tokenizer
        self.preprocessor.fit_tokenizer(df_processed[text_column].tolist())
        
        # Convert texts to sequences
        X = self.preprocessor.texts_to_sequences(df_processed[text_column].tolist())
        
        # Encode labels
        y_emotion = self.preprocessor.encode_labels(
            df_processed['emotion'].tolist(), 'emotion'
        )
        y_hate = self.preprocessor.encode_labels(
            df_processed['hate'].tolist(), 'hate'
        )
        y_violence = self.preprocessor.encode_labels(
            df_processed['violence'].tolist(), 'violence'
        )
        
        # Split data
        X_temp, X_test, y_emotion_temp, y_emotion_test, y_hate_temp, y_hate_test, y_violence_temp, y_violence_test = train_test_split(
            X, y_emotion, y_hate, y_violence,
            test_size=test_size,
            random_state=random_state,
            stratify=y_emotion  # Stratify on emotion for balanced splits
        )
        
        X_train, X_val, y_emotion_train, y_emotion_val, y_hate_train, y_hate_val, y_violence_train, y_violence_val = train_test_split(
            X_temp, y_emotion_temp, y_hate_temp, y_violence_temp,
            test_size=val_size/(1-test_size),
            random_state=random_state,
            stratify=y_emotion_temp
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_emotion_train': y_emotion_train,
            'y_emotion_val': y_emotion_val,
            'y_emotion_test': y_emotion_test,
            'y_hate_train': y_hate_train,
            'y_hate_val': y_hate_val,
            'y_hate_test': y_hate_test,
            'y_violence_train': y_violence_train,
            'y_violence_val': y_violence_val,
            'y_violence_test': y_violence_test,
            'vocab_size': len(self.preprocessor.tokenizer.word_index) + 1,
            'max_length': self.preprocessor.max_length,
            'class_names': {
                'emotion': ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo'],
                'hate': ['hate', 'offensive', 'neutral'],
                'violence': ['sex_viol', 'phys_viol', 'no_viol']
            }
        }
    
    def save_processed_data(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Save processed data to pickle file.
        
        Args:
            data: Processed data dictionary
            filepath: Path to save data
        """
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filepath: str) -> Dict[str, Any]:
        """
        Load processed data from pickle file.
        
        Args:
            filepath: Path to load data from
            
        Returns:
            Processed data dictionary
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Processed data loaded from {filepath}")
        return data


# Convenience function for quick processing
def quick_process_data(csv_path: str,
                      text_column: str = 'text',
                      label_columns: List[str] = None,
                      max_length: int = 100,
                      vocab_size: int = 10000,
                      test_size: float = 0.2,
                      val_size: float = 0.1) -> Dict[str, Any]:
    """
    Quick data processing function.
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column
        label_columns: List of label column names
        max_length: Maximum sequence length
        vocab_size: Maximum vocabulary size
        test_size: Test set size ratio
        val_size: Validation set size ratio
        
    Returns:
        Processed data dictionary
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create preprocessor
    preprocessor = TextPreprocessor(
        max_length=max_length,
        vocab_size=vocab_size
    )
    
    # Create processor
    processor = DataProcessor(preprocessor)
    
    # Process data
    data = processor.process_dataset(
        df, text_column, label_columns, test_size, val_size
    )
    
    return data, preprocessor, processor














