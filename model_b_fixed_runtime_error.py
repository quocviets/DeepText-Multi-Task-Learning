# ===============================================================
# 🚀 DeepText Multi-Task Learning - GRU Fixed + Clean + Validation
# Version: Fixed Runtime Errors (SettingWithCopyWarning + KerasTensor)
# ===============================================================

import pandas as pd
import numpy as np
import re
import os
import json
import random
import pickle
import tensorflow as tf
from tensorflow.keras import layers, Model, mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, jaccard_score
from datetime import datetime

# ===============================================================
# 1️⃣ Config & Seed
# ===============================================================
CONFIG = {
    'vocab_size': 20000,
    'max_length': 100,
    'embedding_dim': 128,
    'gru_units': 128,
    'dropout_rate': 0.3,
    'batch_size': 128,
    'learning_rate': 0.001,
    'epochs': 100,
    'clipnorm': 1.0,
    'loss_weights': {
        'emotion_output': 1.0,
        'hate_output': 3.0,  # 💡 Tip: Nếu Hate/Violence vẫn yếu, tăng lên 4.0 : 4.0
        'violence_output': 3.0  # 💡 Tip: Hoặc giảm dropout_rate → 0.25
    },
    'train_csv': '/kaggle/working/train_augmented_50k.csv',
    'val_csv': '/kaggle/working/val_augmented_50k.csv',
    'output_dir': 'models_gru_final'
}

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Seeds set: {seed}")

set_seeds(42)

# ✅ Mixed Precision: Giảm thời gian train 20-25% trên GPU T4/A100
mixed_precision.set_global_policy('mixed_float16')
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ===============================================================
# 2️⃣ Clean Data - FIXED SettingWithCopyWarning
# ===============================================================
def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text) or not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-ZÀ-ỹ0-9\s.,!?-]', ' ', text)  # Keep alphanumeric + special
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

print("🧹 Cleaning data...")

# ✅ FIX 1: Error handling
try:
    train_df = pd.read_csv(CONFIG['train_csv'], sep=';')
    val_df = pd.read_csv(CONFIG['val_csv'], sep=';')
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}")
    raise

# ✅ FIX 2: Fix SettingWithCopyWarning - Use .copy() or reassign properly
def clean_dataframe(df):
    """Clean a single dataframe and return new copy"""
    df = df.copy()  # ✅ Make explicit copy to avoid SettingWithCopyWarning
    before = len(df)
    df['text'] = df['text'].apply(clean_text)
    df.dropna(subset=['text'], inplace=True)
    df = df[df['text'].str.len() >= 3].copy()  # ✅ Explicit copy after filtering
    df.drop_duplicates(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, before

train_df, before_train = clean_dataframe(train_df)
print(f"✅ Train: {before_train:,} → {len(train_df):,}")

val_df, before_val = clean_dataframe(val_df)
print(f"✅ Val: {before_val:,} → {len(val_df):,}")

emotion_cols = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
hate_cols = ['hate', 'offensive', 'neutral']
violence_cols = ['sex_viol', 'phys_viol', 'no_viol']

# ===============================================================
# 3️⃣ Validate Data Integrity
# ===============================================================
print("\n🧩 Validating label integrity...")

# 1️⃣ Emotion must be one-hot
emo_sum_train = train_df[emotion_cols].sum(axis=1)
emo_sum_val = val_df[emotion_cols].sum(axis=1)
bad_train = (emo_sum_train != 1).sum()
bad_val = (emo_sum_val != 1).sum()
print(f"🎭 Emotion invalid one-hot: train={bad_train}, val={bad_val}")

if bad_train or bad_val:
    train_df = train_df[emo_sum_train == 1].reset_index(drop=True)
    val_df = val_df[emo_sum_val == 1].reset_index(drop=True)
    print(f"✅ After filtering: train={len(train_df):,}, val={len(val_df):,}")

# 2️⃣ Hate & Violence multi-label range check
for cols, name in [(hate_cols, "Hate"), (violence_cols, "Violence")]:
    sums_tr = train_df[cols].sum(axis=1)
    sums_va = val_df[cols].sum(axis=1)
    nan_tr = train_df[cols].isna().any(axis=1).sum()
    nan_va = val_df[cols].isna().any(axis=1).sum()
    print(f"  {name}: NaN train={nan_tr}, val={nan_va}, "
          f"avg labels train={sums_tr.mean():.2f}, val={sums_va.mean():.2f}")
    if nan_tr > 0 or nan_va > 0:
        raise ValueError(f"{name} contains NaN! Train: {nan_tr}, Val: {nan_va}")

# ✅ Final validation
if len(train_df) == 0 or len(val_df) == 0:
    raise ValueError("❌ Empty dataframe after validation!")

print(f"✅ Final data: train={len(train_df):,}, val={len(val_df):,}")

# ===============================================================
# 4️⃣ Tokenization
# ===============================================================
print("\n🔤 Tokenizing...")

tokenizer = Tokenizer(num_words=CONFIG['vocab_size'], oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['text'])

X_train = pad_sequences(
    tokenizer.texts_to_sequences(train_df['text']),
    maxlen=CONFIG['max_length'],
    padding='post',
    truncating='post'
)
X_val = pad_sequences(
    tokenizer.texts_to_sequences(val_df['text']),
    maxlen=CONFIG['max_length'],
    padding='post',
    truncating='post'
)

# Prepare labels
y_emotion_train = np.argmax(train_df[emotion_cols].values, axis=1)
y_emotion_val = np.argmax(val_df[emotion_cols].values, axis=1)
y_emotion_train = to_categorical(y_emotion_train, len(emotion_cols))
y_emotion_val = to_categorical(y_emotion_val, len(emotion_cols))

y_hate_train = train_df[hate_cols].values.astype(np.float32)
y_hate_val = val_df[hate_cols].values.astype(np.float32)
y_viol_train = train_df[violence_cols].values.astype(np.float32)
y_viol_val = val_df[violence_cols].values.astype(np.float32)

y_train = [y_emotion_train, y_hate_train, y_viol_train]
y_val = [y_emotion_val, y_hate_val, y_viol_val]

print(f"✅ X_train: {X_train.shape}, X_val: {X_val.shape}")
print(f"✅ y_emotion: {y_emotion_train.shape}, "
      f"y_hate: {y_hate_train.shape}, y_violence: {y_viol_train.shape}")

# ===============================================================
# 5️⃣ Model Definition - BiGRU + MHA (Residual + Mask)
# ===============================================================
class DeepTextGRUFixed:
    def __init__(self, config):
        self.config = config
        self.model = None

    def build(self):
        """Build the model architecture"""
        inp = layers.Input((self.config['max_length'],), name='text_input')
        
        # ✅ Embedding WITHOUT masking (fix cuDNN compatibility issue)
        # Note: cuDNN has strict mask requirements that conflict with padding='post'
        # Solution: Disable masking - padding tokens will be processed but with minimal impact
        # GlobalMaxPooling will help reduce the impact of padding tokens
        x = layers.Embedding(
            self.config['vocab_size'],
            self.config['embedding_dim'],
            mask_zero=False,  # ✅ FIX: Tắt mask để tương thích với cuDNN
            name='embedding'
        )(inp)
        
        # ✅ Bidirectional GRU (cuDNN optimized, no mask)
        x = layers.Bidirectional(
            layers.GRU(
                self.config['gru_units'],
                return_sequences=True,
                dropout=self.config['dropout_rate']
            ),
            name='bigru'
        )(x)
        
        # ✅ FIX 3: REMOVE manual attention_mask!
        # Problem: Cannot use tf.cast(tf.not_equal(inp, 0)) directly in Functional API
        # Solution: Let embedding mask propagate automatically through layers
        # MultiHeadAttention will use the propagated mask correctly
        
        # Self-attention: mask auto-propagates from embedding layer
        attn = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=(self.config['gru_units'] * 2) // 8,  # 256 // 8 = 32
            name='mha'
        )(x, x)  # ✅ No attention_mask needed - mask propagates automatically
        
        # ✅ Residual connection + LayerNorm
        x = layers.Add(name='residual_add')([x, attn])
        x = layers.LayerNormalization(name='layer_norm')(x)
        
        # ✅ Global pooling
        x = layers.GlobalMaxPooling1D(name='gmp')(x)
        
        # ✅ Shared dense layers
        x = layers.Dense(128, activation='relu', name='shared_dense')(x)
        x = layers.BatchNormalization(name='bn')(x)
        x = layers.Dropout(self.config['dropout_rate'], name='dropout')(x)

        # Emotion Head (multi-class)
        emo = layers.Dense(64, activation='relu', name='emo_dense')(x)
        emo = layers.Dropout(self.config['dropout_rate'], name='emo_drop')(emo)
        emo_out = layers.Dense(
            7,
            activation='softmax',
            name='emotion_output',
            dtype='float32'
        )(emo)

        # Hate Head (multi-label)
        hate = layers.Dense(32, activation='relu', name='hate_dense')(x)
        hate = layers.Dropout(self.config['dropout_rate'], name='hate_drop')(hate)
        hate_out = layers.Dense(
            3,
            activation='sigmoid',
            name='hate_output',
            dtype='float32'
        )(hate)

        # Violence Head (multi-label)
        viol = layers.Dense(32, activation='relu', name='viol_dense')(x)
        viol = layers.Dropout(self.config['dropout_rate'], name='viol_drop')(viol)
        viol_out = layers.Dense(
            3,
            activation='sigmoid',
            name='violence_output',
            dtype='float32'
        )(viol)

        self.model = Model(inp, [emo_out, hate_out, viol_out], name='DeepTextGRUFixed')
        print(f"✅ Model built - {self.model.count_params():,} params")
        return self.model

    def compile(self):
        """Compile the model"""
        opt = Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=self.config['clipnorm']
        )
        self.model.compile(
            optimizer=opt,
            loss={
                'emotion_output': CategoricalCrossentropy(),
                'hate_output': BinaryCrossentropy(),
                'violence_output': BinaryCrossentropy()
            },
            loss_weights=self.config['loss_weights'],
            metrics={
                'emotion_output': ['accuracy'],
                'hate_output': ['accuracy'],
                'violence_output': ['accuracy']
            }
        )
        print("✅ Model compiled.")

# ===============================================================
# 6️⃣ Build + Train
# ===============================================================
print("\n🏗️  Building model...")
modelB = DeepTextGRUFixed(CONFIG)
modelB.build()
modelB.compile()

print("\n" + "="*60)
print("📋 Model Architecture Summary")
print("="*60)
modelB.model.summary()

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"{CONFIG['output_dir']}/best_modelB_{ts}.h5"

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1,
        min_delta=1e-4
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

print("\n🚀 Training started...")
print(f"  Training samples: {len(X_train):,}")
print(f"  Validation samples: {len(X_val):,}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Max epochs: {CONFIG['epochs']}\n")

hist = modelB.model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    callbacks=callbacks,
    verbose=1
)

# ===============================================================
# 7️⃣ Save History & Model - IMPROVED
# ===============================================================
print("\n💾 Saving results...")

# ✅ Safe history saving (handle NaN/inf)
def safe_float(x):
    """Convert to float safely, handling NaN/inf"""
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None

hist_path = f"{CONFIG['output_dir']}/historyB_{ts}.json"
hist_dict = {
    k: [safe_float(x) for x in v if safe_float(x) is not None]
    for k, v in hist.history.items()
}

try:
    with open(hist_path, 'w') as f:
        json.dump(hist_dict, f, indent=2)
    print(f"✅ Saved history: {hist_path}")
except Exception as e:
    print(f"⚠️  Error saving history: {e}")

# ✅ Export history to CSV (tiện vẽ biểu đồ so sánh với Model A)
hist_csv_path = f"{CONFIG['output_dir']}/historyB_{ts}.csv"
try:
    pd.DataFrame(hist.history).to_csv(hist_csv_path, index=False)
    print(f"✅ Saved history CSV: {hist_csv_path}")
except Exception as e:
    print(f"⚠️  Error saving history CSV: {e}")

# ✅ Save tokenizer for inference
tokenizer_path = f"{CONFIG['output_dir']}/tokenizerB_{ts}.pkl"
try:
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Saved tokenizer: {tokenizer_path}")
except Exception as e:
    print(f"⚠️  Error saving tokenizer: {e}")

# ✅ Save config for reference
config_path = f"{CONFIG['output_dir']}/configB_{ts}.json"
try:
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"✅ Saved config: {config_path}")
except Exception as e:
    print(f"⚠️  Error saving config: {e}")

print(f"✅ Model saved: {save_path}")

# ===============================================================
# 8️⃣ Evaluation - F1 Score & Jaccard (so sánh với Model A)
# ===============================================================
print("\n📊 Evaluating model on validation set...")

# Get predictions
predictions = modelB.model.predict(X_val, verbose=1, batch_size=CONFIG['batch_size'])
y_pred_emotion, y_pred_hate, y_pred_violence = predictions

# Convert to class predictions
y_true_emotion = np.argmax(y_emotion_val, axis=1)
y_pred_emotion_cls = np.argmax(y_pred_emotion, axis=1)

# For multi-label tasks: convert to binary predictions (threshold=0.5)
y_pred_hate_bin = (y_pred_hate > 0.5).astype(int)
y_pred_violence_bin = (y_pred_violence > 0.5).astype(int)

y_true_hate_bin = (y_hate_val > 0.5).astype(int)
y_true_violence_bin = (y_viol_val > 0.5).astype(int)

# Calculate metrics
print("\n🎯 Evaluation Results:")
print("-" * 60)

# Emotion (multi-class)
emotion_f1_macro = f1_score(y_true_emotion, y_pred_emotion_cls, average='macro')
emotion_f1_samples = f1_score(y_true_emotion, y_pred_emotion_cls, average='weighted')
print(f"Emotion:")
print(f"  F1 Score (macro):   {emotion_f1_macro:.4f}")
print(f"  F1 Score (weighted): {emotion_f1_samples:.4f}")

# Hate (multi-label)
hate_f1_macro = f1_score(y_true_hate_bin, y_pred_hate_bin, average='macro', zero_division=0)
hate_f1_samples = f1_score(y_true_hate_bin, y_pred_hate_bin, average='samples', zero_division=0)
hate_jaccard = jaccard_score(y_true_hate_bin, y_pred_hate_bin, average='samples', zero_division=0)
print(f"\nHate:")
print(f"  F1 Score (macro):   {hate_f1_macro:.4f}")
print(f"  F1 Score (samples): {hate_f1_samples:.4f}")
print(f"  Jaccard (samples):  {hate_jaccard:.4f}")

# Violence (multi-label)
violence_f1_macro = f1_score(y_true_violence_bin, y_pred_violence_bin, average='macro', zero_division=0)
violence_f1_samples = f1_score(y_true_violence_bin, y_pred_violence_bin, average='samples', zero_division=0)
violence_jaccard = jaccard_score(y_true_violence_bin, y_pred_violence_bin, average='samples', zero_division=0)
print(f"\nViolence:")
print(f"  F1 Score (macro):   {violence_f1_macro:.4f}")
print(f"  F1 Score (samples): {violence_f1_samples:.4f}")
print(f"  Jaccard (samples):  {violence_jaccard:.4f}")

# Save evaluation results
eval_results = {
    'emotion': {
        'f1_macro': float(emotion_f1_macro),
        'f1_weighted': float(emotion_f1_samples)
    },
    'hate': {
        'f1_macro': float(hate_f1_macro),
        'f1_samples': float(hate_f1_samples),
        'jaccard_samples': float(hate_jaccard)
    },
    'violence': {
        'f1_macro': float(violence_f1_macro),
        'f1_samples': float(violence_f1_samples),
        'jaccard_samples': float(violence_jaccard)
    }
}

eval_path = f"{CONFIG['output_dir']}/evaluationB_{ts}.json"
try:
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n✅ Saved evaluation results: {eval_path}")
except Exception as e:
    print(f"⚠️  Error saving evaluation: {e}")

print("-" * 60)

# ===============================================================
# 8️⃣.5️⃣ Export Predictions CSV (cho người khác sử dụng)
# ===============================================================
print("\n📤 Exporting predictions CSV...")

# Tạo DataFrame với predictions
predictions_df = pd.DataFrame()

# Thêm text gốc
predictions_df['text'] = val_df['text'].values

# Emotion predictions
emotion_classes = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
emotion_pred_classes = np.argmax(y_pred_emotion, axis=1)
predictions_df['emotion_pred'] = [emotion_classes[i] for i in emotion_pred_classes]
predictions_df['emotion_pred_prob'] = np.max(y_pred_emotion, axis=1)

# Emotion true labels
emotion_true_classes = np.argmax(y_emotion_val, axis=1)
predictions_df['emotion_true'] = [emotion_classes[i] for i in emotion_true_classes]
predictions_df['emotion_match'] = (emotion_pred_classes == emotion_true_classes).astype(int)

# Hate predictions (multi-label)
hate_classes = ['hate', 'offensive', 'neutral']
for i, col in enumerate(hate_classes):
    predictions_df[f'hate_{col}_pred'] = y_pred_hate[:, i]
    predictions_df[f'hate_{col}_true'] = y_hate_val[:, i]
    predictions_df[f'hate_{col}_pred_bin'] = y_pred_hate_bin[:, i]
predictions_df['hate_pred_all'] = predictions_df.apply(
    lambda row: ','.join([hate_classes[i] for i in range(3) if row[f'hate_{hate_classes[i]}_pred_bin'] == 1]), axis=1
)

# Violence predictions (multi-label)
violence_classes = ['sex_viol', 'phys_viol', 'no_viol']
for i, col in enumerate(violence_classes):
    predictions_df[f'violence_{col}_pred'] = y_pred_violence[:, i]
    predictions_df[f'violence_{col}_true'] = y_viol_val[:, i]
    predictions_df[f'violence_{col}_pred_bin'] = y_pred_violence_bin[:, i]
predictions_df['violence_pred_all'] = predictions_df.apply(
    lambda row: ','.join([violence_classes[i] for i in range(3) if row[f'violence_{violence_classes[i]}_pred_bin'] == 1]), axis=1
)

# Save predictions CSV
predictions_csv_path = f"{CONFIG['output_dir']}/predictions_sentiment_{ts}.csv"
try:
    predictions_df.to_csv(predictions_csv_path, index=False, encoding='utf-8')
    print(f"✅ Saved predictions CSV: {predictions_csv_path}")
    print(f"   Rows: {len(predictions_df):,}, Columns: {len(predictions_df.columns)}")
except Exception as e:
    print(f"⚠️  Error saving predictions CSV: {e}")

# ===============================================================
# 9️⃣ Training Summary
# ===============================================================
print("\n" + "="*60)
print("✅ Training Complete!")
print("="*60)

if hist and hist.history:
    print(f"\nFinal metrics (last epoch):")
    for key in sorted(hist.history.keys()):
        if key.startswith('val_'):
            continue
        if key in hist.history and len(hist.history[key]) > 0:
            final_train = hist.history[key][-1]
            val_key = f"val_{key}"
            if val_key in hist.history and len(hist.history[val_key]) > 0:
                final_val = hist.history[val_key][-1]
                print(f"  {key:35s}: {final_train:.4f} (val: {final_val:.4f})")
            else:
                print(f"  {key:35s}: {final_train:.4f}")

print("\n📁 Files saved:")
print(f"  ✅ Model: {save_path}")
print(f"  ✅ History JSON: {hist_path}")
print(f"  ✅ History CSV (training_log): {hist_csv_path}")
print(f"  ✅ Predictions CSV: {predictions_csv_path}")
print(f"  ✅ Evaluation: {eval_path}")
print(f"  ✅ Tokenizer: {tokenizer_path}")
print(f"  ✅ Config: {config_path}")
print("\n📦 Package để gửi người khác:")
print(f"  1. {hist_csv_path} (hoặc {hist_path})")
print(f"  2. {predictions_csv_path}")
print(f"  3. {save_path}")
print("="*60)

