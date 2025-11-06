import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

print("=" * 60)
print("PREDICTION SCRIPT - Sentiment Analysis")
print("=" * 60)

# ============================================
# 0️⃣ DEFINE CUSTOM LAYER
# ============================================
@tf.keras.utils.register_keras_serializable()
class Cast(tf.keras.layers.Layer):
    def __init__(self, dtype='float32', **kwargs):
        super(Cast, self).__init__(**kwargs)
        self.target_dtype = dtype
    
    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)
    
    def get_config(self):
        config = super(Cast, self).get_config()
        config.update({'dtype': self.target_dtype})
        return config

# ============================================
# 1️⃣ LOAD MODEL
# ============================================
print("\n📦 Loading model...")
custom_objects = {'Cast': Cast}
model = tf.keras.models.load_model("models/best_model_20251027_085402.h5", 
                                   custom_objects=custom_objects, 
                                   compile=False)
print("✅ Model loaded successfully")
print(f"   Input shape: {model.input_shape}")
print(f"   Outputs: {len(model.outputs)} outputs")

# ============================================
# 2️⃣ LOAD & PREPARE DATA
# ============================================
print("\n📂 Loading validation data...")
val_df = pd.read_csv("val_clean.csv", sep=';')
print(f"✅ Loaded {len(val_df)} samples")

# Lấy text và labels
texts = val_df['text'].astype(str).values

# ⚙️ Định nghĩa label columns (phải giống lúc train)
emotion_cols = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
hate_cols = ['hate', 'offensive', 'neutral']
violence_cols = ['sex_viol', 'phys_viol', 'no_viol']

# Lấy true labels từ dataset
y_emotion_true = val_df[emotion_cols].values  # shape: (n, 7)
y_hate_true = val_df[hate_cols].values       # shape: (n, 3)
y_violence_true = val_df[violence_cols].values  # shape: (n, 3)

print(f"   Emotion labels shape: {y_emotion_true.shape}")
print(f"   Hate labels shape: {y_hate_true.shape}")
print(f"   Violence labels shape: {y_violence_true.shape}")

# ============================================
# 3️⃣ TOKENIZE & PADDING
# ============================================
print("\n🔤 Tokenizing and padding texts...")
# Load training data để fit tokenizer
train_df = pd.read_csv("train_clean.csv", sep=';')
all_train_texts = train_df['text'].astype(str).values

# Tạo tokenizer (phải giống config lúc train)
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(all_train_texts)
print(f"   Vocabulary size: {len(tokenizer.word_index)}")

# Transform validation texts
sequences = tokenizer.texts_to_sequences(texts)
X_test = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
print(f"✅ X_test shape: {X_test.shape}")

# ============================================
# 4️⃣ PREDICT (với batch size nhỏ hơn để chạy nhanh)
# ============================================
print("\n🚀 Predicting...")
print("   (Có thể mất vài phút...)")

# Giảm batch size xuống 64 để ổn định hơn
preds = model.predict(X_test, batch_size=64, verbose=1)

print("✅ Prediction completed!")
print(f"   Output 1 (emotion): {preds[0].shape}")
print(f"   Output 2 (hate): {preds[1].shape}")
print(f"   Output 3 (violence): {preds[2].shape}")

# ============================================
# 5️⃣ PROCESS PREDICTIONS
# ============================================
print("\n📊 Processing predictions...")

# 🎭 Emotion - Multi-class classification (argmax)
emotion_pred_probs = preds[0]  # (n, 7)
emotion_pred_labels = np.argmax(emotion_pred_probs, axis=1)
emotion_true_labels = np.argmax(y_emotion_true, axis=1)

# 😡 Hate - Multi-label classification (threshold)
HATE_THRESHOLD = 0.5
hate_pred_probs = preds[1]  # (n, 3)
hate_pred_binary = (hate_pred_probs > HATE_THRESHOLD).astype(int)

# ⚔️ Violence - Multi-label classification (threshold)
VIOL_THRESHOLD = 0.5
viol_pred_probs = preds[2]  # (n, 3)
viol_pred_binary = (viol_pred_probs > VIOL_THRESHOLD).astype(int)

# ============================================
# 6️⃣ CREATE RESULTS DATAFRAME
# ============================================
print("\n📑 Creating results DataFrame...")

# Tạo DataFrame với text gốc
df_preds = pd.DataFrame({
    "text_id": np.arange(len(X_test)),
    "text": texts,
    "emotion_true": [emotion_cols[t] for t in emotion_true_labels],
    "emotion_pred": [emotion_cols[p] for p in emotion_pred_labels],
})

# Thêm emotion probabilities
for i, col in enumerate(emotion_cols):
    df_preds[f"emotion_prob_{col}"] = emotion_pred_probs[:, i]

# Thêm Hate predictions
for i, col in enumerate(hate_cols):
    df_preds[f"hate_true_{col}"] = y_hate_true[:, i].astype(int)
    df_preds[f"hate_pred_{col}"] = hate_pred_binary[:, i]
    df_preds[f"hate_prob_{col}"] = hate_pred_probs[:, i]

# Thêm Violence predictions
for i, col in enumerate(violence_cols):
    df_preds[f"viol_true_{col}"] = y_violence_true[:, i].astype(int)
    df_preds[f"viol_pred_{col}"] = viol_pred_binary[:, i]
    df_preds[f"viol_prob_{col}"] = viol_pred_probs[:, i]

# ============================================
# 7️⃣ SAVE TO CSV
# ============================================
timestamp = "20251027_085402"
preds_csv_path = f"predictions_sentiment_{timestamp}.csv"
df_preds.to_csv(preds_csv_path, index=False, encoding='utf-8-sig')

print(f"\n✅ Predictions saved: {preds_csv_path}")
print(f"   Total samples: {len(df_preds)}")
print(f"   Total columns: {len(df_preds.columns)}")

# ============================================
# 8️⃣ SHOW SAMPLE RESULTS
# ============================================
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (first 5 rows)")
print("=" * 60)
sample_cols = ['text', 'emotion_true', 'emotion_pred', 
               'hate_true_hate', 'hate_pred_hate',
               'viol_true_sex_viol', 'viol_pred_sex_viol']
print(df_preds[sample_cols].head())

# ============================================
# 9️⃣ CALCULATE METRICS
# ============================================
print("\n" + "=" * 60)
print("PREDICTION STATISTICS")
print("=" * 60)

# Emotion accuracy
emotion_accuracy = (emotion_pred_labels == emotion_true_labels).mean()
print(f"\n🎭 Emotion Accuracy: {emotion_accuracy:.4f}")

# Emotion distribution
print("\n🎭 Emotion Predictions Distribution:")
for label in emotion_cols:
    count = (df_preds['emotion_pred'] == label).sum()
    percentage = (count / len(df_preds)) * 100
    print(f"   {label:12s}: {count:6d} ({percentage:5.2f}%)")

# Hate statistics
print("\n😡 Hate Predictions:")
for col in hate_cols:
    pred_count = df_preds[f'hate_pred_{col}'].sum()
    true_count = df_preds[f'hate_true_{col}'].sum()
    accuracy = (df_preds[f'hate_pred_{col}'] == df_preds[f'hate_true_{col}']).mean()
    print(f"   {col:12s}: pred={pred_count:6d}, true={true_count:6d}, acc={accuracy:.4f}")

# Violence statistics
print("\n⚔️  Violence Predictions:")
for col in violence_cols:
    pred_count = df_preds[f'viol_pred_{col}'].sum()
    true_count = df_preds[f'viol_true_{col}'].sum()
    accuracy = (df_preds[f'viol_pred_{col}'] == df_preds[f'viol_true_{col}']).mean()
    print(f"   {col:12s}: pred={pred_count:6d}, true={true_count:6d}, acc={accuracy:.4f}")

print("\n" + "=" * 60)
print("✅ COMPLETED!")
print("=" * 60)
print(f"\nFiles created:")
print(f"  1. {preds_csv_path}")
print(f"  2. models/training_log.csv (already exists)")
print(f"\n📝 Next steps:")
print(f"  - Review predictions: pd.read_csv('{preds_csv_path}')")
print(f"  - Review training log: pd.read_csv('models/training_log.csv')")
print(f"  - Model ready for new predictions: model = tf.keras.models.load_model('models/best_model_20251027_085402.h5')")

