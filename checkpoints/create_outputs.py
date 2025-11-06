import pandas as pd
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# ============================================
# PHẦN 1: TẠO TRAINING_LOG.CSV TỪ HISTORY.JSON
# ============================================
print("=" * 60)
print("PHẦN 1: Tạo training_log.csv từ history.json")
print("=" * 60)

# Đọc file history JSON
history_path = "models/history_20251027_085402.json"
with open(history_path, 'r') as f:
    history_data = json.load(f)

# Tạo DataFrame từ history
history_df = pd.DataFrame(history_data)

# Thêm cột epoch (bắt đầu từ 1)
history_df.insert(0, 'epoch', range(1, len(history_df) + 1))

# Lưu ra training_log.csv
training_log_path = "models/training_log.csv"
history_df.to_csv(training_log_path, index=False)
print(f"✅ Đã tạo file: {training_log_path}")
print(f"   Số epochs: {len(history_df)}")
print(f"   Các cột: {', '.join(history_df.columns)}")
print()

# ============================================
# PHẦN 2: LOAD MODEL VÀ TẠO PREDICTIONS
# ============================================
print("=" * 60)
print("PHẦN 2: Load model và tạo predictions")
print("=" * 60)

# Load model đã train
model_path = "models/best_model_20251027_085402.h5"
print(f"📦 Đang load model từ: {model_path}")
model = load_model(model_path)
print("✅ Model đã được load thành công!")
print(f"   Model inputs: {model.input_shape}")
print(f"   Model outputs: {[output.shape for output in model.outputs]}")
print()

# Đọc dữ liệu validation (sử dụng làm test set)
print("📂 Đang đọc dữ liệu validation...")
val_df = pd.read_csv("val_clean.csv", sep=';')
print(f"✅ Đã đọc {len(val_df)} mẫu từ val_clean.csv")
print()

# Chuẩn bị text data
texts = val_df['text'].astype(str).values

# Tokenize và padding (phải match với training)
print("🔤 Đang tokenize và padding text...")
# Đọc toàn bộ train data để fit tokenizer giống như khi train
train_df = pd.read_csv("train_clean.csv", sep=';')
all_texts = train_df['text'].astype(str).values

# Tạo tokenizer
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(all_texts)

# Transform validation texts
sequences = tokenizer.texts_to_sequences(texts)
X_val = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
print(f"✅ Text đã được xử lý: shape = {X_val.shape}")
print()

# Predict
print("🔮 Đang thực hiện prediction...")
predictions = model.predict(X_val, batch_size=128, verbose=1)
print()

# ============================================
# PHẦN 3: XỬ LÝ VÀ LƯU PREDICTIONS
# ============================================
print("=" * 60)
print("PHẦN 3: Xử lý và lưu predictions")
print("=" * 60)

# Model có 3 outputs: emotion (7 classes), hate (3 classes), violence (3 classes)
emotion_pred = predictions[0]  # (samples, 7)
hate_pred = predictions[1]     # (samples, 3)
violence_pred = predictions[2] # (samples, 3)

print(f"Emotion predictions shape: {emotion_pred.shape}")
print(f"Hate predictions shape: {hate_pred.shape}")
print(f"Violence predictions shape: {violence_pred.shape}")
print()

# Tạo DataFrame kết quả
result_df = pd.DataFrame()
result_df['text'] = texts

# Thêm emotion predictions (7 cột: sad, joy, love, angry, fear, surprise, no_emo)
emotion_labels = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
for i, label in enumerate(emotion_labels):
    result_df[f'pred_{label}'] = emotion_pred[:, i]
    result_df[f'pred_{label}_binary'] = (emotion_pred[:, i] > 0.5).astype(int)

# Thêm hate predictions (3 cột: hate, offensive, neutral)
hate_labels = ['hate', 'offensive', 'neutral']
for i, label in enumerate(hate_labels):
    result_df[f'pred_{label}'] = hate_pred[:, i]
    result_df[f'pred_{label}_binary'] = (hate_pred[:, i] > 0.5).astype(int)

# Thêm violence predictions (3 cột: sex_viol, phys_viol, no_viol)
violence_labels = ['sex_viol', 'phys_viol', 'no_viol']
for i, label in enumerate(violence_labels):
    result_df[f'pred_{label}'] = violence_pred[:, i]
    result_df[f'pred_{label}_binary'] = (violence_pred[:, i] > 0.5).astype(int)

# Thêm ground truth labels (nếu có)
for label in emotion_labels + hate_labels + violence_labels:
    if label in val_df.columns:
        result_df[f'true_{label}'] = val_df[label].values

# Lưu file predictions
predictions_path = "predictions_sentiment.csv"
result_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
print(f"✅ Đã lưu predictions: {predictions_path}")
print(f"   Số mẫu: {len(result_df)}")
print(f"   Số cột: {len(result_df.columns)}")
print()

# In thống kê
print("📊 Thống kê predictions:")
print("-" * 60)

# Emotion statistics
print("\n🎭 Emotion Predictions:")
for label in emotion_labels:
    count = result_df[f'pred_{label}_binary'].sum()
    percentage = (count / len(result_df)) * 100
    print(f"  {label:12s}: {count:6d} samples ({percentage:5.2f}%)")

# Hate statistics
print("\n💢 Hate Predictions:")
for label in hate_labels:
    count = result_df[f'pred_{label}_binary'].sum()
    percentage = (count / len(result_df)) * 100
    print(f"  {label:12s}: {count:6d} samples ({percentage:5.2f}%)")

# Violence statistics
print("\n⚠️  Violence Predictions:")
for label in violence_labels:
    count = result_df[f'pred_{label}_binary'].sum()
    percentage = (count / len(result_df)) * 100
    print(f"  {label:12s}: {count:6d} samples ({percentage:5.2f}%)")

print()
print("=" * 60)
print("✅ HOÀN THÀNH TẤT CẢ!")
print("=" * 60)
print(f"\nCác file đã tạo:")
print(f"  1. {training_log_path}")
print(f"  2. {predictions_path}")
print()
print("📝 Bạn có thể:")
print("  - Xem training history: pd.read_csv('models/training_log.csv')")
print("  - Xem predictions: pd.read_csv('predictions_sentiment.csv')")
print("  - Load model để predict thêm: model = load_model('models/best_model_20251027_085402.h5')")


