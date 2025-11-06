import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# ============================================
# LOAD MODEL VỚI CUSTOM OBJECTS
# ============================================
print("=" * 60)
print("Load model và tạo predictions")
print("=" * 60)

# Đăng ký custom objects để load model
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

# Load model với custom objects
model_path = "models/best_model_20251027_085402.h5"
print(f"📦 Đang load model từ: {model_path}")

try:
    custom_objects = {'Cast': Cast}
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    print("✅ Model đã được load thành công!")
    print(f"   Model inputs: {model.input_shape}")
    if isinstance(model.output, list):
        print(f"   Model outputs: {len(model.outputs)} outputs")
        for i, out in enumerate(model.outputs):
            print(f"      Output {i+1}: {out.shape}")
    else:
        print(f"   Model output: {model.output.shape}")
    print()
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    print("\n⚠️ Thử phương pháp khác...")
    
    # Thử load với compile=False và safe_mode=False
    try:
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'Cast': Cast},
            compile=False,
            safe_mode=False
        )
        print("✅ Model đã được load thành công (safe_mode=False)!")
    except Exception as e2:
        print(f"❌ Vẫn không load được: {e2}")
        print("\n💡 Gợi ý:")
        print("  - Model có thể cần các custom layers khác")
        print("  - Hoặc cần rebuild model với code training gốc")
        exit(1)

# ============================================
# ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ============================================
print("📂 Đang đọc dữ liệu validation/test...")
val_df = pd.read_csv("val_clean.csv", sep=';')
print(f"✅ Đã đọc {len(val_df)} mẫu từ val_clean.csv")
print()

# Chuẩn bị text data
texts = val_df['text'].astype(str).values

# Tokenize và padding (phải match với training)
print("🔤 Đang tokenize và padding text...")
train_df = pd.read_csv("train_clean.csv", sep=';')
all_texts = train_df['text'].astype(str).values

# Tạo tokenizer với cấu hình thông dụng
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(all_texts)

# Transform validation texts
sequences = tokenizer.texts_to_sequences(texts)
X_val = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
print(f"✅ Text đã được xử lý: shape = {X_val.shape}")
print()

# ============================================
# PREDICT
# ============================================
print("🔮 Đang thực hiện prediction...")
try:
    predictions = model.predict(X_val, batch_size=128, verbose=1)
    print("✅ Prediction hoàn thành!")
    print()
except Exception as e:
    print(f"❌ Lỗi khi predict: {e}")
    exit(1)

# ============================================
# XỬ LÝ VÀ LƯU PREDICTIONS
# ============================================
print("=" * 60)
print("Xử lý và lưu predictions")
print("=" * 60)

# Kiểm tra số outputs
if isinstance(predictions, list):
    print(f"Model có {len(predictions)} outputs")
    for i, pred in enumerate(predictions):
        print(f"  Output {i+1}: {pred.shape}")
else:
    print(f"Model có 1 output: {predictions.shape}")
    predictions = [predictions]

# Giả định: Output 1 = emotion, Output 2 = hate, Output 3 = violence
emotion_pred = predictions[0]
hate_pred = predictions[1] if len(predictions) > 1 else None
violence_pred = predictions[2] if len(predictions) > 2 else None

# Tạo DataFrame kết quả
result_df = pd.DataFrame()
result_df['text'] = texts

# Labels
emotion_labels = ['sad', 'joy', 'love', 'angry', 'fear', 'surprise', 'no_emo']
hate_labels = ['hate', 'offensive', 'neutral']
violence_labels = ['sex_viol', 'phys_viol', 'no_viol']

# Thêm emotion predictions
print(f"\n🎭 Xử lý Emotion predictions ({emotion_pred.shape})...")
for i, label in enumerate(emotion_labels[:emotion_pred.shape[1]]):
    result_df[f'pred_{label}'] = emotion_pred[:, i]
    result_df[f'pred_{label}_binary'] = (emotion_pred[:, i] > 0.5).astype(int)

# Thêm hate predictions
if hate_pred is not None:
    print(f"💢 Xử lý Hate predictions ({hate_pred.shape})...")
    for i, label in enumerate(hate_labels[:hate_pred.shape[1]]):
        result_df[f'pred_{label}'] = hate_pred[:, i]
        result_df[f'pred_{label}_binary'] = (hate_pred[:, i] > 0.5).astype(int)

# Thêm violence predictions
if violence_pred is not None:
    print(f"⚠️  Xử lý Violence predictions ({violence_pred.shape})...")
    for i, label in enumerate(violence_labels[:violence_pred.shape[1]]):
        result_df[f'pred_{label}'] = violence_pred[:, i]
        result_df[f'pred_{label}_binary'] = (violence_pred[:, i] > 0.5).astype(int)

# Thêm ground truth labels
all_labels = emotion_labels + hate_labels + violence_labels
for label in all_labels:
    if label in val_df.columns:
        result_df[f'true_{label}'] = val_df[label].values

# Lưu file predictions
predictions_path = "predictions_sentiment.csv"
result_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
print(f"\n✅ Đã lưu predictions: {predictions_path}")
print(f"   Số mẫu: {len(result_df)}")
print(f"   Số cột: {len(result_df.columns)}")

# In thống kê
print("\n📊 Thống kê predictions:")
print("-" * 60)

# Emotion statistics
print("\n🎭 Emotion Predictions:")
for label in emotion_labels:
    if f'pred_{label}_binary' in result_df.columns:
        count = result_df[f'pred_{label}_binary'].sum()
        percentage = (count / len(result_df)) * 100
        avg_prob = result_df[f'pred_{label}'].mean()
        print(f"  {label:12s}: {count:6d} samples ({percentage:5.2f}%) - avg prob: {avg_prob:.4f}")

# Hate statistics
if hate_pred is not None:
    print("\n💢 Hate Predictions:")
    for label in hate_labels:
        if f'pred_{label}_binary' in result_df.columns:
            count = result_df[f'pred_{label}_binary'].sum()
            percentage = (count / len(result_df)) * 100
            avg_prob = result_df[f'pred_{label}'].mean()
            print(f"  {label:12s}: {count:6d} samples ({percentage:5.2f}%) - avg prob: {avg_prob:.4f}")

# Violence statistics
if violence_pred is not None:
    print("\n⚠️  Violence Predictions:")
    for label in violence_labels:
        if f'pred_{label}_binary' in result_df.columns:
            count = result_df[f'pred_{label}_binary'].sum()
            percentage = (count / len(result_df)) * 100
            avg_prob = result_df[f'pred_{label}'].mean()
            print(f"  {label:12s}: {count:6d} samples ({percentage:5.2f}%) - avg prob: {avg_prob:.4f}")

print()
print("=" * 60)
print("✅ HOÀN THÀNH!")
print("=" * 60)


