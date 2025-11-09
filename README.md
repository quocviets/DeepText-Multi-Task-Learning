# DeepText Multi-Task Learning

Hệ thống Deep Learning để phân tích văn bản tiếng Việt với 3 nhiệm vụ đồng thời:
- **🎭 Phân loại cảm xúc** (7 classes)
- **😡 Phát hiện ngôn từ thù địch** (3 classes)
- **⚔️ Phát hiện bạo lực** (3 classes)

## ✨ Tính năng

- ✅ Multi-Task Learning: Phân tích 3 nhiệm vụ đồng thời
- ✅ Kiến trúc tối ưu: Shared Embedding + BiLSTM + Task-Specific Heads
- ✅ Streamlit UI: Giao diện web đẹp, dễ sử dụng
- ✅ Auto-load Model: Tự động load model khi khởi động
- ✅ Batch Prediction: Hỗ trợ phân tích hàng loạt
- ✅ Visualizations: Charts và metrics đẹp mắt

## 🚀 Quick Start

### 1. Cài đặt

```bash
# Clone repository
git clone https://github.com/quocviets/DeepText-Multi-Task-Learning.git
cd DeepText-Multi-Task-Learning

# Cài đặt dependencies
pip install -r requirements.txt
pip install -r ui_app/requirements.txt
```

### 2. Chạy Streamlit UI (Khuyến nghị)

```bash
cd ui_app
streamlit run app.py
```

Mở browser: http://localhost:8501

**Tính năng UI:**
- ✅ Tự động load model khi khởi động
- ✅ Single text prediction với visualizations
- ✅ Batch prediction từ CSV
- ✅ Export kết quả
- ✅ Modern UI với gradients và animations

### 3. Sử dụng Model Service (Programmatic)

```python
from ui_app.model_service import get_model_service

# Load model service
service = get_model_service(
    model_path="checkpoints/models/best_model_20251027_085402.h5",
    train_data_path="checkpoints/train_clean.csv"
)

# Single prediction
result = service.predict("Tôi cảm thấy rất vui vẻ!")
print(result['emotion']['label'])  # joy

# Batch prediction
results = service.predict_batch(["text1", "text2", "text3"])
```

## 📁 Cấu trúc Project

```
DeepText-MTL/
├── ui_app/                    # Streamlit UI Application
│   ├── app.py                 # Main UI app
│   ├── model_service.py       # Model service layer
│   └── requirements.txt       # UI dependencies
│
├── src/                       # Source code
│   ├── model/                 # Model architectures
│   ├── training/              # Training pipeline
│   ├── data_preprocessing/    # Data preprocessing
│   └── utils/                 # Utilities
│
├── checkpoints/               # Model checkpoints
│   └── models/
│       └── best_model_20251027_085402.h5
│
├── data/                      # Datasets
│   ├── raw/                   # Raw data
│   └── processed/             # Processed data
│
├── config_default.json        # Configuration
└── requirements.txt           # Dependencies
```

## 🏗️ Model Architecture

```
Input Text (max_length=100)
        ↓
Shared Embedding (vocab_size=10,000 → embedding_dim=128)
        ↓
Shared BiLSTM (64 units)
        ↓
Shared Dense (128 units) + Dropout
        ↓
┌─────────────┬─────────────┬─────────────┐
│ Emotion     │ Hate Speech │ Violence    │
│ (7 classes) │ (3 classes) │ (3 classes) │
│ Softmax     │ Softmax     │ Softmax     │
└─────────────┴─────────────┴─────────────┘
```

### Task Classes

**Emotion (7 classes):**
- sad, joy, love, angry, fear, surprise, no_emo

**Hate Speech (3 classes):**
- hate, offensive, neutral

**Violence (3 classes):**
- sex_viol, phys_viol, no_viol

## 📊 Training

### Train Model

```python
from src.model.deeptext_multitask import DeepTextMultiTaskClassifier

# Tạo model
model = DeepTextMultiTaskClassifier(
    vocab_size=10000,
    embedding_dim=128,
    lstm_units=64,
    max_length=100,
    dropout_rate=0.3
)

# Build và compile
model.build_model()
model.compile_model()

# Train
history = model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=32
)
```

### Evaluate Model

```python
# Evaluate
results = model.evaluate(X_test, y_test)

# Visualize training
model.plot_training_history()
```

## 🌐 Deploy Streamlit Cloud

### Bước 1: Push lên GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Bước 2: Deploy trên Streamlit Cloud

1. Vào: https://streamlit.io/cloud
2. Đăng nhập với GitHub
3. Click "New app"
4. Chọn repo → Main file: `ui_app/app.py`
5. Click "Deploy"
6. ✅ Nhận link công khai!

**Link sẽ có dạng:** `https://your-app-name.streamlit.app`

## ⚙️ Configuration

### Model Config (`config_default.json`)

```json
{
  "model": {
    "vocab_size": 10000,
    "max_length": 100,
    "embedding_dim": 128,
    "lstm_units": 64,
    "dropout_rate": 0.3
  }
}
```

## 📈 Performance

Model đã được train và đạt performance tốt trên validation set:
- **Emotion Classification**: Accuracy cao
- **Hate Speech Detection**: F1-score tốt
- **Violence Detection**: Precision và Recall cân bằng

## 🔧 Requirements

### Core Dependencies
- Python 3.8+
- TensorFlow 2.8+
- Pandas, NumPy
- Scikit-learn

### UI Dependencies
- Streamlit >= 1.28.0
- Plotly >= 5.0.0

Xem `requirements.txt` và `ui_app/requirements.txt` để biết chi tiết.

## 📝 Usage Examples

### Single Prediction

```python
from ui_app.model_service import get_model_service

service = get_model_service(
    model_path="checkpoints/models/best_model_20251027_085402.h5",
    train_data_path="checkpoints/train_clean.csv"
)

result = service.predict("Tôi cảm thấy rất vui vẻ hôm nay!")

print(f"Emotion: {result['emotion']['label']}")
print(f"Confidence: {result['emotion']['confidence']:.2%}")
print(f"Hate: {result['hate']['labels']}")
print(f"Violence: {result['violence']['labels']}")
```

### Batch Prediction

```python
texts = [
    "Tôi cảm thấy rất vui vẻ!",
    "Đây là một tin nhắn tức giận",
    "Tôi yêu bạn rất nhiều"
]

results = service.predict_batch(texts)
for r in results:
    print(f"Text: {r['text']}")
    print(f"Emotion: {r['emotion']['label']}")
```

## 🎯 Workflow

1. **Data Preparation**: Chuẩn bị dataset với format đúng
2. **Training**: Train model với dữ liệu
3. **Evaluation**: Đánh giá performance
4. **Deployment**: Deploy lên Streamlit Cloud
5. **Usage**: Sử dụng qua UI hoặc API

## 📚 Documentation

- **UI Workflow**: Xem `ui_app/WORKFLOW.md`
- **Deployment Guide**: Xem `ui_app/DEPLOY.md`
- **Troubleshooting**: Xem `ui_app/TROUBLESHOOTING.md`

---

**DeepText Multi-Task Learning** - Phân tích văn bản tiếng Việt với Multi-Task Learning
