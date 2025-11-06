# Workflow: Tích hợp UI/UX với Checkpoint Models

## 📋 Tổng quan

Workflow này mô tả cách tích hợp UI/UX với các checkpoint models của DeepText Multi-Task Learning project. Hệ thống bao gồm:

1. **Model Service Layer** - Load và quản lý models
2. **Streamlit UI Application** - Giao diện web tương tác
3. **Integration Workflow** - Quy trình tích hợp

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit UI (app.py)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Single Input │  │ Batch Input  │  │ Visualization│ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└─────────┼─────────────────┼──────────────────┼─────────┘
          │                 │                  │
          └─────────────────┼──────────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │      Model Service (model_service) │
          │  ┌──────────────────────────────┐  │
          │  │ Load Model from Checkpoint   │  │
          │  │ Load/Fit Tokenizer           │  │
          │  │ Preprocess Text               │  │
          │  │ Model Inference               │  │
          │  │ Post-process Predictions      │  │
          │  └──────────────────────────────┘  │
          └──────────────┬─────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
    ┌─────▼──────┐              ┌──────▼──────┐
    │ Model.h5   │              │ Training    │
    │ Checkpoint │              │ Data (CSV)  │
    └────────────┘              └─────────────┘
```

---

## 🔄 Workflow chi tiết

### Bước 1: Chuẩn bị Dependencies

```bash
# Cài đặt requirements
pip install -r ui_app/requirements.txt
```

### Bước 2: Load Model từ Checkpoint

**Luồng xử lý:**

1. **Khởi tạo ModelService**
   ```python
   model_service = ModelService(
       model_path="DeepText-MTL/checkpoints/models/best_model_20251027_085402.h5",
       config_path="DeepText-MTL/config_default.json"
   )
   ```

2. **Load Config** (optional)
   - Đọc config từ JSON
   - Cập nhật parameters: vocab_size, max_length, classes

3. **Load Model**
   - Load `.h5` file với custom objects (Cast layer)
   - Verify model structure
   - Check input/output shapes

4. **Load Tokenizer**
   - Fit tokenizer từ training data
   - Hoặc load tokenizer đã được saved trước đó

### Bước 3: Preprocess Input Text

**Workflow preprocessing:**

```
Input Text
    ↓
Tokenizer.texts_to_sequences()  →  Convert to sequences
    ↓
pad_sequences()                  →  Padding/Truncating to max_len
    ↓
Numpy Array (batch_size, max_len) → Ready for model input
```

**Code:**
```python
def preprocess_text(text: str) -> np.ndarray:
    sequences = self.tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequences,
        maxlen=self.max_len,
        padding='post',
        truncating='post'
    )
    return padded
```

### Bước 4: Model Inference

**Multi-task prediction:**

```
Input: (batch_size, max_len)
    ↓
Shared Embedding Layer
    ↓
Shared BiLSTM Layer
    ↓
Shared Dense Layer
    ↓
    ├──→ Emotion Head (7 classes, softmax)
    ├──→ Hate Head (3 classes, softmax)
    └──→ Violence Head (3 classes, softmax)
    ↓
Outputs: [emotion_probs, hate_probs, violence_probs]
```

**Code:**
```python
predictions = model.predict(X, verbose=0)
emotion_probs = predictions[0]    # (batch_size, 7)
hate_probs = predictions[1]      # (batch_size, 3)
violence_probs = predictions[2]  # (batch_size, 3)
```

### Bước 5: Post-process Predictions

**Xử lý từng task:**

1. **Emotion** (Multi-class classification):
   ```python
   emotion_idx = np.argmax(emotion_probs, axis=1)
   emotion_label = emotion_classes[emotion_idx]
   confidence = emotion_probs[emotion_idx]
   ```

2. **Hate Speech** (Multi-label classification):
   ```python
   threshold = 0.5
   hate_labels = [
       hate_classes[i] 
       for i in range(len(hate_classes))
       if hate_probs[i] > threshold
   ]
   ```

3. **Violence** (Multi-label classification):
   ```python
   threshold = 0.5
   violence_labels = [
       violence_classes[i]
       for i in range(len(violence_classes))
       if violence_probs[i] > threshold
   ]
   ```

### Bước 6: Hiển thị trong UI

**Streamlit UI workflow:**

1. **Single Prediction:**
   - User nhập text → Click "Phân tích"
   - Hiển thị 3 kết quả với metrics và charts
   - Interactive visualizations với Plotly

2. **Batch Prediction:**
   - Upload CSV hoặc nhập nhiều text
   - Process batch → Display DataFrame
   - Export results to CSV

3. **Visualizations:**
   - Model information
   - Prediction probabilities charts
   - Combined multi-task visualization

---

## 📁 Cấu trúc Project

```
Last_Data/
├── DeepText-MTL/
│   ├── checkpoints/
│   │   ├── models/
│   │   │   └── best_model_20251027_085402.h5  ← Model checkpoint
│   │   └── train_clean.csv                    ← Training data (cho tokenizer)
│   └── config_default.json                    ← Config file
│
└── ui_app/
    ├── app.py                 ← Streamlit UI application
    ├── model_service.py       ← Model service layer
    ├── requirements.txt       ← Dependencies
    └── README.md              ← Documentation
```

---

## 🚀 Cách sử dụng

### 1. Setup Environment

```bash
# Tạo virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
cd ui_app
pip install -r requirements.txt
```

### 2. Chạy Application

```bash
# Chạy Streamlit app
streamlit run app.py

# Hoặc với custom port
streamlit run app.py --server.port 8501
```

### 3. Sử dụng UI

1. **Mở browser:** http://localhost:8501

2. **Load Model từ Sidebar:**
   - Nhập đường dẫn model: `DeepText-MTL/checkpoints/models/best_model_20251027_085402.h5`
   - Nhập đường dẫn config: `DeepText-MTL/config_default.json`
   - Nhập đường dẫn training data: `DeepText-MTL/checkpoints/train_clean.csv`
   - Click "Load Model"

3. **Single Prediction:**
   - Tab "Single Prediction"
   - Nhập text vào text area
   - Click "Phân tích"
   - Xem kết quả với visualizations

4. **Batch Prediction:**
   - Tab "Batch Prediction"
   - Upload CSV hoặc nhập nhiều text
   - Click "Phân tích Batch"
   - Download kết quả CSV

---

## 🔧 Customization

### Thay đổi Model Path

Edit trong `app.py`:
```python
model_path = st.sidebar.text_input(
    "Đường dẫn Model",
    value="your/path/to/model.h5"
)
```

### Thay đổi Thresholds

Edit trong `model_service.py`:
```python
self.hate_threshold = 0.5      # Threshold cho hate speech
self.violence_threshold = 0.5  # Threshold cho violence
```

### Thay đổi UI Theme

Edit trong `app.py`:
```python
st.set_page_config(
    page_title="Your Title",
    page_icon="🎯",
    layout="wide"
)
```

---

## 🐛 Troubleshooting

### Lỗi: Model file not found
- **Nguyên nhân:** Đường dẫn model không đúng
- **Giải pháp:** Kiểm tra lại đường dẫn trong sidebar

### Lỗi: Tokenizer chưa được load
- **Nguyên nhân:** Training data không tìm thấy
- **Giải pháp:** Cung cấp đường dẫn đúng đến `train_clean.csv`

### Lỗi: Custom layer Cast
- **Nguyên nhân:** Model có custom layer cần register
- **Giải pháp:** Đã được handle trong `model_service.py` với `@tf.keras.utils.register_keras_serializable()`

### Lỗi: Memory issues với batch prediction
- **Nguyên nhân:** Batch size quá lớn
- **Giải pháp:** Giảm batch size trong `predict_batch()` method

---

## 📊 Performance Optimization

### 1. Caching Model Loading
```python
@st.cache_resource
def load_model_cached(model_path):
    return get_model_service(model_path)
```

### 2. Batch Processing
- Sử dụng `predict_batch()` thay vì loop qua từng text
- Batch size optimize: 32-64

### 3. GPU Acceleration
- Ensure TensorFlow GPU được cài đặt
- Model sẽ tự động sử dụng GPU nếu available

---

## 🔐 Security Considerations

1. **Input Validation:**
   - Validate text length
   - Sanitize user input
   - Rate limiting cho API calls

2. **Model Protection:**
   - Không expose model files trực tiếp
   - Sử dụng authentication nếu deploy production

3. **Error Handling:**
   - Graceful error messages
   - Không expose internal errors

---

## 🚢 Deployment

### Option 1: Streamlit Cloud
```bash
# Push code lên GitHub
# Deploy trên streamlit.io
```

### Option 2: Docker
```dockerfile
FROM python:3.9

WORKDIR /app
COPY ui_app/requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

### Option 3: Local Server
```bash
# Chạy với production mode
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 📝 Next Steps

1. ✅ Tích hợp với checkpoint models
2. ✅ Tạo UI/UX với Streamlit
3. ✅ Batch prediction support
4. ✅ Visualization với Plotly
5. 🔄 Add authentication
6. 🔄 Add logging & monitoring
7. 🔄 Export model metrics
8. 🔄 A/B testing support

---

## 📚 References

- [DeepText-MTL Model Architecture](../DeepText-MTL/src/model/deeptext_multitask.py)
- [Streamlit Documentation](https://docs.streamlit.io)
- [TensorFlow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras)

