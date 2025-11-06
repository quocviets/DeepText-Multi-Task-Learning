# UI Application - DeepText Multi-Task Learning

Ứng dụng web để tương tác với DeepText Multi-Task Learning models từ checkpoint.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Application

```bash
streamlit run app.py
```

### 3. Open Browser

Mở browser và truy cập: http://localhost:8501

## 📋 Features

- ✅ Load models từ checkpoint (.h5 files)
- ✅ Single text prediction với visualizations
- ✅ Batch prediction từ CSV
- ✅ Interactive charts với Plotly
- ✅ Export results to CSV
- ✅ Model information display

## 📁 File Structure

```
ui_app/
├── app.py              # Streamlit UI application
├── model_service.py    # Model service layer
├── requirements.txt    # Python dependencies
├── WORKFLOW.md        # Chi tiết workflow tích hợp
└── README.md          # This file
```

## 🔧 Configuration

### Model Paths

Trong sidebar của app, bạn có thể cấu hình:
- **Model Path**: Đường dẫn đến file model (.h5)
- **Config Path**: Đường dẫn đến config file (optional)
- **Training Data Path**: Đường dẫn đến training data để fit tokenizer

### Default Paths

- Model: `DeepText-MTL/checkpoints/models/best_model_20251027_085402.h5`
- Config: `DeepText-MTL/config_default.json`
- Training Data: `DeepText-MTL/checkpoints/train_clean.csv`

## 📚 Usage

### Single Prediction

1. Load model từ sidebar
2. Tab "Single Prediction"
3. Nhập text vào text area
4. Click "Phân tích"
5. Xem kết quả với visualizations

### Batch Prediction

1. Tab "Batch Prediction"
2. Upload CSV file (có cột 'text') hoặc nhập nhiều text
3. Click "Phân tích Batch"
4. Download kết quả CSV

### Visualizations

- Tab "Visualizations" để xem model information
- Tab "About" để xem documentation

## 🐛 Troubleshooting

Xem [WORKFLOW.md](WORKFLOW.md) để biết chi tiết troubleshooting.

## 📖 Documentation

Chi tiết workflow tích hợp: [WORKFLOW.md](WORKFLOW.md)

