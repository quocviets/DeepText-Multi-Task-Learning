# 🔧 Hướng dẫn giải quyết lỗi Tokenizer

## Lỗi: "Tokenizer chưa được load"

### ✅ Cách khắc phục:

1. **Đảm bảo đường dẫn Training Data đúng:**
   - Đường dẫn mặc định: `DeepText-MTL/checkpoints/train_clean.csv`
   - Hoặc đường dẫn tuyệt đối: `C:\Users\lequo\Downloads\Last_Data\DeepText-MTL\checkpoints\train_clean.csv`

2. **Kiểm tra file tồn tại:**
   ```python
   import os
   path = "DeepText-MTL/checkpoints/train_clean.csv"
   print(f"File exists: {os.path.exists(path)}")
   ```

3. **Trong Streamlit UI:**
   - Mở sidebar
   - Nhập đường dẫn Training Data vào ô "Đường dẫn Training Data"
   - Click "Load Model"
   - Nếu vẫn lỗi, thử đường dẫn tuyệt đối

### 📋 Workflow đúng:

```
1. Load Model ✅
   └─> Path: DeepText-MTL/checkpoints/models/best_model_20251027_085402.h5

2. Load Config (optional) ✅
   └─> Path: DeepText-MTL/config_default.json

3. Load Tokenizer ⚠️ QUAN TRỌNG!
   └─> Path: DeepText-MTL/checkpoints/train_clean.csv
   └─> Phải tồn tại và có cột 'text'
```

### 🐛 Debug:

Nếu vẫn gặp lỗi, kiểm tra:

1. **File CSV có đúng format không?**
   - Phải có cột 'text' hoặc cột đầu tiên chứa text
   - Separator có thể là `;` hoặc `,`

2. **Encoding:**
   - File phải là UTF-8

3. **Console logs:**
   - Xem console để biết tokenizer đã load từ đâu
   - Vocabulary size phải > 0

### 💡 Tips:

- Khi chạy từ thư mục `ui_app/`, đường dẫn relative sẽ là:
  - `../DeepText-MTL/checkpoints/train_clean.csv`

- Khi chạy từ thư mục root, đường dẫn sẽ là:
  - `DeepText-MTL/checkpoints/train_clean.csv`

### ✅ Đã được fix:

- ✅ Tự động tìm training data trong nhiều đường dẫn
- ✅ Xử lý nhiều format CSV (separator `;`, `,`, `\t`)
- ✅ Tự động detect cột text
- ✅ Hiển thị lỗi rõ ràng nếu không tìm thấy
- ✅ Validation tốt hơn trong UI

