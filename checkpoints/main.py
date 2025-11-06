import pandas as pd
import json
import os

# 🔧 Đường dẫn file history gốc (bạn sửa lại nếu khác)
history_path = "models/history_20251027_085402.csv"
output_dir = "models"

# Đọc file CSV gốc
history_df = pd.read_csv(history_path)

# Ghi lại đúng định dạng Kaggle (epoch, loss, acc, val_loss, ...)
csv_log_path = os.path.join(output_dir, "training_log_final.csv")
history_df.to_csv(csv_log_path, index_label="epoch")
print(f"✅ Training log CSV saved: {csv_log_path}")

# Ghi ra file JSON
json_log_path = os.path.join(output_dir, "history_final.json")
history_df.to_json(json_log_path, orient="records", indent=2)
print(f"✅ History JSON saved: {json_log_path}")
