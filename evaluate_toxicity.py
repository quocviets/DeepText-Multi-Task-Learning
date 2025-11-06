# ===============================================================
# 🔍 Evaluate Toxicity Model - Hate & Violence Tasks
# Task Evaluator: Đánh giá riêng hate_head & violence_head
# ===============================================================

import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

# ===============================================================
# 1️⃣ Config
# ===============================================================
CONFIG = {
    'model_path': 'models_gru_final/best_modelB_*.h5',  # ⚠️ Thay đổi đường dẫn
    'tokenizer_path': 'models_gru_final/tokenizerB_*.pkl',  # ⚠️ Thay đổi đường dẫn
    'test_csv': 'val_augmented_50k.csv',  # ⚠️ Test/Validation data có labels
    'output_dir': 'evaluation_results',
    'max_length': 100,
    'threshold': 0.5  # Threshold cho binary classification
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(f"{CONFIG['output_dir']}/plots", exist_ok=True)

# ===============================================================
# 2️⃣ Load Model & Tokenizer
# ===============================================================
print("📦 Loading model and tokenizer...")

# Tìm file model mới nhất
import glob
model_files = glob.glob(CONFIG['model_path'].replace('*', '*'))
if not model_files:
    raise FileNotFoundError(f"❌ Model not found: {CONFIG['model_path']}")
model_path = max(model_files, key=os.path.getctime)
print(f"✅ Loading model: {model_path}")
model = load_model(model_path)

# Tìm tokenizer
tokenizer_files = glob.glob(CONFIG['tokenizer_path'].replace('*', '*'))
if not tokenizer_files:
    raise FileNotFoundError(f"❌ Tokenizer not found: {CONFIG['tokenizer_path']}")
tokenizer_path = max(tokenizer_files, key=os.path.getctime)
print(f"✅ Loading tokenizer: {tokenizer_path}")
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Load config nếu có
config_files = glob.glob('models_gru_final/configB_*.json')
if config_files:
    config_path = max(config_files, key=os.path.getctime)
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    print(f"✅ Loaded config: {config_path}")
    if 'loss_weights' in model_config:
        CONFIG['loss_weights'] = model_config['loss_weights']
        print(f"   Loss weights: {CONFIG['loss_weights']}")

# ===============================================================
# 3️⃣ Load Test Data
# ===============================================================
print(f"\n📊 Loading test data: {CONFIG['test_csv']}")
test_df = pd.read_csv(CONFIG['test_csv'], sep=';')

# Preprocess text
test_df['text'] = test_df['text'].astype(str).str.lower().str.strip()

# Prepare sequences
X_test = pad_sequences(
    tokenizer.texts_to_sequences(test_df['text']),
    maxlen=CONFIG['max_length'],
    padding='post',
    truncating='post'
)

# Prepare labels
hate_cols = ['hate', 'offensive', 'neutral']
violence_cols = ['sex_viol', 'phys_viol', 'no_viol']

y_hate_true = test_df[hate_cols].values.astype(np.float32)
y_viol_true = test_df[violence_cols].values.astype(np.float32)

print(f"✅ Test samples: {len(X_test):,}")

# ===============================================================
# 4️⃣ Predict
# ===============================================================
print("\n🔮 Running predictions...")
predictions = model.predict(X_test, verbose=1, batch_size=128)
_, y_hate_pred, y_viol_pred = predictions  # Skip emotion output

print(f"✅ Predictions shape: Hate {y_hate_pred.shape}, Violence {y_viol_pred.shape}")

# Convert to binary predictions
y_hate_pred_bin = (y_hate_pred > CONFIG['threshold']).astype(int)
y_viol_pred_bin = (y_viol_pred > CONFIG['threshold']).astype(int)
y_hate_true_bin = (y_hate_true > CONFIG['threshold']).astype(int)
y_viol_true_bin = (y_viol_true > CONFIG['threshold']).astype(int)

# ===============================================================
# 5️⃣ Calculate Metrics - Hate Task
# ===============================================================
print("\n" + "="*60)
print("📊 HATE TASK EVALUATION")
print("="*60)

hate_metrics = {}
hate_f1_macro = f1_score(y_hate_true_bin, y_hate_pred_bin, average='macro', zero_division=0)
hate_f1_samples = f1_score(y_hate_true_bin, y_hate_pred_bin, average='samples', zero_division=0)
hate_f1_micro = f1_score(y_hate_true_bin, y_hate_pred_bin, average='micro', zero_division=0)

print(f"\n🎯 F1 Scores:")
print(f"  Macro:   {hate_f1_macro:.4f}")
print(f"  Samples: {hate_f1_samples:.4f}")
print(f"  Micro:   {hate_f1_micro:.4f}")

hate_metrics['f1_macro'] = float(hate_f1_macro)
hate_metrics['f1_samples'] = float(hate_f1_samples)
hate_metrics['f1_micro'] = float(hate_f1_micro)

# Per-class metrics
print(f"\n📋 Per-class F1:")
for i, col in enumerate(hate_cols):
    f1 = f1_score(y_hate_true_bin[:, i], y_hate_pred_bin[:, i], zero_division=0)
    precision = precision_score(y_hate_true_bin[:, i], y_hate_pred_bin[:, i], zero_division=0)
    recall = recall_score(y_hate_true_bin[:, i], y_hate_pred_bin[:, i], zero_division=0)
    print(f"  {col:12s}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    hate_metrics[f'{col}_f1'] = float(f1)
    hate_metrics[f'{col}_precision'] = float(precision)
    hate_metrics[f'{col}_recall'] = float(recall)

# ROC AUC
print(f"\n📈 ROC AUC:")
hate_auc_scores = {}
for i, col in enumerate(hate_cols):
    try:
        auc_score = roc_auc_score(y_hate_true[:, i], y_hate_pred[:, i])
        hate_auc_scores[col] = auc_score
        print(f"  {col:12s}: {auc_score:.4f}")
        hate_metrics[f'{col}_auc'] = float(auc_score)
    except ValueError as e:
        print(f"  {col:12s}: N/A (only one class present)")

hate_auc_macro = np.mean(list(hate_auc_scores.values())) if hate_auc_scores else 0
print(f"  {'Macro Average':12s}: {hate_auc_macro:.4f}")
hate_metrics['auc_macro'] = float(hate_auc_macro)

# ===============================================================
# 6️⃣ Calculate Metrics - Violence Task
# ===============================================================
print("\n" + "="*60)
print("📊 VIOLENCE TASK EVALUATION")
print("="*60)

violence_metrics = {}
violence_f1_macro = f1_score(y_viol_true_bin, y_viol_pred_bin, average='macro', zero_division=0)
violence_f1_samples = f1_score(y_viol_true_bin, y_viol_pred_bin, average='samples', zero_division=0)
violence_f1_micro = f1_score(y_viol_true_bin, y_viol_pred_bin, average='micro', zero_division=0)

print(f"\n🎯 F1 Scores:")
print(f"  Macro:   {violence_f1_macro:.4f}")
print(f"  Samples: {violence_f1_samples:.4f}")
print(f"  Micro:   {violence_f1_micro:.4f}")

violence_metrics['f1_macro'] = float(violence_f1_macro)
violence_metrics['f1_samples'] = float(violence_f1_samples)
violence_metrics['f1_micro'] = float(violence_f1_micro)

# Per-class metrics
print(f"\n📋 Per-class F1:")
for i, col in enumerate(violence_cols):
    f1 = f1_score(y_viol_true_bin[:, i], y_viol_pred_bin[:, i], zero_division=0)
    precision = precision_score(y_viol_true_bin[:, i], y_viol_pred_bin[:, i], zero_division=0)
    recall = recall_score(y_viol_true_bin[:, i], y_viol_pred_bin[:, i], zero_division=0)
    print(f"  {col:12s}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    violence_metrics[f'{col}_f1'] = float(f1)
    violence_metrics[f'{col}_precision'] = float(precision)
    violence_metrics[f'{col}_recall'] = float(recall)

# ROC AUC
print(f"\n📈 ROC AUC:")
violence_auc_scores = {}
for i, col in enumerate(violence_cols):
    try:
        auc_score = roc_auc_score(y_viol_true[:, i], y_viol_pred[:, i])
        violence_auc_scores[col] = auc_score
        print(f"  {col:12s}: {auc_score:.4f}")
        violence_metrics[f'{col}_auc'] = float(auc_score)
    except ValueError as e:
        print(f"  {col:12s}: N/A (only one class present)")

violence_auc_macro = np.mean(list(violence_auc_scores.values())) if violence_auc_scores else 0
print(f"  {'Macro Average':12s}: {violence_auc_macro:.4f}")
violence_metrics['auc_macro'] = float(violence_auc_macro)

# ===============================================================
# 7️⃣ Plot ROC Curves
# ===============================================================
print("\n📊 Generating ROC curves...")

# Hate ROC
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for i, col in enumerate(hate_cols):
    try:
        fpr, tpr, _ = roc_curve(y_hate_true[:, i], y_hate_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{col} (AUC = {roc_auc:.3f})', linewidth=2)
    except:
        continue

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Hate Task', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)

# Violence ROC
plt.subplot(1, 2, 2)
for i, col in enumerate(violence_cols):
    try:
        fpr, tpr, _ = roc_curve(y_viol_true[:, i], y_viol_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{col} (AUC = {roc_auc:.3f})', linewidth=2)
    except:
        continue

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Violence Task', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()
roc_path = f"{CONFIG['output_dir']}/plots/roc_curves.png"
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {roc_path}")

# ===============================================================
# 8️⃣ Safety Analysis - False Positives
# ===============================================================
print("\n" + "="*60)
print("🛡️ SAFETY ANALYSIS")
print("="*60)

# False Positives cho Hate
print(f"\n⚠️ False Positives (Predicted positive but actually negative):")
hate_fp = np.sum((y_hate_pred_bin == 1) & (y_hate_true_bin == 0), axis=0)
hate_total_p = np.sum(y_hate_pred_bin == 1, axis=0)
for i, col in enumerate(hate_cols):
    fp_rate = (hate_fp[i] / hate_total_p[i] * 100) if hate_total_p[i] > 0 else 0
    print(f"  Hate {col:10s}: {hate_fp[i]:,} FP / {hate_total_p[i]:,} predicted = {fp_rate:.2f}%")

violence_fp = np.sum((y_viol_pred_bin == 1) & (y_viol_true_bin == 0), axis=0)
violence_total_p = np.sum(y_viol_pred_bin == 1, axis=0)
for i, col in enumerate(violence_cols):
    fp_rate = (violence_fp[i] / violence_total_p[i] * 100) if violence_total_p[i] > 0 else 0
    print(f"  Violence {col:6s}: {violence_fp[i]:,} FP / {violence_total_p[i]:,} predicted = {fp_rate:.2f}%")

# High-risk cases (True positive nhưng model miss)
print(f"\n🚨 High-Risk Cases (True positive nhưng model miss - False Negative):")
hate_fn = np.sum((y_hate_pred_bin == 0) & (y_hate_true_bin == 1), axis=0)
hate_total_tp = np.sum(y_hate_true_bin == 1, axis=0)
for i, col in enumerate(hate_cols):
    fn_rate = (hate_fn[i] / hate_total_tp[i] * 100) if hate_total_tp[i] > 0 else 0
    print(f"  Hate {col:10s}: {hate_fn[i]:,} FN / {hate_total_tp[i]:,} actual = {fn_rate:.2f}%")

violence_fn = np.sum((y_viol_pred_bin == 0) & (y_viol_true_bin == 1), axis=0)
violence_total_tp = np.sum(y_viol_true_bin == 1, axis=0)
for i, col in enumerate(violence_cols):
    fn_rate = (violence_fn[i] / violence_total_tp[i] * 100) if violence_total_tp[i] > 0 else 0
    print(f"  Violence {col:6s}: {violence_fn[i]:,} FN / {violence_total_tp[i]:,} actual = {fn_rate:.2f}%")

# ===============================================================
# 9️⃣ Loss Weights Analysis & Suggestions
# ===============================================================
print("\n" + "="*60)
print("⚖️ LOSS WEIGHTS ANALYSIS")
print("="*60)

current_weights = CONFIG.get('loss_weights', {'emotion_output': 1.0, 'hate_output': 3.0, 'violence_output': 3.0})
print(f"\n📊 Current Loss Weights:")
print(f"  Emotion:  {current_weights.get('emotion_output', 'N/A')}")
print(f"  Hate:     {current_weights.get('hate_output', 'N/A')}")
print(f"  Violence: {current_weights.get('violence_output', 'N/A')}")

print(f"\n💡 Suggestions:")
# So sánh performance
hate_avg = (hate_f1_macro + hate_auc_macro) / 2
violence_avg = (violence_f1_macro + violence_auc_macro) / 2

if hate_avg < 0.7:
    print(f"  ⚠️ Hate performance thấp ({hate_avg:.3f})")
    print(f"     → Đề xuất TĂNG hate_output weight: {current_weights.get('hate_output', 3.0)} → 4.0 hoặc 5.0")
elif hate_avg > 0.85:
    print(f"  ✅ Hate performance tốt ({hate_avg:.3f})")
    print(f"     → Có thể giảm weight nếu cần tập trung vào tasks khác")

if violence_avg < 0.7:
    print(f"  ⚠️ Violence performance thấp ({violence_avg:.3f})")
    print(f"     → Đề xuất TĂNG violence_output weight: {current_weights.get('violence_output', 3.0)} → 4.0 hoặc 5.0")
elif violence_avg > 0.85:
    print(f"  ✅ Violence performance tốt ({violence_avg:.3f})")
    print(f"     → Có thể giảm weight nếu cần tập trung vào tasks khác")

# Cân bằng
diff = abs(hate_avg - violence_avg)
if diff > 0.1:
    print(f"\n  ⚠️ Mất cân bằng giữa Hate ({hate_avg:.3f}) và Violence ({violence_avg:.3f})")
    if hate_avg < violence_avg:
        print(f"     → Tăng hate_output weight để cân bằng")
    else:
        print(f"     → Tăng violence_output weight để cân bằng")
else:
    print(f"\n  ✅ Cân bằng tốt giữa Hate ({hate_avg:.3f}) và Violence ({violence_avg:.3f})")

# ===============================================================
# 🔟 Save Results
# ===============================================================
print("\n💾 Saving results...")

results = {
    'hate': hate_metrics,
    'violence': violence_metrics,
    'safety': {
        'hate_fp': {hate_cols[i]: int(hate_fp[i]) for i in range(len(hate_cols))},
        'violence_fp': {violence_cols[i]: int(violence_fp[i]) for i in range(len(violence_cols))},
        'hate_fn': {hate_cols[i]: int(hate_fn[i]) for i in range(len(hate_cols))},
        'violence_fn': {violence_cols[i]: int(violence_fn[i]) for i in range(len(violence_cols))}
    },
    'current_loss_weights': current_weights,
    'suggestions': {
        'hate_avg_performance': float(hate_avg),
        'violence_avg_performance': float(violence_avg),
        'performance_balance': float(diff)
    }
}

results_path = f"{CONFIG['output_dir']}/evaluation_toxicity_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved: {results_path}")

# Save summary CSV
summary_df = pd.DataFrame({
    'Task': ['Hate'] * len(hate_cols) + ['Violence'] * len(violence_cols),
    'Class': hate_cols + violence_cols,
    'F1': [hate_metrics.get(f'{col}_f1', 0) for col in hate_cols] + 
           [violence_metrics.get(f'{col}_f1', 0) for col in violence_cols],
    'Precision': [hate_metrics.get(f'{col}_precision', 0) for col in hate_cols] + 
                 [violence_metrics.get(f'{col}_precision', 0) for col in violence_cols],
    'Recall': [hate_metrics.get(f'{col}_recall', 0) for col in hate_cols] + 
              [violence_metrics.get(f'{col}_recall', 0) for col in violence_cols],
    'AUC': [hate_metrics.get(f'{col}_auc', 0) for col in hate_cols] + 
           [violence_metrics.get(f'{col}_auc', 0) for col in violence_cols]
})

summary_path = f"{CONFIG['output_dir']}/evaluation_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"✅ Saved: {summary_path}")

# ===============================================================
# 1️⃣1️⃣ Summary
# ===============================================================
print("\n" + "="*60)
print("✅ EVALUATION COMPLETE!")
print("="*60)
print(f"\n📁 Results saved in: {CONFIG['output_dir']}/")
print(f"  - {results_path}")
print(f"  - {summary_path}")
print(f"  - {roc_path}")
print("\n💡 Next steps:")
print("  1. Review ROC curves và metrics")
print("  2. Điều chỉnh loss_weights nếu cần")
print("  3. So sánh với single-task models (nếu có)")
print("  4. Fine-tune threshold nếu cần (hiện tại: {})".format(CONFIG['threshold']))
print("="*60)



