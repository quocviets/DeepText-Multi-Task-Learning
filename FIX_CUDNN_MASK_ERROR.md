# 🔧 Fix cuDNN Mask Error

## ❌ Lỗi Gặp Phải

```
InvalidArgumentError: assertion failed: [You are passing a RNN mask that does not correspond to right-padded sequences, while using cuDNN, which is not supported. With cuDNN, RNN masks can only be used for right-padding...
```

## 🔍 Nguyên Nhân

1. **Embedding với `mask_zero=True`**: Tạo mask để ignore padding tokens
2. **Mask propagate đến GRU**: Mask tự động truyền từ embedding → GRU
3. **cuDNN yêu cầu strict**: cuDNN (GPU acceleration) chỉ hỗ trợ mask dạng right-padding strict
4. **Conflict**: Mask format không đúng yêu cầu của cuDNN

### Chi Tiết:
- **cuDNN mask format**: Phải là `[True, True, True, False, False, False]` (contiguous)
- **Mask từ embedding**: Có thể không đúng format này khi có padding='post'
- **Result**: cuDNN reject mask → Error

## ✅ Giải Pháp

### Option 1: Tắt cuDNN (KHUYẾN NGHỊ)

```python
x = layers.Bidirectional(
    layers.GRU(
        self.config['gru_units'],
        return_sequences=True,
        dropout=self.config['dropout_rate'],
        use_cudnn=False  # ✅ Tắt cuDNN
    ),
    name='bigru'
)(x)
```

**Ưu điểm:**
- ✅ Mask hoạt động đúng
- ✅ Đơn giản, không cần thay đổi nhiều
- ✅ Vẫn nhanh trên GPU (không quá chậm)

**Nhược điểm:**
- ⚠️ Có thể chậm hơn một chút (nhưng không đáng kể)

### Option 2: Tắt Mask (KHÔNG KHUYẾN NGHỊ)

```python
# Không dùng mask_zero
x = layers.Embedding(
    self.config['vocab_size'],
    self.config['embedding_dim'],
    mask_zero=False,  # ❌ Mất tính năng masking
    name='embedding'
)(inp)
```

**Nhược điểm:**
- ❌ Padding tokens được xử lý như real tokens
- ❌ Mất tính năng quan trọng
- ❌ Có thể ảnh hưởng đến accuracy

### Option 3: Sử dụng LSTM thay vì GRU (Nếu cần)

Một số trường hợp LSTM không có vấn đề này, nhưng không guarantee.

## 📊 So Sánh

| Option | Mask | Speed | Recommendation |
|--------|------|-------|----------------|
| **use_cudnn=False** | ✅ Hoạt động | ⚠️ Hơi chậm hơn | ⭐⭐⭐⭐⭐ |
| Tắt mask | ❌ Không có | ✅ Nhanh nhất | ❌ Không nên |
| LSTM | ✅ Hoạt động | ⚠️ Khác GRU | ⚠️ Thay đổi model |

## 🎯 Kết Luận

**Giải pháp tốt nhất**: **Tắt cuDNN** (`use_cudnn=False`)

- Mask vẫn hoạt động đúng
- Chỉ chậm hơn một chút (không đáng kể trên GPU hiện đại)
- Code đơn giản, ổn định

**Đã fix trong code**: ✅



