# 🌐 Hướng Dẫn Tạo Link Công Khai

## ⚡ Cách Nhanh Nhất: Ngrok

### 1. Cài đặt Ngrok:
```bash
# Windows: Download từ https://ngrok.com/download
# Hoặc với Chocolatey:
choco install ngrok

# Hoặc với Scoop:
scoop install ngrok
```

### 2. Đăng ký và lấy token:
- Vào: https://dashboard.ngrok.com/signup
- Đăng ký tài khoản miễn phí
- Copy auth token

### 3. Config token:
```bash
ngrok config add-authtoken YOUR_TOKEN_HERE
```

### 4. Chạy app:
```bash
# Terminal 1: Chạy Streamlit
cd ui_app
streamlit run app.py

# Terminal 2: Chạy ngrok
ngrok http 8501
```

### 5. Lấy link:
- Link sẽ hiện trong terminal ngrok
- Ví dụ: `https://abc123.ngrok.io`
- **Copy link này → Ai có link đều vào được!**

---

## 🚀 Cách Tốt Nhất: Streamlit Cloud (Miễn phí)

### 1. Push code lên GitHub:
```bash
git init
git add .
git commit -m "Streamlit app"
git remote add origin YOUR_REPO_URL
git push -u origin main
```

### 2. Deploy:
- Vào: https://streamlit.io/cloud
- Đăng nhập với GitHub
- Click "New app"
- Chọn repo → `ui_app/app.py`
- Click "Deploy"
- ⏳ Đợi vài phút
- ✅ Nhận link: `https://your-app.streamlit.app`

### Ưu điểm:
- ✅ **Hoàn toàn miễn phí**
- ✅ Link không bao giờ đổi
- ✅ Tự động update khi push code
- ✅ Không cần máy bạn chạy

---

## 📱 Chia sẻ link:

Sau khi có link công khai (từ ngrok hoặc Streamlit Cloud):

1. Copy link
2. Gửi cho bất kỳ ai
3. Họ mở link → Vào được app ngay!

**Ví dụ link:**
```
https://deeptext-mtl.streamlit.app  ← Streamlit Cloud
https://abc123.ngrok.io             ← Ngrok
```

---

## 💡 Tips:

- **Ngrok**: Link đổi mỗi lần chạy (free plan)
- **Streamlit Cloud**: Link cố định, không bao giờ đổi
- **Serveo**: Không cần đăng ký nhưng không ổn định lắm

**Chọn Streamlit Cloud nếu muốn link cố định và ổn định!**

