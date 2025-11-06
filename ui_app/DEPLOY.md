# 🚀 Deploy Streamlit App Công Khai (Ai có link đều vào được)

## Option 1: Streamlit Cloud (Khuyến nghị - Miễn phí, Dễ nhất)

### Bước 1: Chuẩn bị GitHub Repo

1. Tạo repo mới trên GitHub
2. Push code lên:
   ```bash
   git init
   git add .
   git commit -m "Streamlit app"
   git remote add origin YOUR_REPO_URL
   git push -u origin main
   ```

### Bước 2: Deploy trên Streamlit Cloud

1. Đăng ký tại: https://streamlit.io/cloud
2. Click "New app"
3. Chọn GitHub repo của bạn
4. Cấu hình:
   - **Main file path**: `ui_app/app.py`
   - **Branch**: `main`
   - **Python version**: `3.9+`
5. Click "Deploy"
6. ⏳ Đợi vài phút để deploy
7. ✅ Nhận link công khai: `https://YOUR_APP_NAME.streamlit.app`

### Ưu điểm:
- ✅ Hoàn toàn miễn phí
- ✅ Link công khai ngay
- ✅ Tự động update khi push code
- ✅ Không cần server
- ✅ SSL tự động

---

## Option 2: Ngrok (Nhanh, Dùng ngay)

### Bước 1: Cài đặt Ngrok

1. Download: https://ngrok.com/download
2. Giải nén và thêm vào PATH
3. Đăng ký tài khoản miễn phí: https://dashboard.ngrok.com/signup
4. Lấy auth token từ dashboard
5. Chạy: `ngrok config add-authtoken YOUR_TOKEN`

### Bước 2: Chạy App

```bash
# Terminal 1: Chạy Streamlit
cd ui_app
streamlit run app.py

# Terminal 2: Chạy ngrok
ngrok http 8501
```

Hoặc dùng script tự động:
```bash
python ui_app/run_public.py
```

### Lấy Link:
- Xem terminal ngrok → có link công khai
- Hoặc mở: http://localhost:4040

### Ví dụ link:
```
https://abc123.ngrok.io  ← Link này ai cũng vào được!
```

### ⚠️ Lưu ý:
- Link ngrok FREE sẽ thay đổi mỗi lần chạy (trừ khi mua plan)
- Cần chạy cả Streamlit và ngrok cùng lúc
- App chỉ chạy khi máy bạn bật

---

## Option 3: Serveo (Không cần đăng ký)

### Chạy:
```bash
# Terminal 1: Streamlit
cd ui_app
streamlit run app.py --server.port 8501

# Terminal 2: Serveo
ssh -R 80:localhost:8501 serveo.net
```

### Link sẽ hiện trong terminal, ví dụ:
```
https://abc123.serveo.net
```

---

## Option 4: Local Network (Chỉ trong cùng WiFi)

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Sau đó lấy IP máy bạn:
- Windows: `ipconfig` → IPv4 Address
- Mac/Linux: `ifconfig` → inet

Người khác truy cập: `http://YOUR_IP:8501`

---

## 📋 Checklist Deploy

- [ ] Code đã push lên GitHub (cho Streamlit Cloud)
- [ ] `requirements.txt` đầy đủ dependencies
- [ ] Model files path đúng (hoặc upload lên cloud storage)
- [ ] Training data path đúng
- [ ] Test local trước khi deploy

---

## 🔧 Fix Issues

### Model files không tìm thấy:
- Upload model lên GitHub repo
- Hoặc dùng cloud storage (S3, Google Drive) và load từ URL

### Port đã được sử dụng:
```bash
# Đổi port
streamlit run app.py --server.port 8502
```

### Ngrok không chạy:
- Kiểm tra auth token đã config chưa
- Kiểm tra port 4040 không bị block

---

## 🎯 Khuyến nghị

**Cho production:** Streamlit Cloud (miễn phí, ổn định)

**Cho demo/test:** Ngrok (nhanh, dễ)

Bạn muốn tôi setup cách nào? 🚀

