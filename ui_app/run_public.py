"""
Script để chạy Streamlit app với ngrok - Tạo link công khai
Ai có link đều truy cập được!

Cách dùng:
1. Cài ngrok: https://ngrok.com/download
2. Đăng ký tài khoản miễn phí và lấy auth token
3. Chạy: ngrok config add-authtoken YOUR_TOKEN
4. Chạy script này: python run_public.py
"""

import subprocess
import sys
import time
import os
import webbrowser

def check_ngrok_installed():
    """Kiểm tra ngrok đã được cài đặt chưa"""
    try:
        result = subprocess.run(['ngrok', 'version'], 
                              capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False

def run_streamlit():
    """Chạy Streamlit app"""
    print("🚀 Đang khởi động Streamlit...")
    streamlit_process = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 'app.py', 
         '--server.port', '8501', '--server.address', 'localhost'],
        cwd='ui_app'
    )
    return streamlit_process

def run_ngrok():
    """Chạy ngrok tunnel"""
    print("🌐 Đang khởi động ngrok tunnel...")
    ngrok_process = subprocess.Popen(
        ['ngrok', 'http', '8501'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(3)  # Đợi ngrok khởi động
    
    # Lấy public URL từ ngrok API
    try:
        import requests
        response = requests.get('http://localhost:4040/api/tunnels')
        if response.status_code == 200:
            tunnels = response.json().get('tunnels', [])
            if tunnels:
                public_url = tunnels[0]['public_url']
                print(f"\n✅ Link công khai: {public_url}")
                print(f"\n📋 Copy link này và chia sẻ cho mọi người!")
                print(f"🔗 {public_url}")
                return public_url
    except:
        print("\n⚠️  Không thể lấy link tự động. Kiểm tra: http://localhost:4040")
        print("   Hoặc chạy: ngrok http 8501")
    
    return None

def main():
    print("=" * 60)
    print("🌐 Tạo Link Công Khai cho Streamlit App")
    print("=" * 60)
    
    # Kiểm tra ngrok
    if not check_ngrok_installed():
        print("\n❌ Ngrok chưa được cài đặt!")
        print("\n📥 Cài đặt ngrok:")
        print("   1. Download: https://ngrok.com/download")
        print("   2. Giải nén và thêm vào PATH")
        print("   3. Đăng ký tài khoản miễn phí: https://dashboard.ngrok.com/signup")
        print("   4. Lấy auth token và chạy: ngrok config add-authtoken YOUR_TOKEN")
        print("\n💡 Hoặc sử dụng Streamlit Cloud (miễn phí, không cần ngrok)")
        print("   Xem file DEPLOY.md để biết cách deploy")
        return
    
    # Chạy Streamlit
    streamlit_process = run_streamlit()
    
    try:
        # Chạy ngrok
        public_url = run_ngrok()
        
        if public_url:
            # Mở browser
            time.sleep(2)
            webbrowser.open(public_url)
        
        print("\n" + "=" * 60)
        print("✅ App đang chạy!")
        print("   - Local: http://localhost:8501")
        if public_url:
            print(f"   - Public: {public_url}")
        print("\n⚠️  Nhấn Ctrl+C để dừng")
        print("=" * 60)
        
        # Giữ script chạy
        streamlit_process.wait()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Đang dừng...")
        streamlit_process.terminate()
        print("✅ Đã dừng!")

if __name__ == "__main__":
    main()

