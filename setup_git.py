"""
Script tự động setup Git và chuẩn bị push lên GitHub
"""

import subprocess
import os
import sys

def run_command(cmd, cwd=None):
    """Chạy command và hiển thị output"""
    print(f"\n🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
    return result.returncode == 0

def main():
    print("=" * 60)
    print("🚀 Setup Git để Push lên GitHub")
    print("=" * 60)
    
    # Kiểm tra đang ở đâu
    current_dir = os.getcwd()
    print(f"\n📁 Current directory: {current_dir}")
    
    # Kiểm tra có phải git repo không
    if not os.path.exists('.git'):
        print("\n⚠️  Chưa có git repo. Đang khởi tạo...")
        if not run_command('git init'):
            print("❌ Không thể khởi tạo git repo")
            return
    
    # Check git status
    print("\n📊 Git status:")
    run_command('git status')
    
    # Kiểm tra .gitignore
    if not os.path.exists('.gitignore'):
        print("\n⚠️  Chưa có .gitignore. Vui lòng tạo file .gitignore trước!")
        return
    
    print("\n" + "=" * 60)
    print("✅ Setup hoàn tất!")
    print("\n📝 Next steps:")
    print("1. git add .")
    print("2. git commit -m 'Initial commit'")
    print("3. Tạo repo mới trên GitHub")
    print("4. git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git")
    print("5. git push -u origin main")
    print("\n💡 Sau đó deploy trên Streamlit Cloud!")
    print("=" * 60)

if __name__ == "__main__":
    main()

