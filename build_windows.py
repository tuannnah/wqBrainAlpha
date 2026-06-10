import PyInstaller.__main__
import os
import sys
import shutil

from console_encoding import configure_utf8_console

configure_utf8_console()

# Đảm bảo thư mục dist tồn tại.
if not os.path.exists('dist'):
    os.makedirs('dist')

# Các module runtime của pipeline nghiên cứu Alpha tự động.
RUNTIME_MODULES = [
    'worldquant_client.py',
    'account_storage.py',
    'research_config.py',
    'research_models.py',
    'metadata_store.py',
    'metadata_sync.py',
    'research_store.py',
    'deepseek_client.py',
    'candidate_selector.py',
    'expression_parser.py',
    'expression_validator.py',
    'qualification.py',
    'alpha_prompts.py',
    'research_engine.py',
    'run_control.py',
    'logging_setup.py',
    'brain_batch_alpha.py',
    'console_encoding.py',
]

# Tham số dòng lệnh cho PyInstaller.
args = [
    'main.py',  # Entry point chính.
    '--name=Alpha_Tool',  # Tên file thực thi.
    '--onefile',  # Đóng gói thành một file duy nhất.
    '--console',  # Chạy với cửa sổ console.
    '--add-data=research_config.json{0}.'.format(os.pathsep),  # Cấu hình nghiên cứu.
    '--collect-all=lark',  # Thu thập toàn bộ grammar/data của lark.
    '--clean',  # Dọn file tạm của PyInstaller.
    '--noconfirm',  # Ghi đè không hỏi lại.
]
args[5:5] = [
    '--add-data={0}{1}.'.format(module, os.pathsep) for module in RUNTIME_MODULES
]

# Thêm icon nếu file tồn tại.
if os.path.exists('icon.ico'):
    args.append('--icon=icon.ico')

# Chạy lệnh đóng gói.
PyInstaller.__main__.run(args)

# Sau khi đóng gói, sao chép hoặc tạo file cấu hình trong thư mục dist.
print("\nĐang xử lý file cấu hình...")
try:
    if os.path.exists('alpha_ids.txt'):
        shutil.copy2('alpha_ids.txt', 'dist/')
        print("✅ Sao chép alpha_ids.txt thành công")
    else:
        with open('dist/alpha_ids.txt', 'w') as f:
            ...
        print("✅ Đã tạo file alpha_ids.txt rỗng")

except Exception as e:
    print(f"❌ Lỗi khi xử lý file cấu hình: {str(e)}")
