import zipapp
import os
import shutil
import subprocess
import sys


def create_zipapp():
    # Tạo thư mục build tạm.
    build_dir = "build"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    # Sao chép file nguồn (toàn bộ module runtime của pipeline nghiên cứu).
    source_files = [
        "main.py",
        "console_encoding.py",
        "brain_batch_alpha.py",
        "worldquant_client.py",
        "account_storage.py",
        "research_config.py",
        "research_models.py",
        "metadata_store.py",
        "metadata_sync.py",
        "research_store.py",
        "deepseek_client.py",
        "candidate_selector.py",
        "expression_parser.py",
        "expression_validator.py",
        "qualification.py",
        "alpha_prompts.py",
        "research_engine.py",
        "run_control.py",
        "logging_setup.py",
    ]

    for file in source_files:
        shutil.copy2(file, build_dir)

    # Sao chép file cấu hình nghiên cứu.
    config_files = [
        "research_config.json",
    ]

    for file in config_files:
        if os.path.exists(file):
            shutil.copy2(file, build_dir)
        else:
            print(f"Warning: không tìm thấy {file}, file sẽ được tạo khi chạy lần đầu")

    # Tạo requirements.txt cho gói zipapp.
    with open(os.path.join(build_dir, "requirements.txt"), "w") as f:
        f.write("requests>=2.31.0\npandas>=2.0.0\nlark>=1.1.9,<2\n")

    # Tạo __main__.py để chạy chương trình trong zipapp.
    with open(os.path.join(build_dir, "__main__.py"), "w") as f:
        f.write("""
import sys
import os


def install_deps():
    import subprocess
    import pkg_resources

    required = {'requests>=2.31.0', 'pandas>=2.0.0', 'lark>=1.1.9,<2'}
    installed = {f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set}
    missing = required - installed

    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])


if __name__ == '__main__':
    # Cài dependency còn thiếu.
    install_deps()

    # Import và chạy chương trình chính.
    from main import main
    main()
""")

    # Tạo file zipapp có thể chạy.
    output = "Alpha_Tool.pyz"
    if os.path.exists(output):
        os.remove(output)

    zipapp.create_archive(
        build_dir,
        output,
        main="__main__:main",
        compressed=True
    )

    print(f"\n✅ Tạo thành công {output}")
    print("Cách sử dụng:")
    print(f"python {output}")


if __name__ == "__main__":
    create_zipapp()
