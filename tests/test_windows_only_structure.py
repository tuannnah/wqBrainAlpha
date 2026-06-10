import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WindowsOnlyStructureTest(unittest.TestCase):
    def test_macos_build_files_are_absent(self):
        self.assertFalse((ROOT / "mac").exists())

        for relative_path in ("README.md", ".gitignore"):
            text = (ROOT / relative_path).read_text(encoding="utf-8").lower()
            for marker in ("mac/", "build_mac", "create_icns", "icon.icns"):
                self.assertNotIn(marker, text, f"{marker!r} found in {relative_path}")

    def test_build_scripts_khong_dung_file_credentials(self):
        for relative_path in ("build.py", "build_windows.py", "create_zipapp.py"):
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn(
                "brain_credentials.txt",
                text,
                f"File credentials vẫn được dùng trong {relative_path}",
            )

    def test_builds_include_new_runtime_modules_and_config(self):
        required = [
            "worldquant_client.py",
            "metadata_store.py",
            "metadata_sync.py",
            "research_store.py",
            "deepseek_client.py",
            "research_engine.py",
            "research_config.json",
        ]
        for script_name in ("build.py", "build_windows.py", "create_zipapp.py"):
            text = (ROOT / script_name).read_text(encoding="utf-8")
            for name in required:
                self.assertIn(name, text, f"{script_name}: {name}")

    def test_builds_exclude_removed_hardcoded_modules(self):
        for script_name in ("build.py", "build_windows.py", "create_zipapp.py"):
            text = (ROOT / script_name).read_text(encoding="utf-8")
            self.assertNotIn("dataset_config.py", text)
            self.assertNotIn("alpha_strategy.py", text)

    def test_gitignore_excludes_runtime_data(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for marker in ("data/", "*.sqlite", "logs/"):
            self.assertIn(marker, text)

    def test_readme_huong_dan_dang_nhap_tuong_tac(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8").lower()

        self.assertNotIn("tạo file `brain_credentials.txt`", text)
        self.assertNotIn('["your_email@example.com", "your_password"]', text)

        for marker in (
            "mật khẩu không hiển thị",
            "không được lưu",
            "xác thực qr",
            "trình duyệt",
            "nhấn enter",
        ):
            self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
