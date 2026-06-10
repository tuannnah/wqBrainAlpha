import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ResearchDocumentationTest(unittest.TestCase):
    def test_readme_documents_required_operation(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for phrase in (
            "DEEPSEEK_API_KEY",
            "Tạo Metadata DB mới",
            "Chọn Metadata DB cũ",
            "quit",
            "PENDING_REVIEW",
            "không tự submit",
        ):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
