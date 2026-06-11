import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIPPED_DIRS = {".git", ".venv", "__pycache__", "build", "dist"}
TEXT_SUFFIXES = {".md", ".py", ".spec", ".txt", ".yml", ".yaml", ".json"}


def has_cjk(text):
    return any("\u4e00" <= char <= "\u9fff" for char in text)


class VietnameseLocalizationTest(unittest.TestCase):
    def test_source_filenames_do_not_use_chinese_characters(self):
        offenders = []
        for path in ROOT.rglob("*"):
            if any(part in SKIPPED_DIRS for part in path.parts):
                continue
            if has_cjk(path.name):
                offenders.append(str(path.relative_to(ROOT)))

        self.assertEqual(offenders, [])

    def test_text_files_do_not_contain_chinese_characters(self):
        offenders = []
        for path in ROOT.rglob("*"):
            if any(part in SKIPPED_DIRS for part in path.parts):
                continue
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue

            text = path.read_text(encoding="utf-8")
            if has_cjk(text):
                offenders.append(str(path.relative_to(ROOT)))

        self.assertEqual(offenders, [])
