import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NoHardcodedResearchTest(unittest.TestCase):
    def test_runtime_does_not_import_legacy_dataset_or_strategy_modules(self):
        runtime_files = [
            "main.py",
            "worldquant_client.py",
            "research_engine.py",
        ]
        for relative in runtime_files:
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("dataset_config", text)
            self.assertNotIn("AlphaStrategy", text)

    def test_hardcoded_modules_are_removed(self):
        self.assertFalse((ROOT / "dataset_config.py").exists())
        self.assertFalse((ROOT / "alpha_strategy.py").exists())


if __name__ == "__main__":
    unittest.main()
