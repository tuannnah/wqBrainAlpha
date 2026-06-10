import json
import tempfile
import unittest
from pathlib import Path

from research_config import ConfigError, ResearchConfig, load_config


class ResearchConfigTest(unittest.TestCase):
    def test_loads_defaults_and_rejects_invalid_limits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "research_config.json"
            path.write_text(json.dumps({
                "root_alphas_per_batch": 5,
                "max_batches_per_idea": 3,
                "max_parents": 2,
                "variants_per_parent": 5,
                "quality_gate_ratio": 0.8,
                "turnover_hard_limit": 0.9,
                "similarity_threshold": 0.9,
                "target_qualified_per_run": 10
            }), encoding="utf-8")

            config = load_config(path)

            self.assertEqual(config.root_alphas_per_batch, 5)
            self.assertEqual(config.deepseek_model, "deepseek-v4-pro")

            path.write_text('{"root_alphas_per_batch": 0}', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_creates_default_file_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "research_config.json"

            config = load_config(path)

            self.assertTrue(path.exists())
            self.assertEqual(config, ResearchConfig())

    def test_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "research_config.json"
            path.write_text('{"khong_ton_tai": 1}', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
