import hashlib
import tempfile
import unittest
from pathlib import Path

from account_storage import build_account_paths, normalize_email


class AccountStorageTest(unittest.TestCase):
    def test_email_is_normalized_and_never_appears_in_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = build_account_paths(" User@Example.COM ", Path(temp_dir))

            expected_hash = hashlib.sha256(
                b"user@example.com"
            ).hexdigest()
            self.assertEqual(normalize_email(" User@Example.COM "), "user@example.com")
            self.assertEqual(paths.account_id, expected_hash)
            self.assertNotIn("user@example.com", str(paths.account_root))
            self.assertEqual(paths.metadata_dir, paths.account_root / "metadata")
            self.assertEqual(paths.research_db, paths.account_root / "research.sqlite")

    def test_directories_are_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = build_account_paths("user@example.com", Path(temp_dir))

            self.assertTrue(paths.metadata_dir.is_dir())
            self.assertTrue(paths.logs_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
