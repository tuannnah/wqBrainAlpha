import io
import tempfile
import unittest
from pathlib import Path

from logging_setup import create_run_logger
from research_store import ResearchStore


class ResearchLoggingTest(unittest.TestCase):
    def test_event_reaches_console_file_and_database_without_secrets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            store = ResearchStore.create(logs_dir / "research.sqlite")
            run_id = store.start_run("snap", {})
            stream = io.StringIO()
            logger = create_run_logger(
                run_id=run_id,
                logs_dir=logs_dir,
                research_store=store,
                stream=stream,
                max_bytes=100000,
                backup_count=2,
            )
            try:
                logger.event("DEEPSEEK", "Đang tạo Alpha", {"api_key": "secret", "batch": 1})

                self.assertIn("[DEEPSEEK] Đang tạo Alpha", stream.getvalue())
                self.assertNotIn("secret", stream.getvalue())
                log_text = (logs_dir / f"{run_id}.log").read_text(encoding="utf-8")
                self.assertNotIn("secret", log_text)
                self.assertEqual(store.list_events(run_id)[0]["event_type"], "DEEPSEEK")
            finally:
                logger.close()
                store.close()


if __name__ == "__main__":
    unittest.main()
