import tempfile
import unittest
from pathlib import Path

from research_store import ResearchStore


class ResearchStoreTest(unittest.TestCase):
    def test_records_alpha_lineage_tokens_and_pending_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore.create(Path(temp_dir) / "research.sqlite")
            try:
                run_id = store.start_run("snapshot-1", {"target_qualified_per_run": 10})
                idea_id = store.create_idea(run_id, "Price reversal", "DEEPSEEK")
                hypothesis_id = store.create_hypothesis(
                    idea_id,
                    "Extreme short-term losses revert",
                    "Behavioral overreaction",
                    ["pv1"],
                    ["returns", "close"],
                )
                request_id = store.record_llm_request(
                    run_id=run_id,
                    request_type="ROOT_ALPHA",
                    model="deepseek-v4-pro",
                    prompt={"idea": "Price reversal"},
                    response={"alphas": []},
                    usage={"prompt_tokens": 100, "completion_tokens": 20},
                )
                parent_id = store.create_alpha(
                    run_id, hypothesis_id, "rank(-returns)", "hash1", "finger1",
                    {"region": "USA"}, ["pv1"], None, 0, None,
                )
                child_id = store.create_alpha(
                    run_id, hypothesis_id, "rank(ts_mean(-returns, 5))",
                    "hash2", "finger2", {"region": "USA"}, ["pv1"],
                    parent_id, 1, "CHANGE_TIME_WINDOW",
                )
                store.enqueue_review(child_id, "wq-alpha-1")

                self.assertIsNotNone(request_id)
                self.assertEqual(store.get_alpha(child_id)["parent_id"], parent_id)
                self.assertEqual(store.count_qualified_for_run(run_id), 1)
                self.assertEqual(
                    store.list_pending_review()[0]["status"], "PENDING_REVIEW"
                )
                self.assertEqual(store.find_expression_hashes(), {"hash1", "hash2"})
            finally:
                store.close()

    def test_idea_lifecycle_and_next_pending(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore.create(Path(temp_dir) / "research.sqlite")
            try:
                run_id = store.start_run("snapshot-1", {})
                first = store.create_idea(run_id, "Idea 1", "DEEPSEEK")
                store.create_idea(run_id, "Idea 2", "DEEPSEEK")

                self.assertEqual(store.next_pending_idea(run_id)["id"], first)
                store.mark_idea_exhausted(first, "no parent after 3 batches")
                self.assertEqual(store.next_pending_idea(run_id)["content"], "Idea 2")
            finally:
                store.close()

    def test_secrets_are_redacted_in_storage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "research.sqlite"
            store = ResearchStore.create(path)
            try:
                run_id = store.start_run("snapshot-1", {})
                store.record_llm_request(
                    run_id=run_id,
                    request_type="ROOT_ALPHA",
                    model="deepseek-v4-pro",
                    prompt={"authorization": "Bearer secret-key", "idea": "x"},
                    response={"api_key": "sk-12345", "alphas": []},
                    usage={},
                )
            finally:
                store.close()

            data = path.read_bytes()
            self.assertNotIn(b"secret-key", data)
            self.assertNotIn(b"sk-12345", data)


if __name__ == "__main__":
    unittest.main()
