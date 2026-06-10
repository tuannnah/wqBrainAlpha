import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from metadata_store import MetadataStore
from metadata_sync import MetadataSynchronizer
from research_models import Scope
from worldquant_client import WorldQuantApiError, WorldQuantRateLimitError


class FakeWorldQuantClient:
    def __init__(self, scopes=None, datasets=None, fields=None, categories=None,
                 operators=None, fail_on=None, rate_limit_endpoint=None):
        self._scopes = scopes if scopes is not None else [
            Scope("EQUITY", "USA", 1, "TOP3000")
        ]
        self._datasets = datasets if datasets is not None else [
            {"id": "ds1", "name": "Dataset", "description": "desc"}
        ]
        self._fields = fields if fields is not None else [{
            "id": "field1",
            "dataset": {"id": "ds1"},
            "description": "field",
            "type": "MATRIX",
        }]
        self._categories = categories if categories is not None else [
            {"id": "cat1", "name": "Category"}
        ]
        self._operators = operators if operators is not None else [
            {"name": "rank", "definition": "rank(x)"}
        ]
        self.fail_on = fail_on
        self.rate_limit_endpoint = rate_limit_endpoint
        self.dataset_calls = 0
        self.field_calls = 0
        self._rate_limited = set()

    def get_configuration(self):
        return {}

    def extract_scopes(self, configuration):
        return list(self._scopes)

    def iter_datasets(self, scope, limit=50):
        self.dataset_calls += 1
        if self.rate_limit_endpoint == "data-sets" and "data-sets" not in self._rate_limited:
            self._rate_limited.add("data-sets")
            raise WorldQuantRateLimitError(2)
        if self.fail_on == "data-sets":
            raise WorldQuantApiError("/data-sets", 500)
        return list(self._datasets)

    def iter_data_fields(self, scope, dataset_id=None, limit=50):
        self.field_calls += 1
        if self.fail_on == "data-fields":
            raise WorldQuantApiError("/data-fields", 500)
        return list(self._fields)

    def get_categories(self):
        if self.fail_on == "data-categories":
            raise WorldQuantApiError("/data-categories", 500)
        return list(self._categories)

    def get_operators(self):
        if self.fail_on == "operators":
            raise WorldQuantApiError("/operators", 500)
        return list(self._operators)


class MetadataSynchronizerTest(unittest.TestCase):
    def test_publishes_ready_snapshot_only_after_full_sync(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_dir = Path(temp_dir)
            client = FakeWorldQuantClient()
            result = MetadataSynchronizer(
                client, metadata_dir, sleep_func=Mock()
            ).create_snapshot("My DB")

            self.assertEqual(result.status, "READY")
            self.assertTrue(result.path.exists())
            self.assertFalse(
                result.path.with_suffix(".sqlite.tmp").exists()
            )
            self.assertEqual(
                [s.label for s in MetadataStore.list_ready(metadata_dir)],
                ["My DB"],
            )

    def test_failed_sync_is_not_listed_as_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_dir = Path(temp_dir)
            client = FakeWorldQuantClient(fail_on="data-fields")
            result = MetadataSynchronizer(
                client, metadata_dir, sleep_func=Mock(), max_retries=0
            ).create_snapshot("Broken")

            self.assertEqual(result.status, "FAILED")
            self.assertEqual(MetadataStore.list_ready(metadata_dir), [])

    def test_resume_skips_completed_endpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_dir = Path(temp_dir)

            failing = FakeWorldQuantClient(fail_on="data-fields")
            first = MetadataSynchronizer(
                failing, metadata_dir, sleep_func=Mock(), max_retries=0
            ).create_snapshot("Resumable")
            self.assertEqual(first.status, "FAILED")
            self.assertEqual(failing.dataset_calls, 1)

            healthy = FakeWorldQuantClient()
            second = MetadataSynchronizer(
                healthy, metadata_dir, sleep_func=Mock()
            ).create_snapshot("Resumable", resume_path=first.path)

            self.assertEqual(second.status, "READY")
            self.assertEqual(healthy.dataset_calls, 0)

    def test_rate_limit_sleeps_then_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_dir = Path(temp_dir)
            sleep_mock = Mock()
            client = FakeWorldQuantClient(rate_limit_endpoint="data-sets")

            result = MetadataSynchronizer(
                client, metadata_dir, sleep_func=sleep_mock
            ).create_snapshot("RateLimited")

            self.assertEqual(result.status, "READY")
            sleep_mock.assert_any_call(2)
            self.assertEqual(client.dataset_calls, 2)


if __name__ == "__main__":
    unittest.main()
