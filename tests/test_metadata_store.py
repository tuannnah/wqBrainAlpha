import tempfile
import unittest
from pathlib import Path

from metadata_store import MetadataIntegrityError, MetadataStore
from research_models import Scope


class MetadataStoreTest(unittest.TestCase):
    def _populate(self, store):
        scope = Scope("EQUITY", "USA", 1, "TOP3000")
        scope_id = store.upsert_scope(scope)
        store.upsert_dataset({
            "id": "fundamental6",
            "name": "Fundamentals",
            "description": "Balance sheet and earnings",
            "category": {"id": "fundamental"},
        }, scope_id)
        store.upsert_data_field({
            "id": "assets",
            "dataset": {"id": "fundamental6"},
            "description": "Total assets",
            "type": "MATRIX",
        }, scope_id)
        store.upsert_operator({
            "name": "ts_rank",
            "scope": ["REGULAR"],
            "definition": "ts_rank(x, d)",
            "description": "Time-series rank",
        })

    def test_snapshot_requires_integrity_before_ready_and_supports_fts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "snapshot.sqlite"
            store = MetadataStore.create(path, "id-1", "USA tháng 6")
            self._populate(store)

            self.assertEqual(store.status(), "SYNCING")
            store.complete_snapshot()
            self.assertEqual(store.status(), "READY")
            self.assertEqual(
                [item["id"] for item in store.search_fields("total assets", 10)],
                ["assets"],
            )
            store.close()

    def test_empty_snapshot_fails_integrity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.sqlite"
            store = MetadataStore.create(path, "id-empty", "Empty")
            with self.assertRaises(MetadataIntegrityError):
                store.complete_snapshot()
            store.close()

    def test_list_ready_only_returns_ready_snapshots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_dir = Path(temp_dir)

            ready = MetadataStore.create(
                metadata_dir / "ready-id.sqlite", "ready-id", "Tên trùng"
            )
            self._populate(ready)
            ready.complete_snapshot()
            ready.close()

            syncing = MetadataStore.create(
                metadata_dir / "syncing-id.sqlite", "syncing-id", "Tên trùng"
            )
            self._populate(syncing)
            syncing.close()

            snapshots = MetadataStore.list_ready(metadata_dir)
            self.assertEqual([item.snapshot_id for item in snapshots], ["ready-id"])
            self.assertEqual(snapshots[0].label, "Tên trùng")


if __name__ == "__main__":
    unittest.main()
