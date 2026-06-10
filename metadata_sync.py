"""Đồng bộ toàn bộ metadata WorldQuant vào một snapshot SQLite, có checkpoint."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from metadata_store import MetadataStore
from worldquant_client import WorldQuantApiError, WorldQuantRateLimitError


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_id: str
    path: Path
    status: str
    error_message: Optional[str] = None


def default_label():
    return datetime.now().strftime("Snapshot %Y-%m-%d %H-%M")


class MetadataSynchronizer:
    def __init__(self, client, metadata_dir, sleep_func, max_retries=2,
                 rate_limit_backoff_seconds=2.0):
        self.client = client
        self.metadata_dir = Path(metadata_dir)
        self.sleep = sleep_func
        self.max_retries = max_retries
        self.rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self._known_datasets = set()

    # -- Public ------------------------------------------------------------

    def create_snapshot(self, label, resume_path=None):
        self._known_datasets = set()
        if resume_path is not None:
            temp_path = Path(resume_path)
            store = MetadataStore.open(temp_path)
            snapshot_id = store.snapshot_id
        else:
            snapshot_id = str(uuid.uuid4())
            temp_path = self.metadata_dir / f"{snapshot_id}.sqlite.tmp"
            store = MetadataStore.create(
                temp_path, snapshot_id, label or default_label()
            )

        try:
            configuration = self._call_with_retry(self.client.get_configuration)
            for scope in self.client.extract_scopes(configuration):
                scope_id = store.upsert_scope(scope)
                self._sync_datasets(store, scope, scope_id)
                self._sync_fields(store, scope, scope_id)
            self._sync_categories(store)
            self._sync_operators(store)
            store.complete_snapshot()
            store.close()
            final_path = self.metadata_dir / f"{snapshot_id}.sqlite"
            temp_path.replace(final_path)
            return SnapshotResult(snapshot_id, final_path, "READY")
        except Exception as exc:  # noqa: BLE001 - chuyển mọi lỗi thành FAILED
            store.fail_snapshot(str(exc))
            store.close()
            return SnapshotResult(snapshot_id, temp_path, "FAILED", str(exc))

    # -- Endpoints ---------------------------------------------------------

    @staticmethod
    def _scope_key(scope):
        return f"{scope.instrument_type}|{scope.region}|{scope.delay}|{scope.universe}"

    def _sync_datasets(self, store, scope, scope_id):
        scope_key = self._scope_key(scope)
        if store.is_page_completed("data-sets", scope_key, 0):
            return
        rows = self._call_with_retry(lambda: list(self.client.iter_datasets(scope)))
        for row in rows:
            store.upsert_dataset(row, scope_id)
            self._known_datasets.add(row["id"])
        store.record_sync_event("data-sets", scope_key, 0, "COMPLETED", len(rows))

    def _sync_fields(self, store, scope, scope_id):
        scope_key = self._scope_key(scope)
        if store.is_page_completed("data-fields", scope_key, 0):
            return
        rows = self._call_with_retry(
            lambda: list(self.client.iter_data_fields(scope))
        )
        for row in rows:
            dataset = row.get("dataset")
            dataset_id = (
                dataset.get("id") if isinstance(dataset, dict)
                else row.get("datasetId")
            )
            if dataset_id and dataset_id not in self._known_datasets:
                stub = dataset if isinstance(dataset, dict) else {"id": dataset_id}
                store.ensure_dataset_stub(dataset_id, stub)
                self._known_datasets.add(dataset_id)
            store.upsert_data_field(row, scope_id)
        store.record_sync_event("data-fields", scope_key, 0, "COMPLETED", len(rows))

    def _sync_categories(self, store):
        if store.is_page_completed("data-categories", None, 0):
            return
        rows = self._as_list(self._call_with_retry(self.client.get_categories))
        for row in rows:
            store.upsert_category(row)
        store.record_sync_event("data-categories", None, 0, "COMPLETED", len(rows))

    def _sync_operators(self, store):
        if store.is_page_completed("operators", None, 0):
            return
        rows = self._as_list(self._call_with_retry(self.client.get_operators))
        for row in rows:
            store.upsert_operator(row)
        store.record_sync_event("operators", None, 0, "COMPLETED", len(rows))

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _as_list(payload):
        if isinstance(payload, dict):
            return payload.get("results", []) or []
        return payload or []

    def _call_with_retry(self, operation):
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except WorldQuantRateLimitError as exc:
                if attempt == self.max_retries:
                    raise
                self.sleep(exc.retry_after_seconds)
            except WorldQuantApiError:
                if attempt == self.max_retries:
                    raise
                self.sleep(self.rate_limit_backoff_seconds * (attempt + 1))
