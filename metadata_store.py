"""Lưu trữ metadata WorldQuant theo từng snapshot SQLite có phiên bản."""

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from research_models import Scope


SCHEMA_VERSION = 1


class MetadataIntegrityError(RuntimeError):
    """Snapshot không đạt kiểm tra toàn vẹn."""


@dataclass(frozen=True)
class SnapshotInfo:
    snapshot_id: str
    label: str
    status: str
    created_at: str
    completed_at: str
    dataset_count: int
    field_count: int
    operator_count: int
    path: Path


def _now():
    return datetime.now(timezone.utc).isoformat()


def _canonical(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class MetadataStore:
    SCHEMA = """
    PRAGMA foreign_keys = ON;
    CREATE TABLE snapshot (
        id TEXT PRIMARY KEY,
        label TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('SYNCING', 'READY', 'FAILED')),
        schema_version INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        dataset_count INTEGER NOT NULL DEFAULT 0,
        field_count INTEGER NOT NULL DEFAULT 0,
        operator_count INTEGER NOT NULL DEFAULT 0,
        error_message TEXT
    );
    CREATE TABLE scopes (
        id INTEGER PRIMARY KEY,
        instrument_type TEXT NOT NULL,
        region TEXT NOT NULL,
        delay INTEGER NOT NULL,
        universe TEXT NOT NULL,
        UNIQUE(instrument_type, region, delay, universe)
    );
    CREATE TABLE categories (
        id TEXT PRIMARY KEY,
        name TEXT,
        parent_id TEXT,
        raw_json TEXT NOT NULL
    );
    CREATE TABLE datasets (
        id TEXT PRIMARY KEY,
        name TEXT,
        description TEXT,
        category_id TEXT,
        raw_json TEXT NOT NULL
    );
    CREATE TABLE dataset_scopes (
        dataset_id TEXT NOT NULL REFERENCES datasets(id),
        scope_id INTEGER NOT NULL REFERENCES scopes(id),
        raw_json TEXT NOT NULL,
        PRIMARY KEY(dataset_id, scope_id)
    );
    CREATE TABLE data_fields (
        id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL REFERENCES datasets(id),
        description TEXT,
        field_type TEXT NOT NULL,
        unit TEXT,
        raw_json TEXT NOT NULL
    );
    CREATE TABLE data_field_scopes (
        field_id TEXT NOT NULL REFERENCES data_fields(id),
        scope_id INTEGER NOT NULL REFERENCES scopes(id),
        coverage REAL,
        date_coverage REAL,
        raw_json TEXT NOT NULL,
        PRIMARY KEY(field_id, scope_id)
    );
    CREATE TABLE operators (
        name TEXT PRIMARY KEY,
        operator_scope TEXT,
        definition TEXT,
        description TEXT,
        level TEXT,
        raw_json TEXT NOT NULL
    );
    CREATE TABLE sync_events (
        id INTEGER PRIMARY KEY,
        endpoint TEXT NOT NULL,
        scope_key TEXT,
        offset INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        record_count INTEGER NOT NULL DEFAULT 0,
        error_message TEXT,
        created_at TEXT NOT NULL
    );
    CREATE VIRTUAL TABLE data_fields_fts USING fts5(
        field_id UNINDEXED,
        description,
        dataset_id
    );
    """

    def __init__(self, connection, path, snapshot_id):
        self.connection = connection
        self.path = Path(path)
        self.snapshot_id = snapshot_id

    # -- Lifecycle ---------------------------------------------------------

    @classmethod
    def _connect(cls, path):
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @classmethod
    def create(cls, path, snapshot_id, label):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = cls._connect(path)
        connection.executescript(cls.SCHEMA)
        connection.execute(
            "INSERT INTO snapshot(id, label, status, schema_version, created_at)"
            " VALUES(?, ?, 'SYNCING', ?, ?)",
            (snapshot_id, label, SCHEMA_VERSION, _now()),
        )
        connection.commit()
        return cls(connection, path, snapshot_id)

    @classmethod
    def open(cls, path):
        path = Path(path)
        connection = cls._connect(path)
        row = connection.execute("SELECT id FROM snapshot LIMIT 1").fetchone()
        snapshot_id = row["id"] if row else None
        return cls(connection, path, snapshot_id)

    def close(self):
        self.connection.close()

    def status(self):
        row = self.connection.execute(
            "SELECT status FROM snapshot WHERE id = ?", (self.snapshot_id,)
        ).fetchone()
        return row["status"] if row else None

    # -- Upserts -----------------------------------------------------------

    def upsert_scope(self, scope):
        self.connection.execute(
            "INSERT OR IGNORE INTO scopes(instrument_type, region, delay, universe)"
            " VALUES(?, ?, ?, ?)",
            (scope.instrument_type, scope.region, scope.delay, scope.universe),
        )
        row = self.connection.execute(
            "SELECT id FROM scopes WHERE instrument_type = ? AND region = ?"
            " AND delay = ? AND universe = ?",
            (scope.instrument_type, scope.region, scope.delay, scope.universe),
        ).fetchone()
        self.connection.commit()
        return row["id"]

    def upsert_category(self, payload):
        parent = payload.get("parent")
        parent_id = parent.get("id") if isinstance(parent, dict) else payload.get("parentId")
        self.connection.execute(
            "INSERT INTO categories(id, name, parent_id, raw_json) VALUES(?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET name=excluded.name,"
            " parent_id=excluded.parent_id, raw_json=excluded.raw_json",
            (payload["id"], payload.get("name"), parent_id, _canonical(payload)),
        )
        self.connection.commit()

    def upsert_dataset(self, payload, scope_id=None):
        category = payload.get("category")
        category_id = category.get("id") if isinstance(category, dict) else payload.get("categoryId")
        self.connection.execute(
            "INSERT INTO datasets(id, name, description, category_id, raw_json)"
            " VALUES(?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET name=excluded.name,"
            " description=excluded.description, category_id=excluded.category_id,"
            " raw_json=excluded.raw_json",
            (
                payload["id"],
                payload.get("name"),
                payload.get("description"),
                category_id,
                _canonical(payload),
            ),
        )
        if scope_id is not None:
            self.connection.execute(
                "INSERT OR REPLACE INTO dataset_scopes(dataset_id, scope_id, raw_json)"
                " VALUES(?, ?, ?)",
                (payload["id"], scope_id, _canonical(payload)),
            )
        self.connection.commit()

    def ensure_dataset_stub(self, dataset_id, payload=None):
        """Chèn dataset tối thiểu nếu chưa tồn tại (không ghi đè bản đầy đủ)."""

        payload = payload or {"id": dataset_id}
        self.connection.execute(
            "INSERT OR IGNORE INTO datasets(id, name, description, category_id, raw_json)"
            " VALUES(?, ?, ?, ?, ?)",
            (
                dataset_id,
                payload.get("name"),
                payload.get("description"),
                None,
                _canonical(payload),
            ),
        )
        self.connection.commit()

    def upsert_data_field(self, payload, scope_id=None):
        dataset = payload.get("dataset")
        dataset_id = dataset.get("id") if isinstance(dataset, dict) else payload.get("datasetId")
        field_id = payload["id"]
        description = payload.get("description")
        field_type = payload.get("type") or payload.get("field_type") or "MATRIX"
        self.connection.execute(
            "INSERT INTO data_fields(id, dataset_id, description, field_type, unit, raw_json)"
            " VALUES(?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET dataset_id=excluded.dataset_id,"
            " description=excluded.description, field_type=excluded.field_type,"
            " unit=excluded.unit, raw_json=excluded.raw_json",
            (
                field_id,
                dataset_id,
                description,
                field_type,
                payload.get("unit"),
                _canonical(payload),
            ),
        )
        self.connection.execute(
            "DELETE FROM data_fields_fts WHERE field_id = ?", (field_id,)
        )
        self.connection.execute(
            "INSERT INTO data_fields_fts(field_id, description, dataset_id)"
            " VALUES(?, ?, ?)",
            (field_id, description or "", dataset_id or ""),
        )
        if scope_id is not None:
            coverage = payload.get("coverage")
            date_coverage = payload.get("dateCoverage") or payload.get("userCount")
            self.connection.execute(
                "INSERT OR REPLACE INTO data_field_scopes"
                "(field_id, scope_id, coverage, date_coverage, raw_json)"
                " VALUES(?, ?, ?, ?, ?)",
                (field_id, scope_id, coverage, date_coverage, _canonical(payload)),
            )
        self.connection.commit()

    def upsert_operator(self, payload):
        scope = payload.get("scope")
        operator_scope = ",".join(scope) if isinstance(scope, list) else scope
        self.connection.execute(
            "INSERT INTO operators(name, operator_scope, definition, description, level, raw_json)"
            " VALUES(?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET operator_scope=excluded.operator_scope,"
            " definition=excluded.definition, description=excluded.description,"
            " level=excluded.level, raw_json=excluded.raw_json",
            (
                payload["name"],
                operator_scope,
                payload.get("definition"),
                payload.get("description"),
                payload.get("level"),
                _canonical(payload),
            ),
        )
        self.connection.commit()

    def record_sync_event(self, endpoint, scope_key, offset, status, record_count=0,
                          error_message=None):
        self.connection.execute(
            "INSERT INTO sync_events"
            "(endpoint, scope_key, offset, status, record_count, error_message, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?)",
            (endpoint, scope_key, offset, status, record_count, error_message, _now()),
        )
        self.connection.commit()

    def is_page_completed(self, endpoint, scope_key, offset):
        row = self.connection.execute(
            "SELECT 1 FROM sync_events WHERE endpoint = ? AND scope_key IS ?"
            " AND offset = ? AND status = 'COMPLETED' LIMIT 1",
            (endpoint, scope_key, offset),
        ).fetchone()
        return row is not None

    # -- Queries -----------------------------------------------------------

    def _count(self, table):
        return self.connection.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]

    @staticmethod
    def _build_match(query):
        tokens = re.findall(r"[A-Za-z0-9]+", query or "")
        if not tokens:
            return None
        return " OR ".join(f'"{token}"' for token in tokens)

    def search_fields(self, query, limit, dataset_ids=None):
        match = self._build_match(query)
        if not match:
            return []
        sql = (
            "SELECT id, dataset_id, description, field_type FROM data_fields"
            " WHERE id IN (SELECT field_id FROM data_fields_fts"
            " WHERE data_fields_fts MATCH ?)"
        )
        params = [match]
        if dataset_ids:
            placeholders = ",".join("?" for _ in dataset_ids)
            sql += f" AND dataset_id IN ({placeholders})"
            params.extend(dataset_ids)
        sql += " LIMIT ?"
        params.append(limit)
        rows = self.connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def dataset_catalog(self, limit=None, excluded_ids=None):
        sql = (
            "SELECT d.id, d.name, d.description, d.category_id,"
            " COUNT(f.id) AS field_count FROM datasets d"
            " LEFT JOIN data_fields f ON f.dataset_id = d.id"
        )
        params = []
        if excluded_ids:
            placeholders = ",".join("?" for _ in excluded_ids)
            sql += f" WHERE d.id NOT IN ({placeholders})"
            params.extend(excluded_ids)
        sql += " GROUP BY d.id ORDER BY field_count DESC, d.id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [dict(row) for row in self.connection.execute(sql, params).fetchall()]

    def fields_in_datasets(self, dataset_ids, limit):
        if not dataset_ids:
            return []
        placeholders = ",".join("?" for _ in dataset_ids)
        rows = self.connection.execute(
            "SELECT id, dataset_id, description, field_type FROM data_fields"
            f" WHERE dataset_id IN ({placeholders}) ORDER BY id LIMIT ?",
            [*dataset_ids, limit],
        ).fetchall()
        return [dict(row) for row in rows]

    def field_records(self, field_ids):
        if not field_ids:
            return []
        placeholders = ",".join("?" for _ in field_ids)
        rows = self.connection.execute(
            "SELECT id, dataset_id, description, field_type FROM data_fields"
            f" WHERE id IN ({placeholders})",
            list(field_ids),
        ).fetchall()
        return [dict(row) for row in rows]

    def operators_for_types(self, field_types=None):
        rows = self.connection.execute(
            "SELECT name, operator_scope, definition, description, level FROM operators"
            " ORDER BY name"
        ).fetchall()
        return [dict(row) for row in rows]

    def scope_for_dataset(self, dataset_id):
        rows = self.connection.execute(
            "SELECT s.instrument_type, s.region, s.delay, s.universe FROM scopes s"
            " JOIN dataset_scopes ds ON ds.scope_id = s.id WHERE ds.dataset_id = ?",
            (dataset_id,),
        ).fetchall()
        return [
            Scope(row["instrument_type"], row["region"], row["delay"], row["universe"])
            for row in rows
        ]

    # -- Completion --------------------------------------------------------

    def complete_snapshot(self):
        foreign_key_errors = self.connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
        if foreign_key_errors:
            raise MetadataIntegrityError(str([tuple(r) for r in foreign_key_errors]))
        counts = {
            "dataset_count": self._count("datasets"),
            "field_count": self._count("data_fields"),
            "operator_count": self._count("operators"),
        }
        if min(counts.values()) == 0:
            raise MetadataIntegrityError("Snapshot thiếu dataset, field hoặc operator")
        self.connection.execute(
            "UPDATE snapshot SET status='READY', completed_at=?,"
            " dataset_count=?, field_count=?, operator_count=? WHERE id=?",
            (
                _now(),
                counts["dataset_count"],
                counts["field_count"],
                counts["operator_count"],
                self.snapshot_id,
            ),
        )
        self.connection.commit()

    def fail_snapshot(self, error_message):
        self.connection.execute(
            "UPDATE snapshot SET status='FAILED', error_message=? WHERE id=?",
            (error_message, self.snapshot_id),
        )
        self.connection.commit()

    # -- Discovery ---------------------------------------------------------

    @classmethod
    def list_ready(cls, metadata_dir):
        metadata_dir = Path(metadata_dir)
        snapshots = []
        for path in sorted(metadata_dir.glob("*.sqlite")):
            try:
                connection = cls._connect(path)
                row = connection.execute(
                    "SELECT id, label, status, created_at, completed_at,"
                    " dataset_count, field_count, operator_count FROM snapshot LIMIT 1"
                ).fetchone()
                connection.close()
            except sqlite3.DatabaseError:
                continue
            if not row or row["status"] != "READY":
                continue
            snapshots.append(SnapshotInfo(
                snapshot_id=row["id"],
                label=row["label"],
                status=row["status"],
                created_at=row["created_at"],
                completed_at=row["completed_at"],
                dataset_count=row["dataset_count"],
                field_count=row["field_count"],
                operator_count=row["operator_count"],
                path=path,
            ))
        snapshots.sort(key=lambda item: item.created_at)
        return snapshots
