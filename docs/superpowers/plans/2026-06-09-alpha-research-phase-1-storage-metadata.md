# Alpha Research Phase 1: Storage And Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build account-isolated storage, versioned metadata snapshots, and a resumable full-account WorldQuant metadata synchronizer.

**Architecture:** Hash the normalized email into an account directory under Windows local application data. Store every metadata version in its own SQLite file, and expose metadata through a dedicated store. A new WorldQuant client owns authentication plus paginated metadata API calls; the synchronizer writes to a temporary snapshot and publishes it only after integrity checks.

**Tech Stack:** Python 3.12, `requests`, `sqlite3`, FTS5, `unittest`.

---

### Task 1: Shared Models, Configuration, And Account Paths

**Files:**
- Create: `research_models.py`
- Create: `research_config.py`
- Create: `research_config.json`
- Create: `account_storage.py`
- Create: `tests/test_research_config.py`
- Create: `tests/test_account_storage.py`

- [ ] **Step 1: Write failing configuration and account-isolation tests**

```python
# tests/test_research_config.py
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


# tests/test_account_storage.py
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
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_config tests.test_account_storage -v
```

Expected: import errors for the four new modules.

- [ ] **Step 3: Implement immutable models and validated JSON config**

```python
# research_models.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Scope:
    instrument_type: str
    region: str
    delay: int
    universe: str


@dataclass(frozen=True)
class AlphaDraft:
    hypothesis: str
    rationale: str
    expression: str
    dataset_ids: List[str]
    field_ids: List[str]
    operator_names: List[str]
    settings: Dict[str, Any]
    parent_id: Optional[int] = None
    generation: int = 0
    improvement_direction: Optional[str] = None


@dataclass(frozen=True)
class SimulationResult:
    worldquant_alpha_id: Optional[str]
    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)
```

```python
# research_config.py
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ResearchConfig:
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_timeout_seconds: int = 90
    deepseek_max_retries: int = 2
    deepseek_max_output_tokens: int = 6000
    root_alphas_per_batch: int = 5
    max_batches_per_idea: int = 3
    max_parents: int = 2
    variants_per_parent: int = 5
    quality_gate_ratio: float = 0.8
    sharpe_threshold: float = 1.5
    fitness_threshold: float = 1.0
    turnover_min: float = 0.01
    turnover_hard_limit: float = 0.9
    similarity_threshold: float = 0.9
    candidate_fields_min: int = 20
    candidate_fields_max: int = 50
    target_qualified_per_run: int = 10
    simulation_poll_timeout_seconds: int = 900
    simulation_delay_seconds: float = 5.0
    rate_limit_backoff_seconds: float = 2.0
    raw_response_max_chars: int = 200_000
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5

    def validate(self):
        positive = (
            "deepseek_timeout_seconds",
            "deepseek_max_output_tokens",
            "root_alphas_per_batch",
            "max_batches_per_idea",
            "max_parents",
            "variants_per_parent",
            "candidate_fields_min",
            "candidate_fields_max",
            "target_qualified_per_run",
            "simulation_poll_timeout_seconds",
            "raw_response_max_chars",
            "log_max_bytes",
            "log_backup_count",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} phải lớn hơn 0")
        for name in (
            "quality_gate_ratio",
            "turnover_hard_limit",
            "similarity_threshold",
        ):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ConfigError(f"{name} phải nằm trong khoảng (0, 1]")
        for name in (
            "sharpe_threshold",
            "fitness_threshold",
            "turnover_min",
            "simulation_delay_seconds",
            "rate_limit_backoff_seconds",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} không được âm")
        if self.candidate_fields_min > self.candidate_fields_max:
            raise ConfigError("candidate_fields_min không được lớn hơn candidate_fields_max")
        if self.variants_per_parent > 5:
            raise ConfigError("variants_per_parent không được lớn hơn 5")


def load_config(path):
    path = Path(path)
    defaults = ResearchConfig()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(defaults), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return defaults
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(ResearchConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"Khóa config không hỗ trợ: {sorted(unknown)}")
    config = ResearchConfig(**raw)
    config.validate()
    return config
```

Create `research_config.json` with the exact serialized defaults from
`ResearchConfig`.

- [ ] **Step 4: Implement account paths**

```python
# account_storage.py
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path


APP_DIR_NAME = "WorldQuantBrainAlpha"


@dataclass(frozen=True)
class AccountPaths:
    account_id: str
    base_dir: Path
    account_root: Path
    metadata_dir: Path
    research_db: Path
    logs_dir: Path
    config_path: Path

    def ensure(self):
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def normalize_email(email):
    return email.strip().lower()


def default_base_dir():
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_DIR_NAME
    return Path.home() / "AppData" / "Local" / APP_DIR_NAME


def build_account_paths(email, base_dir=None):
    normalized = normalize_email(email)
    account_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    base = Path(base_dir) if base_dir else default_base_dir()
    account_root = base / "accounts" / account_id
    paths = AccountPaths(
        account_id=account_id,
        base_dir=base,
        account_root=account_root,
        metadata_dir=account_root / "metadata",
        research_db=account_root / "research.sqlite",
        logs_dir=account_root / "logs",
        config_path=base / "research_config.json",
    )
    paths.ensure()
    return paths
```

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_config tests.test_account_storage -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add research_models.py research_config.py research_config.json account_storage.py tests/test_research_config.py tests/test_account_storage.py
git commit -m "feat: add research configuration and account storage"
```

### Task 2: Metadata Snapshot Store And FTS Search

**Files:**
- Create: `metadata_store.py`
- Create: `tests/test_metadata_store.py`

- [ ] **Step 1: Write failing snapshot lifecycle and search tests**

```python
import tempfile
import unittest
from pathlib import Path

from metadata_store import MetadataStore
from research_models import Scope


class MetadataStoreTest(unittest.TestCase):
    def test_snapshot_requires_integrity_before_ready_and_supports_fts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "snapshot.sqlite"
            store = MetadataStore.create(path, "id-1", "USA tháng 6")
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

            self.assertEqual(store.status(), "SYNCING")
            store.complete_snapshot()
            self.assertEqual(store.status(), "READY")
            self.assertEqual(
                [item["id"] for item in store.search_fields("total assets", 10)],
                ["assets"],
            )
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_metadata_store -v
```

Expected: `ModuleNotFoundError: metadata_store`.

- [ ] **Step 3: Implement SQLite schema and transactional upserts**

Create `MetadataStore.SCHEMA` with:

```sql
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
```

Implement `create`, `open`, `close`, `status`, `upsert_scope`,
`upsert_category`, `upsert_dataset`, `upsert_data_field`, `upsert_operator`,
`record_sync_event`, `search_fields`, and `complete_snapshot`. Store raw API
objects with:

```python
json.dumps(payload, ensure_ascii=False, sort_keys=True)
```

`complete_snapshot()` must run:

```python
foreign_key_errors = self.connection.execute(
    "PRAGMA foreign_key_check"
).fetchall()
if foreign_key_errors:
    raise MetadataIntegrityError(str(foreign_key_errors))
counts = {
    "dataset_count": self._count("datasets"),
    "field_count": self._count("data_fields"),
    "operator_count": self._count("operators"),
}
if min(counts.values()) == 0:
    raise MetadataIntegrityError("Snapshot thiếu dataset, field hoặc operator")
```

- [ ] **Step 4: Run test and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_metadata_store -v
```

Expected: all tests pass.

- [ ] **Step 5: Add duplicate labels and ready-snapshot discovery**

Add a test that creates two SQLite files with the same label and distinct IDs,
marks only one `READY`, then verifies:

```python
snapshots = MetadataStore.list_ready(metadata_dir)
self.assertEqual([item.snapshot_id for item in snapshots], ["ready-id"])
self.assertEqual(snapshots[0].label, "Tên trùng")
```

Implement discovery by scanning `*.sqlite`, opening each read-only enough to
read `snapshot`, and skipping unreadable, `SYNCING`, or `FAILED` files.

- [ ] **Step 6: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_metadata_store -v
git add metadata_store.py tests/test_metadata_store.py
git commit -m "feat: add versioned metadata snapshot store"
```

### Task 3: WorldQuant Client Metadata And Simulation API

**Files:**
- Create: `worldquant_client.py`
- Create: `tests/test_worldquant_client.py`
- Modify: `brain_batch_alpha.py`
- Modify: `tests/test_interactive_login.py`

- [ ] **Step 1: Write failing tests for authentication compatibility and pages**

```python
from unittest.mock import Mock

from research_models import Scope
from worldquant_client import WorldQuantClient


def response(status, data, headers=None):
    item = Mock(status_code=status, headers=headers or {})
    item.json.return_value = data
    return item


class WorldQuantMetadataClientTest(unittest.TestCase):
    def test_iterates_all_pages_with_scope_parameters(self):
        session = Mock()
        session.post.return_value = response(201, {})
        session.get.side_effect = [
            response(200, {"count": 3, "results": [{"id": "a"}, {"id": "b"}]}),
            response(200, {"count": 3, "results": [{"id": "c"}]}),
        ]
        client = WorldQuantClient("user@example.com", "secret", session=session)

        rows = list(client.iter_data_fields(
            Scope("EQUITY", "USA", 1, "TOP3000"),
            limit=2,
        ))

        self.assertEqual([row["id"] for row in rows], ["a", "b", "c"])
        self.assertEqual(session.get.call_args_list[0].kwargs["params"]["offset"], 0)
        self.assertEqual(session.get.call_args_list[1].kwargs["params"]["offset"], 2)
```

Also test:

- `/configuration`, `/data-sets`, `/data-categories`, and `/operators`;
- timeout on every request;
- HTTP 429 exposes `Retry-After` through `WorldQuantRateLimitError`;
- `simulate_alpha(payload)` returns `SimulationResult`, including compile and
  authorization failures instead of printing and returning `None`.

- [ ] **Step 2: Run tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_worldquant_client -v
```

Expected: import error for `worldquant_client`.

- [ ] **Step 3: Move authentication into `WorldQuantClient`**

Copy the already-tested authentication and verification behavior from
`BrainBatchAlpha` into `WorldQuantClient`. Keep constructor dependency
injection:

```python
def __init__(
    self,
    email,
    password,
    session=None,
    browser_open=None,
    confirmation_input=None,
    sleep_func=sleep,
):
    self.session = session or requests.Session()
    self.session.auth = HTTPBasicAuth(email, password)
    self.browser_open = browser_open or webbrowser.open
    self.confirmation_input = confirmation_input or input
    self.sleep = sleep_func
    self._setup_authentication()
```

Make `BrainBatchAlpha` inherit from `WorldQuantClient` temporarily:

```python
from worldquant_client import AuthenticationError, WorldQuantClient


class BrainBatchAlpha(WorldQuantClient):
    """Compatibility adapter for the legacy menu until Phase 4."""
```

Do not remove legacy simulation and submit methods yet.

- [ ] **Step 4: Implement metadata pagination and scope extraction**

Use one request helper:

```python
def _request_json(self, method, path, **kwargs):
    response = self.session.request(
        method,
        f"{self.API_BASE_URL}{path}",
        timeout=self.REQUEST_TIMEOUT_SECONDS,
        **kwargs,
    )
    if response.status_code == 429:
        raise WorldQuantRateLimitError(
            float(response.headers.get("Retry-After", 1))
        )
    if response.status_code >= 400:
        raise WorldQuantApiError(path, response.status_code)
    return response.json()
```

Implement page iteration with `limit` and `offset`. Implement
`extract_scopes(configuration)` against the tested fixture under:

```text
actions.POST.settings.children
```

Read `instrumentType`, then region choices, then universe and delay choices for
each instrument/region pair, and return unique `Scope` values.

- [ ] **Step 5: Implement structured simulation**

`simulate_alpha(payload)` posts `/simulations`, follows `Location`, respects
`Retry-After`, stops at `simulation_poll_timeout_seconds`, fetches
`/alphas/{id}`, and returns:

```python
SimulationResult(
    worldquant_alpha_id=alpha_id,
    status="COMPLETED",
    metrics=alpha_data.get("is", {}),
    checks=alpha_data.get("is", {}).get("checks", []),
    raw_response=alpha_data,
)
```

For non-201 creation responses, map BRAIN response checks/messages to
`COMPILE_ERROR`, `DATASET_AUTHORIZATION_ERROR`, or `REQUEST_ERROR`.

- [ ] **Step 6: Run authentication and client tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_worldquant_client tests.test_interactive_login -v
```

Expected: all tests pass and existing interactive login behavior is unchanged.

- [ ] **Step 7: Commit**

```powershell
git add worldquant_client.py brain_batch_alpha.py tests/test_worldquant_client.py tests/test_interactive_login.py
git commit -m "feat: add structured WorldQuant API client"
```

### Task 4: Resumable Full-Account Metadata Synchronization

**Files:**
- Create: `metadata_sync.py`
- Create: `tests/test_metadata_sync.py`

- [ ] **Step 1: Write failing publish, failure, and resume tests**

```python
class MetadataSynchronizerTest(unittest.TestCase):
    def test_publishes_ready_snapshot_only_after_full_sync(self):
        client = FakeWorldQuantClient(
            scopes=[Scope("EQUITY", "USA", 1, "TOP3000")],
            datasets=[{"id": "ds1", "name": "Dataset", "description": "desc"}],
            fields=[{
                "id": "field1",
                "dataset": {"id": "ds1"},
                "description": "field",
                "type": "MATRIX",
            }],
            categories=[{"id": "cat1", "name": "Category"}],
            operators=[{"name": "rank", "definition": "rank(x)"}],
        )
        result = MetadataSynchronizer(client, metadata_dir).create_snapshot("My DB")

        self.assertEqual(result.status, "READY")
        self.assertTrue(result.path.exists())
        self.assertFalse(result.path.with_suffix(".sqlite.tmp").exists())

    def test_failed_sync_is_not_listed_as_ready(self):
        client = FakeWorldQuantClient(fail_on="data-fields")
        result = MetadataSynchronizer(client, metadata_dir).create_snapshot("Broken")

        self.assertEqual(result.status, "FAILED")
        self.assertEqual(MetadataStore.list_ready(metadata_dir), [])
```

Add a resume test where the first attempt records a completed dataset page,
then fails on fields. Reopen the temporary DB and assert the second attempt
does not request the completed page again.

Add a rate-limit test where the first page call raises
`WorldQuantRateLimitError(2)`, the synchronizer calls the injected sleep
function with `2`, and the second call succeeds.

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_metadata_sync -v
```

Expected: import error for `metadata_sync`.

- [ ] **Step 3: Implement synchronization state machine**

```python
class MetadataSynchronizer:
    def create_snapshot(self, label):
        snapshot_id = str(uuid.uuid4())
        final_path = self.metadata_dir / f"{snapshot_id}.sqlite"
        temp_path = self.metadata_dir / f"{snapshot_id}.sqlite.tmp"
        store = MetadataStore.create(temp_path, snapshot_id, label or default_label())
        try:
            configuration = self.client.get_configuration()
            for scope in self.client.extract_scopes(configuration):
                if self.stop_requested():
                    raise MetadataSyncStopped()
                scope_id = store.upsert_scope(scope)
                self._sync_datasets(store, scope, scope_id)
                self._sync_fields(store, scope, scope_id)
            self._sync_categories(store)
            self._sync_operators(store)
            store.complete_snapshot()
            store.close()
            temp_path.replace(final_path)
            return SnapshotResult(snapshot_id, final_path, "READY")
        except Exception as exc:
            store.fail_snapshot(str(exc))
            store.close()
            failed_path = self.metadata_dir / f"{snapshot_id}.failed.sqlite"
            temp_path.replace(failed_path)
            return SnapshotResult(snapshot_id, failed_path, "FAILED", str(exc))
```

Use `sync_events` as checkpoints keyed by endpoint, serialized scope, and
offset. Before requesting a page, skip it when a matching `COMPLETED` event
exists.

Wrap every page request with:

```python
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
```

Pass retry limits, backoff, and `sleep_func` into `MetadataSynchronizer` so the
tests do not wait in real time.

- [ ] **Step 4: Handle dataset references from field responses**

Before inserting a field, ensure its dataset exists. If `/data-fields` returns
a dataset not present in `/data-sets`, insert the embedded dataset stub with
its ID/name and raw JSON, then insert the field. This keeps foreign keys valid
without inventing dataset IDs.

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_metadata_sync tests.test_metadata_store tests.test_worldquant_client -v
git add metadata_sync.py tests/test_metadata_sync.py
git commit -m "feat: synchronize full WorldQuant metadata snapshots"
```

### Task 5: Phase 1 Verification

**Files:**
- Verify: all Phase 1 files

- [ ] **Step 1: Run focused suite**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_config tests.test_account_storage tests.test_metadata_store tests.test_worldquant_client tests.test_metadata_sync -v
```

Expected: `OK`.

- [ ] **Step 2: Run full project verification**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

Expected: all commands exit `0`.
