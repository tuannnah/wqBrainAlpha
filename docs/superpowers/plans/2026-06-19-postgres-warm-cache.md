# Postgres + Warm-Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chuyển backend lưu trữ từ SQLite sang PostgreSQL, di trú dữ liệu hiện có, và thêm lệnh `warm-cache` tải sẵn toàn bộ data WQB (resume được) vào DB.

**Architecture:** Ba phần độc lập, tuần tự. Phần 1 chỉ là cấu hình + driver (logic DB đã backend-agnostic). Phần 2 thêm `src/storage/migrate.py` copy bảng SQLite→Postgres bằng `merge` (idempotent). Phần 3 thêm `src/data/universe_matrix.py` (ma trận hằng) + `src/data/warm_cache.py` (bộ chạy resume tận dụng `FieldRepository.ensure`/`OperatorRepository.ensure` và retry 429 sẵn có ở `WQBrainClient`).

**Tech Stack:** Python, SQLAlchemy 2.0, Typer (CLI), rich (in bảng), loguru, pytest. DB: PostgreSQL qua `psycopg[binary]>=3`.

## Global Constraints

- TDD bắt buộc: viết test fail trước, rồi mới code (theo memory feedback_workflow).
- Code, comment, commit message, giao tiếp đều bằng **tiếng Việt** (giữ dấu đầy đủ).
- Mỗi task = một commit độc lập.
- Test KHÔNG gọi mạng thật: dùng `tests/fakes.FakeClient`/`FakeResponse` và SQLite in-memory (`sqlite:///:memory:`).
- Không sửa hành vi SQLite hiện có (không hồi quy test cũ).
- Không mở rộng `_migrate_add_columns` sang Postgres (YAGNI — Postgres bắt đầu sạch).
- Driver Postgres: dùng URL `postgresql+psycopg://user:pass@host:port/dbname`.

---

## File Structure

- `requirements.txt` — thêm `psycopg[binary]>=3`. (Task 1)
- `.env.example` — thêm dòng mẫu URL Postgres (comment). (Task 1)
- `src/storage/migrate.py` — **MỚI**: `migrate_all(source_engine, dest_engine)`. (Task 2)
- `main.py` — thêm lệnh Typer `migrate-sqlite` (Task 2) và `warm-cache` (Task 6).
- `src/data/universe_matrix.py` — **MỚI**: hằng `WQB_MATRIX` + `iter_scopes(...)`. (Task 3)
- `src/data/fields.py` — `FieldFetchError` mang `status_code`; thêm `mark_no_access(...)`. (Task 4)
- `src/data/warm_cache.py` — **MỚI**: `WarmCacheReport` + `warm_cache(...)`. (Task 5)
- `tests/test_db_postgres.py` — Task 1
- `tests/test_migrate.py` — Task 2
- `tests/test_universe_matrix.py` — Task 3
- `tests/test_fields_operators.py` — bổ sung cho Task 4
- `tests/test_warm_cache.py` — Task 5

---

## Task 1: Hỗ trợ backend Postgres

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example:25-26`
- Test: `tests/test_db_postgres.py`

**Interfaces:**
- Consumes: `src.storage.db.make_engine(database_url)` (đã có).
- Produces: không API mới; đảm bảo `make_engine` chấp nhận URL Postgres và trả engine backend `postgresql` không gắn pragma SQLite.

- [ ] **Step 1: Viết test fail**

Tạo `tests/test_db_postgres.py`:

```python
"""Test make_engine nhận URL Postgres mà không áp cấu hình riêng SQLite."""

from __future__ import annotations

from src.storage.db import make_engine


def test_make_engine_postgres_backend():
    # create_engine là lazy: không kết nối thật, chỉ phân giải dialect.
    engine = make_engine("postgresql+psycopg://u:p@localhost:5432/wq")
    assert engine.url.get_backend_name() == "postgresql"
    # Không set check_same_thread (đó là cờ riêng của SQLite).
    assert "check_same_thread" not in engine.url.query


def test_make_engine_sqlite_van_hoat_dong():
    engine = make_engine("sqlite:///:memory:")
    assert engine.url.get_backend_name() == "sqlite"
```

- [ ] **Step 2: Chạy test để xác nhận trạng thái**

Run: `python -m pytest tests/test_db_postgres.py -v`
Expected: `test_make_engine_sqlite_van_hoat_dong` PASS; `test_make_engine_postgres_backend` FAIL hoặc ERROR vì chưa cài driver `psycopg` (SQLAlchemy không phân giải được dialect `postgresql+psycopg`).

- [ ] **Step 3: Cài driver + cập nhật requirements/.env**

Thêm vào cuối phần dependencies trong `requirements.txt` một dòng:

```
psycopg[binary]>=3
```

Cài vào môi trường hiện tại:

Run: `pip install "psycopg[binary]>=3"`

Sửa `.env.example`, thay khối Database (dòng 25-26 hiện tại) thành:

```
# Database
# SQLite (mặc định, không cần cài thêm):
DATABASE_URL=sqlite:///wq_alpha.db
# PostgreSQL (đổi sang dòng dưới sau khi đã migrate; cần psycopg[binary]):
# DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/wq_alpha
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `python -m pytest tests/test_db_postgres.py -v`
Expected: cả hai test PASS.

- [ ] **Step 5: Không hồi quy bộ test cũ**

Run: `python -m pytest tests/test_storage.py tests/test_fields_operators.py -q`
Expected: tất cả PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example tests/test_db_postgres.py
git commit -m "feat(db): hỗ trợ backend PostgreSQL qua psycopg (driver + mẫu URL)"
```

---

## Task 2: Migrate dữ liệu SQLite → Postgres

**Files:**
- Create: `src/storage/migrate.py`
- Modify: `main.py` (thêm lệnh `migrate-sqlite`, import `make_engine` + `migrate_all`)
- Test: `tests/test_migrate.py`

**Interfaces:**
- Consumes: `src.storage.db.init_db`, `make_engine`; models trong `src.storage.models`.
- Produces:
  - `migrate_all(source_engine, dest_engine) -> dict[str, int]` — copy mọi bảng theo thứ tự FK, trả `{tên_bảng: số_rows}`. Idempotent (merge theo PK). Bỏ qua bảng không tồn tại ở nguồn (đếm 0).
  - Lệnh CLI `migrate-sqlite --source <url> --dest <url>`.

- [ ] **Step 1: Viết test fail**

Tạo `tests/test_migrate.py`:

```python
"""Test migrate_all copy đúng & idempotent giữa hai engine."""

from __future__ import annotations

from sqlalchemy import create_engine

from src.simulation.simulator import SimulationResult
from src.storage.db import init_db, make_session_factory
from src.storage.migrate import migrate_all
from src.storage.models import AlphaModel, OperatorModel, SimulationModel
from src.storage.repository import AlphaRepository


def _engine():
    return create_engine(
        "sqlite:///:memory:", future=True, connect_args={"check_same_thread": False}
    )


def _seed_source(engine):
    sf = make_session_factory(engine)
    repo = AlphaRepository(sf)
    aid = repo.save_alpha("rank(close)", source="llm", description="seed")
    repo.save_simulation(
        SimulationResult(
            expression="rank(close)", alpha_id=aid, status="passed",
            sharpe=1.5, fitness=1.0, turnover=0.2, raw={"is": {}},
        ),
        region="USA", universe="TOP3000",
    )
    s = sf()
    try:
        s.merge(OperatorModel(name="rank", definition="rank(x)", arity=1))
        s.commit()
    finally:
        s.close()


def test_migrate_all_copy_dung_so_rows():
    src = init_db(_engine())
    _seed_source(src)
    dst = _engine()  # chưa init: migrate_all tự init schema đích

    counts = migrate_all(src, dst)

    assert counts["alphas"] == 1
    assert counts["simulations"] == 1
    assert counts["operators"] == 1

    dsf = make_session_factory(dst)
    s = dsf()
    try:
        assert s.query(AlphaModel).count() == 1
        assert s.query(SimulationModel).count() == 1
        assert s.query(OperatorModel).count() == 1
    finally:
        s.close()


def test_migrate_all_idempotent():
    src = init_db(_engine())
    _seed_source(src)
    dst = _engine()

    migrate_all(src, dst)
    migrate_all(src, dst)  # chạy lần hai không nhân đôi

    dsf = make_session_factory(dst)
    s = dsf()
    try:
        assert s.query(AlphaModel).count() == 1
        assert s.query(SimulationModel).count() == 1
    finally:
        s.close()


def test_migrate_all_bo_qua_bang_thieu_o_nguon():
    # Nguồn chỉ có schema, không có bảng tùy biến nào ngoài bộ models -> vẫn chạy.
    src = init_db(_engine())
    dst = _engine()
    counts = migrate_all(src, dst)
    assert counts["alphas"] == 0  # rỗng nhưng không lỗi
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_migrate.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: src.storage.migrate`.

- [ ] **Step 3: Viết `src/storage/migrate.py`**

```python
"""Di trú dữ liệu giữa hai engine (vd SQLite -> PostgreSQL), idempotent."""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from src.storage.db import init_db
from src.storage.models import (
    AlphaModel,
    DataFieldModel,
    FailureModel,
    FetchStateModel,
    InvalidFieldModel,
    OperatorModel,
    SimulationModel,
    SubmissionModel,
)

# Thứ tự tôn trọng khóa ngoại: bảng được tham chiếu đứng trước.
MIGRATION_ORDER = [
    DataFieldModel,
    FetchStateModel,
    OperatorModel,
    InvalidFieldModel,
    AlphaModel,        # simulations/submissions tham chiếu alphas
    SimulationModel,
    FailureModel,
    SubmissionModel,
]


def migrate_all(source_engine: Engine, dest_engine: Engine) -> dict[str, int]:
    """Copy mọi bảng models từ source sang dest. Trả {tên_bảng: số_rows}.

    Dùng merge theo khóa chính -> chạy lại an toàn (không nhân đôi). Bảng không
    tồn tại ở nguồn (DB cũ thiếu) được bỏ qua với count 0.
    """
    init_db(dest_engine)
    src_tables = set(inspect(source_engine).get_table_names())
    SrcSession = sessionmaker(bind=source_engine, future=True)
    DstSession = sessionmaker(bind=dest_engine, future=True)

    counts: dict[str, int] = {}
    src = SrcSession()
    dst = DstSession()
    try:
        for model in MIGRATION_ORDER:
            table = model.__tablename__
            if table not in src_tables:
                counts[table] = 0
                continue
            rows = src.query(model).all()
            for row in rows:
                data = {c.name: getattr(row, c.name) for c in model.__table__.columns}
                dst.merge(model(**data))
            dst.commit()
            counts[table] = len(rows)
    finally:
        src.close()
        dst.close()
    return counts
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `python -m pytest tests/test_migrate.py -v`
Expected: 3 test PASS.

- [ ] **Step 5: Thêm lệnh `migrate-sqlite` vào `main.py`**

Trong `main.py`, thêm vào khối import storage (cạnh dòng `from src.storage.db import init_db, make_engine, make_session_factory`):

```python
from src.storage.migrate import migrate_all
```

Thêm command mới (đặt sau lệnh `login`):

```python
@app.command("migrate-sqlite")
def migrate_sqlite(
    source: str = typer.Option("sqlite:///wq_alpha.db", help="URL DB nguồn (SQLite)"),
    dest: str = typer.Option("", help="URL DB đích; rỗng = dùng DATABASE_URL"),
) -> None:
    """Copy toàn bộ dữ liệu từ SQLite sang DB đích (Postgres), idempotent."""
    _setup_logging()
    dest_url = dest or settings.database_url
    if dest_url == source:
        console.print("[red]❌ DB đích trùng DB nguồn — không có gì để migrate.[/red]")
        raise typer.Exit(code=1)
    counts = migrate_all(make_engine(source), make_engine(dest_url))
    table = Table(title="Đã migrate")
    table.add_column("Bảng")
    table.add_column("Số rows", justify="right")
    for name, n in counts.items():
        table.add_row(name, str(n))
    console.print(table)
    console.print(f"[green]OK[/green] {source} -> {dest_url}")
```

- [ ] **Step 6: Smoke test lệnh CLI (import được, không lỗi cú pháp)**

Run: `python -c "import main; print('migrate-sqlite' in [c.name for c in main.app.registered_commands])"`
Expected: in `True`.

- [ ] **Step 7: Commit**

```bash
git add src/storage/migrate.py main.py tests/test_migrate.py
git commit -m "feat(migrate): lệnh migrate-sqlite copy dữ liệu SQLite sang Postgres (idempotent)"
```

---

## Task 3: Ma trận tổ hợp `universe_matrix.py`

**Files:**
- Create: `src/data/universe_matrix.py`
- Test: `tests/test_universe_matrix.py`

**Interfaces:**
- Produces:
  - `WQB_MATRIX: dict[str, dict]` — `region -> {"universes": list[str], "delays": list[int]}`.
  - `iter_scopes(regions: list[str] | None = None, delays: list[int] | None = None) -> Iterator[tuple[str, str, int]]` — sinh `(region, universe, delay)`, lọc theo `regions`/`delays` nếu truyền.

- [ ] **Step 1: Viết test fail**

Tạo `tests/test_universe_matrix.py`:

```python
"""Test ma trận tổ hợp WQB và iter_scopes."""

from __future__ import annotations

from src.data.universe_matrix import WQB_MATRIX, iter_scopes


def test_iter_scopes_tat_ca_khong_rong():
    scopes = list(iter_scopes())
    assert len(scopes) > 0
    # Mỗi phần tử là bộ ba (region, universe, delay).
    region, universe, delay = scopes[0]
    assert isinstance(region, str) and isinstance(universe, str)
    assert isinstance(delay, int)


def test_iter_scopes_loc_theo_region():
    usa = list(iter_scopes(regions=["USA"]))
    assert usa, "USA phải có trong WQB_MATRIX"
    assert all(s[0] == "USA" for s in usa)
    # Bằng số universe * số delay của USA.
    cfg = WQB_MATRIX["USA"]
    assert len(usa) == len(cfg["universes"]) * len(cfg["delays"])


def test_iter_scopes_loc_theo_delay():
    d1 = list(iter_scopes(regions=["USA"], delays=[1]))
    assert d1
    assert all(s[2] == 1 for s in d1)
    assert len(d1) == len(WQB_MATRIX["USA"]["universes"])


def test_iter_scopes_region_khong_phan_biet_hoa_thuong():
    assert list(iter_scopes(regions=["usa"])) == list(iter_scopes(regions=["USA"]))
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_universe_matrix.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: src.data.universe_matrix`.

- [ ] **Step 3: Viết `src/data/universe_matrix.py`**

```python
"""Ma trận region/universe/delay đã biết của WQB + sinh tổ hợp để warm-cache.

Bảng hằng là nguồn sự thật để DUYỆT; tổ hợp tài khoản không có quyền sẽ được
phát hiện qua probe lúc fetch (warm_cache đánh dấu no_access). Vì vậy bảng không
cần khớp tuyệt đối với quyền tài khoản — chỉ cần bao phủ rộng. Bổ sung/sửa khi
WQB thay đổi danh mục universe.
"""

from __future__ import annotations

from typing import Iterator

# region -> universes hay dùng + các delay khả dụng.
WQB_MATRIX: dict[str, dict] = {
    "USA": {"universes": ["TOP3000", "TOP1000", "TOP500", "TOP200"], "delays": [0, 1]},
    "EUR": {"universes": ["TOP2500", "TOP1200", "TOP800", "TOP400"], "delays": [0, 1]},
    "GLB": {"universes": ["TOP3000", "MINVOL1M"], "delays": [1]},
    "ASI": {"universes": ["MINVOL1M", "ILLIQUID_MINVOL1M"], "delays": [1]},
    "CHN": {"universes": ["TOP2000U"], "delays": [0, 1]},
    "JPN": {"universes": ["TOP1600", "TOP1200", "TOP800"], "delays": [0, 1]},
    "KOR": {"universes": ["TOP600"], "delays": [0, 1]},
    "TWN": {"universes": ["TOP500"], "delays": [0, 1]},
    "HKG": {"universes": ["TOP800", "TOP500"], "delays": [0, 1]},
    "AMR": {"universes": ["TOP600"], "delays": [1]},
}


def iter_scopes(
    regions: list[str] | None = None,
    delays: list[int] | None = None,
) -> Iterator[tuple[str, str, int]]:
    """Sinh (region, universe, delay) cho mọi tổ hợp trong WQB_MATRIX.

    regions: lọc theo danh sách region (không phân biệt hoa/thường); None = tất cả.
    delays:  lọc theo danh sách delay; None = tất cả delay của từng region.
    """
    region_filter = {r.upper() for r in regions} if regions else None
    delay_filter = set(delays) if delays is not None else None
    for region, cfg in WQB_MATRIX.items():
        if region_filter is not None and region not in region_filter:
            continue
        for universe in cfg["universes"]:
            for delay in cfg["delays"]:
                if delay_filter is not None and delay not in delay_filter:
                    continue
                yield (region, universe, delay)
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `python -m pytest tests/test_universe_matrix.py -v`
Expected: 4 test PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/universe_matrix.py tests/test_universe_matrix.py
git commit -m "feat(data): ma trận WQB_MATRIX + iter_scopes sinh tổ hợp region/universe/delay"
```

---

## Task 4: `FieldFetchError.status_code` + `FieldRepository.mark_no_access`

**Files:**
- Modify: `src/data/fields.py:21-23` (class `FieldFetchError`), `:181-190` (raise có status_code), thêm method `mark_no_access`.
- Test: `tests/test_fields_operators.py` (thêm test mới ở cuối file)

**Interfaces:**
- Consumes: `FetchStateModel`, `_now`, `_key` (đã có trong `fields.py`).
- Produces:
  - `FieldFetchError(message, status_code=None)` với thuộc tính `.status_code`.
  - `FieldRepository.mark_no_access(region, universe, delay) -> None` — ghi `FetchStateModel.status="no_access"`.

- [ ] **Step 1: Viết test fail**

Thêm vào cuối `tests/test_fields_operators.py`:

```python
def test_field_fetch_error_mang_status_code():
    from src.data.fields import FieldFetchError

    err = FieldFetchError("không có quyền", status_code=403)
    assert err.status_code == 403
    assert FieldFetchError("x").status_code is None


def test_fetch_403_raise_kem_status_code():
    from src.data.fields import FieldFetchError

    engine = init_db(_engine())
    sf = make_session_factory(engine)
    client = FakeClient()
    client.queue_get(FakeResponse(403, text="forbidden"))
    repo = FieldRepository(client, sf)
    try:
        repo.fetch_all("USA", "TOP3000", 1)
        assert False, "phải raise"
    except FieldFetchError as exc:
        assert exc.status_code == 403


def test_mark_no_access_ghi_trang_thai():
    from src.storage.models import FetchStateModel

    engine = init_db(_engine())
    sf = make_session_factory(engine)
    repo = FieldRepository(None, sf)
    repo.mark_no_access("EUR", "TOP400", 0)
    state = repo.get_state("EUR", "TOP400", 0)
    assert state is not None
    assert state.status == "no_access"
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_fields_operators.py -k "status_code or no_access or 403" -v`
Expected: FAIL — `FieldFetchError` chưa nhận `status_code`; `mark_no_access` chưa tồn tại.

- [ ] **Step 3: Sửa `src/data/fields.py`**

Thay class `FieldFetchError` (dòng 21-23) thành:

```python
class FieldFetchError(RuntimeError):
    """Lỗi khi tải data-fields từ WorldQuant.

    status_code: mã HTTP nếu lỗi đến từ phản hồi server (401/403/429/4xx/5xx);
    None nếu lỗi không gắn với HTTP.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
```

Trong `_fetch_all_pages`, sửa hai chỗ `raise FieldFetchError(...)` (khối `if resp.status_code >= 400:`) để truyền `status_code`:

```python
            if resp.status_code >= 400:
                logger.error("GET /data-fields lỗi {}: {}", resp.status_code, resp.text[:500])
                if resp.status_code == 429:
                    raise FieldFetchError(
                        "Bị giới hạn tần suất (429) sau nhiều lần thử. Hãy chờ vài phút rồi tải lại.",
                        status_code=429,
                    )
                raise FieldFetchError(
                    f"Không tải được data-fields (HTTP {resp.status_code}). "
                    "Kiểm tra region/universe/delay hợp lệ và tài khoản có quyền.",
                    status_code=resp.status_code,
                )
```

Thêm method `mark_no_access` vào `FieldRepository` (đặt cạnh `_update_state`, trong nhóm fetch/store):

```python
    def mark_no_access(self, region: str, universe: str, delay: int) -> None:
        """Đánh dấu scope tài khoản không truy cập được (resume sẽ bỏ qua nhanh)."""
        session: Session = self.session_factory()
        try:
            key = self._key(region, universe, delay)
            state = session.get(FetchStateModel, key) or FetchStateModel(key=key)
            state.entity = "data_fields"
            state.region, state.universe, state.delay = region, universe, delay
            state.fetched_at = _now()
            state.status = "no_access"
            session.merge(state)
            session.commit()
        finally:
            session.close()
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `python -m pytest tests/test_fields_operators.py -v`
Expected: tất cả PASS (gồm 3 test mới + các test cũ không hồi quy).

- [ ] **Step 5: Commit**

```bash
git add src/data/fields.py tests/test_fields_operators.py
git commit -m "feat(fields): FieldFetchError mang status_code + mark_no_access cho warm-cache"
```

---

## Task 5: Bộ chạy `warm_cache`

**Files:**
- Create: `src/data/warm_cache.py`
- Test: `tests/test_warm_cache.py`

**Interfaces:**
- Consumes:
  - `FieldRepository.ensure(region, universe, delay, force=False) -> (list[DataField], bool)`.
  - `FieldRepository.get_state(region, universe, delay) -> FetchStateModel | None`.
  - `FieldRepository.mark_no_access(region, universe, delay)`.
  - `OperatorRepository.ensure(force=False) -> (list[Operator], bool)`.
  - `FieldFetchError` (có `.status_code`).
- Produces:
  - `WarmCacheReport` dataclass: `fetched: int`, `skipped: int`, `no_access: int`, `errors: list[tuple]`, `operators: int`.
  - `warm_cache(field_repo, operator_repo, scopes, *, force=False, sleep_s=2.0, sleep_func=None, on_event=None) -> WarmCacheReport`.

- [ ] **Step 1: Viết test fail**

Tạo `tests/test_warm_cache.py`:

```python
"""Test warm_cache: resume, probe no_access, gom lỗi."""

from __future__ import annotations

from sqlalchemy import create_engine

from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository
from src.data.warm_cache import WarmCacheReport, warm_cache
from src.storage.db import init_db, make_session_factory
from tests.fakes import FakeClient, FakeResponse


def _engine():
    return create_engine(
        "sqlite:///:memory:", future=True, connect_args={"check_same_thread": False}
    )


def _seed_operators(sf):
    oc = FakeClient()
    oc.queue_get(
        FakeResponse(200, json_data={"results": [{"name": "rank", "definition": "rank(x)"}]})
    )
    OperatorRepository(oc, sf).fetch_all()


def _noop_sleep(_s):
    pass


def test_warm_cache_fetch_moi():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)

    fc = FakeClient()
    fc.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "close"}]}))
    field_repo = FieldRepository(fc, sf)
    op_repo = OperatorRepository(FakeClient(), sf)  # operators đã cache -> không gọi API

    report = warm_cache(
        field_repo, op_repo, [("USA", "TOP3000", 1)], sleep_func=_noop_sleep
    )
    assert isinstance(report, WarmCacheReport)
    assert report.fetched == 1
    assert report.skipped == 0
    assert report.operators == 1


def test_warm_cache_resume_bo_qua_scope_da_complete():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)
    # Seed scope đã complete.
    seed = FakeClient()
    seed.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "close"}]}))
    FieldRepository(seed, sf).fetch_all("USA", "TOP3000", 1, page_size=10)

    # Client rỗng: nếu gọi API sẽ IndexError -> chứng minh không gọi.
    field_repo = FieldRepository(FakeClient(), sf)
    op_repo = OperatorRepository(FakeClient(), sf)
    report = warm_cache(
        field_repo, op_repo, [("USA", "TOP3000", 1)], sleep_func=_noop_sleep
    )
    assert report.skipped == 1
    assert report.fetched == 0


def test_warm_cache_empty_danh_dau_no_access():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)

    fc = FakeClient()
    fc.queue_get(FakeResponse(200, json_data={"count": 0, "results": []}))
    field_repo = FieldRepository(fc, sf)
    op_repo = OperatorRepository(FakeClient(), sf)
    report = warm_cache(
        field_repo, op_repo, [("ASI", "MINVOL1M", 1)], sleep_func=_noop_sleep
    )
    assert report.no_access == 1
    assert report.fetched == 0
    assert field_repo.get_state("ASI", "MINVOL1M", 1).status == "no_access"


def test_warm_cache_403_danh_dau_no_access():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)

    fc = FakeClient()
    fc.queue_get(FakeResponse(403, text="forbidden"))
    field_repo = FieldRepository(fc, sf)
    op_repo = OperatorRepository(FakeClient(), sf)
    report = warm_cache(
        field_repo, op_repo, [("CHN", "TOP2000U", 0)], sleep_func=_noop_sleep
    )
    assert report.no_access == 1
    assert field_repo.get_state("CHN", "TOP2000U", 0).status == "no_access"


def test_warm_cache_resume_bo_qua_no_access():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)
    field_repo = FieldRepository(FakeClient(), sf)  # client rỗng
    field_repo.mark_no_access("EUR", "TOP400", 0)
    op_repo = OperatorRepository(FakeClient(), sf)

    report = warm_cache(
        field_repo, op_repo, [("EUR", "TOP400", 0)], sleep_func=_noop_sleep
    )
    assert report.no_access == 1
    assert report.fetched == 0


def test_warm_cache_loi_http_khac_gom_vao_errors():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)
    fc = FakeClient()
    fc.queue_get(FakeResponse(500, text="server error"))
    field_repo = FieldRepository(fc, sf)
    op_repo = OperatorRepository(FakeClient(), sf)
    report = warm_cache(
        field_repo, op_repo, [("USA", "TOP3000", 1)], sleep_func=_noop_sleep
    )
    assert len(report.errors) == 1
    assert report.errors[0][0] == ("USA", "TOP3000", 1)
    assert report.no_access == 0
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_warm_cache.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: src.data.warm_cache`.

- [ ] **Step 3: Viết `src/data/warm_cache.py`**

```python
"""Tải sẵn (warm) toàn bộ datafields + operators cho nhiều scope, resume được.

Tận dụng FieldRepository/OperatorRepository (đã có cache + TTL) và retry 429 sẵn
ở WQBrainClient. Mỗi scope tự lưu trạng thái nên chạy lại chỉ làm phần còn thiếu.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from loguru import logger

from src.data.fields import FieldFetchError, FieldRepository
from src.data.operators import OperatorRepository

Scope = tuple[str, str, int]


@dataclass
class WarmCacheReport:
    fetched: int = 0          # số scope fetch mới thành công (có field)
    skipped: int = 0          # số scope đã cache còn hạn -> bỏ qua
    no_access: int = 0        # số scope không quyền/empty -> đánh dấu no_access
    errors: list = field(default_factory=list)  # list[tuple[Scope, str]]
    operators: int = 0        # số operator đã đảm bảo trong cache


def warm_cache(
    field_repo: FieldRepository,
    operator_repo: OperatorRepository,
    scopes: Iterable[Scope],
    *,
    force: bool = False,
    sleep_s: float = 2.0,
    sleep_func: Callable[[float], None] | None = None,
    on_event: Callable[[str, Scope], None] | None = None,
) -> WarmCacheReport:
    """Duyệt scopes, fetch field còn thiếu, đánh dấu no_access cho scope không quyền.

    force=True: bỏ qua cache, tải lại tất cả (kể cả scope đã no_access).
    sleep_s: nghỉ giữa các scope CÓ gọi API (giảm rủi ro 429).
    on_event(kind, scope): callback tiến độ; kind in
        {"fetched","skip_cached","skip_no_access","no_access","error"}.
    """
    sleep_func = sleep_func or _time.sleep
    report = WarmCacheReport()

    ops, _ = operator_repo.ensure(force=force)
    report.operators = len(ops)

    def _emit(kind: str, scope: Scope) -> None:
        if on_event is not None:
            on_event(kind, scope)

    for scope in scopes:
        region, universe, delay = scope

        if not force:
            state = field_repo.get_state(region, universe, delay)
            if state is not None and state.status == "no_access":
                report.no_access += 1
                _emit("skip_no_access", scope)
                continue

        try:
            fields, fetched = field_repo.ensure(region, universe, delay, force=force)
        except FieldFetchError as exc:
            if getattr(exc, "status_code", None) in (401, 403):
                field_repo.mark_no_access(region, universe, delay)
                report.no_access += 1
                _emit("no_access", scope)
            else:
                report.errors.append((scope, str(exc)))
                _emit("error", scope)
                logger.warning("warm-cache lỗi {}: {}", scope, exc)
            continue

        if not fetched:
            report.skipped += 1
            _emit("skip_cached", scope)
            continue

        if not fields:
            field_repo.mark_no_access(region, universe, delay)
            report.no_access += 1
            _emit("no_access", scope)
        else:
            report.fetched += 1
            _emit("fetched", scope)

        sleep_func(sleep_s)

    return report
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `python -m pytest tests/test_warm_cache.py -v`
Expected: 6 test PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/warm_cache.py tests/test_warm_cache.py
git commit -m "feat(warm-cache): bộ chạy tải sẵn datafields/operators có resume + probe no_access"
```

---

## Task 6: Lệnh CLI `warm-cache`

**Files:**
- Modify: `main.py` (thêm command `warm-cache` + import)
- Test: smoke test qua `python -c`

**Interfaces:**
- Consumes: `iter_scopes` (Task 3), `warm_cache`/`WarmCacheReport` (Task 5), `FieldRepository`, `OperatorRepository`, `_make_client`, `init_db`, `make_session_factory`.
- Produces: lệnh `warm-cache --regions --delays --force --sleep`.

- [ ] **Step 1: Thêm import vào `main.py`**

Cạnh các import data (sau `from src.data.operators import ...`):

```python
from src.data.universe_matrix import iter_scopes
from src.data.warm_cache import warm_cache
```

- [ ] **Step 2: Thêm command `warm-cache` vào `main.py`**

Đặt sau lệnh `probe-fields`:

```python
@app.command("warm-cache")
def warm_cache_cmd(
    regions: str = typer.Option("", help="CSV region cần tải; rỗng = tất cả trong WQB_MATRIX"),
    delays: str = typer.Option("0,1", help="CSV delay cần tải"),
    force: bool = typer.Option(False, help="Tải lại tất cả, bỏ qua cache"),
    sleep: float = typer.Option(2.0, help="Giây nghỉ giữa các scope có gọi API"),
) -> None:
    """Tải sẵn toàn bộ datafields + operators vào DB (resume được)."""
    _setup_logging()
    region_list = [r.strip() for r in regions.split(",") if r.strip()] or None
    delay_list = [int(d.strip()) for d in delays.split(",") if d.strip()]

    engine = init_db(make_engine())
    sf = make_session_factory(engine)
    client = _make_client()
    client.authenticate()
    field_repo = FieldRepository(client, sf)
    op_repo = OperatorRepository(client, sf)

    scopes = list(iter_scopes(regions=region_list, delays=delay_list))
    console.print(f"[cyan]Bắt đầu warm-cache {len(scopes)} tổ hợp...[/cyan]")

    def _on_event(kind: str, scope) -> None:
        console.print(f"  [{kind}] {scope[0]}/{scope[1]}/delay={scope[2]}")

    report = warm_cache(
        field_repo, op_repo, scopes, force=force, sleep_s=sleep, on_event=_on_event
    )

    table = Table(title="Kết quả warm-cache")
    table.add_column("Hạng mục")
    table.add_column("Số lượng", justify="right")
    table.add_row("Operators", str(report.operators))
    table.add_row("Fetch mới", str(report.fetched))
    table.add_row("Bỏ qua (đã cache)", str(report.skipped))
    table.add_row("Không quyền", str(report.no_access))
    table.add_row("Lỗi", str(len(report.errors)))
    console.print(table)
    for scope, msg in report.errors:
        console.print(f"[red]  lỗi {scope}: {msg}[/red]")
```

- [ ] **Step 3: Smoke test lệnh đăng ký được**

Run: `python -c "import main; print('warm-cache' in [c.name for c in main.app.registered_commands])"`
Expected: in `True`.

- [ ] **Step 4: Kiểm tra toàn bộ test xanh**

Run: `python -m pytest -q`
Expected: tất cả PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(cli): lệnh warm-cache tải sẵn toàn bộ data WQB vào DB (resume)"
```

---

## Self-Review

**Spec coverage:**
- Phần 1 (backend Postgres) → Task 1. ✓
- Phần 2 (migrate SQLite→Postgres, idempotent, thứ tự FK) → Task 2. ✓
- Phần 3 ma trận hằng + probe → Task 3 (`WQB_MATRIX`/`iter_scopes`); resume + no_access → Task 4 (`mark_no_access`, `status_code`) + Task 5 (`warm_cache`); CLI → Task 6. ✓
- Tận dụng retry 429 sẵn có → không lặp lại logic, dùng `client` hiện tại. ✓
- Test không gọi mạng (FakeClient) → mọi task. ✓

**Type consistency:**
- `FieldRepository.ensure(region, universe, delay, force=...)` dùng nhất quán ở Task 5/6.
- `OperatorRepository.ensure(force=...)` → `(list, bool)` nhất quán.
- `FieldFetchError(status_code=...)` định nghĩa Task 4, dùng Task 5.
- `WarmCacheReport` field names (`fetched/skipped/no_access/errors/operators`) khớp giữa Task 5 (định nghĩa) và Task 6 (đọc).
- `iter_scopes(regions=, delays=)` khớp Task 3 ↔ Task 6.
- `migrate_all(source_engine, dest_engine)` khớp Task 2 ↔ lệnh CLI.

**Placeholder scan:** không có TBD/TODO; mọi step có code/command cụ thể.
