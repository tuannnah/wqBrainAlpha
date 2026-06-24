# Phase 5 — Database Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (khuyến nghị) hoặc superpowers:executing-plans để thực thi từng task. Task 5.1→5.2→5.3
> phụ thuộc chuỗi (models → migration idempotent → repository dùng models đó); Task 5.4
> (`ResultCache`) phụ thuộc 5.3 (`result_cache_get/put`); Task 5.5 (dead-field wiring) phụ
> thuộc 5.3 (`add_dead_field`/`is_dead_field`) nhưng độc lập với 5.4 — có thể làm song song
> với 5.4 nếu có 2 agent. Task 5.6 là review + merge + push cuối cùng, luôn chạy sau hết.

**Goal:** Mở rộng tầng storage hiện có (KHÔNG phá DB cũ — `init_db` idempotent qua
`create_all` + `_migrate_add_columns`) để MiniBrain có nơi lưu **mọi** outcome của
expression search: `ExpressionModel` (de-dup theo canonical hash), `EvaluationModel` (một
lần chạy backtest dưới một `PortfolioConfig` + cửa sổ data — lưu cả pass và fail, kèm seed
để tái lập R8), `PoolPnlModel` (PnL vector của alpha pass, phục vụ Phase 6 self-corr),
`DeadFieldModel` (field bị Brain từ chối — blacklist tự học), `BrainRecordModel` (ground
truth Brain để Phase 4.5 calibration đọc). Thêm `AlphaRepository`-style methods mới (không
sửa method cũ) và `ResultCache` (DB-backed `canonical_hash+config+window -> AlphaMetrics`)
để re-score một expression đã biết là miễn phí (B12 tier 3).

**Architecture:** Mở rộng `src/storage/models.py` (thêm class mới, append cuối file, không
sửa model cũ) bằng cùng pattern SQLAlchemy ORM hiện có (`declarative_base`, `Column`,
`_utcnow`). Mở rộng `src/storage/repository.py` bằng một class mới
`MiniBrainRepository` (không sửa `AlphaRepository`/`InvalidFieldRepository` cũ — tránh xung
đột, hai repo phục vụ hai luồng dữ liệu khác nhau: luồng Brain-sim cũ vs luồng MiniBrain
local mới) theo đúng session pattern hiện có (`self.session_factory()` → try/finally
`session.close()`). `src/cache/result_cache.py` là module **mới** (package `src/cache/`
chưa tồn tại — tạo `__init__.py`), bọc `MiniBrainRepository.result_cache_get/put`, không tự
mở session riêng (tái dùng repo được inject). Theo dependency rule (master plan B1):
`src/storage` và `src/cache` KHÔNG import `src/gp`/`src/llm`; `src/storage/models.py`
KHÔNG import `src/lang`/`src/backtest` (lưu dữ liệu đã serialize — JSON string/blob — không
lưu object Python sống, giữ migrate.py port được Postgres).

**Tech Stack:** Python 3.12, SQLAlchemy ORM (đã có), numpy (pack/unpack blob), pytest, ruff,
mypy --strict. Không thêm dependency mới.

## Global Constraints

- Python 3.12; cú pháp hiện đại (`match`, `X | None`, `type` alias, `@dataclass(frozen=True, slots=True)`, `Protocol`).
- Full type hints; `mypy --strict` clean; `ruff` clean; không unused import.
- **No look-ahead:** time-series ops chỉ đọc rows ≤ t; thiếu lịch sử → NaN.
- **No survivorship:** universe mask per-day; out-of-universe = NaN (không phải 0).
- **Delay-1:** `pnl_t = nansum(weights_{t-1} * returns_t)`.
- **Stage separation:** expression = signal core; neut/decay/trunc/scale/delay ở `PortfolioConfig`.
- **Thresholds chỉ ở `config/thresholds.py`** — không hardcode gate number ở call site.
- **Determinism:** randomness qua seed inject; ghi seed vào DB.
- **WQ operator fidelity:** tra skill `worldquant-brain` trước khi viết FASTEXPR/operator.
- **TDD:** test trước, đỏ → code tối thiểu → xanh → commit. Mỗi phase = 1 nhánh git → merge → push.
- **Per-phase ritual:** Design → Implement → Explain → Review (test+ruff+mypy) → Gate → Journal (`PROGRESS.md`).

## Pre-condition (đọc trước khi bắt đầu)

Phase 5 mở rộng storage hiện có — **đọc thật** các file sau trước khi sửa, đã đọc để viết
plan này:

- `src/storage/models.py` (133 dòng) — `Base = declarative_base()`, các model hiện có
  (`DataFieldModel`, `FetchStateModel`, `OperatorModel`, `AlphaModel`, `SimulationModel`,
  `FailureModel`, `InvalidFieldModel`, `SubmissionModel`). Model mới **append cuối file**,
  cùng style (`Column`, `_utcnow` cho `created_at`, không dùng `Mapped`/`mapped_column`).
- `src/storage/db.py` (112 dòng) — `init_db(engine)` gọi `Base.metadata.create_all(engine)`
  rồi `_migrate_add_columns(engine)` (chỉ ADD COLUMN cho bảng đã tồn tại, bảng mới thì
  `create_all` tự tạo đủ cột — **không cần sửa `db.py`/`migrate.py` để bảng mới được tạo**,
  chỉ cần model đăng ký vào `Base`). `make_session_factory(engine)` trả
  `sessionmaker[Session]` — dùng để inject vào repository mới giống `AlphaRepository`.
- `src/storage/repository.py` (209 dòng) — pattern: mỗi method tự mở `session =
  self.session_factory()`, `try/finally: session.close()`, `session.commit()` sau
  `session.add`/`session.merge`. `expr_hash(expression, config)` dùng sha256 hex — Phase 5
  **không** trùng tên hàm này; `MiniBrainRepository` dùng `canonical_hash` (đã có sẵn từ
  `CanonicalHasher`, Phase 1) làm khóa thay vì tính lại hash riêng.
- `src/storage/migrate.py` (99 dòng) — `MIGRATION_ORDER` liệt kê model theo thứ tự FK cho
  `migrate_all` (copy SQLite→Postgres). Model Phase 5 **phải** được thêm vào danh sách này
  theo đúng thứ tự FK (`ExpressionModel` trước `EvaluationModel` trước `PoolPnlModel`;
  `DeadFieldModel`/`BrainRecordModel` không FK, thêm ở cuối) — nếu bỏ qua, `migrate_all` im
  lặng không copy dữ liệu Phase 5 khi chuyển sang Postgres (lỗi thầm lặng, phải tránh).
- `src/lang/visitors.py` — `DepthVisitor`, `FieldCollector`, `CanonicalHasher`,
  `ComplexityVisitor` đều implement `NodeVisitor[T]`, gọi qua `node.accept(VisitorClass())`
  (KHÔNG có `__call__`). `upsert_expression` test dùng `parse()` + các visitor này thật, vì
  chúng đã tồn tại từ Phase 1 (đã merge vào `main`).
- `src/backtest/metrics_local.py` — `AlphaMetrics` (frozen,slots): `sharpe: float,
  annual_return: float, turnover: float, max_drawdown: float, fitness: float,
  per_year_sharpe: dict[int, float], weight_concentration: float`.
- `src/backtest/config.py` — `PortfolioConfig` (frozen,slots,hashable):
  `neutralization: Neutralization, decay: int, truncation: float, scale_book: float,
  delay: int`. `Neutralization` là `Enum`.

Trước khi code Task 5.1, xác nhận trạng thái nhánh:

```bash
venv/Scripts/python.exe -c "from src.backtest.metrics_local import AlphaMetrics; from src.backtest.config import PortfolioConfig; from src.lang.visitors import CanonicalHasher, DepthVisitor, FieldCollector, ComplexityVisitor; from src.lang.parser import parse; print('ok')"
```

Nếu `ModuleNotFoundError` → Phase 1–4 chưa merge vào `main`. Task 5.1–5.2 (models +
migration list) không phụ thuộc các module này nên vẫn làm được; Task 5.3+ (repository
dùng `AlphaMetrics`/`parse`/visitors trong test) bị khoá — dừng, báo cáo, đợi merge.

---

### Task 5.1: Models mới (`src/storage/models.py`)

**Files:**
- Modify: `src/storage/models.py`
- Test: `tests/unit/test_storage_models_minibrain.py`

**Interfaces:**
- Consumes: `Base` (đã có, `declarative_base()`), `_utcnow` (đã có).
- Produces (append cuối `src/storage/models.py`):
  - `class ExpressionModel(Base)` — `__tablename__ = "expressions"`. Cột: `id: Integer
    primary_key autoincrement`, `canonical_hash: String unique not null indexed`,
    `expr_string: Text not null`, `depth: Integer not null`, `complexity: Integer not
    null`, `fields_json: Text not null`, `created_at: DateTime default=_utcnow`.
  - `class EvaluationModel(Base)` — `__tablename__ = "evaluations"`. Cột: `id: Integer
    primary_key autoincrement`, `expression_id: Integer ForeignKey("expressions.id") not
    null indexed`, `config_json: Text not null`, `data_window: String not null`, `sharpe:
    Float`, `annual_return: Float`, `turnover: Float`, `max_drawdown: Float`, `fitness:
    Float`, `weight_concentration: Float`, `per_year_json: Text`, `self_corr_max: Float`,
    `status: String not null indexed` (`'passed'|'failed_gate'|'invalid'|'error'`),
    `fail_reasons: Text` (nullable — JSON list, rỗng khi pass), `seed: Integer` (nullable),
    `created_at: DateTime default=_utcnow`. `UniqueConstraint("expression_id",
    "config_json", "data_window", name="uq_evaluation_expr_config_window")`.
  - `class PoolPnlModel(Base)` — `__tablename__ = "pool_pnl"`. Cột: `evaluation_id: Integer
    primary_key ForeignKey("evaluations.id")`, `dates_blob: LargeBinary not null`,
    `pnl_blob: LargeBinary not null`.
  - `class DeadFieldModel(Base)` — `__tablename__ = "dead_fields_minibrain"` (đặt tên khác
    `invalid_fields` hiện có — đây là field GP đề xuất bị Brain reject, ngữ nghĩa gần giống
    `InvalidFieldModel` nhưng thuộc luồng MiniBrain riêng theo B11; **không tái dùng**
    `InvalidFieldModel` để tránh hai luồng ghi tranh khóa chính khác cấu trúc — pattern
    `field_id` (3 cột khóa kép) khác `name` (1 cột) của B11). Cột: `name: String primary_key`,
    `reason: Text`, `created_at: DateTime default=_utcnow`.
  - `class BrainRecordModel(Base)` — `__tablename__ = "brain_records"`. Cột: `id: Integer
    primary_key autoincrement`, `expr_string: Text not null`, `brain_sharpe: Float`,
    `brain_fitness: Float`, `brain_turnover: Float`, `brain_self_corr: Float`,
    `submitted: Integer` (0/1, SQLite không có bool riêng — dùng `Integer` đồng bộ style
    hiện có không có cột Boolean nào trong file), `created_at: DateTime default=_utcnow`.
  - Index tường minh (đúng B11): `Index("idx_eval_expr", EvaluationModel.expression_id)`,
    `Index("idx_eval_status", EvaluationModel.status)`, `Index("idx_eval_sharpe",
    EvaluationModel.sharpe)` — khai báo qua `Column(..., index=True)` trên
    `expression_id`/`status` (SQLAlchemy tự tạo index cùng tên ngắn) và thêm
    `sqlalchemy.Index` rời cho `sharpe` (không đặt `index=True` trực tiếp trên cột numeric
    đang dùng cho range query để tên index khớp B11 `idx_eval_sharpe`).

- [ ] **Step 1: Tạo nhánh từ main sạch**

```bash
git checkout main && git pull --ff-only 2>/dev/null; git checkout -b phase-5-database
git status
```
Expected: "On branch phase-5-database", working tree clean.

- [ ] **Step 2: Viết test đỏ**

```python
# tests/unit/test_storage_models_minibrain.py
"""Test models MiniBrain mới: bảng tạo đúng cột, FK, unique constraint (B11 schema)."""

from __future__ import annotations

from sqlalchemy import create_engine, inspect

from src.storage.db import init_db
from src.storage.models import (
    BrainRecordModel,
    DeadFieldModel,
    EvaluationModel,
    ExpressionModel,
    PoolPnlModel,
)


def _fresh_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    return engine


def test_all_minibrain_tables_created():
    engine = _fresh_engine()
    tables = set(inspect(engine).get_table_names())
    assert {"expressions", "evaluations", "pool_pnl", "dead_fields_minibrain",
            "brain_records"} <= tables


def test_expression_table_columns():
    engine = _fresh_engine()
    cols = {c["name"] for c in inspect(engine).get_columns("expressions")}
    assert cols == {"id", "canonical_hash", "expr_string", "depth", "complexity",
                     "fields_json", "created_at"}


def test_evaluation_table_columns_and_fk():
    engine = _fresh_engine()
    cols = {c["name"] for c in inspect(engine).get_columns("evaluations")}
    assert cols == {
        "id", "expression_id", "config_json", "data_window", "sharpe", "annual_return",
        "turnover", "max_drawdown", "fitness", "weight_concentration", "per_year_json",
        "self_corr_max", "status", "fail_reasons", "seed", "created_at",
    }
    fks = inspect(engine).get_foreign_keys("evaluations")
    assert any(fk["referred_table"] == "expressions" for fk in fks)


def test_evaluation_unique_constraint_blocks_duplicate():
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    engine = _fresh_engine()
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    expr = ExpressionModel(
        canonical_hash="h1", expr_string="close", depth=1, complexity=1, fields_json="[]",
    )
    session.add(expr)
    session.commit()
    session.add(EvaluationModel(
        expression_id=expr.id, config_json="{}", data_window="2020..2021", status="passed",
    ))
    session.commit()
    session.add(EvaluationModel(
        expression_id=expr.id, config_json="{}", data_window="2020..2021", status="passed",
    ))
    try:
        session.commit()
        raised = False
    except IntegrityError:
        session.rollback()
        raised = True
    finally:
        session.close()
    assert raised


def test_pool_pnl_fk_to_evaluation():
    engine = _fresh_engine()
    cols = {c["name"] for c in inspect(engine).get_columns("pool_pnl")}
    assert cols == {"evaluation_id", "dates_blob", "pnl_blob"}
    fks = inspect(engine).get_foreign_keys("pool_pnl")
    assert any(fk["referred_table"] == "evaluations" for fk in fks)


def test_dead_field_and_brain_record_columns():
    engine = _fresh_engine()
    dead_cols = {c["name"] for c in inspect(engine).get_columns("dead_fields_minibrain")}
    assert dead_cols == {"name", "reason", "created_at"}
    brain_cols = {c["name"] for c in inspect(engine).get_columns("brain_records")}
    assert brain_cols == {
        "id", "expr_string", "brain_sharpe", "brain_fitness", "brain_turnover",
        "brain_self_corr", "submitted", "created_at",
    }


def test_canonical_hash_is_unique():
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    engine = _fresh_engine()
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    session.add(ExpressionModel(
        canonical_hash="dup", expr_string="close", depth=1, complexity=1, fields_json="[]",
    ))
    session.commit()
    session.add(ExpressionModel(
        canonical_hash="dup", expr_string="open", depth=1, complexity=1, fields_json="[]",
    ))
    try:
        session.commit()
        raised = False
    except IntegrityError:
        session.rollback()
        raised = True
    finally:
        session.close()
    assert raised
```

- [ ] **Step 3: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_storage_models_minibrain.py -v
```
Expected: FAIL `ImportError: cannot import name 'ExpressionModel' from 'src.storage.models'`.

- [ ] **Step 4: Thêm model mới vào `src/storage/models.py`**

```python
# Thêm vào cuối src/storage/models.py (sau SubmissionModel), thêm import đầu file:
# từ "Column, DateTime, Float, ForeignKey, Integer, String, Text" mở rộng thành
# "Column, DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String, Text,
#  UniqueConstraint"

class ExpressionModel(Base):
    """Biểu thức canonical đã từng được đánh giá (de-dup theo canonical_hash, Phase 1
    CanonicalHasher). Một expression có thể có nhiều EvaluationModel (config/window khác
    nhau)."""

    __tablename__ = "expressions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_hash = Column(String, nullable=False, unique=True, index=True)
    expr_string = Column(Text, nullable=False)
    depth = Column(Integer, nullable=False)
    complexity = Column(Integer, nullable=False)
    fields_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=_utcnow)


class EvaluationModel(Base):
    """Một lần backtest cụ thể (expression + PortfolioConfig + cửa sổ data). Lưu CẢ pass
    và fail (B11: avoid-list cần biết alpha nào đã thử và vì sao fail) + seed (R8)."""

    __tablename__ = "evaluations"
    __table_args__ = (
        UniqueConstraint(
            "expression_id", "config_json", "data_window",
            name="uq_evaluation_expr_config_window",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    expression_id = Column(Integer, ForeignKey("expressions.id"), nullable=False, index=True)
    config_json = Column(Text, nullable=False)
    data_window = Column(String, nullable=False)
    sharpe = Column(Float)
    annual_return = Column(Float)
    turnover = Column(Float)
    max_drawdown = Column(Float)
    fitness = Column(Float)
    weight_concentration = Column(Float)
    per_year_json = Column(Text)
    self_corr_max = Column(Float)
    status = Column(String, nullable=False, index=True)
    fail_reasons = Column(Text)
    seed = Column(Integer)
    created_at = Column(DateTime, default=_utcnow)


Index("idx_eval_sharpe", EvaluationModel.sharpe)


class PoolPnlModel(Base):
    """PnL vector hằng ngày của 1 evaluation PASSED — pool self-correlation cục bộ (Phase
    6). Khóa chính = evaluation_id (1 alpha pass góp 1 vector PnL vào pool)."""

    __tablename__ = "pool_pnl"

    evaluation_id = Column(Integer, ForeignKey("evaluations.id"), primary_key=True)
    dates_blob = Column(LargeBinary, nullable=False)
    pnl_blob = Column(LargeBinary, nullable=False)


class DeadFieldModel(Base):
    """Field GP/LLM đề xuất bị coi là 'chết' theo nghĩa MiniBrain (khác InvalidFieldModel
    của luồng Brain-sim cũ — đây dùng để chặn GP đề xuất lại field đã biết vô dụng/sai khi
    chạy local, không phải field bị Brain API từ chối)."""

    __tablename__ = "dead_fields_minibrain"

    name = Column(String, primary_key=True)
    reason = Column(Text)
    created_at = Column(DateTime, default=_utcnow)


class BrainRecordModel(Base):
    """Ground truth Brain-sim cho CalibrationHarness (Phase 4.5): expression + metrics thật
    từ Brain, để so Spearman ρ với metrics local."""

    __tablename__ = "brain_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    expr_string = Column(Text, nullable=False)
    brain_sharpe = Column(Float)
    brain_fitness = Column(Float)
    brain_turnover = Column(Float)
    brain_self_corr = Column(Float)
    submitted = Column(Integer)  # 0/1
    created_at = Column(DateTime, default=_utcnow)
```

- [ ] **Step 5: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_storage_models_minibrain.py -v
```
Expected: PASS (7 test).

- [ ] **Step 6: Chạy lại TOÀN BỘ test storage cũ — xác nhận không phá DB cũ**

```bash
venv/Scripts/python.exe -m pytest tests/unit/ -k "storage or migrate or repository" -v
```
Expected: PASS — model mới không sửa model cũ, `init_db` vẫn idempotent qua `create_all`.

- [ ] **Step 7: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/storage/models.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/storage/models.py
```
Expected: sạch.

- [ ] **Step 8: Commit**

```bash
git add src/storage/models.py tests/unit/test_storage_models_minibrain.py
git commit -m "feat(storage): models MiniBrain — Expression/Evaluation/PoolPnl/DeadField/BrainRecord (B11)"
```

---

### Task 5.2: Migration — đăng ký vào `MIGRATION_ORDER` (idempotent, port Postgres)

**Files:**
- Modify: `src/storage/migrate.py`
- Test: `tests/unit/test_migrate_minibrain.py`

**Interfaces:**
- Consumes: model mới (5.1).
- Produces: `MIGRATION_ORDER` (đã có, list) mở rộng thêm `ExpressionModel,
  EvaluationModel, PoolPnlModel, DeadFieldModel, BrainRecordModel` theo đúng thứ tự FK
  (`ExpressionModel` trước `EvaluationModel` trước `PoolPnlModel`; `DeadFieldModel`/
  `BrainRecordModel` không phụ thuộc gì, đặt cuối). Không có hàm mới — `init_db`
  (`src/storage/db.py`, đã có, KHÔNG sửa) tự tạo bảng mới qua `create_all` vì model đã đăng
  ký vào `Base`; Task 5.2 chỉ đảm bảo `migrate_all` (SQLite→Postgres) không bỏ sót dữ liệu
  Phase 5.

- [ ] **Step 1: Viết test đỏ — `migrate_all` copy đủ 5 bảng mới**

```python
# tests/unit/test_migrate_minibrain.py
"""Test migrate_all copy đủ bảng MiniBrain mới (Expression/Evaluation/PoolPnl/DeadField/
BrainRecord), tôn trọng thứ tự FK, idempotent khi chạy lại."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.db import init_db
from src.storage.migrate import migrate_all
from src.storage.models import (
    BrainRecordModel,
    DeadFieldModel,
    EvaluationModel,
    ExpressionModel,
    PoolPnlModel,
)


def _seeded_source():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    Session = sessionmaker(bind=engine, future=True)
    s = Session()
    expr = ExpressionModel(
        canonical_hash="h1", expr_string="close", depth=1, complexity=1, fields_json="[]",
    )
    s.add(expr)
    s.commit()
    ev = EvaluationModel(
        expression_id=expr.id, config_json="{}", data_window="w", status="passed",
    )
    s.add(ev)
    s.commit()
    s.add(PoolPnlModel(evaluation_id=ev.id, dates_blob=b"\x00", pnl_blob=b"\x00"))
    s.add(DeadFieldModel(name="bad_field", reason="rejected"))
    s.add(BrainRecordModel(expr_string="close", brain_sharpe=1.0, submitted=1))
    s.commit()
    s.close()
    return engine


def test_migrate_all_copies_minibrain_tables():
    src = _seeded_source()
    dst = create_engine("sqlite:///:memory:", future=True)
    counts = migrate_all(src, dst)
    assert counts["expressions"] == 1
    assert counts["evaluations"] == 1
    assert counts["pool_pnl"] == 1
    assert counts["dead_fields_minibrain"] == 1
    assert counts["brain_records"] == 1


def test_migrate_all_is_idempotent_on_minibrain_tables():
    src = _seeded_source()
    dst = create_engine("sqlite:///:memory:", future=True)
    migrate_all(src, dst)
    counts2 = migrate_all(src, dst)  # chạy lại — không lỗi unique/FK
    assert counts2["expressions"] == 1
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_migrate_minibrain.py -v
```
Expected: FAIL `KeyError: 'expressions'` (chưa có trong `MIGRATION_ORDER` nên
`migrate_all` không copy — counts dict thiếu key).

- [ ] **Step 3: Sửa `src/storage/migrate.py`**

```python
# Sửa import ở đầu file, thêm 5 model mới:
from src.storage.models import (
    AlphaModel,
    BrainRecordModel,
    DataFieldModel,
    DeadFieldModel,
    EvaluationModel,
    ExpressionModel,
    FailureModel,
    FetchStateModel,
    InvalidFieldModel,
    OperatorModel,
    PoolPnlModel,
    SimulationModel,
    SubmissionModel,
)

# Mở rộng MIGRATION_ORDER (thứ tự tôn trọng FK: bảng tham chiếu đứng trước):
MIGRATION_ORDER = [
    DataFieldModel,
    FetchStateModel,
    OperatorModel,
    InvalidFieldModel,
    AlphaModel,        # simulations/submissions tham chiếu alphas
    SimulationModel,
    FailureModel,
    SubmissionModel,
    ExpressionModel,   # evaluations tham chiếu expressions
    EvaluationModel,   # pool_pnl tham chiếu evaluations
    PoolPnlModel,
    DeadFieldModel,
    BrainRecordModel,
]
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_migrate_minibrain.py -v
```
Expected: PASS (2 test).

- [ ] **Step 5: Chạy lại test migrate cũ — không phá hành vi cũ**

```bash
venv/Scripts/python.exe -m pytest tests/unit/ -k migrate -v
```
Expected: PASS toàn bộ (cũ + mới).

- [ ] **Step 6: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/storage/migrate.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/storage/migrate.py
```
Expected: sạch.

- [ ] **Step 7: Commit**

```bash
git add src/storage/migrate.py tests/unit/test_migrate_minibrain.py
git commit -m "feat(storage): migrate_all copy bảng MiniBrain (Expression/Evaluation/PoolPnl/DeadField/BrainRecord)"
```

---

### Task 5.3: `MiniBrainRepository` (`src/storage/repository.py`)

**Files:**
- Modify: `src/storage/repository.py`
- Test: `tests/unit/test_minibrain_repository.py`

**Interfaces:**
- Consumes: model mới (5.1), `AlphaMetrics` (`src/backtest/metrics_local.py`, Phase 4),
  `PortfolioConfig` (`src/backtest/config.py`, Phase 3) — chỉ dùng trong test/caller để xây
  `config_json`, repository tự nó **không import** `src/backtest` (giữ dependency rule:
  storage là tầng thấp nhất, không phụ thuộc tầng nghiệp vụ phía trên — caller serialize
  `PortfolioConfig` thành JSON string TRƯỚC khi gọi repository).
- Produces: `class MiniBrainRepository` (append cuối `src/storage/repository.py`, cạnh
  `AlphaRepository`/`InvalidFieldRepository` hiện có, cùng constructor pattern
  `__init__(self, session_factory)`):
  - `def upsert_expression(self, expr_string: str, canonical_hash: str, depth: int,
    complexity: int, fields: set[str]) -> int` — `session.query(ExpressionModel).filter_by
    (canonical_hash=...).first()`; nếu có → trả `id` có sẵn (không insert trùng, không
    update — canonical_hash là khóa bất biến của 1 biểu thức); nếu chưa có → insert mới,
    `fields_json = json.dumps(sorted(fields))`, trả `id` mới.
  - `def record_evaluation(self, expression_id: int, config_json: str, data_window: str,
    metrics: AlphaMetrics | None, self_corr_max: float | None, status: str, fail_reasons:
    list[str], seed: int | None) -> int` — lưu **cả pass lẫn fail**: nếu `metrics is None`
    (fail trước khi backtest sinh được metrics — vd parse lỗi), mọi cột metric numeric =
    `None`; nếu có `metrics`, map field-by-field
    (`sharpe/annual_return/turnover/max_drawdown/fitness/weight_concentration` trực tiếp,
    `per_year_json = json.dumps(metrics.per_year_sharpe)`). `fail_reasons` rỗng `[]` khi
    `status == "passed"`, JSON-dump khi có lý do. Dùng `session.merge` theo unique
    constraint `(expression_id, config_json, data_window)`: nếu hàng đã tồn tại (re-eval
    đúng tổ hợp) → ghi đè (cập nhật outcome mới nhất, không nhân đôi); SQLAlchemy `merge`
    cần khóa chính khớp — vì PK là `id` tự tăng không phải unique constraint, dùng truy vấn
    `filter_by` thủ công trước: tồn tại → update field rồi commit; không tồn tại → insert.
    Trả `id` evaluation.
  - `def load_pool(self) -> dict[int, npt.NDArray[np.float64]]` — đọc TẤT CẢ
    `PoolPnlModel`, với mỗi row: `dates = np.frombuffer(row.dates_blob,
    dtype="datetime64[D]")` (chỉ để xác nhận đọc được, không trả về — `max_corr` ở Phase 6
    chỉ cần pnl theo `evaluation_id`), `pnl = np.frombuffer(row.pnl_blob,
    dtype=np.float64)`. Trả `{evaluation_id: pnl_array}`.
  - `def save_pool_pnl(self, evaluation_id: int, dates: npt.NDArray[np.datetime64], pnl:
    npt.NDArray[np.float64]) -> None` — pack: `dates.astype("datetime64[D]").tobytes()`,
    `pnl.astype(np.float64).tobytes()`; `session.merge(PoolPnlModel(evaluation_id=...,
    dates_blob=..., pnl_blob=...))` (merge vì gọi lại cùng evaluation_id phải ghi đè, không
    lỗi PK trùng).
  - `def add_dead_field(self, name: str, reason: str = "") -> None` — `session.merge
    (DeadFieldModel(name=name, reason=reason))` (idempotent).
  - `def is_dead_field(self, name: str) -> bool` — `session.query(DeadFieldModel).filter_by
    (name=name).first() is not None`.
  - `def result_cache_get(self, canonical_hash: str, config_json: str, data_window: str) ->
    AlphaMetrics | None` — join `ExpressionModel.canonical_hash == canonical_hash` →
    `EvaluationModel` filter `config_json`/`data_window`/`status == "passed"` (chỉ cache
    hit cho kết quả PASS — fail không có metrics đầy đủ để tái dùng an toàn) → nếu thiếu
    hàng hoặc `sharpe is None` → `None`; ngược lại dựng lại `AlphaMetrics` từ cột (load
    `per_year_json` qua `json.loads`, ép key về `int`).
  - `def result_cache_put(self, canonical_hash: str, expr_string: str, depth: int,
    complexity: int, fields: set[str], config_json: str, data_window: str, metrics:
    AlphaMetrics, seed: int | None) -> int` — gọi `upsert_expression` rồi
    `record_evaluation(status="passed", fail_reasons=[])`, trả evaluation id. (Hàm tiện ích
    gộp 2 bước cho `ResultCache.put`, Task 5.4.)
  - `def top_n(self, n: int) -> list[tuple[str, float, float]]` — query join
    `EvaluationModel.status == "passed"` với `ExpressionModel`, order by `sharpe desc
    nullslast`, limit `n`, trả `[(expr_string, sharpe, fitness)]` (giống style
    `top_simulated` đã có ở `AlphaRepository`, áp cho luồng MiniBrain).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_minibrain_repository.py
"""Test MiniBrainRepository: upsert_expression dedup, record_evaluation (pass&fail),
load_pool/save_pool_pnl round-trip, dead_field, result_cache hit/miss, top_n."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.metrics_local import AlphaMetrics
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


@pytest.fixture
def repo():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    session_factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(session_factory)


def _metrics(sharpe=1.5) -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=sharpe, annual_return=0.1, turnover=0.2, max_drawdown=-0.05,
        fitness=2.0, per_year_sharpe={2021: 1.2, 2022: 1.8}, weight_concentration=0.05,
    )


def _cfg_json() -> str:
    cfg = PortfolioConfig(neutralization=Neutralization.SECTOR)
    return json.dumps({"neutralization": cfg.neutralization.name, "decay": cfg.decay,
                        "truncation": cfg.truncation, "scale_book": cfg.scale_book,
                        "delay": cfg.delay})


def test_upsert_expression_dedups_by_canonical_hash(repo):
    id1 = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    id2 = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    assert id1 == id2


def test_upsert_expression_distinct_hash_creates_new_row(repo):
    id1 = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    id2 = repo.upsert_expression("open", "hash2", depth=1, complexity=1, fields={"open"})
    assert id1 != id2


def test_record_evaluation_passed_stores_full_metrics(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    eval_id = repo.record_evaluation(
        expr_id, _cfg_json(), "2020..2021", _metrics(), self_corr_max=0.1,
        status="passed", fail_reasons=[], seed=42,
    )
    assert isinstance(eval_id, int)


def test_record_evaluation_failed_stores_reasons_without_metrics(repo):
    expr_id = repo.upsert_expression("bad(", "hash_bad", depth=0, complexity=0, fields=set())
    eval_id = repo.record_evaluation(
        expr_id, _cfg_json(), "2020..2021", metrics=None, self_corr_max=None,
        status="invalid", fail_reasons=["parse lỗi: unexpected token"], seed=None,
    )
    assert isinstance(eval_id, int)


def test_record_evaluation_upsert_same_key_updates_not_duplicates(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(1.0), 0.1, "passed", [], 1)
    repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(2.0), 0.1, "passed", [], 1)
    cached = repo.result_cache_get("hash1", _cfg_json(), "w1")
    assert cached is not None
    assert cached.sharpe == pytest.approx(2.0)  # ghi đè, không nhân đôi


def test_save_and_load_pool_pnl_roundtrip(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    eval_id = repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(), 0.1, "passed", [], 1)
    dates = np.array(["2021-01-01", "2021-01-02", "2021-01-03"], dtype="datetime64[D]")
    pnl = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    repo.save_pool_pnl(eval_id, dates, pnl)
    pool = repo.load_pool()
    assert eval_id in pool
    np.testing.assert_allclose(pool[eval_id], pnl)


def test_dead_field_add_and_check(repo):
    assert repo.is_dead_field("bad_field") is False
    repo.add_dead_field("bad_field", reason="brain rejected")
    assert repo.is_dead_field("bad_field") is True


def test_dead_field_add_is_idempotent(repo):
    repo.add_dead_field("bad_field", reason="r1")
    repo.add_dead_field("bad_field", reason="r2")  # ghi đè, không lỗi PK trùng
    assert repo.is_dead_field("bad_field") is True


def test_result_cache_miss_returns_none(repo):
    assert repo.result_cache_get("never_seen_hash", _cfg_json(), "w1") is None


def test_result_cache_hit_after_passed_evaluation(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(1.7), 0.1, "passed", [], 9)
    cached = repo.result_cache_get("hash1", _cfg_json(), "w1")
    assert cached is not None
    assert cached.sharpe == pytest.approx(1.7)
    assert cached.per_year_sharpe == {2021: 1.2, 2022: 1.8}


def test_result_cache_no_hit_for_failed_evaluation(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    repo.record_evaluation(expr_id, _cfg_json(), "w1", None, None, "invalid", ["x"], None)
    assert repo.result_cache_get("hash1", _cfg_json(), "w1") is None


def test_result_cache_put_then_get(repo):
    m = _metrics(2.5)
    repo.result_cache_put(
        "hash_new", "ts_mean(close, 5)", depth=2, complexity=3, fields={"close"},
        config_json=_cfg_json(), data_window="w1", metrics=m, seed=7,
    )
    cached = repo.result_cache_get("hash_new", _cfg_json(), "w1")
    assert cached is not None
    assert cached.sharpe == pytest.approx(2.5)


def test_top_n_orders_by_sharpe_desc_passed_only(repo):
    id_a = repo.upsert_expression("a", "ha", depth=1, complexity=1, fields=set())
    id_b = repo.upsert_expression("b", "hb", depth=1, complexity=1, fields=set())
    repo.record_evaluation(id_a, _cfg_json(), "w1", _metrics(1.0), 0.1, "passed", [], 1)
    repo.record_evaluation(id_b, _cfg_json(), "w1", _metrics(3.0), 0.1, "passed", [], 1)
    top = repo.top_n(5)
    assert top[0][0] == "b"
    assert top[0][1] == pytest.approx(3.0)
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_minibrain_repository.py -v
```
Expected: FAIL `ImportError: cannot import name 'MiniBrainRepository' from
'src.storage.repository'`.

- [ ] **Step 3: Thêm `MiniBrainRepository` vào `src/storage/repository.py`**

```python
# Thêm import ở đầu file (giữ import cũ, mở rộng):
import numpy as np
import numpy.typing as npt

from src.backtest.metrics_local import AlphaMetrics
from src.storage.models import (
    AlphaModel,
    DeadFieldModel,
    EvaluationModel,
    ExpressionModel,
    FailureModel,
    InvalidFieldModel,
    PoolPnlModel,
    SimulationModel,
)

# Thêm class mới ở cuối file:

class MiniBrainRepository:
    """Repository cho luồng MiniBrain local (Expression/Evaluation/PoolPnl/DeadField).
    Tách khỏi AlphaRepository (luồng Brain-sim cũ) — hai luồng dữ liệu độc lập, schema
    khác nhau, không chia sẻ session pattern ngoài cấu trúc try/finally."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def upsert_expression(
        self, expr_string: str, canonical_hash: str, depth: int, complexity: int,
        fields: set[str],
    ) -> int:
        session = self.session_factory()
        try:
            existing = (
                session.query(ExpressionModel)
                .filter_by(canonical_hash=canonical_hash)
                .first()
            )
            if existing is not None:
                return existing.id
            row = ExpressionModel(
                canonical_hash=canonical_hash, expr_string=expr_string, depth=depth,
                complexity=complexity, fields_json=json.dumps(sorted(fields)),
            )
            session.add(row)
            session.commit()
            return row.id
        finally:
            session.close()

    def record_evaluation(
        self, expression_id: int, config_json: str, data_window: str,
        metrics: AlphaMetrics | None, self_corr_max: float | None, status: str,
        fail_reasons: list[str], seed: int | None,
    ) -> int:
        session = self.session_factory()
        try:
            existing = (
                session.query(EvaluationModel)
                .filter_by(
                    expression_id=expression_id, config_json=config_json,
                    data_window=data_window,
                )
                .first()
            )
            row = existing or EvaluationModel(
                expression_id=expression_id, config_json=config_json,
                data_window=data_window,
            )
            row.status = status
            row.fail_reasons = json.dumps(fail_reasons, ensure_ascii=False)
            row.self_corr_max = self_corr_max
            row.seed = seed
            if metrics is not None:
                row.sharpe = metrics.sharpe
                row.annual_return = metrics.annual_return
                row.turnover = metrics.turnover
                row.max_drawdown = metrics.max_drawdown
                row.fitness = metrics.fitness
                row.weight_concentration = metrics.weight_concentration
                row.per_year_json = json.dumps(metrics.per_year_sharpe)
            else:
                row.sharpe = None
                row.annual_return = None
                row.turnover = None
                row.max_drawdown = None
                row.fitness = None
                row.weight_concentration = None
                row.per_year_json = None
            if existing is None:
                session.add(row)
            session.commit()
            return row.id
        finally:
            session.close()

    def load_pool(self) -> dict[int, npt.NDArray[np.float64]]:
        session = self.session_factory()
        try:
            rows = session.query(PoolPnlModel).all()
            return {
                row.evaluation_id: np.frombuffer(row.pnl_blob, dtype=np.float64)
                for row in rows
            }
        finally:
            session.close()

    def save_pool_pnl(
        self, evaluation_id: int, dates: npt.NDArray[np.datetime64],
        pnl: npt.NDArray[np.float64],
    ) -> None:
        session = self.session_factory()
        try:
            session.merge(
                PoolPnlModel(
                    evaluation_id=evaluation_id,
                    dates_blob=dates.astype("datetime64[D]").tobytes(),
                    pnl_blob=pnl.astype(np.float64).tobytes(),
                )
            )
            session.commit()
        finally:
            session.close()

    def add_dead_field(self, name: str, reason: str = "") -> None:
        session = self.session_factory()
        try:
            session.merge(DeadFieldModel(name=name, reason=reason))
            session.commit()
        finally:
            session.close()

    def is_dead_field(self, name: str) -> bool:
        session = self.session_factory()
        try:
            return (
                session.query(DeadFieldModel).filter_by(name=name).first() is not None
            )
        finally:
            session.close()

    def result_cache_get(
        self, canonical_hash: str, config_json: str, data_window: str,
    ) -> AlphaMetrics | None:
        session = self.session_factory()
        try:
            row = (
                session.query(EvaluationModel)
                .join(ExpressionModel, EvaluationModel.expression_id == ExpressionModel.id)
                .filter(
                    ExpressionModel.canonical_hash == canonical_hash,
                    EvaluationModel.config_json == config_json,
                    EvaluationModel.data_window == data_window,
                    EvaluationModel.status == "passed",
                )
                .first()
            )
            if row is None or row.sharpe is None:
                return None
            per_year = {
                int(k): v for k, v in json.loads(row.per_year_json or "{}").items()
            }
            return AlphaMetrics(
                sharpe=row.sharpe, annual_return=row.annual_return, turnover=row.turnover,
                max_drawdown=row.max_drawdown, fitness=row.fitness,
                per_year_sharpe=per_year, weight_concentration=row.weight_concentration,
            )
        finally:
            session.close()

    def result_cache_put(
        self, canonical_hash: str, expr_string: str, depth: int, complexity: int,
        fields: set[str], config_json: str, data_window: str, metrics: AlphaMetrics,
        seed: int | None,
    ) -> int:
        expr_id = self.upsert_expression(expr_string, canonical_hash, depth, complexity, fields)
        return self.record_evaluation(
            expr_id, config_json, data_window, metrics, self_corr_max=None,
            status="passed", fail_reasons=[], seed=seed,
        )

    def top_n(self, n: int) -> list[tuple[str, float, float]]:
        session = self.session_factory()
        try:
            rows = (
                session.query(ExpressionModel.expr_string, EvaluationModel.sharpe,
                              EvaluationModel.fitness)
                .join(EvaluationModel, EvaluationModel.expression_id == ExpressionModel.id)
                .filter(EvaluationModel.status == "passed")
                .order_by(EvaluationModel.sharpe.desc().nullslast())
                .limit(n)
                .all()
            )
            return [(r[0], r[1], r[2] if r[2] is not None else 0.0) for r in rows]
        finally:
            session.close()
```

> **Lưu ý self_corr_max ở `result_cache_put`:** truyền `None` vì `ResultCache` (Task 5.4)
> chỉ cache metrics, không phải gate self-corr (self-corr phụ thuộc pool tại thời điểm eval,
> không phải thuộc tính bất biến của expression — cache nó sẽ stale khi pool đổi). Nếu
> caller cần lưu self_corr cùng lúc, gọi `record_evaluation` trực tiếp thay vì
> `result_cache_put`.

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_minibrain_repository.py -v
```
Expected: PASS (13 test).

- [ ] **Step 5: Chạy lại test repository cũ — không phá luồng Brain-sim**

```bash
venv/Scripts/python.exe -m pytest tests/unit/ -k "repository or storage" -v
```
Expected: PASS toàn bộ.

- [ ] **Step 6: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/storage/repository.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/storage/repository.py
```
Expected: sạch. Nếu mypy phàn nàn `session.query(...)` thiếu type (SQLAlchemy ORM động) —
đây là hạn chế đã có với code cũ (`AlphaRepository` cũng không có annotation trả về cho
`session`); giữ nhất quán, không ép `# type: ignore` tràn lan — chỉ thêm nếu mypy thật sự
fail trên dòng cụ thể, ghi rõ lý do trong comment.

- [ ] **Step 7: Commit**

```bash
git add src/storage/repository.py tests/unit/test_minibrain_repository.py
git commit -m "feat(storage): MiniBrainRepository — upsert_expression/record_evaluation/pool/dead_field/result_cache/top_n"
```

---

### Task 5.4: `ResultCache` (`src/cache/result_cache.py`)

**Files:**
- Create: `src/cache/__init__.py`
- Create: `src/cache/result_cache.py`
- Test: `tests/unit/test_result_cache.py`

**Interfaces:**
- Consumes: `MiniBrainRepository` (5.3), `AlphaMetrics` (Phase 4).
- Produces: `class ResultCache` — bọc mỏng quanh `MiniBrainRepository`
  (`__init__(self, repo: MiniBrainRepository)`), API tách biệt key (`canonical_hash`,
  `config_json`, `data_window`) khỏi việc serialize expression — `ResultCache` KHÔNG biết
  `expr_string`/`depth`/`complexity`/`fields` ở `get`, chỉ cần ở `put` (vì cache-miss nghĩa
  là expression có thể chưa từng được `upsert_expression`):
  - `def get(self, canonical_hash: str, config_json: str, data_window: str) -> AlphaMetrics
    | None` — gọi thẳng `self.repo.result_cache_get(...)`. Đây là method tồn tại để
    `ResultCache` là điểm truy cập DUY NHẤT mà GP/pipeline (Phase 7/8) gọi vào, không gọi
    trực tiếp repository — giữ tầng cache là một lớp trừu tượng độc lập (B12 tier 3) có thể
    đổi backend (vd thêm in-process LRU phía trước DB) mà không sửa caller.
  - `def put(self, canonical_hash: str, expr_string: str, depth: int, complexity: int,
    fields: set[str], config_json: str, data_window: str, metrics: AlphaMetrics, seed: int
    | None = None) -> None` — gọi `self.repo.result_cache_put(...)`, bỏ qua giá trị trả về
    (caller không cần `evaluation_id` ở tầng cache).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_result_cache.py
"""Test ResultCache: bọc MiniBrainRepository, hit sau put, miss khi key khác."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.metrics_local import AlphaMetrics
from src.cache.result_cache import ResultCache
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


@pytest.fixture
def cache():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    session_factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    repo = MiniBrainRepository(session_factory)
    return ResultCache(repo)


def _metrics() -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.3, annual_return=0.08, turnover=0.3, max_drawdown=-0.02,
        fitness=1.9, per_year_sharpe={2022: 1.1}, weight_concentration=0.04,
    )


def test_get_returns_none_on_cold_cache(cache):
    assert cache.get("hash_x", "{}", "w1") is None


def test_put_then_get_hits(cache):
    cache.put(
        "hash_x", "ts_mean(close, 5)", depth=2, complexity=3, fields={"close"},
        config_json="{}", data_window="w1", metrics=_metrics(), seed=3,
    )
    hit = cache.get("hash_x", "{}", "w1")
    assert hit is not None
    assert hit.sharpe == pytest.approx(1.3)
    assert hit.per_year_sharpe == {2022: 1.1}


def test_different_config_json_is_a_miss(cache):
    cache.put(
        "hash_x", "close", depth=1, complexity=1, fields={"close"},
        config_json="{}", data_window="w1", metrics=_metrics(), seed=None,
    )
    assert cache.get("hash_x", '{"decay": 5}', "w1") is None


def test_different_data_window_is_a_miss(cache):
    cache.put(
        "hash_x", "close", depth=1, complexity=1, fields={"close"},
        config_json="{}", data_window="w1", metrics=_metrics(), seed=None,
    )
    assert cache.get("hash_x", "{}", "w2") is None
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_result_cache.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.cache'`.

- [ ] **Step 3: Tạo `src/cache/__init__.py` + `src/cache/result_cache.py`**

```python
# src/cache/__init__.py
"""Tầng cache MiniBrain (B12): field cache (Phase 0, parquet), sub-expression cache (Phase
2, in-memory LRU), result cache (đây — DB-backed, Phase 5)."""
```

```python
# src/cache/result_cache.py
"""ResultCache — lớp DB-backed (B12 tier 3): canonical_hash+config+window -> AlphaMetrics.

Bọc MiniBrainRepository để GP/pipeline (Phase 7/8) có một điểm truy cập cache duy nhất,
không gọi trực tiếp repository — cho phép đổi backend cache (vd thêm LRU phía trước) mà
không sửa call site. Re-scoring một expression đã biết là MIỄN PHÍ khi cache hit.
"""

from __future__ import annotations

from src.backtest.metrics_local import AlphaMetrics
from src.storage.repository import MiniBrainRepository


class ResultCache:
    """Cache kết quả backtest theo khóa (canonical_hash, config_json, data_window)."""

    def __init__(self, repo: MiniBrainRepository) -> None:
        self.repo = repo

    def get(
        self, canonical_hash: str, config_json: str, data_window: str,
    ) -> AlphaMetrics | None:
        return self.repo.result_cache_get(canonical_hash, config_json, data_window)

    def put(
        self, canonical_hash: str, expr_string: str, depth: int, complexity: int,
        fields: set[str], config_json: str, data_window: str, metrics: AlphaMetrics,
        seed: int | None = None,
    ) -> None:
        self.repo.result_cache_put(
            canonical_hash, expr_string, depth, complexity, fields, config_json,
            data_window, metrics, seed,
        )
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_result_cache.py -v
```
Expected: PASS (4 test).

- [ ] **Step 5: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/cache/
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/cache/result_cache.py
```
Expected: sạch.

- [ ] **Step 6: Commit**

```bash
git add src/cache/__init__.py src/cache/result_cache.py tests/unit/test_result_cache.py
git commit -m "feat(cache): ResultCache DB-backed canonical_hash+config+window -> AlphaMetrics (B12)"
```

---

### Task 5.5: Tích hợp thật — `upsert_expression` từ AST thật (Phase 1 visitors) + round-trip end-to-end

**Files:**
- Create: `tests/integration/test_storage_minibrain_integration.py`

**Interfaces:**
- Consumes: `parse` (`src/lang/parser.py`), `DepthVisitor`, `FieldCollector`,
  `CanonicalHasher`, `ComplexityVisitor` (`src/lang/visitors.py`, Phase 1 — tất cả đã merge
  vào `main`), `MiniBrainRepository` (5.3), `ResultCache` (5.4).
- Produces: không có module mới — 1 test integration chứng minh: parse một expression thật
  → tính `depth`/`fields`/`canonical_hash`/`complexity` bằng visitor thật (không hardcode
  giá trị giả) → `upsert_expression` → `record_evaluation` (pass) → `ResultCache.get` hit
  → `record_evaluation` lần 2 với expression khác, status `"failed_gate"` → xác nhận
  `fail_reasons` đọc lại đúng từ DB. Đây là bằng chứng Task 5.1–5.4 nối với Phase 1 thật,
  không chỉ test cách ly bằng giá trị hash thủ công.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/integration/test_storage_minibrain_integration.py
"""Integration: parse (Phase 1 thật) -> visitors thật -> MiniBrainRepository -> ResultCache.
Không hardcode canonical_hash/depth/complexity — tính bằng visitor thật trên AST thật."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.metrics_local import AlphaMetrics
from src.cache.result_cache import ResultCache
from src.lang.parser import parse
from src.lang.visitors import CanonicalHasher, ComplexityVisitor, DepthVisitor, FieldCollector
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


def _make_repo():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    session_factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(session_factory)


def test_parse_visit_upsert_cache_roundtrip_with_real_ast():
    expr_string = "ts_mean(close, 5)"
    node = parse(expr_string)
    depth = node.accept(DepthVisitor())
    fields = node.accept(FieldCollector())
    canonical_hash = node.accept(CanonicalHasher())
    complexity = node.accept(ComplexityVisitor())

    repo = _make_repo()
    expr_id = repo.upsert_expression(expr_string, canonical_hash, depth, complexity, fields)
    assert isinstance(expr_id, int)

    cfg_json = json.dumps({"delay": 1})
    metrics = AlphaMetrics(
        sharpe=1.4, annual_return=0.09, turnover=0.25, max_drawdown=-0.03,
        fitness=1.7, per_year_sharpe={2023: 1.4}, weight_concentration=0.06,
    )
    repo.record_evaluation(
        expr_id, cfg_json, "2023..2024", metrics, self_corr_max=0.2,
        status="passed", fail_reasons=[], seed=11,
    )

    cache = ResultCache(repo)
    hit = cache.get(canonical_hash, cfg_json, "2023..2024")
    assert hit is not None
    assert hit.sharpe == pytest.approx(1.4)
    assert hit.per_year_sharpe == {2023: 1.4}


def test_failed_expression_recorded_with_reasons_not_cached():
    expr_string = "ts_mean(volume, 999)"  # window lớn -> giả định pass parse, fail gate
    node = parse(expr_string)
    depth = node.accept(DepthVisitor())
    fields = node.accept(FieldCollector())
    canonical_hash = node.accept(CanonicalHasher())
    complexity = node.accept(ComplexityVisitor())

    repo = _make_repo()
    expr_id = repo.upsert_expression(expr_string, canonical_hash, depth, complexity, fields)
    repo.record_evaluation(
        expr_id, json.dumps({"delay": 1}), "2023..2024", metrics=None,
        self_corr_max=None, status="failed_gate",
        fail_reasons=["self_corr 0.91 >= SELF_CORR_MAX 0.70"], seed=None,
    )

    cache = ResultCache(repo)
    assert cache.get(canonical_hash, json.dumps({"delay": 1}), "2023..2024") is None


def test_dedup_real_canonical_hash_for_commutative_expression():
    """CanonicalHasher (Phase 1) sort commutative args -> 'add(close, volume)' và
    'add(volume, close)' phải cho CÙNG canonical_hash -> upsert_expression dedup."""
    repo = _make_repo()
    hash_a = parse("add(close, volume)").accept(CanonicalHasher())
    hash_b = parse("add(volume, close)").accept(CanonicalHasher())
    assert hash_a == hash_b  # xác nhận tiền đề trước khi test dedup qua repo

    id1 = repo.upsert_expression("add(close, volume)", hash_a, 2, 3, {"close", "volume"})
    id2 = repo.upsert_expression("add(volume, close)", hash_b, 2, 3, {"close", "volume"})
    assert id1 == id2
```

- [ ] **Step 2: Chạy test — FAIL hoặc lỗi (kiểm tra giả định `add` là commutative op đã
  đăng ký trong registry Phase 1; nếu `add` không tồn tại trong grammar/registry hiện tại,
  đổi sang op commutative thật có sẵn — đọc `src/lang/registry.py` để xác nhận trước khi
  sửa, không suy diễn)**

```bash
venv/Scripts/python.exe -m pytest tests/integration/test_storage_minibrain_integration.py -v
```
Expected ban đầu: có thể FAIL ở `test_dedup_real_canonical_hash_for_commutative_expression`
nếu `add`/commutative-tagging khác giả định — sửa biểu thức test cho khớp registry thật,
giữ nguyên ý định (chứng minh dedup qua hash thật của 2 cách viết tương đương).

- [ ] **Step 3: Sửa test cho khớp API/registry thật, chạy lại đến PASS**

```bash
venv/Scripts/python.exe -m pytest tests/integration/test_storage_minibrain_integration.py -v
```
Expected: PASS (3 test).

- [ ] **Step 4: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check tests/integration/test_storage_minibrain_integration.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent tests/integration/test_storage_minibrain_integration.py
```
Expected: sạch.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_storage_minibrain_integration.py
git commit -m "test(storage): integration parse->visitors thật->MiniBrainRepository->ResultCache"
```

---

### Task 5.6: Review + merge + push

**Files:** không có file code mới — review toàn nhánh.

- [ ] **Step 1: Chạy toàn bộ test suite**

```bash
venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: PASS toàn bộ (Phase 5 mới + mọi test cũ từ Phase 0-4 + luồng Brain-sim cũ).

- [ ] **Step 2: ruff + mypy toàn repo phần đã đổi**

```bash
venv/Scripts/python.exe -m ruff check src/storage/ src/cache/ tests/unit/test_storage_models_minibrain.py tests/unit/test_migrate_minibrain.py tests/unit/test_minibrain_repository.py tests/unit/test_result_cache.py tests/integration/test_storage_minibrain_integration.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/storage/ src/cache/
```
Expected: cả hai sạch.

- [ ] **Step 3: Xác nhận DB cũ không bị phá — chạy thử `init_db` trên 1 DB sqlite có dữ liệu giả lập (model cũ), rồi mở lại bằng code mới**

```bash
venv/Scripts/python.exe -c "
from sqlalchemy import create_engine
from src.storage.db import init_db
from src.storage.models import AlphaModel
from sqlalchemy.orm import sessionmaker
e = create_engine('sqlite:///:memory:', future=True)
init_db(e)
S = sessionmaker(bind=e, future=True)
s = S()
s.add(AlphaModel(id='x1', expression='close', source='manual'))
s.commit()
s.close()
init_db(e)  # gọi lại lần 2 -- idempotent, không lỗi
print('init_db idempotent OK, AlphaModel cũ vẫn còn:', S().query(AlphaModel).count())
"
```
Expected: in `init_db idempotent OK, AlphaModel cũ vẫn còn: 1` — xác nhận `create_all` +
`_migrate_add_columns` không phá bảng cũ khi thêm model mới.

- [ ] **Step 4: Cập nhật `docs/superpowers/plans/2026-06-24-minibrain-integration-master-plan.md` — đánh dấu Phase 5 done (nếu repo có file PROGRESS.md riêng, ghi vào đó thay vì sửa master plan)**

Kiểm tra có `PROGRESS.md` ở gốc repo hay không (`ls PROGRESS.md`); nếu có, append mục
"Phase 5 — Database: done, ngày <hôm nay>, tóm tắt: ExpressionModel/EvaluationModel/
PoolPnlModel/DeadFieldModel/BrainRecordModel + MiniBrainRepository + ResultCache." Nếu
không có file này trong repo, bỏ qua bước này (không tự tạo file ngoài phạm vi plan).

- [ ] **Step 5: Merge vào main**

```bash
git checkout main
git pull --ff-only
git merge --no-ff phase-5-database -m "merge: Phase 5 — Database (MiniBrain storage + result cache)"
git push origin main
```

- [ ] **Step 6: Self-review cuối (ghi vào commit message hoặc báo cáo cho user)**

Tự kiểm danh sách sau, báo cáo PASS/FAIL từng dòng:
- [ ] `init_db` vẫn idempotent, DB cũ (luồng Brain-sim: `AlphaModel`/`SimulationModel`/...)
  không bị động đến cột/dữ liệu.
- [ ] Mọi outcome (pass VÀ fail) ghi được qua `record_evaluation`, có `fail_reasons` +
  `seed` cho fail (R8 reproducibility).
- [ ] `load_pool`/`save_pool_pnl` round-trip đúng dtype (`float64` cho pnl,
  `datetime64[D]` cho dates) — không mất độ chính xác qua `tobytes()/frombuffer()`.
- [ ] `result_cache_get` chỉ hit cho `status == "passed"` (không trả metrics rỗng/None cho
  fail).
- [ ] `migrate_all` (Postgres port) copy đủ 5 bảng mới theo đúng thứ tự FK.
- [ ] `MIGRATION_ORDER`, model mới không sửa/xóa model cũ nào trong
  `src/storage/models.py`.
- [ ] `mypy --strict --follow-imports=silent` sạch trên `src/storage/`, `src/cache/`.
- [ ] `ruff` sạch, không unused import.
- [ ] Code + commit message giữ dấu tiếng Việt đúng chính tả.
- [ ] Nhánh `phase-5-database` đã merge `--no-ff` vào `main` và push thành công.

---

## Self-review (bao phủ B11 + B12 result cache)

- `ExpressionModel`/`EvaluationModel`/`PoolPnlModel`/`DeadFieldModel`/`BrainRecordModel`
  đúng cột B11 (kể cả `UNIQUE(expression_id, config_json, data_window)`, index
  `idx_eval_expr`/`idx_eval_status`/`idx_eval_sharpe`). ✔ Task 5.1.
- Lưu cả pass và fail (avoid-list) + seed (R8). ✔ Task 5.3 (`record_evaluation` nhận
  `metrics: AlphaMetrics | None`, `fail_reasons`, `seed` luôn được lưu dù pass/fail).
- `load_pool`/`save_pool_pnl` pack/unpack blob bằng numpy `tobytes()/frombuffer()`, dtype
  ghi rõ (`float64`, `datetime64[D]`). ✔ Task 5.3.
- `result_cache_get/put` (B12 tier 3) qua `ResultCache` bọc repository, key
  `canonical_hash+config+window`. ✔ Task 5.4.
- `dead_field` add/check tự học blacklist. ✔ Task 5.3.
- `top_n` cho shortlist sau này (Phase 8). ✔ Task 5.3.
- Schema port Postgres: không dùng kiểu SQLite-only (chỉ `Integer/String/Text/Float/
  DateTime/LargeBinary` — đều có tương đương Postgres); `migrate_all` cập nhật đủ. ✔ Task
  5.2.
- `migrate.py` idempotent (`session.merge` theo PK, không insert trùng khi chạy lại). ✔
  Task 5.2 test `test_migrate_all_is_idempotent_on_minibrain_tables`.
- Không phá DB cũ (luồng Brain-sim, `AlphaRepository`/`InvalidFieldRepository`/model cũ
  nguyên vẹn). ✔ Task 5.1 Step 6, Task 5.6 Step 3.
- Tích hợp thật với Phase 1 visitors (không chỉ test cách ly bằng hash giả). ✔ Task 5.5.
- Python 3.12, type hints đầy đủ, mypy --strict sạch, ruff sạch, tiếng Việt giữ dấu trong
  docstring/comment. ✔ xuyên suốt tất cả Task.
