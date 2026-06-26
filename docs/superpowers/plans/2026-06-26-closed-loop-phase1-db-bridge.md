# Closed-Loop Phase 1 — Cầu DB (Brain SIM ↔ MiniBrain) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (khuyến nghị) hoặc superpowers:executing-plans. Steps dùng checkbox (`- [ ]`). Đây là
> Phase 1 của feature "Vòng kín AI + MiniBrain" (spec
> `docs/superpowers/specs/2026-06-26-ai-minibrain-closed-loop-design.md`). Các phase sau
> (orchestrator, feedback, menu, prompt) có plan riêng.

**Goal:** Thêm cầu DB liên kết một expression MiniBrain (qua `canonical_hash`) với kết quả
SIM thật trên WorldQuant Brain, để vòng kín lưu được "expression này đã sim Brain → sharpe/
fitness/self_corr thật" và feed ngược (so local↔Brain, decorrelate tầng 2, calibrate).

**Architecture:** Một bảng mới `brain_sim_links` (model `BrainSimLinkModel` trong
`src/storage/models.py`, tự tạo qua `init_db` vì kế thừa `Base`) + các method trên
`MiniBrainRepository` (`record_brain_sim`, `load_brain_sims`, `brain_pnl_pool`). Không sửa
schema cũ (`AlphaModel`/`SimulationModel`) — cầu mới keyed theo `canonical_hash` (danh tính
chung của expression), độc lập hai thế giới.

**Tech Stack:** Python 3.12, SQLAlchemy (declarative `Base`, `Column`), pytest, SQLite
in-memory cho test.

## Global Constraints

- Python 3.12; full type hints; `mypy --strict --follow-imports=silent` clean trên code mới
  (lưu ý: `src/storage/models.py` có mypy debt baseline `declarative_base()` legacy — KHÔNG
  làm phát sinh lỗi MỚI ngoài baseline đó); `ruff` clean; không unused import.
- Determinism, no look-ahead: không áp dụng trực tiếp ở tầng DB này.
- **Tiếng Việt giữ dấu đúng chính tả** trong mọi docstring/comment mới.
- TDD: test trước (đỏ) → code tối thiểu (xanh) → commit. Mỗi task = 1 commit.
- KHÔNG sửa `AlphaModel`/`SimulationModel`/`EvaluationModel` hiện có (chỉ THÊM model + method).

## Pre-condition (chữ ký thật đã xác minh)

```python
# src/storage/models.py — declarative Base, các model kế thừa Base; init_db tạo mọi bảng Base.
class ExpressionModel(Base):   # __tablename__="expressions"; id (Integer PK autoincrement);
    canonical_hash (String, unique, index); expr_string; depth; complexity; fields_json; created_at
class EvaluationModel(Base):   # __tablename__="evaluations"; expression_id FK->expressions.id; ...
# _utcnow() là factory default datetime đã có trong models.py (dùng cho created_at).
# Base, Column, String, Integer, Float, Text, DateTime, ForeignKey đã import sẵn ở đầu models.py.

# src/storage/repository.py
class MiniBrainRepository:
    def __init__(self, session_factory): self.session_factory = session_factory
    # các method hiện có: upsert_expression(...)->int, record_evaluation(...)->int,
    #   load_pool()->dict[int, tuple[Dates, NDArray[float64]]], save_pool_pnl(eval_id, dates, pnl)
    # session pattern: s = self.session_factory(); try: ...; s.commit(); finally: s.close()

# src/storage/db.py
def init_db(engine) -> engine   # tạo mọi bảng Base
# test pattern (xem tests/unit/test_gp_engine.py): create_engine("sqlite:///:memory:") ->
#   init_db(engine) -> sessionmaker(bind=engine, future=True, expire_on_commit=False)

# src/local_types.py: Dates = NDArray[datetime64]
```

## File Structure

- **Modify** `src/storage/models.py` (~20 dòng): thêm `class BrainSimLinkModel(Base)`.
- **Modify** `src/storage/repository.py` (~70 dòng): thêm `record_brain_sim`, `load_brain_sims`,
  `brain_pnl_pool` vào `MiniBrainRepository`.
- **Create** `tests/unit/test_brain_sim_link.py` (~130 dòng): test model + 3 method.

---

### Task 1: `BrainSimLinkModel` — bảng liên kết Brain SIM ↔ expression MiniBrain

**Files:**
- Modify: `src/storage/models.py`
- Test: `tests/unit/test_brain_sim_link.py`

**Interfaces:**
- Consumes: `Base`, `Column`, `Integer`, `String`, `Float`, `Text`, `DateTime`, `ForeignKey`,
  `_utcnow` (đã có trong `src/storage/models.py`).
- Produces:
  ```python
  class BrainSimLinkModel(Base):
      __tablename__ = "brain_sim_links"
      id: int                      # PK autoincrement
      canonical_hash: str          # index — danh tính expression MiniBrain (khớp ExpressionModel.canonical_hash)
      expr_string: str             # bản FASTEXPR đã nộp sim (tiện đọc/đối chiếu)
      wq_alpha_id: str | None      # id alpha trên nền tảng WQ
      region: str
      universe: str
      sharpe: float | None         # metric THẬT từ Brain
      fitness: float | None
      turnover: float | None
      self_corr: float | None      # self-correlation THẬT do Brain báo (decorrelate tầng 2)
      status: str                  # 'passed' | 'failed' | 'error'
      raw_json: str | None         # full JSON kết quả sim (audit)
      created_at: datetime
  ```

- [ ] **Step 1: Viết test đỏ `tests/unit/test_brain_sim_link.py` (phần model)**

```python
"""Test cầu DB Brain SIM ↔ MiniBrain: model BrainSimLinkModel + method repository
record_brain_sim/load_brain_sims/brain_pnl_pool. SQLite in-memory, không mạng."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.db import init_db
from src.storage.models import BrainSimLinkModel
from src.storage.repository import MiniBrainRepository


@pytest.fixture
def repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    sf = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(sf)


def test_brain_sim_link_table_created_and_insertable(repo) -> None:  # noqa: ANN001
    """Bảng brain_sim_links được init_db tạo; chèn 1 row đọc lại đúng giá trị."""
    s = repo.session_factory()
    try:
        row = BrainSimLinkModel(
            canonical_hash="h1", expr_string="rank(close)", wq_alpha_id="WQ123",
            region="USA", universe="TOP3000", sharpe=1.5, fitness=1.2, turnover=0.3,
            self_corr=0.4, status="passed", raw_json="{}",
        )
        s.add(row)
        s.commit()
        got = s.query(BrainSimLinkModel).filter_by(canonical_hash="h1").one()
        assert got.wq_alpha_id == "WQ123"
        assert got.sharpe == 1.5
        assert got.self_corr == 0.4
        assert got.created_at is not None
    finally:
        s.close()
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_brain_sim_link.py -q
```
Expected: FAIL `ImportError: cannot import name 'BrainSimLinkModel'`.

- [ ] **Step 3: Thêm `BrainSimLinkModel` vào `src/storage/models.py`**

Thêm sau `class PoolPnlModel` (hoặc cuối nhóm MiniBrain models), dùng đúng import có sẵn ở đầu
file (`Base`, `Column`, `Integer`, `String`, `Float`, `Text`, `DateTime`, `_utcnow`):

```python
class BrainSimLinkModel(Base):
    """Cầu liên kết một expression MiniBrain (theo ``canonical_hash``) với kết quả SIM THẬT
    trên WorldQuant Brain. Tách khỏi ``AlphaModel``/``SimulationModel`` (luồng LLM cũ) — cầu
    này keyed theo ``canonical_hash`` là danh tính chung của expression MiniBrain, phục vụ
    feedback vòng kín (so local↔Brain, decorrelate tầng 2 bằng ``self_corr`` Brain thật)."""

    __tablename__ = "brain_sim_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_hash = Column(String, nullable=False, index=True)
    expr_string = Column(Text, nullable=False)
    wq_alpha_id = Column(String)
    region = Column(String)
    universe = Column(String)
    sharpe = Column(Float)
    fitness = Column(Float)
    turnover = Column(Float)
    self_corr = Column(Float)
    status = Column(String, nullable=False)
    raw_json = Column(Text)
    created_at = Column(DateTime, default=_utcnow)
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_brain_sim_link.py -q
```
Expected: 1 PASS.

- [ ] **Step 5: Kiểm dấu tiếng Việt + commit**

```bash
git add src/storage/models.py tests/unit/test_brain_sim_link.py
git commit -m "feat(storage): BrainSimLinkModel - cau DB Brain SIM <-> expression MiniBrain"
```

---

### Task 2: Repository methods `record_brain_sim` / `load_brain_sims` / `brain_pnl_pool`

**Files:**
- Modify: `src/storage/repository.py`
- Test: `tests/unit/test_brain_sim_link.py` (thêm test)

**Interfaces:**
- Consumes: `BrainSimLinkModel` (Task 1), `PoolPnlModel` (đã có — để `brain_pnl_pool` đọc PnL
  thật nếu lưu kèm), `MiniBrainRepository.session_factory`.
- Produces (thêm vào `class MiniBrainRepository`):
  ```python
  def record_brain_sim(
      self, canonical_hash: str, expr_string: str, *, wq_alpha_id: str | None,
      region: str, universe: str, sharpe: float | None, fitness: float | None,
      turnover: float | None, self_corr: float | None, status: str,
      raw_json: str | None = None,
  ) -> int: ...
  # merge theo (canonical_hash, region, universe): đã có -> cập nhật; chưa -> insert. Trả id.

  def load_brain_sims(self) -> list[BrainSimLinkModel]: ...
  # mọi link đã ghi (cho calibrate feedback + avoid-list).

  def brain_pnl_pool(self) -> dict[str, float]: ...
  # {canonical_hash: self_corr} cho các link status='passed' có self_corr != None
  # (tra cứu nhanh self-corr Brain thật của các alpha đã nộp — decorrelate tầng 2).
  ```

- [ ] **Step 1: Viết test đỏ (thêm vào `tests/unit/test_brain_sim_link.py`)**

```python
def test_record_brain_sim_inserts_then_updates_by_key(repo) -> None:  # noqa: ANN001
    """record_brain_sim merge theo (canonical_hash, region, universe): lần 2 cập nhật,
    KHÔNG nhân đôi row."""
    id1 = repo.record_brain_sim(
        "hA", "rank(close)", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=1.0, fitness=0.9, turnover=0.2, self_corr=0.3, status="passed",
    )
    id2 = repo.record_brain_sim(
        "hA", "rank(close)", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=1.8, fitness=1.5, turnover=0.25, self_corr=0.35, status="passed",
    )
    assert id1 == id2  # cùng key -> cùng row
    sims = repo.load_brain_sims()
    assert len(sims) == 1
    assert sims[0].sharpe == 1.8  # đã cập nhật giá trị mới


def test_load_brain_sims_returns_all(repo) -> None:  # noqa: ANN001
    repo.record_brain_sim("h1", "close", wq_alpha_id=None, region="USA", universe="TOP3000",
                          sharpe=1.0, fitness=1.0, turnover=0.1, self_corr=0.1, status="passed")
    repo.record_brain_sim("h2", "open", wq_alpha_id=None, region="USA", universe="TOP3000",
                          sharpe=None, fitness=None, turnover=None, self_corr=None, status="error")
    assert len(repo.load_brain_sims()) == 2


def test_brain_pnl_pool_only_passed_with_self_corr(repo) -> None:  # noqa: ANN001
    """brain_pnl_pool chỉ trả link passed có self_corr != None."""
    repo.record_brain_sim("hp", "close", wq_alpha_id="W", region="USA", universe="TOP3000",
                          sharpe=1.0, fitness=1.0, turnover=0.1, self_corr=0.5, status="passed")
    repo.record_brain_sim("hf", "open", wq_alpha_id="W2", region="USA", universe="TOP3000",
                          sharpe=0.0, fitness=0.0, turnover=0.0, self_corr=None, status="failed")
    pool = repo.brain_pnl_pool()
    assert pool == {"hp": 0.5}
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_brain_sim_link.py -q
```
Expected: FAIL `AttributeError: 'MiniBrainRepository' object has no attribute 'record_brain_sim'`.

- [ ] **Step 3: Thêm 3 method vào `class MiniBrainRepository` (`src/storage/repository.py`)**

Thêm import `BrainSimLinkModel` vào khối import models đã có ở đầu file (gộp vào dòng
`from src.storage.models import (...)` hiện có — KHÔNG tạo dòng trùng). Thêm method (theo
đúng session pattern hiện có trong file):

```python
    def record_brain_sim(
        self, canonical_hash: str, expr_string: str, *, wq_alpha_id: str | None,
        region: str, universe: str, sharpe: float | None, fitness: float | None,
        turnover: float | None, self_corr: float | None, status: str,
        raw_json: str | None = None,
    ) -> int:
        """Ghi kết quả SIM Brain cho 1 expression MiniBrain. Merge theo khóa duy nhất
        (canonical_hash, region, universe): đã có -> cập nhật outcome mới nhất (không nhân
        đôi); chưa có -> insert. Trả id row."""
        session = self.session_factory()
        try:
            existing = (
                session.query(BrainSimLinkModel)
                .filter_by(canonical_hash=canonical_hash, region=region, universe=universe)
                .first()
            )
            row = existing or BrainSimLinkModel(
                canonical_hash=canonical_hash, region=region, universe=universe,
            )
            row.expr_string = expr_string
            row.wq_alpha_id = wq_alpha_id
            row.sharpe = sharpe
            row.fitness = fitness
            row.turnover = turnover
            row.self_corr = self_corr
            row.status = status
            row.raw_json = raw_json
            if existing is None:
                session.add(row)
            session.commit()
            return row.id  # type: ignore[return-value]
        finally:
            session.close()

    def load_brain_sims(self) -> list[BrainSimLinkModel]:
        """Trả mọi link Brain SIM đã ghi (cho calibrate feedback + avoid-list)."""
        session = self.session_factory()
        try:
            return session.query(BrainSimLinkModel).all()
        finally:
            session.close()

    def brain_pnl_pool(self) -> dict[str, float]:
        """Trả {canonical_hash: self_corr} cho các link status='passed' có self_corr != None
        — tra cứu nhanh self-corr Brain THẬT của alpha đã nộp (decorrelate tầng 2)."""
        session = self.session_factory()
        try:
            rows = (
                session.query(BrainSimLinkModel)
                .filter(BrainSimLinkModel.status == "passed")
                .filter(BrainSimLinkModel.self_corr.isnot(None))
                .all()
            )
            return {r.canonical_hash: float(r.self_corr) for r in rows}
        finally:
            session.close()
```

- [ ] **Step 4: Chạy test — PASS (4 test tổng)**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_brain_sim_link.py -q
```
Expected: 4 PASS.

- [ ] **Step 5: ruff + mypy + kiểm dấu tiếng Việt**

```bash
venv/Scripts/python.exe -m ruff check src/storage/repository.py src/storage/models.py tests/unit/test_brain_sim_link.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/storage/repository.py
```
Expected: ruff sạch. mypy: chỉ các lỗi baseline tiền-tồn của `src/storage` (vd
`declarative_base()` legacy) — KHÔNG lỗi MỚI từ 3 method/model vừa thêm. Nếu có lỗi mới (vd
`row.id` no-any-return), thêm `# type: ignore[...]` đúng mã lỗi như pattern các method khác
trong file (`upsert_expression`/`record_evaluation` đã có `# type: ignore`).

- [ ] **Step 6: Chạy regression nhanh storage + commit**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_brain_sim_link.py tests/integration/test_storage_minibrain_integration.py -q
git add src/storage/repository.py tests/unit/test_brain_sim_link.py
git commit -m "feat(storage): record_brain_sim/load_brain_sims/brain_pnl_pool cho cau DB vong kin"
```

---

## Self-review

**Spec coverage (Phase 1 scope):**
- [x] Cầu DB liên kết evaluation MiniBrain ↔ kết quả SIM Brain (qua canonical_hash) — Task 1.
- [x] Lưu self_corr Brain thật (decorrelate tầng 2 nguồn) — `BrainSimLinkModel.self_corr` +
  `brain_pnl_pool` (Task 1/2).
- [x] Đọc lại cho feedback calibrate/avoid-list — `load_brain_sims` (Task 2).
- [x] Không sửa schema cũ — chỉ THÊM model + method.

**Placeholder scan:** ✅ Mọi step có code/lệnh cụ thể. Không TBD.

**Type consistency:**
- `record_brain_sim(canonical_hash, expr_string, *, wq_alpha_id, region, universe, sharpe,
  fitness, turnover, self_corr, status, raw_json=None) -> int` — nhất quán giữa Task 2
  Interfaces, test (Step 1), impl (Step 3).
- `BrainSimLinkModel` field names (canonical_hash/wq_alpha_id/sharpe/fitness/turnover/
  self_corr/status/raw_json) — nhất quán giữa Task 1 model, Task 2 method, mọi test.
- `brain_pnl_pool() -> dict[str, float]` (hash→self_corr) — khớp test
  `test_brain_pnl_pool_only_passed_with_self_corr`.

**Risks / gotchas:**
1. `row.id` sau commit có thể khiến mypy than no-any-return — xử như pattern `# type: ignore`
   đã dùng ở `upsert_expression`/`record_evaluation` cùng file.
2. `self_corr.isnot(None)` là cú pháp SQLAlchemy đúng cho filter NULL (không dùng `!= None`
   trần — ruff/flake8 cảnh báo E711).
