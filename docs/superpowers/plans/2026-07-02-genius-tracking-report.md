# BRAIN Genius Tracking Report (Sub-project G) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Báo cáo READ-ONLY (không phải gate) cho 4/6 tiêu chí tie-break của BRAIN Genius (avg
distinct Operators/Alpha thấp hơn tốt hơn, avg distinct Fields/Alpha, Total distinct Operators
cao hơn tốt hơn, Total distinct Fields) từ các alpha ĐÃ NỘP — theo
`docs/worldquantbrain/docs/consultant-information/brain-genius.md`.

**Phạm vi KHÔNG làm** (thiếu dữ liệu cached, làm liều sẽ ra số sai lệch với Genius thật):
- **"Pyramid" (region×delay×dataset-CATEGORY, ≥3 alpha/pyramid)** — `SimulationModel` KHÔNG có
  cột `delay`, và dataset CATEGORY (fundamental/analyst/price-volume...) không được cache ở đâu
  trong DB hiện tại (`DataFieldModel` chỉ có `dataset_id`, không có category) — cần fetch/lưu
  thêm dữ liệu mới hoàn toàn để làm đúng, ngoài phạm vi 1 task nhỏ, để lại cho lần sau nếu cần.
- **Maximum Simulation Streak, Community leader (forum likes/referral)** — dữ liệu ở nền tảng
  WQ, không có trong DB local, không có API đã biết để lấy.
- **Combined Alpha Performance** — cần Out-sample Sharpe của TOÀN pool đã nộp, WQ tự tính
  "refresh mỗi 4-6 tuần", không có công thức tái tạo local đáng tin cậy.

**Architecture:** `src/scoring/genius_report.py` — dùng lại `OperatorCollector`/
`FieldCollector` (sub-project D) trên biểu thức của MỌI alpha đã `SubmissionModel.status ==
"submitted"` (join qua `wq_alpha_id`, giống cách `SubmissionManager.select_candidates()` đã
làm).

**Tech Stack:** Python + SQLAlchemy (đọc DB), không I/O mạng.

## Global Constraints

- TDD bắt buộc: test FAIL trước, code tối thiểu, xác nhận PASS.
- Code/comment/commit tiếng Việt có dấu.
- 1 task = 1 commit.
- Chạy test: `venv/Scripts/python -m pytest`.
- `ts_backfill`/`group_backfill` KHÔNG tính vào đếm operator (tài liệu Genius ghi rõ, cùng
  ngoại lệ với Power Pool sub-project A) — field KHÔNG có ngoại lệ nào theo tài liệu Genius
  (khác Power Pool/Single-Dataset có loại 6-7 grouping field) — đếm field ở đây KHÔNG loại
  grouping field, cố ý khác `count_operators_fields` của sub-project A.

---

### Task 1: 4 hàm report (avg/total distinct operators & fields của alpha đã nộp)

**Files:**
- Create: `src/scoring/genius_report.py`
- Test: `tests/unit/test_genius_report.py`

**Interfaces:**
- Consumes: `parse_expression`, `OperatorCollector`, `FieldCollector`
  (`src/lang/parser.py`/`src/lang/visitors.py`), `AlphaModel`/`SimulationModel`/
  `SubmissionModel` (`src/storage/models.py`).
- Produces: `average_distinct_operators_per_alpha(session_factory) -> float | None`,
  `average_distinct_fields_per_alpha(session_factory) -> float | None`,
  `total_distinct_operators(session_factory) -> int`,
  `total_distinct_fields(session_factory) -> int` — hàm report thuần, chưa có caller CLI trong
  plan này (ngoài phạm vi — nối vào `miniquant` là việc của bạn sau).

- [ ] **Step 1: Viết test FAIL**

Tạo file `tests/unit/test_genius_report.py`:

```python
"""Test báo cáo BRAIN Genius (sub-project G) — 4 metric tie-break tính được LOCAL từ alpha đã
nộp. KHÔNG phải gate, chỉ để tham khảo."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from src.scoring.genius_report import (
    average_distinct_fields_per_alpha,
    average_distinct_operators_per_alpha,
    total_distinct_fields,
    total_distinct_operators,
)
from src.storage.db import init_db, make_session_factory
from src.storage.models import AlphaModel, SimulationModel, SubmissionModel


def _engine():
    return create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})


def _seed_submitted(session_factory):
    session = session_factory()
    try:
        session.add(AlphaModel(id="a1", expression="rank(add(close, open))", source="ga"))
        session.add(SimulationModel(
            id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000", status="passed",
        ))
        session.add(SubmissionModel(id="sub1", alpha_id="WQ1", status="submitted"))

        session.add(AlphaModel(id="a2", expression="ts_delta(close, 5)", source="ga"))
        session.add(SimulationModel(
            id="s2", alpha_id="a2", wq_alpha_id="WQ2", region="USA", universe="TOP3000", status="passed",
        ))
        session.add(SubmissionModel(id="sub2", alpha_id="WQ2", status="submitted"))

        # chưa nộp -> KHÔNG được tính vào report
        session.add(AlphaModel(id="a3", expression="rank(rank(rank(close)))", source="ga"))
        session.add(SimulationModel(
            id="s3", alpha_id="a3", wq_alpha_id="WQ3", region="USA", universe="TOP3000", status="passed",
        ))
        session.commit()
    finally:
        session.close()


def test_average_distinct_operators_per_alpha():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_submitted(sf)
    # a1: {rank, add}=2 operator; a2: {ts_delta}=1 operator -> avg (2+1)/2 = 1.5
    assert average_distinct_operators_per_alpha(sf) == pytest.approx(1.5)


def test_average_distinct_fields_per_alpha():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_submitted(sf)
    # a1: {close, open}=2 field; a2: {close}=1 field -> avg (2+1)/2 = 1.5
    assert average_distinct_fields_per_alpha(sf) == pytest.approx(1.5)


def test_total_distinct_operators():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_submitted(sf)
    # union {rank, add} | {ts_delta} = {rank, add, ts_delta} = 3
    assert total_distinct_operators(sf) == 3


def test_total_distinct_fields():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_submitted(sf)
    # union {close, open} | {close} = {close, open} = 2
    assert total_distinct_fields(sf) == 2


def test_none_khi_chua_co_alpha_nao_nop():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    assert average_distinct_operators_per_alpha(sf) is None
    assert average_distinct_fields_per_alpha(sf) is None
    assert total_distinct_operators(sf) == 0
    assert total_distinct_fields(sf) == 0


def test_ts_backfill_group_backfill_khong_tinh_vao_operator():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    session.add(AlphaModel(id="a1", expression="rank(ts_backfill(group_backfill(close, sector), 5))", source="ga"))
    session.add(SimulationModel(
        id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000", status="passed",
    ))
    session.add(SubmissionModel(id="sub1", alpha_id="WQ1", status="submitted"))
    session.commit()
    session.close()
    assert average_distinct_operators_per_alpha(sf) == pytest.approx(1.0)  # chỉ 'rank'
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/unit/test_genius_report.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.scoring.genius_report'`

- [ ] **Step 3: Cài tối thiểu**

Tạo `src/scoring/genius_report.py`:

```python
"""Báo cáo READ-ONLY cho tiêu chí tie-break BRAIN Genius tính được LOCAL từ alpha đã nộp —
KHÔNG phải gate nộp. Nguồn: docs/worldquantbrain/docs/consultant-information/brain-genius.md
(mục "What happens if more consultants qualify..."). Phạm vi KHÔNG làm (pyramid, streak,
community leader, Combined Alpha Performance) ghi ở
docs/superpowers/plans/2026-07-02-genius-tracking-report.md."""

from __future__ import annotations

from src.lang.parser import parse_expression
from src.lang.visitors import FieldCollector, OperatorCollector
from src.storage.models import AlphaModel, SimulationModel, SubmissionModel

# ts_backfill/group_backfill KHÔNG tính vào Avg/Total distinct Operators (tài liệu Genius).
_EXEMPT_OPERATORS = {"ts_backfill", "group_backfill"}


def _submitted_expressions(session_factory) -> list[str]:
    """Biểu thức của mọi alpha đã `status == "submitted"` (không trùng lặp theo wq_alpha_id)."""
    session = session_factory()
    try:
        wq_ids = {
            row[0]
            for row in session.query(SubmissionModel.alpha_id)
            .filter(SubmissionModel.status == "submitted")
            .all()
        }
        if not wq_ids:
            return []
        rows = (
            session.query(AlphaModel.expression)
            .join(SimulationModel, SimulationModel.alpha_id == AlphaModel.id)
            .filter(SimulationModel.wq_alpha_id.in_(wq_ids))
            .distinct()
            .all()
        )
        return [r[0] for r in rows]
    finally:
        session.close()


def average_distinct_operators_per_alpha(session_factory) -> float | None:
    exprs = _submitted_expressions(session_factory)
    if not exprs:
        return None
    counts = [
        len(OperatorCollector().visit(parse_expression(e)) - _EXEMPT_OPERATORS) for e in exprs
    ]
    return sum(counts) / len(counts)


def average_distinct_fields_per_alpha(session_factory) -> float | None:
    exprs = _submitted_expressions(session_factory)
    if not exprs:
        return None
    counts = [len(FieldCollector().visit(parse_expression(e))) for e in exprs]
    return sum(counts) / len(counts)


def total_distinct_operators(session_factory) -> int:
    exprs = _submitted_expressions(session_factory)
    all_ops: set[str] = set()
    for e in exprs:
        all_ops |= OperatorCollector().visit(parse_expression(e)) - _EXEMPT_OPERATORS
    return len(all_ops)


def total_distinct_fields(session_factory) -> int:
    exprs = _submitted_expressions(session_factory)
    all_fields: set[str] = set()
    for e in exprs:
        all_fields |= FieldCollector().visit(parse_expression(e))
    return len(all_fields)
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/unit/test_genius_report.py -v`
Expected: PASS (6/6)

- [ ] **Step 5: Chạy toàn bộ suite, xác nhận không vỡ gì**

Run: `venv/Scripts/python -m pytest tests/ -q`
Expected: PASS hết, trừ 1 fail có sẵn không liên quan (`test_make_engine_postgres_backend`).

- [ ] **Step 6: Commit**

```bash
git add src/scoring/genius_report.py tests/unit/test_genius_report.py
git commit -m "feat(scoring): them bao cao BRAIN Genius (avg/total distinct operators/fields)"
```

---

## Self-Review (đã chạy)

- **Spec coverage**: mục "Sub-project G" trong roadmap spec — "report đơn giản: avg distinct
  operators/fields mỗi alpha" = Task 1 (4 hàm); "đếm pyramid đã hình thành" — CỐ Ý bỏ, lý do ghi
  rõ ngay đầu plan (thiếu cột `delay` + dataset category chưa cache, không phải sơ sót).
- **Placeholder scan**: sạch.
- **Type consistency**: cả 4 hàm dùng chung `_submitted_expressions()`, cùng ký hiệu
  `session_factory` xuyên suốt; `None` cho "chưa có alpha nào" (avg) vs `0` cho "chưa có gì để
  hợp" (total) — nhất quán với ngữ nghĩa từng kiểu trả về.
