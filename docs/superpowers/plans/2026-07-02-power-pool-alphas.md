# Power Pool Alphas (Sub-project A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kiểm tra điều kiện Power Pool tính được LOCAL (Sharpe≥1.0, operator unique≤8, field
unique≤3), ghép mô tả Idea/Rationale ≥100 ký tự từ Hypothesis đã có, và đếm quota Power Pool
thuần (10/tháng, 1/ngày) từ dữ liệu đã nộp — làm nền để gói vào pipeline `miniquant` sau.

**Phạm vi KHÔNG làm** (đã ghi trong roadmap spec, dữ liệu không đủ tin cậy để tự dựng mà không
xác minh với API thật — làm liều sẽ silently sai):
- **Power Pool Correlation** (<0.5, khác self-correlation 0.7 thường) — endpoint thật chưa xác
  nhận.
- **Power Pool Theme matching** — danh sách theme thật chưa tìm thấy ở đâu trong tài liệu/API
  đã đọc (đã grep package `wqb-mcp` không ra).
- **Turnover/Sub-universe/Robust-universe test PASS** qua tên check cụ thể trong
  `SimulationModel.failed_checks` (sub-project B) — chỉ CHẮC CHẮN biết tên check `"LOW_SHARPE"`
  tồn tại (từ test có sẵn), các tên khác (`SUB_UNIVERSE`, `ROBUST_UNIVERSE`,...) là ĐOÁN, không
  xác nhận được — nên KHÔNG lọc theo tên check đoán mò (rủi ro loại nhầm/nhận nhầm).
- Gắn tag `PowerPoolSelected` thật khi nộp — nối vào pipeline nộp thật là việc của bạn
  (`miniquant`) sau này, dùng `SubmissionManager.set_properties()` (sub-project C) đã có sẵn.

**Architecture:** `src/scoring/power_pool.py` — hàm thuần, tái dùng `OperatorCollector`/
`FieldCollector` (sub-project D) để đếm operator/field. `src/submission/power_pool_quota.py` —
đếm số submission đã tag `PowerPoolSelected` (cột `tags` từ sub-project C) trong khoảng thời
gian, phục vụ tự kiểm tra quota trước khi gọi `submit()`.

**Tech Stack:** Python thuần (scoring), SQLAlchemy (quota, đọc DB).

## Global Constraints

- TDD bắt buộc: test FAIL trước, code tối thiểu, xác nhận PASS.
- Code/comment/commit tiếng Việt có dấu.
- Mỗi task = 1 commit.
- Chạy test: `venv/Scripts/python -m pytest`.
- 7 grouping field loại khỏi đếm field Power Pool: `country, industry, subindustry, currency,
  market, sector, exchange` — CÓ `currency`, khác 6 field của Single Dataset Alphas (sub-project
  D, không có `currency`) — đây là 2 danh sách khác nhau theo đúng tài liệu gốc, không gộp.

---

### Task 1: `count_operators_fields()` — đếm operator/field unique theo luật Power Pool

**Files:**
- Create: `src/scoring/power_pool.py`
- Test: `tests/unit/test_power_pool.py`

**Interfaces:**
- Consumes: `parse_expression` (`src/lang/parser.py`), `OperatorCollector`, `FieldCollector`
  (`src/lang/visitors.py`, sub-project D).
- Produces: `count_operators_fields(expr: str) -> tuple[int, int]` — dùng bởi Task 2.

- [ ] **Step 1: Viết test FAIL**

Tạo file `tests/unit/test_power_pool.py`:

```python
"""Test điều kiện Power Pool Alphas (sub-project A) — chỉ phần tính được LOCAL, KHÔNG gồm
Power Pool Correlation/Theme (xem docstring plan/module)."""

from __future__ import annotations

from src.scoring.power_pool import count_operators_fields


def test_dem_operator_field_co_ban():
    n_op, n_field = count_operators_fields("rank(add(close, open))")
    assert n_op == 2  # rank, add
    assert n_field == 2  # close, open


def test_loai_tru_ts_backfill_group_backfill_khoi_dem_operator():
    n_op, _ = count_operators_fields("rank(ts_backfill(group_backfill(close, sector), 5))")
    assert n_op == 1  # chỉ 'rank' tính; ts_backfill/group_backfill không tính theo tài liệu


def test_loai_tru_grouping_field_khoi_dem_field():
    _, n_field = count_operators_fields("group_rank(close, sector)")
    assert n_field == 1  # 'sector' là grouping field, bị loại; chỉ 'close' được tính


def test_operator_field_unique_khong_dem_lap():
    n_op, n_field = count_operators_fields("add(rank(close), rank(close))")
    assert n_op == 2  # add, rank (không đếm rank 2 lần)
    assert n_field == 1  # close (không đếm 2 lần)
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/unit/test_power_pool.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.scoring.power_pool'`

- [ ] **Step 3: Cài tối thiểu**

Tạo `src/scoring/power_pool.py`:

```python
"""Điều kiện Power Pool Alphas — CHỈ phần tính được LOCAL (Sharpe, operator/field unique).
KHÔNG gồm Power Pool Correlation/Theme matching (endpoint/danh sách thật chưa xác nhận, xem
docs/superpowers/plans/2026-07-02-power-pool-alphas.md mục "Phạm vi KHÔNG làm").

Nguồn tiêu chí: docs/worldquantbrain/docs/consultant-information/power-pool-alphas.md."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.lang.parser import parse_expression
from src.lang.visitors import FieldCollector, OperatorCollector

MIN_SHARPE = 1.0
MAX_UNIQUE_OPERATORS = 8
MAX_UNIQUE_FIELDS = 3
MIN_DESCRIPTION_LEN = 100

# Operator KHÔNG tính vào giới hạn 8 (theo tài liệu Power Pool).
_EXEMPT_OPERATORS = {"ts_backfill", "group_backfill"}
# 7 grouping field KHÔNG tính vào giới hạn 3 field — CÓ 'currency' (khác Single Dataset Alphas,
# sub-project D, không có 'currency' — 2 danh sách khác nhau theo đúng tài liệu gốc).
_GROUPING_FIELDS = {
    "country", "industry", "subindustry", "currency", "market", "sector", "exchange",
}


def count_operators_fields(expr: str) -> tuple[int, int]:
    """Trả (số operator unique trừ ngoại lệ, số field unique trừ grouping field) — 2 con số
    quyết định giới hạn Power Pool (<=8 operator, <=3 field)."""
    node = parse_expression(expr)
    operators = OperatorCollector().visit(node) - _EXEMPT_OPERATORS
    fields = FieldCollector().visit(node) - _GROUPING_FIELDS
    return len(operators), len(fields)
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/unit/test_power_pool.py -v`
Expected: PASS (4/4)

- [ ] **Step 5: Commit**

```bash
git add src/scoring/power_pool.py tests/unit/test_power_pool.py
git commit -m "feat(scoring): them count_operators_fields cho dieu kien Power Pool"
```

---

### Task 2: `check_power_pool_eligibility()` — gộp Sharpe + operator/field thành 1 verdict

**Files:**
- Modify: `src/scoring/power_pool.py` (thêm dataclass + hàm)
- Test: `tests/unit/test_power_pool.py`

**Interfaces:**
- Consumes: `count_operators_fields()` (Task 1).
- Produces: `PowerPoolEligibility` (dataclass: `eligible: bool`, `reasons: list[str]`,
  `n_operators: int`, `n_fields: int`), `check_power_pool_eligibility(expr: str, sharpe: float
  | None) -> PowerPoolEligibility` — dùng bởi luồng nộp thật của bạn sau này (ngoài phạm vi
  plan).

- [ ] **Step 1: Viết test FAIL**

Thêm vào `tests/unit/test_power_pool.py`:

```python
from src.scoring.power_pool import check_power_pool_eligibility


def test_du_dieu_kien_power_pool():
    result = check_power_pool_eligibility("rank(add(close, open))", sharpe=1.2)
    assert result.eligible is True
    assert result.reasons == []
    assert result.n_operators == 2
    assert result.n_fields == 2


def test_khong_du_vi_sharpe_thap():
    result = check_power_pool_eligibility("rank(close)", sharpe=0.5)
    assert result.eligible is False
    assert any("Sharpe" in r for r in result.reasons)


def test_khong_du_vi_sharpe_none():
    result = check_power_pool_eligibility("rank(close)", sharpe=None)
    assert result.eligible is False


def test_khong_du_vi_qua_nhieu_operator():
    expr = "close"
    for i in range(9):
        expr = f"op{i}({expr})"
    result = check_power_pool_eligibility(expr, sharpe=1.5)
    assert result.eligible is False
    assert any("operator" in r for r in result.reasons)


def test_khong_du_vi_qua_nhieu_field():
    result = check_power_pool_eligibility("add(add(add(f1, f2), f3), f4)", sharpe=1.5)
    assert result.eligible is False
    assert any("field" in r for r in result.reasons)
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/unit/test_power_pool.py -k eligibility -v`
Expected: FAIL với `ImportError: cannot import name 'check_power_pool_eligibility'`

- [ ] **Step 3: Cài tối thiểu**

Thêm vào cuối `src/scoring/power_pool.py`:

```python
@dataclass(frozen=True)
class PowerPoolEligibility:
    eligible: bool
    reasons: list[str] = field(default_factory=list)  # rỗng nếu eligible
    n_operators: int = 0
    n_fields: int = 0


def check_power_pool_eligibility(expr: str, sharpe: float | None) -> PowerPoolEligibility:
    """Kiểm tra 3 tiêu chí Power Pool tính được LOCAL: Sharpe>=1.0, operator unique<=8, field
    unique<=3. KHÔNG gồm Power Pool Correlation/Theme/Turnover-SubUniverse-RobustUniverse test
    (xem docstring module)."""
    reasons: list[str] = []
    n_operators, n_fields = count_operators_fields(expr)

    if sharpe is None or sharpe < MIN_SHARPE:
        reasons.append(f"Sharpe {sharpe} < {MIN_SHARPE}")
    if n_operators > MAX_UNIQUE_OPERATORS:
        reasons.append(f"{n_operators} operator unique > {MAX_UNIQUE_OPERATORS}")
    if n_fields > MAX_UNIQUE_FIELDS:
        reasons.append(f"{n_fields} field unique > {MAX_UNIQUE_FIELDS}")

    return PowerPoolEligibility(
        eligible=not reasons, reasons=reasons, n_operators=n_operators, n_fields=n_fields
    )
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/unit/test_power_pool.py -v`
Expected: PASS (9/9 — 4 của Task 1 + 5 của Task 2)

- [ ] **Step 5: Commit**

```bash
git add src/scoring/power_pool.py tests/unit/test_power_pool.py
git commit -m "feat(scoring): them check_power_pool_eligibility (Sharpe + operator/field)"
```

---

### Task 3: Mô tả Idea/Rationale cho Power Pool (>=100 ký tự)

**Files:**
- Modify: `src/scoring/power_pool.py` (thêm 2 hàm)
- Test: `tests/unit/test_power_pool.py`

**Interfaces:**
- Consumes: `src.llm.hypothesis.Hypothesis` (dataclass 4 field: `observation`, `background`,
  `economic_rationale`, `implementation_spec` — đã có sẵn từ GĐ2).
- Produces: `build_power_pool_description(hypothesis: Hypothesis) -> str`,
  `is_valid_power_pool_description(text: str) -> bool`.

- [ ] **Step 1: Viết test FAIL**

Thêm vào `tests/unit/test_power_pool.py`:

```python
from src.llm.hypothesis import Hypothesis
from src.scoring.power_pool import build_power_pool_description, is_valid_power_pool_description


def test_build_description_ghep_dung_mau_wq():
    h = Hypothesis(
        observation="Gia co phieu co xu huong dao chieu sau chuoi giam manh trong ngan han.",
        background="Ly thuyet mean-reversion tren thi truong von ngan han.",
        economic_rationale="Nha dau tu phan ung thai qua roi dieu chinh lai theo thoi gian.",
        implementation_spec="Dung field close, cua so 5 ngay, chuan hoa bang rank.",
    )
    desc = build_power_pool_description(h)
    assert "Idea:" in desc
    assert "Rationale for data used:" in desc
    assert "Rationale for operators used:" in desc
    assert is_valid_power_pool_description(desc) is True


def test_is_valid_description_do_dai():
    assert is_valid_power_pool_description("a" * 99) is False
    assert is_valid_power_pool_description("a" * 100) is True
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/unit/test_power_pool.py -k description -v`
Expected: FAIL với `ImportError: cannot import name 'build_power_pool_description'`

- [ ] **Step 3: Cài tối thiểu**

Thêm vào đầu `src/scoring/power_pool.py`, sau các import hiện có:

```python
from src.llm.hypothesis import Hypothesis
```

Thêm vào cuối `src/scoring/power_pool.py`:

```python
def build_power_pool_description(hypothesis: Hypothesis) -> str:
    """Ghép Idea/Rationale theo mẫu WQ Brain (Idea / Rationale for data used / Rationale for
    operators used, >=100 ký tự) từ Hypothesis 4 phần đã sinh sẵn (GĐ2, HypothesisGenerator).
    KHÔNG gọi LLM thêm — ghép trực tiếp nội dung đã có, ánh xạ gần đúng (implementation_spec
    thường nêu cả field lẫn tham số -> dùng cho phần 'data used'; economic_rationale ánh xạ
    sang 'operators used' vì đây là phần diễn giải cách vận hành tín hiệu)."""
    parts = [
        f"Idea: {hypothesis.observation} {hypothesis.background}".strip(),
        f"Rationale for data used: {hypothesis.implementation_spec}".strip(),
        f"Rationale for operators used: {hypothesis.economic_rationale}".strip(),
    ]
    return "\n".join(p for p in parts if p)


def is_valid_power_pool_description(text: str) -> bool:
    return len(text) >= MIN_DESCRIPTION_LEN
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/unit/test_power_pool.py -v`
Expected: PASS (11/11)

- [ ] **Step 5: Commit**

```bash
git add src/scoring/power_pool.py tests/unit/test_power_pool.py
git commit -m "feat(scoring): them build_power_pool_description tu Hypothesis co san"
```

---

### Task 4: Đếm quota Power Pool thuần (10/tháng, 1/ngày)

**Files:**
- Create: `src/submission/power_pool_quota.py`
- Test: `tests/unit/test_power_pool_quota.py`

**Interfaces:**
- Consumes: `SubmissionModel.tags`/`.status`/`.submitted_at` (cột `tags` từ sub-project C).
- Produces: `count_pure_power_pool_submissions(session_factory, since: datetime) -> int` — gọi
  2 lần với `since` khác nhau (đầu ngày / đầu tháng) để tự kiểm tra quota trước khi nộp — việc
  GỌI thật thuộc luồng e2e của bạn (`miniquant`), ngoài phạm vi plan này.

- [ ] **Step 1: Viết test FAIL**

Tạo file `tests/unit/test_power_pool_quota.py`:

```python
"""Test đếm quota Power Pool thuần (sub-project A, Task 4) — đọc cột tags đã lưu qua
SubmissionManager.set_properties() (sub-project C)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from sqlalchemy import create_engine

from src.storage.db import init_db, make_session_factory
from src.storage.models import SubmissionModel
from src.submission.power_pool_quota import count_pure_power_pool_submissions


def _engine():
    return create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})


def test_dem_dung_so_lan_nop_power_pool_thuan_trong_khoang():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    now = datetime.utcnow()
    session.add(SubmissionModel(
        id="s1", alpha_id="WQ1", status="submitted",
        tags=json.dumps(["PowerPoolSelected"]), submitted_at=now,
    ))
    session.add(SubmissionModel(
        id="s2", alpha_id="WQ2", status="submitted",
        tags=json.dumps(["khac"]), submitted_at=now,
    ))  # không có tag Power Pool -> không đếm
    session.add(SubmissionModel(
        id="s3", alpha_id="WQ3", status="rejected",
        tags=json.dumps(["PowerPoolSelected"]), submitted_at=now,
    ))  # bị reject, không phải đã nộp thành công -> không đếm
    session.add(SubmissionModel(
        id="s4", alpha_id="WQ4", status="submitted",
        tags=json.dumps(["PowerPoolSelected"]), submitted_at=now - timedelta(days=40),
    ))  # ngoài khoảng thời gian -> không đếm
    session.commit()
    session.close()

    count = count_pure_power_pool_submissions(sf, since=now - timedelta(days=1))
    assert count == 1


def test_dem_0_khi_khong_co_submission_nao():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    count = count_pure_power_pool_submissions(sf, since=datetime.utcnow() - timedelta(days=1))
    assert count == 0
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/unit/test_power_pool_quota.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.submission.power_pool_quota'`

- [ ] **Step 3: Cài tối thiểu**

Tạo `src/submission/power_pool_quota.py`:

```python
"""Đếm số lần đã nộp 'Power Pool thuần' (tag PowerPoolSelected) trong khoảng thời gian — phục
vụ tự kiểm tra quota 10/tháng + 1/ngày trước khi gọi submit() (sub-project A, Task 4).
Nguồn tiêu chí: docs/worldquantbrain/docs/consultant-information/power-pool-alphas.md
(mục "Submission Quotas After 3 Months"). CHỈ đếm dựa trên cột `SubmissionModel.tags` đã lưu
qua SubmissionManager.set_properties() (sub-project C) — KHÔNG tự gắn tag ở đây."""

from __future__ import annotations

import json
from datetime import datetime

from src.storage.models import SubmissionModel

POWER_POOL_TAG = "PowerPoolSelected"


def count_pure_power_pool_submissions(session_factory, since: datetime) -> int:
    """Đếm submission `status == "submitted"` có tag `PowerPoolSelected`, kể từ `since`."""
    session = session_factory()
    try:
        rows = (
            session.query(SubmissionModel.tags)
            .filter(SubmissionModel.status == "submitted")
            .filter(SubmissionModel.submitted_at >= since)
            .all()
        )
    finally:
        session.close()

    count = 0
    for (tags_json,) in rows:
        if not tags_json:
            continue
        try:
            tags = json.loads(tags_json)
        except (ValueError, TypeError):
            continue
        if isinstance(tags, list) and POWER_POOL_TAG in tags:
            count += 1
    return count
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/unit/test_power_pool_quota.py -v`
Expected: PASS (2/2)

- [ ] **Step 5: Chạy toàn bộ suite, xác nhận không vỡ gì**

Run: `venv/Scripts/python -m pytest tests/ -q`
Expected: PASS hết, trừ 1 fail có sẵn không liên quan (`test_make_engine_postgres_backend`).

- [ ] **Step 6: Commit**

```bash
git add src/submission/power_pool_quota.py tests/unit/test_power_pool_quota.py
git commit -m "feat(submission): them count_pure_power_pool_submissions (quota 10/thang 1/ngay)"
```

---

## Self-Review (đã chạy)

- **Spec coverage**: mục "Sub-project A" trong roadmap spec — đếm operator/field (việc 1) =
  Task 1/2; mô tả Idea/Rationale (việc 4) = Task 3; quota (việc 5) = Task 4; Power Pool
  Correlation (việc 2), Theme matching (việc 3), tag tự động (việc 6) — CỐ Ý không có task, lý
  do ghi rõ ngay đầu plan (dữ liệu/endpoint chưa xác nhận), không phải thiếu sót.
- **Placeholder scan**: sạch, mọi step có code đầy đủ.
- **Type consistency**: `count_operators_fields` (Task 1) được `check_power_pool_eligibility`
  (Task 2) gọi lại đúng chữ ký `(expr: str) -> tuple[int, int]`; `PowerPoolEligibility` dùng
  nhất quán field `eligible/reasons/n_operators/n_fields` trong test và code.
