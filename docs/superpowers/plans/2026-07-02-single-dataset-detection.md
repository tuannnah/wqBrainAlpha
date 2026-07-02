# Single Dataset Alpha Detection (Sub-project D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hàm thuần phát hiện một alpha có phải "Single Dataset Alpha" không (mọi field trừ 6
grouping field đến từ đúng 1 dataset) — dùng để GẮN NHÃN/tham khảo (vd LLM ưu tiên hướng đơn
dataset — tài liệu: "less prone to overfitting"), **KHÔNG dùng để gate nộp**: WQ Brain tự áp
ngưỡng nới lỏng cho Single Dataset Alpha khi tính `is.checks`/`status` (xem đính chính
sub-project B trong `docs/superpowers/specs/2026-07-02-submission-compliance-roadmap-design.md`)
— tool không cần tự quyết định "được nộp hay không" dựa trên phát hiện này.

**Architecture:** Thêm `OperatorCollector` (visitor mới, cùng họ `FieldCollector` đã có trong
`src/lang/visitors.py`) để trích tên operator dùng trong biểu thức — hạ tầng này DÙNG CHUNG với
sub-project A (đếm operator unique cho Power Pool) nên làm 1 lần ở đây. Sau đó
`src/scoring/dataset_usage.py` kết hợp `FieldCollector` + `OperatorCollector` + map
field→dataset để suy dataset duy nhất (hoặc `None` nếu dùng >1 dataset).

**Tech Stack:** Python thuần, không I/O, không phụ thuộc DB/network.

## Global Constraints

- TDD bắt buộc: test FAIL trước, code tối thiểu, xác nhận PASS.
- Code/comment/commit tiếng Việt có dấu.
- Mỗi task = 1 commit.
- Chạy test: `venv/Scripts/python -m pytest`.
- KHÔNG nối vào `SubmissionManager`/bất kỳ gate nộp nào — đây là hàm thông tin thuần.

---

### Task 1: `OperatorCollector` visitor

**Files:**
- Modify: `src/lang/visitors.py` (thêm class sau `FieldCollector`, dòng ~50)
- Test: `tests/unit/test_lang_visitors.py` (file mới)

**Interfaces:**
- Consumes: `Node`, `Call`, `Constant`, `Field`, `NodeVisitor` (đã có trong `src/lang/ast.py`,
  import sẵn ở đầu `visitors.py`).
- Produces: `OperatorCollector().visit(node) -> set[str]` — tập tên operator (Call.op) dùng
  trong cây — dùng bởi Task 2 và (sau này) sub-project A.

- [ ] **Step 1: Viết test FAIL**

Tạo file mới `tests/unit/test_lang_visitors.py`:

```python
"""Test các visitor mới trên AST (OperatorCollector)."""

from __future__ import annotations

from src.lang.parser import parse_expression
from src.lang.visitors import OperatorCollector


def test_operator_collector_don_gian():
    node = parse_expression("rank(close)")
    assert OperatorCollector().visit(node) == {"rank"}


def test_operator_collector_long_nhau():
    node = parse_expression("rank(add(ts_delta(close, 5), open))")
    assert OperatorCollector().visit(node) == {"rank", "add", "ts_delta"}


def test_operator_collector_khong_co_operator():
    node = parse_expression("close")
    assert OperatorCollector().visit(node) == set()


def test_operator_collector_dem_operator_lap_lai_chi_1_lan():
    node = parse_expression("add(rank(close), rank(open))")
    assert OperatorCollector().visit(node) == {"add", "rank"}
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/unit/test_lang_visitors.py -v`
Expected: FAIL với `ImportError: cannot import name 'OperatorCollector'`

- [ ] **Step 3: Cài tối thiểu**

Trong `src/lang/visitors.py`, thêm class ngay sau `FieldCollector` (trước `class Serializer`):

```python
class OperatorCollector(NodeVisitor["set[str]"]):
    """Tập tên operator (Call.op) dùng trong cây — phục vụ đếm operator unique cho
    Power Pool eligibility (sub-project A) và phát hiện single-dataset alpha
    (sub-project D, operator inst_pnl/convert tính là dùng dataset pv1)."""

    def visit(self, node: Node) -> set[str]:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> set[str]:
        return set()

    def visit_field(self, node: Field) -> set[str]:
        return set()

    def visit_call(self, node: Call) -> set[str]:
        result: set[str] = {node.op}
        for c in node.children():
            result |= c.accept(self)
        return result
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/unit/test_lang_visitors.py -v`
Expected: PASS (4/4)

- [ ] **Step 5: Commit**

```bash
git add src/lang/visitors.py tests/unit/test_lang_visitors.py
git commit -m "feat(lang): them OperatorCollector visitor (tap ten operator dung trong AST)"
```

---

### Task 2: `src/scoring/dataset_usage.py` — phát hiện Single Dataset Alpha

**Files:**
- Create: `src/scoring/dataset_usage.py`
- Test: `tests/unit/test_dataset_usage.py`

**Interfaces:**
- Consumes: `parse_expression(expr: str) -> Node` (`src/lang/parser.py`), `FieldCollector`,
  `OperatorCollector` (Task 1).
- Produces: `dataset_of_alpha(expr: str, field_dataset: dict[str, str]) -> str | None`,
  `is_single_dataset_alpha(expr: str, field_dataset: dict[str, str]) -> bool` — hàm thuần,
  chưa có caller trong plan này (sub-project A/LLM prompt sẽ dùng sau).

- [ ] **Step 1: Viết test FAIL**

Tạo file `tests/unit/test_dataset_usage.py`:

```python
"""Test phát hiện Single Dataset Alpha (sub-project D) — hàm thông tin thuần, KHÔNG gate nộp."""

from __future__ import annotations

from src.scoring.dataset_usage import dataset_of_alpha, is_single_dataset_alpha


def test_single_dataset_khi_moi_field_cung_dataset():
    fd = {"close": "pv1", "open": "pv1"}
    assert dataset_of_alpha("rank(add(close, open))", fd) == "pv1"
    assert is_single_dataset_alpha("rank(add(close, open))", fd) is True


def test_khong_single_dataset_khi_2_dataset_khac_nhau():
    fd = {"close": "pv1", "eps": "fundamental6"}
    assert dataset_of_alpha("rank(add(close, eps))", fd) is None
    assert is_single_dataset_alpha("rank(add(close, eps))", fd) is False


def test_grouping_field_bi_bo_qua_khi_tinh_dataset():
    fd = {"close": "pv1"}
    assert dataset_of_alpha("group_rank(close, sector)", fd) == "pv1"


def test_field_khong_ro_dataset_tra_none():
    fd = {"close": "pv1"}
    assert dataset_of_alpha("rank(add(close, unknown_field))", fd) is None


def test_inst_pnl_operator_them_pv1_lam_thanh_2_dataset():
    fd = {"eps": "fundamental6"}
    assert dataset_of_alpha("inst_pnl(eps, 5)", fd) is None


def test_inst_pnl_khop_pv1_van_la_single_dataset():
    fd = {"close": "pv1"}
    assert dataset_of_alpha("inst_pnl(close, 5)", fd) == "pv1"
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/unit/test_dataset_usage.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.scoring.dataset_usage'`

- [ ] **Step 3: Cài tối thiểu**

Tạo `src/scoring/dataset_usage.py`:

```python
"""Phát hiện 'Single Dataset Alpha' (WQ Brain: mọi field trừ 6 grouping field đến từ ĐÚNG 1
dataset) — dùng để gắn nhãn/tham khảo khi sinh alpha, KHÔNG dùng để gate nộp (WQ Brain tự áp
ngưỡng nới lỏng cho loại này qua `is.checks`, xem
docs/worldquantbrain/docs/consultant-information/single-dataset-alphas.md và đính chính
sub-project B trong docs/superpowers/specs/2026-07-02-submission-compliance-roadmap-design.md)."""

from __future__ import annotations

from src.lang.parser import parse_expression
from src.lang.visitors import FieldCollector, OperatorCollector

# 6 grouping field được miễn trừ khi tính dataset (theo single-dataset-alphas.md — khác danh
# sách Power Pool có thêm 'currency', xem lưu ý trong roadmap spec).
_GROUPING_FIELDS = {"country", "exchange", "market", "sector", "industry", "subindustry"}
# 2 operator luôn tính là dùng dataset 'pv1' (theo tài liệu).
_PV1_OPERATORS = {"inst_pnl", "convert"}


def dataset_of_alpha(expr: str, field_dataset: dict[str, str]) -> str | None:
    """Trả dataset_id DUY NHẤT nếu `expr` là single-dataset; `None` nếu dùng >1 dataset hoặc
    có field không xác định được dataset trong `field_dataset`."""
    node = parse_expression(expr)
    fields = FieldCollector().visit(node)
    operators = OperatorCollector().visit(node)

    datasets: set[str] = set()
    for field_id in fields:
        if field_id in _GROUPING_FIELDS:
            continue
        dataset_id = field_dataset.get(field_id)
        if dataset_id is None:
            return None
        datasets.add(dataset_id)
    if operators & _PV1_OPERATORS:
        datasets.add("pv1")
    if len(datasets) == 1:
        return next(iter(datasets))
    return None


def is_single_dataset_alpha(expr: str, field_dataset: dict[str, str]) -> bool:
    return dataset_of_alpha(expr, field_dataset) is not None
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/unit/test_dataset_usage.py -v`
Expected: PASS (6/6)

- [ ] **Step 5: Chạy toàn bộ suite, xác nhận không vỡ gì**

Run: `venv/Scripts/python -m pytest tests/ -q`
Expected: PASS hết, trừ 1 fail có sẵn không liên quan (`test_make_engine_postgres_backend`).

- [ ] **Step 6: Commit**

```bash
git add src/scoring/dataset_usage.py tests/unit/test_dataset_usage.py
git commit -m "feat(scoring): them dataset_of_alpha/is_single_dataset_alpha (sub-project D)"
```

---

## Self-Review (đã chạy)

- **Spec coverage**: mục "Sub-project D" trong roadmap spec — việc 1 (phát hiện single
  dataset, trừ 5 grouping field) = Task 2; việc inst_pnl/convert=pv1 = Task 2; việc "áp ngưỡng
  Last-2Y-Sharpe" — CỐ Ý bỏ, vì đã xác nhận (đính chính sub-project B) WQ tự áp ngưỡng đó qua
  `status`, tool không cần tự gate.
- **Placeholder scan**: sạch.
- **Type consistency**: `dataset_of_alpha`/`is_single_dataset_alpha` chữ ký nhất quán giữa mô
  tả Interfaces và code; `OperatorCollector` cùng pattern `FieldCollector` (visit trả `set[str]`).
