# Hợp nhất 2 bộ sinh biểu thức LLM qua `expr_synth` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tách phần lõi dựng prompt + vòng repair của hai bộ sinh biểu thức LLM (`generator.py`, `translator.py`) ra module chung `src/llm/expr_synth.py`, và thêm `autowrap_vector_fields` sửa tất định lỗi VECTOR→MATRIX.

**Architecture:** Module mới gồm các hàm thuần (nhận repo/prefilter làm tham số). `AlphaTranslator` và `LLMAlphaGenerator` giữ vai trò công khai, chỉ uỷ thác phần lõi. Auto-wrap chạy trong vòng repair, TRƯỚC `prefilter.check`, bằng AST (`src.generation.ast_utils`).

**Tech Stack:** Python 3.12, pytest, loguru, dataclass AST có sẵn.

## Global Constraints

- TDD bắt buộc: viết test đỏ trước, code tối thiểu cho xanh, rồi commit.
- Code, comment, commit message bằng tiếng Việt (giữ đủ dấu).
- Mỗi task một commit; không bỏ qua hook.
- KHÔNG đụng: cách dựng prefilter, path sinh ý tưởng (`generate_ideas`, `build_ideas_system_prompt`, `_idea_field_context`, `_feedback_context`), `loop.py`, `refiner.py`, `max_depth`.
- Hành vi công khai `AlphaTranslator.translate` phải bất biến (chỉ dời ruột code).
- Spec nguồn: `docs/superpowers/specs/2026-06-20-unify-llm-expr-synth-design.md`.

---

## File Structure

- **Create** `src/llm/expr_synth.py` — module chung: `autowrap_vector_fields`, `build_symbol_context`, `build_syntax_constraints`, `suggest_fields`, `repair_to_expression`, hằng `MAX_REPAIR_ATTEMPTS`, `FEWSHOT_EXAMPLES`, helper nội bộ `_tokens/_relevant_fields/_field_type_context/_load_cached`.
- **Create** `tests/test_expr_synth.py` — test cho module chung.
- **Modify** `src/llm/translator.py` — uỷ thác sang `expr_synth`; xoá method đã dời.
- **Modify** `tests/test_translator.py` — di trú `test_suggest_fields` sang gọi `expr_synth`.
- **Modify** `src/llm/generator.py` — `_generate_one` dựng prompt qua `expr_synth` + gọi `repair_to_expression`.
- **Modify** `tests/test_generator.py` — thêm test prompt có FIELD TYPES + auto-wrap trong `_generate_one`.

---

### Task 1: `autowrap_vector_fields` + khung module `expr_synth`

**Files:**
- Create: `src/llm/expr_synth.py`
- Test: `tests/test_expr_synth.py`

**Interfaces:**
- Consumes: `src.generation.ast_utils` (`Leaf`, `Node`, `parse_expression`, `to_expression`).
- Produces:
  - `MAX_REPAIR_ATTEMPTS: int = 3`
  - `FEWSHOT_EXAMPLES: list[str]`
  - `autowrap_vector_fields(expr: str, field_types: dict[str,str] | None, matrix_only_ops: set[str] | None) -> str`

- [ ] **Step 1: Viết test đỏ** — tạo `tests/test_expr_synth.py`:

```python
"""Test module chung expr_synth: auto-wrap, dựng prompt, vòng repair."""

from __future__ import annotations

from src.llm import expr_synth


def test_autowrap_boc_vec_avg_leaf_vector_duoi_matrix_op():
    out = expr_synth.autowrap_vector_fields(
        "ts_zscore(svec, 20)",
        field_types={"svec": "VECTOR"},
        matrix_only_ops={"ts_zscore"},
    )
    assert out == "ts_zscore(vec_avg(svec), 20)"


def test_autowrap_khong_dung_field_matrix_hay_so():
    out = expr_synth.autowrap_vector_fields(
        "ts_zscore(close, 20)",
        field_types={"close": "MATRIX"},
        matrix_only_ops={"ts_zscore"},
    )
    assert out == "ts_zscore(close, 20)"


def test_autowrap_idempotent_khi_da_co_vec_avg():
    out = expr_synth.autowrap_vector_fields(
        "rank(vec_avg(svec))",
        field_types={"svec": "VECTOR"},
        matrix_only_ops={"rank"},
    )
    assert out == "rank(vec_avg(svec))"


def test_autowrap_boc_nhieu_leaf_vector():
    out = expr_synth.autowrap_vector_fields(
        "add(rank(v1), ts_delta(v2, 5))",
        field_types={"v1": "VECTOR", "v2": "VECTOR"},
        matrix_only_ops={"rank", "ts_delta"},
    )
    assert out == "add(rank(vec_avg(v1)), ts_delta(vec_avg(v2), 5))"


def test_autowrap_no_op_khi_thieu_du_lieu_kieu():
    assert expr_synth.autowrap_vector_fields("rank(svec)", None, None) == "rank(svec)"
    assert expr_synth.autowrap_vector_fields("rank(svec)", {}, set()) == "rank(svec)"


def test_autowrap_giu_nguyen_khi_khong_parse_duoc():
    bad = "rank(svec"  # ngoặc lệch
    assert expr_synth.autowrap_vector_fields(bad, {"svec": "VECTOR"}, {"rank"}) == bad
```

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_expr_synth.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.expr_synth'`.

- [ ] **Step 3: Viết module tối thiểu** — tạo `src/llm/expr_synth.py`:

```python
"""Lõi dùng chung của hai bộ sinh biểu thức LLM (generator + translator).

Gom phần trùng lặp: dựng ngữ cảnh prompt (symbol + field type), vòng
prefilter-repair, và auto-wrap field VECTOR. Hai lớp công khai
(LLMAlphaGenerator, AlphaTranslator) chỉ uỷ thác phần lõi cho module này.
"""

from __future__ import annotations

from loguru import logger

from src.generation.ast_utils import Leaf, Node, parse_expression, to_expression

MAX_REPAIR_ATTEMPTS = 3
MAX_FIELDS_IN_PROMPT = 40

# Ví dụ minh hoạ CÚ PHÁP, đa dạng cấu trúc, tránh khung kinh điển trùng Alpha101.
FEWSHOT_EXAMPLES = [
    "ts_decay_linear(rank(ts_std_dev(returns, 20)), 5)",
    "group_neutralize(ts_zscore(vwap, 60), industry)",
    "rank(divide(ts_mean(volume, 10), ts_mean(volume, 60)))",
    "ts_rank(ts_corr(close, volume, 20), 120)",
]


def autowrap_vector_fields(expr: str, field_types, matrix_only_ops) -> str:
    """Bọc vec_avg() quanh leaf field VECTOR bị đưa thẳng vào matrix-only op.

    Khớp ĐÚNG luật pre_filter._check_symbols: với Node có op ∈ matrix_only_ops,
    con TRỰC TIẾP là Leaf field có field_types[name]=='VECTOR' -> thay bằng
    vec_avg(leaf). Thiếu dữ liệu kiểu -> trả nguyên. Không parse được -> trả
    nguyên để prefilter báo lỗi (không nuốt lỗi).
    """
    if not field_types or not matrix_only_ops:
        return expr
    try:
        tree = parse_expression(expr)
    except ValueError:
        return expr

    def _walk(node):
        if isinstance(node, Leaf):
            return node
        wrap_here = node.op in matrix_only_ops
        new_children = []
        for child in node.children:
            child = _walk(child)
            if (
                wrap_here
                and isinstance(child, Leaf)
                and not isinstance(child.value, (int, float))
                and field_types.get(str(child.value)) == "VECTOR"
            ):
                child = Node("vec_avg", [child])
            new_children.append(child)
        node.children = new_children
        return node

    return to_expression(_walk(tree))
```

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_expr_synth.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/llm/expr_synth.py tests/test_expr_synth.py
git commit -m "feat(llm): autowrap_vector_fields sửa tất định lỗi VECTOR→MATRIX"
```

---

### Task 2: Dựng ngữ cảnh prompt dùng chung

**Files:**
- Modify: `src/llm/expr_synth.py`
- Test: `tests/test_expr_synth.py`

**Interfaces:**
- Consumes: `MAX_FIELDS_IN_PROMPT`, `FEWSHOT_EXAMPLES` (Task 1).
- Produces:
  - `build_symbol_context(field_repo, operator_repo, prefilter, scope, relevance_text="") -> str`
  - `build_syntax_constraints(prefilter) -> str`
  - `suggest_fields(field_repo, scope, bad_field, limit=5) -> list[str]`

- [ ] **Step 1: Viết test đỏ** — thêm vào `tests/test_expr_synth.py`:

```python
import re

from src.simulation.pre_filter import PreFilter
from tests.fakes import FakeSymbolRepo


class _Field:
    def __init__(self, id, type="MATRIX", description="", dataset_id=""):
        self.id = id
        self.type = type
        self.description = description
        self.dataset_id = dataset_id


class _FieldRepo:
    def __init__(self, fields):
        self._fields = fields

    def load_cached(self, region=None, universe=None, delay=None):
        return self._fields


def test_build_symbol_context_chen_quy_tac_vector():
    repo = _FieldRepo([_Field("close", "MATRIX"), _Field("svec", "VECTOR")])
    ops = FakeSymbolRepo(["rank", "ts_zscore", "vec_avg", "vec_sum"])
    pf = PreFilter(known_operators={"rank"}, known_fields={"close", "svec"})
    out = expr_synth.build_symbol_context(repo, ops, pf, scope=None, relevance_text="svec")
    assert "FIELD TYPES" in out
    assert "VECTOR" in out
    assert "vec_avg" in out
    assert "ts_zscore(vec_avg(svec)" in out


def test_build_symbol_context_khong_vector_thi_khong_chen_quy_tac():
    repo = _FieldRepo([_Field("close", "MATRIX")])
    ops = FakeSymbolRepo(["rank"])
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    out = expr_synth.build_symbol_context(repo, ops, pf, scope=None)
    assert "QUY TAC VECTOR" not in out


def test_build_syntax_constraints_lay_gioi_han_tu_prefilter():
    pf = PreFilter(known_operators={"rank"}, max_depth=6, max_nodes=30)
    out = expr_synth.build_syntax_constraints(pf)
    assert "6" in out and "30" in out
    low = out.lower()
    assert "vị trí" in low and ("key=value" in low or "std=" in low)


def test_suggest_fields_uu_tien_cung_tien_to_dataset():
    repo = _FieldRepo([
        _Field("opt6_1dorhv_real"), _Field("opt6_close"),
        _Field("news12_sent"), _Field("close"),
    ])
    out = expr_synth.suggest_fields(repo, scope=None, bad_field="opt6_1dorhv")
    assert "opt6_1dorhv_real" in out
    assert out[0].startswith("opt6_")
```

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_expr_synth.py -k "symbol_context or syntax_constraints or suggest_fields" -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_symbol_context'`.

- [ ] **Step 3: Viết code** — thêm vào `src/llm/expr_synth.py` (sau `autowrap_vector_fields`):

```python
import re


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _load_cached(field_repo, scope):
    """Nạp fields: có scope -> load_cached(**scope); không -> load_cached().
    Bao try/except chữ ký để tương thích repo cũ không nhận tham số."""
    if scope:
        return list(field_repo.load_cached(**scope))
    try:
        return list(field_repo.load_cached())
    except TypeError:
        return list(field_repo.load_cached(None, None, None))


def _relevant_fields(cached_fields, text: str) -> list[str]:
    """Xếp hạng fields theo độ liên quan với text (hypothesis/idea/mô tả), cắt
    MAX_FIELDS_IN_PROMPT. Text rỗng -> giữ thứ tự gốc (tương thích)."""
    text_low = (text or "").lower()
    text_tokens = _tokens(text_low)
    scored = []
    for idx, f in enumerate(cached_fields):
        fid = getattr(f, "id", None)
        if not fid:
            continue
        dataset = (getattr(f, "dataset_id", "") or "").lower()
        score = 0
        if fid.lower() in text_low:
            score += 100
        if dataset and dataset in text_low:
            score += 20
        score += len(_tokens(fid + " " + (getattr(f, "description", "") or "")) & text_tokens)
        scored.append((score, idx, fid))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [fid for _, _, fid in scored[:MAX_FIELDS_IN_PROMPT]]


def _field_type_context(selected_fields) -> str:
    by_type: dict[str, list[str]] = {}
    for field in selected_fields:
        fid = getattr(field, "id", None)
        ftype = (getattr(field, "type", "") or "").strip().upper()
        if fid and ftype:
            by_type.setdefault(ftype, []).append(fid)
    if not by_type:
        return ""

    lines = ["FIELD TYPES (dung de tranh sai kieu input):"]
    for ftype in ("MATRIX", "VECTOR", "GROUP", "EVENT"):
        values = by_type.get(ftype)
        if values:
            lines.append(f"- {ftype}: {', '.join(values[:20])}")
    vector_fields = by_type.get("VECTOR") or []
    if vector_fields:
        sample = vector_fields[0]
        lines.append(
            "QUY TAC VECTOR: khong goi truc tiep ts_zscore/ts_mean/ts_rank/rank tren VECTOR field. "
            "Hay giam VECTOR ve MATRIX bang vec_avg(field) hoac vec_sum(field) truoc. "
            f"Vi du: ts_zscore(vec_avg({sample}), 20), rank(vec_avg({sample}))."
        )
    return "\n".join(lines)


def build_symbol_context(field_repo, operator_repo, prefilter, scope, relevance_text: str = "") -> str:
    operators = [o.name for o in operator_repo.load_cached() if getattr(o, "name", None)]
    cached_fields = _load_cached(field_repo, scope)
    fields = _relevant_fields(cached_fields, relevance_text)
    field_by_id = {getattr(f, "id", None): f for f in cached_fields if getattr(f, "id", None)}
    selected_fields = [field_by_id[fid] for fid in fields if fid in field_by_id]
    type_context = _field_type_context(selected_fields)
    op_line = ", ".join(operators[:80]) or "rank, ts_delta, ts_mean, group_neutralize, ts_corr"
    field_line = ", ".join(fields) or "close, open, high, low, volume, vwap, returns"
    examples = "\n".join(f"- {e}" for e in FEWSHOT_EXAMPLES)
    context = (
        f"OPERATORS hợp lệ: {op_line}\n"
        f"FIELDS khả dụng: {field_line}\n"
        "GROUPS cho neutralize: market, sector, industry, subindustry\n"
        f"Ví dụ alpha hợp lệ:\n{examples}"
    )
    if type_context:
        context += f"\n{type_context}"
    return context


def build_syntax_constraints(prefilter) -> str:
    """Ràng buộc cú pháp suy ra từ pre-filter để biểu thức qua lọc ngay."""
    max_depth = getattr(prefilter, "max_depth", 6)
    max_nodes = getattr(prefilter, "max_nodes", 30)
    return (
        "RÀNG BUỘC bắt buộc để qua bộ lọc cú pháp:\n"
        f"- Độ sâu lồng nhau TỐI ĐA {max_depth}; tổng số node TỐI ĐA {max_nodes}. "
        "Ưu tiên biểu thức GỌN và NÔNG, tránh lồng quá nhiều tầng.\n"
        "- CHỈ dùng đối số theo VỊ TRÍ. TUYỆT ĐỐI không dùng đối số có tên kiểu "
        "key=value (vd viết winsorize(x, 3) chứ KHÔNG viết winsorize(x, std=3)).\n"
        "- Đối số chỉ là field/group đã liệt kê, biểu thức con, hoặc SỐ NGUYÊN.\n"
    )


def suggest_fields(field_repo, scope, bad_field: str, limit: int = 5) -> list[str]:
    """Field thật gần 'bad_field' nhất: ưu tiên cùng tiền tố dataset, rồi trùng token."""
    cached = _load_cached(field_repo, scope)
    bad_low = (bad_field or "").lower()
    bad_prefix = bad_low.split("_", 1)[0]
    bad_tokens = set(re.findall(r"[a-z0-9]+", bad_low))
    scored = []
    for f in cached:
        fid = getattr(f, "id", None)
        if not fid:
            continue
        fl = fid.lower()
        score = 0
        if bad_prefix and fl.startswith(bad_prefix):
            score += 50
        score += len(set(re.findall(r"[a-z0-9]+", fl)) & bad_tokens)
        if score:
            scored.append((score, fid))
    scored.sort(key=lambda t: -t[0])
    return [fid for _, fid in scored[:limit]]
```

Lưu ý: gộp `import re` lên đầu file cùng các import khác (không để rải rác).

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_expr_synth.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add src/llm/expr_synth.py tests/test_expr_synth.py
git commit -m "feat(llm): dựng ngữ cảnh prompt + suggest_fields dùng chung trong expr_synth"
```

---

### Task 3: Vòng repair dùng chung `repair_to_expression`

**Files:**
- Modify: `src/llm/expr_synth.py`
- Test: `tests/test_expr_synth.py`

**Interfaces:**
- Consumes: `autowrap_vector_fields`, `suggest_fields`, `MAX_REPAIR_ATTEMPTS`; `src.llm.jsonutil.extract_json`; `src.simulation.simulator.extract_rejected_field`.
- Produces: `repair_to_expression(deepseek, prefilter, field_repo, scope, system, user, task) -> str | None`

- [ ] **Step 1: Viết test đỏ** — thêm vào `tests/test_expr_synth.py`:

```python
import json

from tests.fakes import FakeDeepSeek


def test_repair_tra_expr_khi_pass_lan_dau():
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    ds = FakeDeepSeek([json.dumps({"expression": "rank(close)"})])
    out = expr_synth.repair_to_expression(
        ds, pf, _FieldRepo([_Field("close")]), None, "sys", "usr", task=None
    )
    assert out == "rank(close)"
    assert len(ds.calls) == 1


def test_repair_autowrap_pass_khong_goi_lai_llm():
    pf = PreFilter(
        known_operators={"ts_zscore", "vec_avg"},
        known_fields={"svec"},
        field_types={"svec": "VECTOR"},
        matrix_only_ops={"ts_zscore"},
    )
    ds = FakeDeepSeek([json.dumps({"expression": "ts_zscore(svec, 20)"})])
    out = expr_synth.repair_to_expression(
        ds, pf, _FieldRepo([_Field("svec", "VECTOR")]), None, "sys", "usr", task=None
    )
    assert out == "ts_zscore(vec_avg(svec), 20)"
    assert len(ds.calls) == 1  # auto-wrap sửa, không cần round-trip thêm


def test_repair_them_hint_field_khi_field_bia():
    pf = PreFilter(known_operators={"rank"}, known_fields={"opt6_real"})
    ds = FakeDeepSeek([
        json.dumps({"expression": "rank(opt6_bia)"}),   # field bịa -> fail
        json.dumps({"expression": "rank(opt6_real)"}),  # sửa lại hợp lệ
    ])
    out = expr_synth.repair_to_expression(
        ds, pf, _FieldRepo([_Field("opt6_real")]), None, "sys", "usr", task=None
    )
    assert out == "rank(opt6_real)"
    # lượt user thứ 2 phải chứa hint field thật gần nhất
    assert "opt6_real" in ds.calls[1][1]


def test_repair_tra_none_khi_het_retry():
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    ds = FakeDeepSeek([json.dumps({"expression": "bad_op(x)"})] * 5)
    out = expr_synth.repair_to_expression(
        ds, pf, _FieldRepo([_Field("close")]), None, "sys", "usr", task=None
    )
    assert out is None
    assert len(ds.calls) == expr_synth.MAX_REPAIR_ATTEMPTS
```

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_expr_synth.py -k repair -q`
Expected: FAIL — `AttributeError: ... 'repair_to_expression'`.

- [ ] **Step 3: Viết code** — thêm import lên đầu `src/llm/expr_synth.py`:

```python
from src.llm.jsonutil import extract_json
from src.simulation.simulator import extract_rejected_field
```

và thêm hàm:

```python
def repair_to_expression(deepseek, prefilter, field_repo, scope, system, user, task) -> str | None:
    """Vòng LLM -> auto-wrap -> prefilter.check -> retry kèm hint field thay thế."""
    field_types = getattr(prefilter, "field_types", None)
    matrix_only = getattr(prefilter, "matrix_only_ops", None)
    for attempt in range(MAX_REPAIR_ATTEMPTS):
        data = extract_json(deepseek.complete(system, user, json_mode=True, task=task))
        expr = data.get("expression") if isinstance(data, dict) else None
        if not expr:
            user = 'Trả ĐÚNG JSON {"expression": "..."}.'
            continue
        expr = autowrap_vector_fields(expr, field_types, matrix_only)
        ok, reason = prefilter.check(expr)
        if ok:
            return expr
        logger.info("LLM expr lỗi (lần {}): {} — {}", attempt + 1, expr, reason)
        bad = extract_rejected_field(reason)
        hint = ""
        if bad:
            suggestions = suggest_fields(field_repo, scope, bad)
            if suggestions:
                hint = f" Field có thật gần nhất: {', '.join(suggestions)}."
        user = f'Biểu thức "{expr}" bị lỗi: {reason}.{hint} Sửa lại, trả JSON.'
    return None
```

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_expr_synth.py -q`
Expected: PASS (14 passed).

- [ ] **Step 5: Commit**

```bash
git add src/llm/expr_synth.py tests/test_expr_synth.py
git commit -m "feat(llm): vòng repair_to_expression dùng chung (auto-wrap + hint field)"
```

---

### Task 4: Refactor `translator.py` uỷ thác sang `expr_synth`

**Files:**
- Modify: `src/llm/translator.py`
- Modify: `tests/test_translator.py`

**Interfaces:**
- Consumes: `expr_synth.build_symbol_context`, `build_syntax_constraints`, `repair_to_expression`, `suggest_fields`, `FEWSHOT_EXAMPLES`, `MAX_REPAIR_ATTEMPTS`.
- Produces: `AlphaTranslator.translate`/`_to_expression` giữ chữ ký + output bất biến.

- [ ] **Step 1: Di trú test gọi private** — trong `tests/test_translator.py`, thay `test_suggest_fields_tra_field_that_gan_nhat` thành gọi `expr_synth`:

```python
def test_suggest_fields_tra_field_that_gan_nhat():
    """Field thật gần 'bad_field': cùng tiền tố dataset + trùng token đứng trước."""
    from src.llm import expr_synth
    fields = FakeSymbolRepo(["opt6_1dorhv_real", "opt6_close", "news12_sent", "close"])
    out = expr_synth.suggest_fields(fields, None, "opt6_1dorhv")
    assert "opt6_1dorhv_real" in out
    assert out[0].startswith("opt6_")
```

- [ ] **Step 2: Chạy toàn bộ test translator, xác nhận đỏ ở đúng chỗ refactor**

Run: `python -m pytest tests/test_translator.py -q`
Expected: PASS hiện tại (chưa refactor code) — test mới vẫn đỏ vì `expr_synth.suggest_fields` đã có (Task 2) nên thực ra XANH. Nếu đỏ, đọc lỗi và sửa import.

(Ghi chú: bước này chủ yếu chốt test mới chạy được trước khi đụng `translator.py`.)

- [ ] **Step 3: Refactor `translator.py`** — sửa import đầu file:

```python
from src.llm import expr_synth
from src.llm.hypothesis import Hypothesis
```

Xoá các hằng/khối đã dời và các method `_tokens`, `_relevant_fields`, `_field_type_context`, `_symbol_context`, `_syntax_constraints`, `_suggest_fields`. Thay `MAX_FIELDS_IN_PROMPT`/`MAX_REPAIR_ATTEMPTS`/`FEWSHOT_EXAMPLES` cục bộ bằng tham chiếu từ `expr_synth` (xoá bản trùng trong file). Thay thân `_to_expression`:

```python
    def _to_expression(self, description: str, relevance_text: str = "") -> str | None:
        system = (
            "Bạn là chuyên gia viết biểu thức FASTEXPR trên WorldQuant BRAIN.\n"
            f"{expr_synth.build_symbol_context(self.field_repo, self.operator_repo, self.prefilter, self._scope, relevance_text or description)}\n"
            f"{self._avoid_context()}"
            f"{expr_synth.build_syntax_constraints(self.prefilter)}"
            "Dịch MÔ TẢ thành MỘT biểu thức FASTEXPR dùng đúng operators/fields được liệt kê. "
            'Trả JSON {"expression": "..."}.'
        )
        user = f"MÔ TẢ: {description}\nViết biểu thức FASTEXPR."
        return expr_synth.repair_to_expression(
            self.deepseek, self.prefilter, self.field_repo, self._scope, system, user, task="translate"
        )
```

Giữ nguyên: `AlphaCandidate`, `__init__`, `set_avoid_subtrees`, `set_scope`, `_avoid_context`, `_describe`, `translate`. Giữ import `re` chỉ khi còn dùng (nếu không, xoá). Giữ import `extract_json`/`extract_rejected_field` chỉ khi còn dùng trong file (nếu không, xoá để khỏi lint thừa).

- [ ] **Step 4: Chạy test translator, xác nhận xanh**

Run: `python -m pytest tests/test_translator.py -q`
Expected: PASS (toàn bộ). Đặc biệt `test_prompt_neu_field_type_va_cach_giam_vector_truoc_matrix_ops`, `test_prompt_neu_gioi_han_depth_node_lay_tu_prefilter`, `test_translate_*` còn xanh (output prompt bất biến).

- [ ] **Step 5: Commit**

```bash
git add src/llm/translator.py tests/test_translator.py
git commit -m "refactor(llm): translator uỷ thác dựng prompt + repair cho expr_synth"
```

---

### Task 5: Refactor `generator.py` — fix VECTOR + auto-wrap

**Files:**
- Modify: `src/llm/generator.py`
- Modify: `tests/test_generator.py`

**Interfaces:**
- Consumes: `expr_synth.build_symbol_context`, `build_syntax_constraints`, `repair_to_expression`.
- Produces: `LLMAlphaGenerator._generate_one(idea) -> str | None` (chữ ký giữ nguyên), prompt nay có FIELD TYPES + QUY TẮC VECTOR.

- [ ] **Step 1: Viết test đỏ** — thêm vào `tests/test_generator.py`:

```python
import json

from src.llm.generator import LLMAlphaGenerator
from src.simulation.pre_filter import PreFilter
from tests.fakes import FakeDeepSeek, FakeSymbolRepo


class _TField:
    def __init__(self, id, type="MATRIX"):
        self.id = id
        self.type = type
        self.description = ""
        self.dataset_id = ""


class _TFieldRepo:
    def __init__(self, fields):
        self._fields = fields

    def load_cached(self, region=None, universe=None, delay=None):
        return self._fields


def _gen(ds, fields, pf):
    ops = FakeSymbolRepo(["rank", "ts_zscore", "vec_avg"])
    return LLMAlphaGenerator(ds, _TFieldRepo(fields), ops, pf)


def test_generate_one_prompt_co_quy_tac_vector():
    pf = PreFilter(
        known_operators={"rank", "ts_zscore", "vec_avg"},
        known_fields={"svec"},
        field_types={"svec": "VECTOR"},
        matrix_only_ops={"rank", "ts_zscore"},
    )
    ds = FakeDeepSeek([json.dumps({"expression": "rank(vec_avg(svec))"})])
    _gen(ds, [_TField("svec", "VECTOR")], pf)._generate_one("dùng svec")
    system = ds.calls[0][0]
    assert "VECTOR" in system and "vec_avg" in system


def test_generate_one_autowrap_field_vector():
    pf = PreFilter(
        known_operators={"ts_zscore", "vec_avg"},
        known_fields={"svec"},
        field_types={"svec": "VECTOR"},
        matrix_only_ops={"ts_zscore"},
    )
    ds = FakeDeepSeek([json.dumps({"expression": "ts_zscore(svec, 20)"})])
    out = _gen(ds, [_TField("svec", "VECTOR")], pf)._generate_one("svec")
    assert out == "ts_zscore(vec_avg(svec), 20)"
    assert len(ds.calls) == 1  # auto-wrap sửa ngay, không round-trip thêm
```

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_generator.py -k "generate_one" -q`
Expected: FAIL — prompt cũ không chứa "VECTOR"/`vec_avg`, và `_generate_one` cũ không auto-wrap (`ts_zscore(svec, 20)` bị prefilter loại → trả None).

- [ ] **Step 3: Refactor `generator.py`** — thêm import:

```python
from src.llm import expr_synth
```

**GIỮ** `build_system_prompt` (có test `tests/test_llm.py:82` gọi nó) nhưng đổi ruột sang `expr_synth`, thêm tham số `relevance_text` mặc định rỗng (tương thích ngược: test gọi không tham số). `_generate_one` gọi lại nó với `idea` làm relevance:

```python
    def build_system_prompt(self, relevance_text: str = "") -> str:
        context = expr_synth.build_symbol_context(
            self.field_repo, self.operator_repo, self.prefilter, None, relevance_text
        )
        constraints = expr_synth.build_syntax_constraints(self.prefilter)
        return (
            "Bạn là chuyên gia thiết kế Alpha trên WorldQuant BRAIN, viết biểu thức FASTEXPR.\n"
            "Cú pháp: hàm(đối_số, ...), toán tử + - * /, rank chuẩn hóa cross-sectional, "
            "tiền tố ts_ là chuỗi thời gian với tham số cửa sổ là số nguyên.\n"
            f"{context}\n{constraints}{self._feedback_context()}"
            'Luôn trả về JSON đúng định dạng: {"expression": "...", "rationale": "..."}. '
            "Chỉ dùng operators và fields được liệt kê."
        )

    def _generate_one(self, idea: str) -> str | None:
        system = self.build_system_prompt(idea)
        user = f'Ý tưởng alpha: "{idea}". Sinh MỘT biểu thức FASTEXPR. Trả JSON.'
        return expr_synth.repair_to_expression(
            self.deepseek, self.prefilter, self.field_repo, None, system, user, task=None
        )
```

Giữ nguyên `generate`, `generate_ideas`, `build_ideas_system_prompt`, `_idea_field_context`, `_feedback_context`, `_parse_ideas`. Lưu ý: sau đổi, `FEWSHOT_EXAMPLES` cục bộ của generator (chỉ dùng trong `build_system_prompt` cũ) sẽ thừa — dọn ở Task 6.

- [ ] **Step 4: Chạy test generator, xác nhận xanh**

Run: `python -m pytest tests/test_generator.py -q`
Expected: PASS (gồm 2 test mới + các test cũ về ý tưởng).

- [ ] **Step 5: Commit**

```bash
git add src/llm/generator.py tests/test_generator.py
git commit -m "fix(llm): generator dùng expr_synth — prompt có FIELD TYPES + auto-wrap VECTOR"
```

---

### Task 6: Dọn hằng trùng, regression toàn bộ

**Files:**
- Modify: `src/llm/generator.py`, `src/llm/translator.py` (dọn dead code nếu còn)

**Interfaces:**
- Consumes: tất cả task trước.
- Produces: suite xanh, không còn hằng/import thừa.

- [ ] **Step 1: Soát hằng/import thừa**

Run: `grep -rn "MAX_REPAIR_ATTEMPTS\|FEWSHOT_EXAMPLES\|MAX_FIELDS_IN_PROMPT" src/llm/generator.py src/llm/translator.py`
Với mỗi hằng còn khai báo trong generator/translator nhưng KHÔNG còn dùng trong file đó → xoá khai báo. `FEWSHOT_EXAMPLES` trong generator: chỉ xoá nếu `grep -n "FEWSHOT_EXAMPLES" src/llm/generator.py` cho thấy không còn nơi dùng (vd `build_ideas_system_prompt`); nếu còn dùng cho sinh ý tưởng → GIỮ.

- [ ] **Step 2: Soát import thừa** (không có pyflakes — rà thủ công)

Với mỗi import ở đầu `src/llm/translator.py` và `src/llm/generator.py`, kiểm symbol còn được dùng trong chính file đó:
Run ví dụ: `grep -n "parse_expression\|iter_leaves\|extract_json\|extract_rejected_field\|\bre\." src/llm/generator.py`
Symbol nào không còn xuất hiện ngoài dòng import → xoá dòng import đó. Sau khi sửa, đảm bảo file vẫn biên dịch:
Run: `python -m py_compile src/llm/generator.py src/llm/translator.py src/llm/expr_synth.py`
Expected: không lỗi.

- [ ] **Step 3: Chạy toàn bộ test suite**

Run: `python -m pytest -q`
Expected: PASS toàn bộ (bao gồm `test_expr_synth`, `test_translator`, `test_generator`, `test_pre_filter`, `test_refiner`, `test_llm_task_routing`).

- [ ] **Step 4: Commit (nếu có dọn dẹp)**

```bash
git add src/llm/generator.py src/llm/translator.py
git commit -m "chore(llm): dọn hằng/import trùng sau khi tách expr_synth"
```

---

## Self-Review (đã rà khi viết plan)

- **Spec coverage:** module `expr_synth` (T1–T3) ↔ spec mục 3; auto-wrap ↔ mục 3 "Auto-wrap"; refactor translator/generator (T4–T5) ↔ mục 4; test ↔ mục 6; dọn dẹp ↔ mục 4. Đủ.
- **Phi mục tiêu:** không đụng prefilter construction, path ý tưởng, loop/refiner, max_depth — các task tôn trọng.
- **Type consistency:** chữ ký `build_symbol_context(field_repo, operator_repo, prefilter, scope, relevance_text="")`, `repair_to_expression(deepseek, prefilter, field_repo, scope, system, user, task)`, `autowrap_vector_fields(expr, field_types, matrix_only_ops)`, `suggest_fields(field_repo, scope, bad_field, limit=5)` dùng nhất quán xuyên T1–T5.
- **Rủi ro đã lường:** auto-wrap no-op khi thiếu field_types/matrix_only (giữ `test_translate_repair_cu_phap` với `bad_op` xanh); `to_expression` chuẩn hoá lại chuỗi (test auto-wrap assert đúng dạng đã chuẩn hoá).
