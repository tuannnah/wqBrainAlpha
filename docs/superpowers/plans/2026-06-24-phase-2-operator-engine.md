# Phase 2 — Operator Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (khuyến nghị) để thực thi. Mỗi step dùng checkbox (`- [ ]`). Task 2.1–2.2 phải xong trước
> (file `subexpr_cache.py` + khung `evaluator.py`); Task 2.3–2.8 (6 file `operators_local/*.py`)
> độc lập nhau (mỗi file riêng, không đụng chung file) → **chia 6 sub-agent song song** sau khi
> 2.1–2.2 xong. Task 2.9 (wire + integration test) chỉ chạy sau khi cả 6 file operator merge
> xong. Task 2.10 review + merge + push chạy cuối, tuần tự.

**Goal:** Cài impl thật cho toàn bộ operator FASTEXPR-subset đã khai báo skeleton ở Phase 1
(`OperatorSpec.impl` từ `_not_implemented` → hàm thật), dựng `Evaluator(NodeVisitor[Panel])`
duyệt AST → `(T,N)` Panel với universe mask + sub-expression cache theo canonical hash. Kết
thúc phase: `parse(expr_str)` rồi `Evaluator(ctx).evaluate(node)` chạy đúng trên `small_panel`,
NaN-propagate đúng, không look-ahead, registry không còn placeholder nào active trong tập
operator đã liệt kê.

**Architecture:** `src/engine/{evaluator,subexpr_cache}.py` + `src/operators_local/
{arithmetic,cross_sectional,timeseries,group,neutralization,conditional}.py`. Mỗi file
operator tự `@register(...)` qua decorator của `src/lang/registry.py` khi import — ghi đè
spec placeholder Phase 1 cùng tên (registry cho phép ghi đè theo thiết kế B5). `src/engine`
import `src/lang` + `src/data` (Phase 0/1); operator modules import `src/lang/registry` +
`src/local_types`; không file nào trong `src/operators_local` hoặc `src/engine` import
`src/gp`, `src/storage`, `src/llm` (dependency rule master plan).

**Tech Stack:** Python 3.12, NumPy, pytest, ruff, mypy --strict. Không thêm dependency mới.

## Global Constraints

- Python 3.12; cú pháp hiện đại (`match`, `X | None`, `type` alias,
  `@dataclass(frozen=True, slots=True)`, `Protocol`).
- Full type hints; `mypy --strict` clean; `ruff` clean; không unused import.
- **No look-ahead:** time-series ops chỉ đọc rows ≤ t; thiếu lịch sử → NaN.
- **No survivorship:** universe mask per-day; out-of-universe = NaN (không phải 0).
- **Delay-1:** `pnl_t = nansum(weights_{t-1} * returns_t)`.
- **Stage separation:** expression = signal core; neut/decay/trunc/scale/delay ở
  `PortfolioConfig`.
- **Thresholds chỉ ở `config/thresholds.py`** — không hardcode gate number ở call site.
- **Determinism:** randomness qua seed inject; ghi seed vào DB.
- **WQ operator fidelity:** tra skill `worldquant-brain` trước khi viết FASTEXPR/operator.
- **TDD:** test trước, đỏ → code tối thiểu → xanh → commit. Mỗi phase = 1 nhánh git → merge
  → push.
- **Per-phase ritual:** Design → Implement → Explain → Review (test+ruff+mypy) → Gate →
  Journal (`PROGRESS.md`).

## WQ-faithful operator notes (từ MINIBRAIN_DESIGN.md B5 — bắt buộc đúng, đây là correctness)

- `ts_delay` là op shift; **không có `delay`**. `ts_delta(x,d) = x − ts_delay(x,d)`.
- `ts_rank(x,d)` → rank của giá trị hôm nay trong cửa sổ trailing d, normalize ~[0,1],
  **bounded=True**. `ts_zscore` **unbounded** (`bounded=False`).
- `ts_backfill(x,d)` lấp NaN bằng giá trị hợp lệ gần nhất trong d ngày trước; chỉ dùng dữ
  liệu ≤ t.
- `rank/winsorize/scale` là rank/sign-preserving → **không** giảm self-correlation; KHÔNG
  gán nhầm vào category NEUTRALIZATION.
- `group_neutralize(x, group)` trừ mean theo group mỗi ngày; **wrapper config**,
  `gp_usable=False`.
- `regression_neut(y, x)` = residual cross-sectional của y hồi quy theo x mỗi ngày.
  `vector_neut(x, y)` = trừ phần chiếu của x lên y. **Đây là 2 op duy nhất giảm
  self-correlation** — category NEUTRALIZATION.
- `trade_when(trigger, alpha, exit)`: trigger>0 → lấy alpha hôm đó; exit>0 → giữ giá trị
  trước (carry-forward), ngược lại NaN cho tới khi trigger lại kích hoạt.
  `hump(x, thr)`: chặn thay đổi nhỏ hơn `thr` (giảm turnover) — KHÔNG áp lên alpha có
  turnover nhanh là chính bản chất signal (note kiến trúc, không phải gate trong scope
  Phase 2).

## Phạm vi registry Phase 2 (ghi đè + bổ sung so với Phase 1 minimal)

| op | category | signature | bounded | gp_usable | commutative |
|---|---|---|---|---|---|
| `add` | ARITHMETIC | `(PANEL,PANEL)` | False | True | True |
| `subtract` | ARITHMETIC | `(PANEL,PANEL)` | False | True | False |
| `multiply` | ARITHMETIC | `(PANEL,PANEL)` | False | True | True |
| `divide` | ARITHMETIC | `(PANEL,PANEL)` | False | True | False |
| `log` | ARITHMETIC | `(PANEL,)` | False | True | False |
| `abs` | ARITHMETIC | `(PANEL,)` | False | True | False |
| `sign` | ARITHMETIC | `(PANEL,)` | True | True | False |
| `power` | ARITHMETIC | `(PANEL,SCALAR)` | False | True | False |
| `max` | ARITHMETIC | `(PANEL,PANEL)` | False | True | True |
| `min` | ARITHMETIC | `(PANEL,PANEL)` | False | True | True |
| `rank` | CROSS_SECTIONAL | `(PANEL,)` | True | True | False |
| `winsorize` | CROSS_SECTIONAL | `(PANEL,SCALAR)` | False | True | False |
| `scale` | SCALING | `(PANEL,)` | False | False | False |
| `zscore` | CROSS_SECTIONAL | `(PANEL,)` | False | True | False |
| `ts_mean` | TIME_SERIES | `(PANEL,WINDOW)` | False | True | False |
| `ts_std` | TIME_SERIES | `(PANEL,WINDOW)` | False | True | False |
| `ts_delta` | TIME_SERIES | `(PANEL,WINDOW)` | False | True | False |
| `ts_delay` | TIME_SERIES | `(PANEL,WINDOW)` | False | True | False |
| `ts_rank` | TIME_SERIES | `(PANEL,WINDOW)` | True | True | False |
| `ts_zscore` | TIME_SERIES | `(PANEL,WINDOW)` | False | True | False |
| `ts_corr` | TIME_SERIES | `(PANEL,PANEL,WINDOW)` | True | True | False |
| `ts_decay_linear` | TIME_SERIES | `(PANEL,WINDOW)` | False | True | False |
| `ts_backfill` | TIME_SERIES | `(PANEL,WINDOW)` | False | True | False |
| `group_neutralize` | GROUP | `(PANEL,GROUP)` | False | False | False |
| `regression_neut` | NEUTRALIZATION | `(PANEL,PANEL)` | False | True | False |
| `vector_neut` | NEUTRALIZATION | `(PANEL,PANEL)` | False | True | False |
| `trade_when` | CONDITIONAL | `(PANEL,PANEL,PANEL)` | False | True | False |
| `hump` | CONDITIONAL | `(PANEL,SCALAR)` | False | True | False |

`rank` đã có spec placeholder ở Phase 1 (category CROSS_SECTIONAL, bounded=True) — Task 2.4
ghi đè cùng spec, chỉ đổi `impl`. `ts_mean/add/subtract/multiply/divide` tương tự (Task
2.3/2.5 ghi đè placeholder Phase 1, giữ category/bounded/signature đã chốt). `scale` dùng
`OpCategory.SCALING` (có sẵn trong enum B5) — không phải `CROSS_SECTIONAL` — vì là wrapper
rescale gross-exposure (rank/sign-preserving, B5), không phải biến đổi thống kê cross-
sectional thuần; `gp_usable=False` để GP không "fix" correlation bằng `scale`.

---

### Task 2.1: `SubexprCache` — LRU theo canonical hash

**Files:**
- Create: `src/engine/__init__.py`
- Create: `src/engine/subexpr_cache.py`
- Test: `tests/unit/test_engine_subexpr_cache.py`

**Interfaces:**
- Produces: `class SubexprCache` với `get(key: str) -> Panel | None`,
  `put(key: str, value: Panel) -> None`, `__len__`, tham số `maxsize: int` ở `__init__`
  (mặc định ví dụ 256). Đơn giản nhất: wrap `collections.OrderedDict` (LRU thủ công, move-
  to-end khi get/put, evict `popitem(last=False)` khi vượt `maxsize`).
- Consumes: không phụ thuộc `src/lang`/`src/data` — module thuần (key là `str` hash đã tính
  sẵn ở caller, value là `Panel` numpy array bất kỳ).

- [ ] **Step 1: Test đỏ — hit/miss/eviction**

```python
# tests/unit/test_engine_subexpr_cache.py
"""Test SubexprCache: LRU theo key string (canonical hash), giữ panel (T,N)."""

from __future__ import annotations

import numpy as np

from src.engine.subexpr_cache import SubexprCache


def test_miss_tra_none() -> None:
    cache = SubexprCache(maxsize=4)
    assert cache.get("hash-a") is None


def test_put_get_hit_tra_dung_panel() -> None:
    cache = SubexprCache(maxsize=4)
    panel = np.array([[1.0, 2.0], [3.0, 4.0]])
    cache.put("hash-a", panel)
    out = cache.get("hash-a")
    assert out is not None
    np.testing.assert_array_equal(out, panel)


def test_vuot_maxsize_evict_key_cu_nhat() -> None:
    cache = SubexprCache(maxsize=2)
    cache.put("a", np.zeros((1, 1)))
    cache.put("b", np.zeros((1, 1)))
    cache.put("c", np.zeros((1, 1)))  # evict "a" (cũ nhất, chưa được get lại)
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None
    assert len(cache) == 2


def test_get_lam_moi_thu_tu_lru() -> None:
    cache = SubexprCache(maxsize=2)
    cache.put("a", np.zeros((1, 1)))
    cache.put("b", np.zeros((1, 1)))
    cache.get("a")  # "a" vừa được dùng -> không còn là cũ nhất
    cache.put("c", np.zeros((1, 1)))  # evict "b" thay vì "a"
    assert cache.get("b") is None
    assert cache.get("a") is not None
```

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_engine_subexpr_cache.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'src.engine'`).

- [ ] **Step 2: Impl thật**

```python
# src/engine/subexpr_cache.py
"""Cache LRU sub-expression theo canonical hash — chia sẻ panel đã eval giữa các node
AST trùng nhau trong cùng một cây hoặc giữa các cá thể GP (B6: throughput win chính
trước khi tối ưu numba)."""

from __future__ import annotations

from collections import OrderedDict

from src.local_types import Panel


class SubexprCache:
    """LRU cache key=canonical hash (str) -> Panel (T,N) đã eval."""

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize phải dương")
        self._maxsize = maxsize
        self._store: OrderedDict[str, Panel] = OrderedDict()

    def get(self, key: str) -> Panel | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: Panel) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)
```

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_engine_subexpr_cache.py -v
```
Expected: PASS (4 test).

- [ ] **Step 3: Commit**

```bash
git add src/engine/__init__.py src/engine/subexpr_cache.py tests/unit/test_engine_subexpr_cache.py
git commit -m "feat(engine): SubexprCache LRU theo canonical hash"
```

---

### Task 2.2: `EvalContext` + `Evaluator` khung (visit_constant/visit_field, dispatch visit_call)

**Files:**
- Create: `src/engine/evaluator.py`
- Test: `tests/unit/test_engine_evaluator.py`

**Interfaces:**
- Consumes: `src.lang.ast.{Node,Constant,Field,Call,NodeVisitor}`,
  `src.lang.registry.{OperatorRegistry,ArgKind,default_registry}`,
  `src.lang.visitors.CanonicalHasher`, `src.data.market_panel.MarketData`,
  `src.engine.subexpr_cache.SubexprCache`, `src.local_types.Panel`.
- Produces:
  ```python
  @dataclass(frozen=True, slots=True)
  class EvalContext:
      data: MarketData
      registry: OperatorRegistry
      cache: SubexprCache | None = None

  class Evaluator(NodeVisitor[Panel]):
      def __init__(self, ctx: EvalContext) -> None: ...
      def evaluate(self, node: Node) -> Panel: ...      # entry point, dùng cache
      def visit_constant(self, node: Constant) -> Panel: ...
      def visit_field(self, node: Field) -> Panel: ...
      def visit_call(self, node: Call) -> Panel: ...
  ```
  `evaluate()` là điểm vào duy nhất nên dùng (không gọi `node.accept(self)` trực tiếp từ
  ngoài) vì nó quản lý cache; `visit_call` đệ quy gọi `self.evaluate(child)` cho con kiểu
  PANEL (không phải `child.accept(self)`) để con cũng được cache.

- [ ] **Step 1: Test đỏ — constant broadcast, field passthrough, cache hit không re-eval**

```python
# tests/unit/test_engine_evaluator.py
"""Test khung Evaluator: constant broadcast (T,N), field đọc từ MarketData, cache theo
canonical hash. KHÔNG test operator cụ thể ở đây (đó là golden test Task 2.3-2.8) — chỉ
test cơ chế visit_constant/visit_field/dispatch + cache, dùng operator giả lập đơn giản."""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.evaluator import EvalContext, Evaluator
from src.engine.subexpr_cache import SubexprCache
from src.lang.ast import Call, Constant, Field
from src.lang.registry import ArgKind, OpCategory, OperatorRegistry, OperatorSpec


def _registry_voi_double() -> OperatorRegistry:
    """Registry test cục bộ (không đụng REGISTRY toàn cục) với 1 op giả `double(x) = x*2`."""
    reg = OperatorRegistry()
    reg.register(OperatorSpec(
        name="double", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL,), impl=lambda ctx, x: x * 2.0, bounded=False,
    ))
    return reg


def test_visit_constant_broadcast_shape_t_n(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=_registry_voi_double())
    out = Evaluator(ctx).evaluate(Constant(3.0))
    assert out.shape == small_panel.universe.shape
    assert np.all(out == 3.0)


def test_visit_field_doc_dung_du_lieu(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=_registry_voi_double())
    out = Evaluator(ctx).evaluate(Field("close"))
    np.testing.assert_array_equal(np.nan_to_num(out, nan=-1.0),
                                   np.nan_to_num(small_panel.field("close"), nan=-1.0))


def test_visit_call_dispatch_dung_impl(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=_registry_voi_double())
    out = Evaluator(ctx).evaluate(Call("double", (Field("close"),)))
    expected = small_panel.field("close") * 2.0
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_cache_hit_khong_goi_lai_impl(small_panel) -> None:
    calls = {"n": 0}

    def _counting_impl(ctx, x):
        calls["n"] += 1
        return x * 2.0

    reg = OperatorRegistry()
    reg.register(OperatorSpec(
        name="double", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL,), impl=_counting_impl, bounded=False,
    ))
    ctx = EvalContext(data=small_panel, registry=reg, cache=SubexprCache(maxsize=8))
    node = Call("double", (Field("close"),))
    ev = Evaluator(ctx)
    ev.evaluate(node)
    ev.evaluate(node)  # node giống hệt -> cùng canonical hash -> cache hit
    assert calls["n"] == 1


def test_khong_co_cache_van_chay_dung(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=_registry_voi_double(), cache=None)
    out = Evaluator(ctx).evaluate(Call("double", (Field("close"),)))
    assert out.shape == small_panel.universe.shape
```

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_engine_evaluator.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'src.engine.evaluator'`).

- [ ] **Step 2: Impl thật**

```python
# src/engine/evaluator.py
"""Evaluator: duyệt AST (NodeVisitor[Panel]) -> (T,N) Panel. Dispatch qua OperatorRegistry,
áp universe mask (NaN ngoài universe) sau mỗi Call, cache theo canonical hash (B6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.market_panel import MarketData
from src.engine.subexpr_cache import SubexprCache
from src.lang.ast import Call, Constant, Field, Node, NodeVisitor
from src.lang.registry import ArgKind, OperatorRegistry
from src.lang.visitors import CanonicalHasher
from src.local_types import Panel


@dataclass(frozen=True, slots=True)
class EvalContext:
    data: MarketData
    registry: OperatorRegistry
    cache: SubexprCache | None = None


def _apply_universe_mask(panel: Panel, universe: Panel) -> Panel:
    """NaN hóa mọi cell ngoài universe — bất biến B6: out-of-universe luôn NaN."""
    out = panel.copy()
    out[~universe] = np.nan
    return out


def _literal(node: Node) -> float | str:
    """Đọc giá trị literal của arg WINDOW/SCALAR/GROUP — không eval thành Panel.
    WINDOW/SCALAR đọc từ Constant.value; GROUP đọc tên group từ Field.name (group key
    được biểu diễn như Field trong AST vì cũng là một identifier, vd `sector`)."""
    if isinstance(node, Constant):
        return node.value
    if isinstance(node, Field):
        return node.name
    raise TypeError(
        f"arg literal (WINDOW/SCALAR/GROUP) phải là Constant hoặc Field, nhận {type(node)!r}"
    )


class Evaluator(NodeVisitor[Panel]):
    """Duyệt AST sinh Panel (T,N). Dùng `evaluate()` làm điểm vào (quản lý cache);
    `visit_*` không tự gọi lại `evaluate` của con qua `accept` mà qua `self.evaluate`
    để mọi sub-node cũng đi qua cache."""

    def __init__(self, ctx: EvalContext) -> None:
        self._ctx = ctx
        self._hasher = CanonicalHasher(ctx.registry)

    def evaluate(self, node: Node) -> Panel:
        if self._ctx.cache is not None:
            key = self._hasher.visit(node)
            cached = self._ctx.cache.get(key)
            if cached is not None:
                return cached
            result = node.accept(self)
            self._ctx.cache.put(key, result)
            return result
        return node.accept(self)

    def visit_constant(self, node: Constant) -> Panel:
        shape = self._ctx.data.universe.shape
        return np.full(shape, float(node.value), dtype=np.float64)

    def visit_field(self, node: Field) -> Panel:
        return _apply_universe_mask(self._ctx.data.field(node.name), self._ctx.data.universe)

    def visit_call(self, node: Call) -> Panel:
        spec = self._ctx.registry.get(node.op)
        eval_args: list[Panel | float | str] = []
        for arg, kind in zip(node.args, spec.signature, strict=True):
            if kind is ArgKind.PANEL:
                eval_args.append(self.evaluate(arg))
            else:  # WINDOW, SCALAR, GROUP
                eval_args.append(_literal(arg))
        out = spec.impl(self._ctx, *eval_args)
        return _apply_universe_mask(out, self._ctx.data.universe)
```

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_engine_evaluator.py -v
```
Expected: PASS (5 test).

- [ ] **Step 3: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/engine/evaluator.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/engine/evaluator.py
```
Expected: clean cả hai.

- [ ] **Step 4: Commit**

```bash
git add src/engine/evaluator.py tests/unit/test_engine_evaluator.py
git commit -m "feat(engine): EvalContext + Evaluator khung (constant/field/call + cache)"
```

---

### Task 2.3: `src/operators_local/arithmetic.py`

> Có thể chạy song song với 2.4–2.8 (sub-agent riêng) — chỉ chạm file riêng của mình +
> `tests/golden/test_operators_arithmetic.py`. KHÔNG sửa `src/lang/registry.py`.

**Files:**
- Create: `src/operators_local/__init__.py` (nếu chưa có — package marker rỗng, một
  sub-agent tạo, các sub-agent khác bỏ qua nếu đã tồn tại)
- Create: `src/operators_local/arithmetic.py`
- Test: `tests/golden/__init__.py`, `tests/golden/test_operators_arithmetic.py`

**Interfaces:**
- Produces: `add, subtract, multiply, divide, log, abs_, sign, power, max_, min_` —
  hàm `(ctx: EvalContext, *args) -> Panel`, mỗi hàm gắn `@register(name=..., category=
  OpCategory.ARITHMETIC, signature=..., bounded=..., commutative=...)` đúng bảng ở đầu
  file plan (registry này ghi đè placeholder Phase 1 cho `add/subtract/multiply/divide`).
  Đặt tên hàm Python `abs_`/`max_`/`min_` (tránh shadow builtin) nhưng `name="abs"` /
  `"max"` / `"min"` trong registry.
- Consumes: `src.engine.evaluator.EvalContext`, `src.lang.registry.{register,ArgKind,
  OpCategory}`, `src.local_types.Panel`, NumPy.

- [ ] **Step 1: Test đỏ — golden test arithmetic (NaN-propagation + giá trị)**

```python
# tests/golden/test_operators_arithmetic.py
"""Golden test arithmetic ops trên small_panel: giá trị đúng + NaN-propagate."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Constant, Field
from src.lang.registry import default_registry


def test_add_dung_gia_tri(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("add", (Field("close"), Field("volume"))))
    expected = small_panel.field("close") + small_panel.field("volume")
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_divide_nan_khi_mau_la_nan(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("divide", (Field("close"), Field("close"))))
    # mọi cell trong-universe: close/close == 1.0; ngoài universe -> NaN (mask sau impl)
    in_uni = small_panel.universe
    assert np.allclose(out[in_uni], 1.0)
    assert np.all(np.isnan(out[~in_uni]))


def test_log_abs_sign(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    close = small_panel.field("close")
    out_log = Evaluator(ctx).evaluate(Call("log", (Field("close"),)))
    np.testing.assert_allclose(out_log, np.log(close), equal_nan=True)
    out_abs = Evaluator(ctx).evaluate(Call("abs", (Field("close"),)))
    np.testing.assert_allclose(out_abs, np.abs(close), equal_nan=True)
    out_sign = Evaluator(ctx).evaluate(Call("sign", (Field("close"),)))
    np.testing.assert_allclose(out_sign, np.sign(close), equal_nan=True)


def test_power_max_min(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    close = small_panel.field("close")
    out_pow = Evaluator(ctx).evaluate(Call("power", (Field("close"), Constant(2.0))))
    np.testing.assert_allclose(out_pow, close ** 2.0, equal_nan=True)
    out_max = Evaluator(ctx).evaluate(Call("max", (Field("close"), Field("volume"))))
    np.testing.assert_allclose(
        out_max, np.maximum(close, small_panel.field("volume")), equal_nan=True
    )


def test_nan_propagation_qua_chuoi_phep_toan(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(
        Call("add", (Call("log", (Field("close"),)), Constant(1.0)))
    )
    in_uni = small_panel.universe
    assert not np.any(np.isnan(out[in_uni]))  # close luôn >0 trong fixture
    assert np.all(np.isnan(out[~in_uni]))
```

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_arithmetic.py -v
```
Expected: FAIL (operator chưa được import/đăng ký → `KeyError: operator không tồn tại`,
hoặc nếu đã match placeholder Phase 1 thì `NotImplementedError`).

- [ ] **Step 2: Impl thật**

```python
# src/operators_local/arithmetic.py
"""Operator số học: + - * / log abs sign power max min. Tất cả NaN-propagate tự nhiên
qua NumPy (NaN op x = NaN); không cần xử lý universe ở đây — Evaluator áp mask sau impl."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


@register(name="add", category=OpCategory.ARITHMETIC,
           signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=True)
def add(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    return x + y


@register(name="subtract", category=OpCategory.ARITHMETIC,
           signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=False)
def subtract(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    return x - y


@register(name="multiply", category=OpCategory.ARITHMETIC,
           signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=True)
def multiply(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    return x * y


@register(name="divide", category=OpCategory.ARITHMETIC,
           signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=False)
def divide(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    with np.errstate(divide="ignore", invalid="ignore"):
        return x / y


@register(name="log", category=OpCategory.ARITHMETIC,
           signature=(ArgKind.PANEL,), bounded=False, commutative=False)
def log(ctx: EvalContext, x: Panel) -> Panel:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x)


@register(name="abs", category=OpCategory.ARITHMETIC,
           signature=(ArgKind.PANEL,), bounded=False, commutative=False)
def abs_(ctx: EvalContext, x: Panel) -> Panel:
    return np.abs(x)


@register(name="sign", category=OpCategory.ARITHMETIC,
           signature=(ArgKind.PANEL,), bounded=True, commutative=False)
def sign(ctx: EvalContext, x: Panel) -> Panel:
    return np.sign(x)


@register(name="power", category=OpCategory.ARITHMETIC,
           signature=(ArgKind.PANEL, ArgKind.SCALAR), bounded=False, commutative=False)
def power(ctx: EvalContext, x: Panel, p: float) -> Panel:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.power(x, p)


@register(name="max", category=OpCategory.ARITHMETIC,
           signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=True)
def max_(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    return np.maximum(x, y)


@register(name="min", category=OpCategory.ARITHMETIC,
           signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=True)
def min_(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    return np.minimum(x, y)
```

Lưu ý: `default_registry()` chỉ trả registry toàn cục đã có sẵn các spec import-time —
test phải `import src.operators_local.arithmetic` (trực tiếp hoặc gián tiếp) trước khi gọi
`default_registry()` để decorator chạy. Thêm dòng import tường minh đầu file test nếu
`conftest.py` không tự import package `operators_local` (xem Task 2.9 — file
`src/operators_local/__init__.py` nên import tất cả 6 submodule để 1 lần import là đủ).

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_arithmetic.py -v
```
Expected: PASS (5 test).

- [ ] **Step 3: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/operators_local/arithmetic.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/operators_local/arithmetic.py
```

- [ ] **Step 4: Commit**

```bash
git add src/operators_local/__init__.py src/operators_local/arithmetic.py tests/golden/__init__.py tests/golden/test_operators_arithmetic.py
git commit -m "feat(operators): arithmetic (add/sub/mul/div/log/abs/sign/power/max/min)"
```

---

### Task 2.4: `src/operators_local/cross_sectional.py`

> Song song với 2.3, 2.5–2.8.

**Files:**
- Create: `src/operators_local/cross_sectional.py`
- Test: `tests/golden/test_operators_cross_sectional.py`

**Interfaces:**
- Produces: `rank, winsorize, scale, zscore` — `(ctx, x: Panel, ...) -> Panel`, mỗi hàm
  hoạt động **per-row** (mỗi ngày `t` độc lập) chỉ trên cell in-universe; cell ngoài
  universe không tham gia tính rank/mean/std của hàng đó (NaN tự bị `nan`-aware reducer
  loại trừ, Evaluator sẽ mask lại NaN ngoài universe sau).
- Consumes: `EvalContext` (đọc `ctx.data.universe` nếu cần biết per-row N hợp lệ — thực ra
  NumPy `nan*`-reducer đã tự loại NaN nên không cần đọc universe trực tiếp, miễn input đã
  NaN ngoài universe, điều Evaluator đảm bảo qua `visit_field`/mask sau `visit_call` con).

- [ ] **Step 1: Test đỏ**

```python
# tests/golden/test_operators_cross_sectional.py
"""Golden test cross-sectional ops: per-row in-universe, rank bounded ~[0,1]."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Constant, Field
from src.lang.registry import default_registry


def test_rank_bounded_0_1_trong_universe(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("rank", (Field("close"),)))
    in_uni = small_panel.universe
    assert np.nanmin(out[in_uni]) >= 0.0
    assert np.nanmax(out[in_uni]) <= 1.0
    assert np.all(np.isnan(out[~in_uni]))


def test_rank_chi_tinh_tren_in_universe_row(small_panel) -> None:
    """Một hàng có universe hẹp hơn (3 mã cuối ngoài universe ở nửa đầu fixture) —
    rank của các mã in-universe không bị ảnh hưởng bởi giá trị các mã ngoài universe."""
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("rank", (Field("close"),)))
    row0 = 0  # 3 mã cuối ngoài universe ở fixture (xem conftest small_panel)
    valid = small_panel.universe[row0]
    ranked_vals = out[row0][valid]
    assert ranked_vals.size == valid.sum()
    assert not np.any(np.isnan(ranked_vals))


def test_zscore_mean_0_std_1_per_row(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("zscore", (Field("close"),)))
    in_uni = small_panel.universe
    row = 100  # hàng universe đầy đủ (nửa sau fixture)
    vals = out[row][in_uni[row]]
    assert abs(float(np.mean(vals))) < 1e-8
    assert abs(float(np.std(vals)) - 1.0) < 1e-6


def test_winsorize_chan_outlier(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("winsorize", (Field("close"), Constant(2.0))))
    in_uni = small_panel.universe
    row = 100
    vals = out[row][in_uni[row]]
    z = (vals - np.mean(vals)) / np.std(vals)
    assert np.nanmax(np.abs(z)) <= 2.0 + 1e-6


def test_scale_tong_abs_bang_1(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("scale", (Field("close"),)))
    in_uni = small_panel.universe
    row = 100
    vals = out[row][in_uni[row]]
    assert abs(float(np.sum(np.abs(vals))) - 1.0) < 1e-6
```

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_cross_sectional.py -v
```
Expected: FAIL (`KeyError`/`NotImplementedError`).

- [ ] **Step 2: Impl thật**

```python
# src/operators_local/cross_sectional.py
"""Operator cross-sectional: rank/winsorize/scale/zscore — per-row (mỗi ngày t), chỉ trên
cell in-universe (NaN tự loại nhờ nan-aware reducer numpy khi input panel có NaN ngoài
universe)."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


@register(name="rank", category=OpCategory.CROSS_SECTIONAL,
           signature=(ArgKind.PANEL,), bounded=True, commutative=False)
def rank(ctx: EvalContext, x: Panel) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row = x[t]
        valid = ~np.isnan(row)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        order = np.argsort(row[valid], kind="stable")
        ranks = np.empty(n_valid, dtype=np.float64)
        denom = n_valid - 1 if n_valid > 1 else 1
        ranks[order] = np.arange(n_valid, dtype=np.float64) / denom
        out[t][valid] = ranks
    return out


@register(name="winsorize", category=OpCategory.CROSS_SECTIONAL,
           signature=(ArgKind.PANEL, ArgKind.SCALAR), bounded=False, commutative=False)
def winsorize(ctx: EvalContext, x: Panel, std_count: float) -> Panel:
    out = x.copy()
    for t in range(x.shape[0]):
        row = out[t]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        mean = float(np.mean(row[valid]))
        std = float(np.std(row[valid]))
        if std == 0.0:
            continue
        lo, hi = mean - std_count * std, mean + std_count * std
        row[valid] = np.clip(row[valid], lo, hi)
    return out


@register(name="zscore", category=OpCategory.CROSS_SECTIONAL,
           signature=(ArgKind.PANEL,), bounded=False, commutative=False)
def zscore(ctx: EvalContext, x: Panel) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row = x[t]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        mean = float(np.mean(row[valid]))
        std = float(np.std(row[valid]))
        if std == 0.0:
            continue
        out[t][valid] = (row[valid] - mean) / std
    return out


@register(name="scale", category=OpCategory.SCALING,
           signature=(ArgKind.PANEL,), bounded=False, gp_usable=False, commutative=False)
def scale(ctx: EvalContext, x: Panel) -> Panel:
    """Rescale per-row để tổng |giá trị| trong-universe = 1 (rank/sign-preserving,
    wrapper config — không tham gia core GP search, B5)."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row = x[t]
        valid = ~np.isnan(row)
        if valid.sum() == 0:
            continue
        total = float(np.sum(np.abs(row[valid])))
        if total == 0.0:
            continue
        out[t][valid] = row[valid] / total
    return out
```

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_cross_sectional.py -v
```
Expected: PASS (5 test).

- [ ] **Step 3: ruff + mypy + commit**

```bash
venv/Scripts/python.exe -m ruff check src/operators_local/cross_sectional.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/operators_local/cross_sectional.py
git add src/operators_local/cross_sectional.py tests/golden/test_operators_cross_sectional.py
git commit -m "feat(operators): cross_sectional (rank/winsorize/scale/zscore) per-row in-universe"
```

---

### Task 2.5: `src/operators_local/timeseries.py`

> Song song với 2.3–2.4, 2.6–2.8. Phần nhạy correctness nhất — **no-look-ahead test bắt
> buộc** cho mọi op.

**Files:**
- Create: `src/operators_local/timeseries.py`
- Test: `tests/golden/test_operators_timeseries.py`

**Interfaces:**
- Produces: `ts_mean, ts_std, ts_delta, ts_delay, ts_rank, ts_zscore, ts_corr,
  ts_decay_linear, ts_backfill` — `(ctx, x: Panel, d: int[, y: Panel]) -> Panel`. Mỗi op
  tại hàng `t` chỉ đọc `x[max(0,t-d+1):t+1]` (hoặc `t-d:t+1` tùy định nghĩa window — chốt
  dưới đây); thiếu đủ `d` quan sát hợp lệ (sau loại NaN) → NaN tại hàng đó.
- Consumes: `EvalContext`; `ts_corr` nhận 2 Panel + window.

**Định nghĩa window chốt (để 9 hàm nhất quán):** cửa sổ trailing độ dài `d` kết thúc tại
`t` bao gồm **rows `[t-d+1, t]`** (đúng `d` quan sát, kể cả `t`). Hàng `t < d-1` chưa đủ
`d` quan sát → NaN (không dùng window ngắn hơn). `ts_delay(x,d)` lấy đúng `x[t-d]` (không
phải window) — nếu `t-d < 0` → NaN.

- [ ] **Step 1: Test đỏ — giá trị + no-look-ahead**

```python
# tests/golden/test_operators_timeseries.py
"""Golden test time-series ops: giá trị đúng trên small_panel, KHÔNG look-ahead (thay đổi
rows > t không đổi kết quả tại row t), thiếu lịch sử -> NaN, ts_rank bounded."""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Constant, Field
from src.lang.registry import default_registry
from src.data.market_panel import MarketData


def _eval(panel: MarketData, node) -> np.ndarray:
    return Evaluator(EvalContext(data=panel, registry=default_registry())).evaluate(node)


@pytest.mark.parametrize("op", ["ts_mean", "ts_std", "ts_zscore", "ts_decay_linear"])
def test_thieu_lich_su_la_nan(small_panel, op) -> None:
    out = _eval(small_panel, Call(op, (Field("close"), Constant(20))))
    assert np.all(np.isnan(out[:19]))  # < d-1=19 quan sát -> NaN


def test_ts_delay_khong_phai_ts_delta(small_panel) -> None:
    out_delay = _eval(small_panel, Call("ts_delay", (Field("close"), Constant(5))))
    out_delta = _eval(small_panel, Call("ts_delta", (Field("close"), Constant(5))))
    close = small_panel.field("close")
    row = 50
    np.testing.assert_allclose(out_delay[row], close[row - 5], equal_nan=True)
    np.testing.assert_allclose(out_delta[row], close[row] - close[row - 5], equal_nan=True)
    assert not np.allclose(np.nan_to_num(out_delay[row]), np.nan_to_num(out_delta[row]))


def test_ts_mean_dung_gia_tri(small_panel) -> None:
    out = _eval(small_panel, Call("ts_mean", (Field("close"), Constant(10))))
    close = small_panel.field("close")
    row = 50
    expected = np.nanmean(close[row - 9 : row + 1], axis=0)
    np.testing.assert_allclose(out[row], expected, equal_nan=True)


def test_ts_rank_bounded_0_1(small_panel) -> None:
    out = _eval(small_panel, Call("ts_rank", (Field("close"), Constant(20))))
    valid = ~np.isnan(out)
    assert np.nanmin(out[valid]) >= 0.0
    assert np.nanmax(out[valid]) <= 1.0


def test_no_look_ahead_doi_tuong_lai_khong_doi_qua_khu(small_panel) -> None:
    """Bất biến cốt lõi: sửa dữ liệu CHỈ ở rows > t không thay đổi kết quả tại row t,
    cho mọi op time-series (test chỉ chọn ts_mean làm đại diện + ts_corr)."""
    row_t = 60
    mutated_close = small_panel.field("close").copy()
    mutated_close[row_t + 1 :] += 999.0  # phá tương lai
    mutated = MarketData(
        dates=small_panel.dates, assets=small_panel.assets,
        fields={**small_panel.fields, "close": mutated_close},
        universe=small_panel.universe, returns=small_panel.returns,
        groups=small_panel.groups,
    )
    out_orig = _eval(small_panel, Call("ts_mean", (Field("close"), Constant(10))))
    out_mut = _eval(mutated, Call("ts_mean", (Field("close"), Constant(10))))
    np.testing.assert_allclose(out_orig[row_t], out_mut[row_t], equal_nan=True)


def test_ts_corr_no_look_ahead(small_panel) -> None:
    row_t = 60
    mutated_volume = small_panel.field("volume").copy()
    mutated_volume[row_t + 1 :] *= 5.0
    mutated = MarketData(
        dates=small_panel.dates, assets=small_panel.assets,
        fields={**small_panel.fields, "volume": mutated_volume},
        universe=small_panel.universe, returns=small_panel.returns,
        groups=small_panel.groups,
    )
    node = Call("ts_corr", (Field("close"), Field("volume"), Constant(15)))
    out_orig = _eval(small_panel, node)
    out_mut = _eval(mutated, node)
    np.testing.assert_allclose(out_orig[row_t], out_mut[row_t], equal_nan=True)


def test_ts_backfill_lap_nan_tu_qua_khu(small_panel) -> None:
    close = small_panel.field("close").copy()
    close[40, 0] = np.nan  # 1 NaN giữa dữ liệu hợp lệ ở cột 0
    mutated = MarketData(
        dates=small_panel.dates, assets=small_panel.assets,
        fields={**small_panel.fields, "close": close},
        universe=small_panel.universe, returns=small_panel.returns,
        groups=small_panel.groups,
    )
    out = _eval(mutated, Call("ts_backfill", (Field("close"), Constant(5))))
    assert not np.isnan(out[40, 0])
    np.testing.assert_allclose(out[40, 0], close[39, 0])
```

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_timeseries.py -v
```
Expected: FAIL.

- [ ] **Step 2: Impl thật**

```python
# src/operators_local/timeseries.py
"""Operator time-series: trailing window [t-d+1, t] (đúng d quan sát, kể cả t); thiếu đủ
lịch sử -> NaN. KHÔNG bao giờ đọc rows > t (no-look-ahead, B6/Global Constraints)."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


def _window_slice(t: int, d: int) -> slice | None:
    start = t - d + 1
    if start < 0:
        return None
    return slice(start, t + 1)


@register(name="ts_mean", category=OpCategory.TIME_SERIES,
           signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_mean(ctx: EvalContext, x: Panel, d: int) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        with np.errstate(invalid="ignore"):
            out[t] = np.nanmean(x[win], axis=0)
    return out


@register(name="ts_std", category=OpCategory.TIME_SERIES,
           signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_std(ctx: EvalContext, x: Panel, d: int) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        with np.errstate(invalid="ignore"):
            out[t] = np.nanstd(x[win], axis=0)
    return out


@register(name="ts_delay", category=OpCategory.TIME_SERIES,
           signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_delay(ctx: EvalContext, x: Panel, d: int) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    if d < x.shape[0]:
        out[d:] = x[: x.shape[0] - d]
    return out


@register(name="ts_delta", category=OpCategory.TIME_SERIES,
           signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_delta(ctx: EvalContext, x: Panel, d: int) -> Panel:
    return x - ts_delay(ctx, x, d)


@register(name="ts_rank", category=OpCategory.TIME_SERIES,
           signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=True, commutative=False)
def ts_rank(ctx: EvalContext, x: Panel, d: int) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        window = x[win]
        for col in range(x.shape[1]):
            series = window[:, col]
            valid = ~np.isnan(series)
            n_valid = int(valid.sum())
            if n_valid == 0 or np.isnan(x[t, col]):
                continue
            vals = series[valid]
            denom = n_valid - 1 if n_valid > 1 else 1
            out[t, col] = float(np.sum(vals <= x[t, col]) - 1) / denom
    return out


@register(name="ts_zscore", category=OpCategory.TIME_SERIES,
           signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_zscore(ctx: EvalContext, x: Panel, d: int) -> Panel:
    mean = ts_mean(ctx, x, d)
    std = ts_std(ctx, x, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (x - mean) / std
    out[std == 0.0] = np.nan
    return out


@register(name="ts_corr", category=OpCategory.TIME_SERIES,
           signature=(ArgKind.PANEL, ArgKind.PANEL, ArgKind.WINDOW), bounded=True,
           commutative=False)
def ts_corr(ctx: EvalContext, x: Panel, y: Panel, d: int) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        wx, wy = x[win], y[win]
        for col in range(x.shape[1]):
            sx, sy = wx[:, col], wy[:, col]
            valid = ~np.isnan(sx) & ~np.isnan(sy)
            if int(valid.sum()) < 2:
                continue
            sxv, syv = sx[valid], sy[valid]
            if np.std(sxv) == 0.0 or np.std(syv) == 0.0:
                continue
            out[t, col] = float(np.corrcoef(sxv, syv)[0, 1])
    return out


@register(name="ts_decay_linear", category=OpCategory.TIME_SERIES,
           signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_decay_linear(ctx: EvalContext, x: Panel, d: int) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    weights = np.arange(1, d + 1, dtype=np.float64)  # xa nhất=1 ... gần nhất(t)=d
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        window = x[win]
        for col in range(x.shape[1]):
            series = window[:, col]
            valid = ~np.isnan(series)
            if not np.any(valid):
                continue
            w = weights[valid]
            out[t, col] = float(np.sum(series[valid] * w) / np.sum(w))
    return out


@register(name="ts_backfill", category=OpCategory.TIME_SERIES,
           signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_backfill(ctx: EvalContext, x: Panel, d: int) -> Panel:
    """Lấp NaN bằng giá trị hợp lệ gần nhất trong d hàng trước (rows <= t); quá d hàng
    không tìm thấy giá trị hợp lệ -> giữ NaN."""
    out = x.copy()
    d = int(d)
    for col in range(x.shape[1]):
        last_valid_row = -1
        for t in range(x.shape[0]):
            if not np.isnan(x[t, col]):
                last_valid_row = t
            elif last_valid_row >= 0 and (t - last_valid_row) <= d:
                out[t, col] = x[last_valid_row, col]
    return out
```

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_timeseries.py -v
```
Expected: PASS (toàn bộ test bao gồm no-look-ahead).

- [ ] **Step 3: ruff + mypy + commit**

```bash
venv/Scripts/python.exe -m ruff check src/operators_local/timeseries.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/operators_local/timeseries.py
git add src/operators_local/timeseries.py tests/golden/test_operators_timeseries.py
git commit -m "feat(operators): timeseries (ts_mean/std/delta/delay/rank/zscore/corr/decay_linear/backfill)"
```

---

### Task 2.6: `src/operators_local/group.py`

> Song song với 2.3–2.5, 2.7–2.8.

**Files:**
- Create: `src/operators_local/group.py`
- Test: `tests/golden/test_operators_group.py`

**Interfaces:**
- Produces: `group_neutralize(ctx, x: Panel, group_name: str) -> Panel`, `@register(
  name="group_neutralize", category=OpCategory.GROUP, signature=(ArgKind.PANEL,
  ArgKind.GROUP), bounded=False, gp_usable=False)`.
- Consumes: `ctx.data.groups[group_name]` (int codes `(T,N)`, từ `MarketData.groups`).

- [ ] **Step 1: Test đỏ**

```python
# tests/golden/test_operators_group.py
"""Golden test group_neutralize: trừ mean theo group mỗi ngày, per-row in-universe."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Field
from src.lang.registry import default_registry


def test_group_neutralize_mean_0_moi_group(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("group_neutralize", (Field("close"), Field("sector"))))
    row = 100  # universe đầy đủ ở nửa sau
    sector_row = small_panel.groups["sector"][row]
    in_uni = small_panel.universe[row]
    for g in np.unique(sector_row[in_uni]):
        mask = in_uni & (sector_row == g)
        if mask.sum() < 1:
            continue
        assert abs(float(np.mean(out[row][mask]))) < 1e-8


def test_group_neutralize_gp_usable_false() -> None:
    spec = default_registry().get("group_neutralize")
    assert spec.gp_usable is False
```

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_group.py -v
```
Expected: FAIL.

- [ ] **Step 2: Impl thật**

```python
# src/operators_local/group.py
"""group_neutralize: trừ mean theo group mỗi ngày (wrapper config, gp_usable=False, B5)."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


@register(name="group_neutralize", category=OpCategory.GROUP,
           signature=(ArgKind.PANEL, ArgKind.GROUP), bounded=False, gp_usable=False,
           commutative=False)
def group_neutralize(ctx: EvalContext, x: Panel, group_name: str) -> Panel:
    groups = ctx.data.groups[group_name]
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row, grp_row = x[t], groups[t]
        valid = ~np.isnan(row)
        if not np.any(valid):
            continue
        for g in np.unique(grp_row[valid]):
            mask = valid & (grp_row == g)
            if not np.any(mask):
                continue
            out[t][mask] = row[mask] - float(np.mean(row[mask]))
    return out
```

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_group.py -v
```
Expected: PASS.

- [ ] **Step 3: ruff + mypy + commit**

```bash
venv/Scripts/python.exe -m ruff check src/operators_local/group.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/operators_local/group.py
git add src/operators_local/group.py tests/golden/test_operators_group.py
git commit -m "feat(operators): group_neutralize (wrapper config, gp_usable=False)"
```

---

### Task 2.7: `src/operators_local/neutralization.py`

> Song song với 2.3–2.6, 2.8.

**Files:**
- Create: `src/operators_local/neutralization.py`
- Test: `tests/golden/test_operators_neutralization.py`

**Interfaces:**
- Produces: `regression_neut(ctx, y: Panel, x: Panel) -> Panel` (residual cross-sectional
  per-row của `y` hồi quy tuyến tính theo `x`, gồm intercept), `vector_neut(ctx, x: Panel,
  y: Panel) -> Panel` (`x` trừ phần chiếu của `x` lên `y`, per-row). Cả hai
  `category=OpCategory.NEUTRALIZATION`, `gp_usable=True` (đây là 2 op duy nhất giảm
  self-corr, B5 — phải khả dụng cho GP đối tượng correlation).

- [ ] **Step 1: Test đỏ**

```python
# tests/golden/test_operators_neutralization.py
"""Golden test regression_neut/vector_neut: residual có corr ~0 với biến neutralize,
per-row in-universe."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Field
from src.lang.registry import default_registry


def test_regression_neut_residual_khong_tuong_quan_voi_x(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(
        Call("regression_neut", (Field("close"), Field("volume")))
    )
    row = 100
    in_uni = small_panel.universe[row]
    resid = out[row][in_uni]
    x = small_panel.field("volume")[row][in_uni]
    corr = np.corrcoef(resid, x)[0, 1]
    assert abs(corr) < 1e-6


def test_vector_neut_truc_giao_voi_y(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("vector_neut", (Field("close"), Field("volume"))))
    row = 100
    in_uni = small_panel.universe[row]
    resid = out[row][in_uni]
    y = small_panel.field("volume")[row][in_uni]
    dot = float(np.dot(resid, y))
    assert abs(dot) < 1e-6


def test_categoria_e_gp_usable(small_panel) -> None:
    spec_r = default_registry().get("regression_neut")
    spec_v = default_registry().get("vector_neut")
    assert spec_r.gp_usable is True
    assert spec_v.gp_usable is True
```

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_neutralization.py -v
```
Expected: FAIL.

- [ ] **Step 2: Impl thật**

```python
# src/operators_local/neutralization.py
"""regression_neut/vector_neut — 2 op DUY NHẤT trong MiniBrain giảm self-correlation
(B5). Mỗi op hoạt động per-row (cross-sectional), chỉ trên cell in-universe."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


@register(name="regression_neut", category=OpCategory.NEUTRALIZATION,
           signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=False)
def regression_neut(ctx: EvalContext, y: Panel, x: Panel) -> Panel:
    """Residual cross-sectional per-row của y hồi quy OLS (với intercept) theo x."""
    out = np.full_like(y, np.nan, dtype=np.float64)
    for t in range(y.shape[0]):
        yr, xr = y[t], x[t]
        valid = ~np.isnan(yr) & ~np.isnan(xr)
        n_valid = int(valid.sum())
        if n_valid < 2:
            continue
        xv, yv = xr[valid], yr[valid]
        if np.std(xv) == 0.0:
            out[t][valid] = yv - float(np.mean(yv))
            continue
        design = np.column_stack([np.ones(n_valid), xv])
        coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
        out[t][valid] = yv - design @ coef
    return out


@register(name="vector_neut", category=OpCategory.NEUTRALIZATION,
           signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=False)
def vector_neut(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    """Trừ phần chiếu của x lên y mỗi hàng: x - (x.y / y.y) * y, chỉ trên in-universe."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        xr, yr = x[t], y[t]
        valid = ~np.isnan(xr) & ~np.isnan(yr)
        if not np.any(valid):
            continue
        xv, yv = xr[valid], yr[valid]
        denom = float(np.dot(yv, yv))
        if denom == 0.0:
            out[t][valid] = xv
            continue
        proj_coef = float(np.dot(xv, yv)) / denom
        out[t][valid] = xv - proj_coef * yv
    return out
```

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_neutralization.py -v
```
Expected: PASS.

- [ ] **Step 3: ruff + mypy + commit**

```bash
venv/Scripts/python.exe -m ruff check src/operators_local/neutralization.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/operators_local/neutralization.py
git add src/operators_local/neutralization.py tests/golden/test_operators_neutralization.py
git commit -m "feat(operators): regression_neut + vector_neut (2 op giảm self-correlation)"
```

---

### Task 2.8: `src/operators_local/conditional.py`

> Song song với 2.3–2.7.

**Files:**
- Create: `src/operators_local/conditional.py`
- Test: `tests/golden/test_operators_conditional.py`

**Interfaces:**
- Produces: `trade_when(ctx, trigger: Panel, alpha: Panel, exit_cond: Panel) -> Panel`
  (trigger>0 → giá trị alpha hôm đó; exit_cond>0 → carry-forward giá trị hợp lệ trước đó;
  ngược lại NaN), `hump(ctx, x: Panel, thr: float) -> Panel` (carry-forward giá trị trước
  nếu `|x_t - x_{t-1}| < thr`, theo cột, no-look-ahead — chỉ dùng giá trị tại `t-1`).

- [ ] **Step 1: Test đỏ**

```python
# tests/golden/test_operators_conditional.py
"""Golden test trade_when/hump: logic carry-forward theo điều kiện, no-look-ahead."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Constant, Field
from src.lang.registry import default_registry


def _ctx_for(panel):
    return EvalContext(data=panel, registry=default_registry())


def test_trade_when_trigger_duong_lay_alpha(small_panel) -> None:
    ctx = _ctx_for(small_panel)
    node = Call("trade_when", (
        Call("sign", (Field("close"),)),  # luôn 1 (close>0) -> trigger luôn kích hoạt
        Field("close"),
        Constant(-1.0),  # exit luôn âm -> không carry-forward (n/a vì trigger luôn ưu tiên)
    ))
    out = Evaluator(ctx).evaluate(node)
    in_uni = small_panel.universe
    np.testing.assert_allclose(out[in_uni], small_panel.field("close")[in_uni], equal_nan=True)


def test_trade_when_khong_trigger_khong_exit_la_nan(small_panel) -> None:
    ctx = _ctx_for(small_panel)
    close = small_panel.field("close")
    node = Call("trade_when", (
        Constant(-1.0),  # trigger luôn <=0 -> không bao giờ lấy alpha mới
        Field("close"),
        Constant(-1.0),  # exit luôn <=0 -> không carry-forward
    ))
    out = Evaluator(ctx).evaluate(node)
    assert np.all(np.isnan(out))


def test_hump_chan_thay_doi_nho(small_panel) -> None:
    ctx = _ctx_for(small_panel)
    out = Evaluator(ctx).evaluate(Call("hump", (Field("close"), Constant(1e9))))
    # threshold siêu lớn -> mọi thay đổi đều bị chặn -> chuỗi const = giá trị đầu tiên hợp lệ
    col = 0
    series = out[:, col]
    valid = ~np.isnan(series)
    vals = series[valid]
    assert np.allclose(vals, vals[0])


def test_hump_no_look_ahead(small_panel) -> None:
    from src.data.market_panel import MarketData

    row_t = 60
    mutated_close = small_panel.field("close").copy()
    mutated_close[row_t + 1 :] += 999.0
    mutated = MarketData(
        dates=small_panel.dates, assets=small_panel.assets,
        fields={**small_panel.fields, "close": mutated_close},
        universe=small_panel.universe, returns=small_panel.returns,
        groups=small_panel.groups,
    )
    node = Call("hump", (Field("close"), Constant(0.01)))
    out_orig = Evaluator(_ctx_for(small_panel)).evaluate(node)
    out_mut = Evaluator(_ctx_for(mutated)).evaluate(node)
    np.testing.assert_allclose(out_orig[row_t], out_mut[row_t], equal_nan=True)
```

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_conditional.py -v
```
Expected: FAIL.

- [ ] **Step 2: Impl thật**

```python
# src/operators_local/conditional.py
"""trade_when/hump — conditioning lever (B5: trade_when là nguồn edge chính qua gating;
hump giảm turnover, không nên áp lên alpha có turnover nhanh là bản chất). Cả hai
carry-forward chỉ dùng giá trị tại rows <= t (no-look-ahead)."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


@register(name="trade_when", category=OpCategory.CONDITIONAL,
           signature=(ArgKind.PANEL, ArgKind.PANEL, ArgKind.PANEL), bounded=False,
           commutative=False)
def trade_when(ctx: EvalContext, trigger: Panel, alpha: Panel, exit_cond: Panel) -> Panel:
    out = np.full_like(alpha, np.nan, dtype=np.float64)
    last_valid = np.full(alpha.shape[1], np.nan, dtype=np.float64)
    for t in range(alpha.shape[0]):
        trig_t, exit_t, alpha_t = trigger[t], exit_cond[t], alpha[t]
        take_new = trig_t > 0
        carry = (~take_new) & (exit_t > 0)
        out[t][take_new] = alpha_t[take_new]
        out[t][carry] = last_valid[carry]
        # còn lại (không trigger, không carry) giữ NaN mặc định
        has_val = ~np.isnan(out[t])
        last_valid = np.where(has_val, out[t], last_valid)
    return out


@register(name="hump", category=OpCategory.CONDITIONAL,
           signature=(ArgKind.PANEL, ArgKind.SCALAR), bounded=False, commutative=False)
def hump(ctx: EvalContext, x: Panel, thr: float) -> Panel:
    out = x.copy()
    for col in range(x.shape[1]):
        last = np.nan
        for t in range(x.shape[0]):
            cur = x[t, col]
            if np.isnan(cur):
                continue
            if np.isnan(last) or abs(cur - last) >= thr:
                last = cur
            out[t, col] = last
    return out
```

```bash
venv/Scripts/python.exe -m pytest tests/golden/test_operators_conditional.py -v
```
Expected: PASS.

- [ ] **Step 3: ruff + mypy + commit**

```bash
venv/Scripts/python.exe -m ruff check src/operators_local/conditional.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/operators_local/conditional.py
git add src/operators_local/conditional.py tests/golden/test_operators_conditional.py
git commit -m "feat(operators): trade_when + hump (conditioning lever)"
```

---

### Task 2.9: Wire registry + integration test (parse → eval)

> Chạy SAU khi 2.3–2.8 đã merge (cần cả 6 file operator để `__init__.py` import đủ).

**Files:**
- Modify: `src/operators_local/__init__.py`
- Create: `tests/integration/__init__.py`, `tests/integration/test_eval_pipeline.py`

**Interfaces:**
- Produces: import `src.operators_local` đăng ký toàn bộ 27 operator vào `REGISTRY` toàn
  cục (side-effect import 6 submodule).
- Consumes: `src.lang.parser.parse`, `src.engine.evaluator.{EvalContext, Evaluator}`,
  `src.lang.registry.default_registry`.

- [ ] **Step 1: Test đỏ — pipeline đầy đủ parse→eval, không placeholder nào còn active**

```python
# tests/integration/test_eval_pipeline.py
"""Integration: parse(expr_str) -> Evaluator.evaluate(node) -> (T,N) Panel đúng
NaN-propagation trên small_panel, dùng registry đầy đủ Phase 2 (không placeholder)."""

from __future__ import annotations

import numpy as np

import src.operators_local  # noqa: F401  side-effect: đăng ký toàn bộ operator
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import parse
from src.lang.registry import default_registry


def test_khong_con_placeholder_not_implemented(small_panel) -> None:
    reg = default_registry()
    ctx = EvalContext(data=small_panel, registry=reg)
    node = parse("rank(close)")
    out = Evaluator(ctx).evaluate(node)  # raise NotImplementedError nếu còn placeholder
    assert out.shape == small_panel.universe.shape


def test_pipeline_bieu_thuc_long(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    node = parse("rank(ts_delta(close, 10))")
    out = Evaluator(ctx).evaluate(node)
    in_uni = small_panel.universe
    assert np.all(np.isnan(out[~in_uni]))
    # hàng có đủ lịch sử (t>=9) và universe đầy đủ phải có rank hợp lệ
    row = 50
    assert not np.any(np.isnan(out[row][in_uni[row]]))


def test_tat_ca_operator_co_impl_khong_placeholder(small_panel) -> None:
    reg = default_registry()
    for name in ["add", "subtract", "multiply", "divide", "log", "abs", "sign", "power",
                 "max", "min", "rank", "winsorize", "scale", "zscore", "ts_mean", "ts_std",
                 "ts_delta", "ts_delay", "ts_rank", "ts_zscore", "ts_corr",
                 "ts_decay_linear", "ts_backfill", "group_neutralize", "regression_neut",
                 "vector_neut", "trade_when", "hump"]:
        spec = reg.get(name)
        assert spec.impl.__name__ != "_not_implemented", f"{name} vẫn là placeholder"


def test_gp_function_set_loai_wrapper_config() -> None:
    reg = default_registry()
    gp_names = {s.name for s in reg.gp_function_set()}
    assert "group_neutralize" not in gp_names
    assert "scale" not in gp_names
    assert "regression_neut" in gp_names
    assert "vector_neut" in gp_names
```

```bash
venv/Scripts/python.exe -m pytest tests/integration/test_eval_pipeline.py -v
```
Expected: FAIL nếu `src/operators_local/__init__.py` chưa import 6 submodule (`KeyError`
hoặc placeholder check fail).

- [ ] **Step 2: Wire `__init__.py`**

```python
# src/operators_local/__init__.py
"""Side-effect import: mỗi submodule tự @register() operator vào REGISTRY toàn cục khi
import. Import package này (hoặc bất kỳ submodule) là đủ để có toàn bộ operator Phase 2."""

from __future__ import annotations

from src.operators_local import (  # noqa: F401
    arithmetic,
    conditional,
    cross_sectional,
    group,
    neutralization,
    timeseries,
)
```

```bash
venv/Scripts/python.exe -m pytest tests/integration/test_eval_pipeline.py -v
```
Expected: PASS (4 test).

- [ ] **Step 3: Chạy lại toàn bộ test suite Phase 0–2**

```bash
venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: PASS toàn bộ, không regression Phase 0/1.

- [ ] **Step 4: ruff + mypy toàn repo phần liên quan + commit**

```bash
venv/Scripts/python.exe -m ruff check src/engine src/operators_local tests/golden tests/integration
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/engine src/operators_local
git add src/operators_local/__init__.py tests/integration/__init__.py tests/integration/test_eval_pipeline.py
git commit -m "feat(engine): wire toàn bộ operator + integration test parse->eval"
```

---

### Task 2.10: Review + merge + push

**Files:** không tạo file mới — review toàn bộ diff `phase-2-operator-engine` so với `main`.

- [ ] **Step 1: Self-review checklist**

```bash
venv/Scripts/python.exe -m pytest tests/ -v
venv/Scripts/python.exe -m ruff check src/ tests/
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/
git log --oneline main..phase-2-operator-engine
git diff main..phase-2-operator-engine --stat
```

Kiểm tra thủ công (đối chiếu lại bảng registry + B5/B6):
- [ ] Không còn `impl` nào trỏ tới `_not_implemented` cho 27 operator đã liệt kê.
- [ ] `ts_delay` ≠ `ts_delta`; không có operator tên `delay` trần.
- [ ] `ts_rank`/`rank`/`ts_corr` có `bounded=True`; `ts_zscore`/`zscore` có
  `bounded=False`.
- [ ] `group_neutralize`/`scale` có `gp_usable=False`; `regression_neut`/`vector_neut` có
  `gp_usable=True` và là **duy nhất** 2 op category `NEUTRALIZATION`.
- [ ] Mọi op time-series có test no-look-ahead (đổi rows>t không đổi row t) — đối chiếu
  Task 2.5/2.8 đã cover `ts_mean`, `ts_corr`, `hump`; nếu thiếu ở `ts_rank`/`ts_zscore`/
  `ts_decay_linear`/`ts_backfill`/`ts_delay`/`ts_std`/`ts_delta`, bổ sung trước khi merge
  (logic dùng chung `_window_slice`/`ts_delay` nên rủi ro thấp, nhưng test tường minh là
  bằng chứng).
- [ ] Type hint đầy đủ; không `Any` rò ra ngoài biên `impl: Callable[..., Any]` của
  `OperatorSpec` (đã có từ Phase 1, không đổi).
- [ ] Tiếng Việt trong docstring/commit giữ dấu đúng.
- [ ] Không file nào trong `src/engine`/`src/operators_local` import `src.gp`/`src.storage`/
  `src.llm`.

- [ ] **Step 2: Merge + push**

```bash
git checkout main
git merge --ff-only phase-2-operator-engine
git push
```

Nếu không fast-forward được (main đã tiến), dùng `git merge --no-ff phase-2-operator-engine`
sau khi xác nhận không conflict, rồi push.

---

## Self-review kế hoạch (trước khi giao cho sub-agent)

- **Spec coverage:** 27/27 operator liệt kê ở master plan Phase 2 (arithmetic 10,
  cross_sectional 4, timeseries 9, group 1, neutralization 2, conditional 2 — khớp đề bài;
  `+ - * / log abs sign power max min` = 10 phép arithmetic đúng yêu cầu) đều có Task +
  golden test riêng.
- **Placeholder:** Task 2.9 Step 1 test thứ 3 chặn cứng việc còn `impl.__name__ ==
  "_not_implemented"` cho từng operator — không thể merge nếu còn sót.
- **Type consistency:** `EvalContext`/`Evaluator` dùng đúng chữ ký B6 (`data, registry,
  cache`); mọi hàm operator ký `(ctx: EvalContext, *Panel, **literal) -> Panel` nhất quán
  cả 6 file.
- **Correctness rủi ro cao nhất:** `ts_rank`/`ts_corr`/`ts_decay_linear` dùng vòng lặp
  per-column thuần Python (O(T·N·d)) — đủ nhanh cho `small_panel` (120×30) và golden test,
  nhưng đây là điểm tối ưu numba/vectorize đầu tiên nếu Phase 7 (GP) cần throughput cao hơn
  (note kiến trúc, không phải việc của Phase 2).
- **Ambiguity đã tự quyết:**
  1. Định nghĩa window trailing `[t-d+1, t]` (đúng d quan sát, kể cả t) — chốt tường minh
     vì spec gốc (master plan + B5/B6) không nói rõ biên window.
  2. `scale` đổi category sang `OpCategory.SCALING` (đã có sẵn trong enum B5) thay vì
     `CROSS_SECTIONAL` — khớp đúng vai "wrapper rescale gross-exposure" mà note B5 mô tả
     (rank/sign-preserving, không nên ở chung nhóm với rank/winsorize/zscore vốn vẫn
     `gp_usable=True`).
  3. Literal GROUP arg biểu diễn bằng `Field(name)` trong AST (không thêm node type mới)
     vì Phase 1 chưa định nghĩa cú pháp riêng cho group key — parser Phase 1 parse
     `group_neutralize(x, sector)` với `sector` như identifier thường, khớp grammar hiện
     có; `_literal()` trong Evaluator đọc `Field.name` làm string cho arg GROUP.
  4. `trade_when`/`hump` không có gate/threshold riêng ở `config/thresholds.py` (Global
     Constraint "thresholds chỉ ở config/thresholds.py" áp dụng cho **gate số** dùng để
     pass/fail alpha, không áp dụng cho tham số operator do GP/user truyền trực tiếp vào
     biểu thức — `thr` của `hump` là SCALAR literal trong AST, không phải gate).

**Concerns cần lưu ý khi thực thi:**
- `ts_rank`/`ts_corr`/`ts_decay_linear`/`group_neutralize` dùng vòng lặp Python lồng cấp
  O(T·N) hoặc O(T·N·d) — chấp nhận được cho `small_panel` (120×30) trong CI, nhưng sẽ cần
  vectorize/numba nếu Phase 7 GP eval hàng nghìn cá thể trên panel lớn (ghi chú, không
  block Phase 2).
- `OperatorSpec.impl: Callable[..., Any]` (kế thừa từ Phase 1) khiến `mypy --strict` không
  thể tự suy ra chữ ký từng operator — mỗi hàm impl vẫn cần full type hint riêng để mypy
  bắt lỗi nội bộ file, nhưng lời gọi qua `spec.impl(...)` ở `Evaluator.visit_call` không
  được mypy kiểm tra chặt (chấp nhận theo thiết kế registry B5, không phải lỗi).
