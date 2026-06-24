# Phase 1 — Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (khuyến nghị) hoặc superpowers:executing-plans để thực thi từng task. Mỗi step dùng
> checkbox (`- [ ]`); chạy tuần tự trong task, có thể song song hoá giữa Task 1.5–1.9 sau khi
> Task 1.1–1.2 xong (đều ghi vào `src/lang/visitors.py`, nên 1 sub-agent gộp viết tuần tự an
> toàn hơn — xem mục Sub-agent ở cuối).

**Goal:** Dựng tầng ngôn ngữ FASTEXPR-subset cho MiniBrain: AST sealed hierarchy
(`Constant/Field/Call` + `NodeVisitor`), `OperatorRegistry` skeleton (khai báo operator, chưa
impl — đó là Phase 2), grammar Lark, parser validate cú pháp/arity/operator-tồn-tại qua
registry, và 5 visitor (`DepthVisitor/FieldCollector/Serializer/CanonicalHasher/
ComplexityVisitor`). Kết thúc phase: migrate toàn bộ caller của `src/generation/ast_utils.py`
sang AST mới rồi xóa `ast_utils.py` + test cũ của nó.

**Architecture:** `src/lang/{ast,registry,parser,visitors}.py` + `src/lang/grammar.lark`.
Theo dependency rule của master plan: `src/lang` KHÔNG import `src/gp`, `src/storage`,
`src/llm`. Registry là single source of truth được parser dùng để validate; Phase 2 sẽ nạp
impl thật cho từng `OperatorSpec`, Phase 1 chỉ đăng ký placeholder (`impl` raise
`NotImplementedError`) đủ để parser/visitor hoạt động và test được.

**Tech Stack:** Python 3.12, `lark` (cài mới — chưa có trong `requirements.txt`/venv), pytest,
ruff, mypy --strict.

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

## Phạm vi Phase 1 đã chốt (ranh giới registry)

Registry ở Phase 1 là **skeleton khai báo**, chưa impl thật (đó là Phase 2). Để parser
validate được arity/operator-tồn-tại, ta đăng ký các `OperatorSpec` tối thiểu cần cho test
parse — `impl` của mỗi spec raise `NotImplementedError("Phase 2 sẽ impl <name>")`:

| op | category | signature | bounded | commutative |
|---|---|---|---|---|
| `rank` | CROSS_SECTIONAL | `(PANEL,)` | True | False |
| `ts_mean` | TIME_SERIES | `(PANEL, WINDOW)` | False | False |
| `add` | ARITHMETIC | `(PANEL, PANEL)` | False | True |
| `subtract` | ARITHMETIC | `(PANEL, PANEL)` | False | False |
| `divide` | ARITHMETIC | `(PANEL, PANEL)` | False | False |

`close`/`open` là `Field` leaf — không cần đăng ký registry (field hợp lệ được parser chấp
nhận theo cú pháp identifier; validate "field có tồn tại trong `available_fields()`" thuộc
phạm vi Phase 2/integration khi đã có `MarketData` thật, **không** thuộc Phase 1). Phép toán
hạ tầng `+ - * /` trong grammar map trực tiếp sang `add/subtract/multiply/divide` (cần thêm
`multiply` vào bảng trên để 4 toán tử đều resolve được — xem Task 1.2).

---

### Task 1.1: Tạo nhánh + cài `lark`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Tạo nhánh từ main sạch**

```bash
git checkout main && git pull --ff-only 2>/dev/null; git checkout -b phase-1-parser
git status
```
Expected: "On branch phase-1-parser", working tree clean.

- [ ] **Step 2: Thêm `lark` vào requirements + cài vào venv**

Thêm vào `requirements.txt`, ngay dưới dòng `pyarrow>=15` (nhóm "MiniBrain local
backtester"):

```
lark>=1.1
```

```bash
venv/Scripts/python.exe -m pip install "lark>=1.1"
venv/Scripts/python.exe -c "import lark; print(lark.__version__)"
```
Expected: in ra version `lark` (vd `1.x.x`), không lỗi `ModuleNotFoundError`.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat(deps): thêm lark cho parser FASTEXPR-subset (Phase 1)"
```

---

### Task 1.2: AST nodes (`src/lang/ast.py`)

**Files:**
- Create: `src/lang/__init__.py`
- Create: `src/lang/ast.py`
- Test: `tests/unit/test_lang_ast.py`

**Interfaces:**
- Produces: `Node`(ABC, `accept(v)`/`children()`), `Constant(value:float)`,
  `Field(name:str)`, `Call(op:str, args:tuple[Node,...])` — tất cả `frozen=True, slots=True`
  trừ `Node` (ABC không slots); `NodeVisitor[T]` Protocol với
  `visit_constant/visit_field/visit_call`.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_lang_ast.py
"""Test cây AST sealed hierarchy: Constant/Field/Call + NodeVisitor."""

from __future__ import annotations

import pytest

from src.lang.ast import Call, Constant, Field, Node, NodeVisitor


class _CountingVisitor:
    """Visitor tối giản để xác nhận dispatch accept() đúng phương thức."""

    def __init__(self) -> None:
        self.constants = 0
        self.fields = 0
        self.calls = 0

    def visit_constant(self, node: Constant) -> str:
        self.constants += 1
        return f"const:{node.value}"

    def visit_field(self, node: Field) -> str:
        self.fields += 1
        return f"field:{node.name}"

    def visit_call(self, node: Call) -> str:
        self.calls += 1
        return f"call:{node.op}"


def test_constant_is_frozen_and_hashable():
    c = Constant(1.5)
    assert c.value == 1.5
    with pytest.raises(AttributeError):
        c.value = 2.0  # type: ignore[misc]
    hash(c)  # không raise


def test_field_children_empty():
    f = Field("close")
    assert f.children() == ()
    assert f.name == "close"


def test_call_children_returns_args():
    a, b = Field("close"), Constant(5.0)
    call = Call(op="ts_mean", args=(a, b))
    assert call.children() == (a, b)
    assert call.op == "ts_mean"


def test_call_is_frozen_and_hashable():
    call = Call(op="rank", args=(Field("close"),))
    with pytest.raises(AttributeError):
        call.op = "ts_mean"  # type: ignore[misc]
    hash(call)  # không raise (args là tuple — hashable)


def test_accept_dispatches_to_correct_visit_method():
    v = _CountingVisitor()
    tree = Call(op="rank", args=(Field("close"), Constant(5.0)))
    result = tree.accept(v)
    assert result == "call:rank"
    assert v.calls == 1
    tree.args[0].accept(v)
    tree.args[1].accept(v)
    assert v.fields == 1 and v.constants == 1


def test_node_is_abstract_base():
    assert issubclass(Constant, Node)
    assert issubclass(Field, Node)
    assert issubclass(Call, Node)
    with pytest.raises(TypeError):
        Node()  # type: ignore[abstract]


def test_node_visitor_is_protocol_runtime_checkable_via_duck_typing():
    # NodeVisitor là Protocol thuần (không @runtime_checkable bắt buộc) —
    # xác nhận _CountingVisitor thỏa cấu trúc bằng cách dùng trực tiếp.
    v: NodeVisitor[str] = _CountingVisitor()
    assert v.visit_field(Field("open")) == "field:open"
```

- [ ] **Step 2: Chạy test — đỏ**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_ast.py -v
```
Expected: FAIL `ModuleNotFoundError: src.lang.ast` (hoặc `src.lang`).

- [ ] **Step 3: Tạo `src/lang/__init__.py` + `src/lang/ast.py`**

```python
# src/lang/__init__.py
"""Tầng ngôn ngữ FASTEXPR-subset: AST, registry operator, grammar, parser, visitors.

Dependency rule (master plan): src/lang KHÔNG import src/gp, src/storage, src/llm.
"""
```

```python
# src/lang/ast.py
"""Cây AST sealed hierarchy cho FASTEXPR-subset.

Ba loại node: `Constant` (literal số), `Field` (tên cột dữ liệu, vd "close"), `Call`
(operator/hàm với danh sách args). Mỗi node bất biến (frozen+slots) để an toàn dùng làm
khóa cache/hash. Visitor pattern (`NodeVisitor` Protocol) tách mọi phân tích (depth, hash,
serialize, eval...) ra khỏi node — open/closed: thêm phân tích mới = thêm visitor, không
sửa node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, TypeVar

T = TypeVar("T")


class NodeVisitor(Protocol[T]):
    """Hợp đồng visitor: một phương thức `visit_*` cho mỗi loại node cụ thể."""

    def visit_constant(self, node: Constant) -> T: ...

    def visit_field(self, node: Field) -> T: ...

    def visit_call(self, node: Call) -> T: ...


class Node(ABC):
    """Nút trừu tượng của AST. Không tự sealed bằng cú pháp Python, nhưng chỉ
    `Constant/Field/Call` được định nghĩa trong module này — coi là sealed theo quy ước."""

    @abstractmethod
    def accept(self, v: NodeVisitor[T]) -> T:
        """Gọi đúng phương thức `visit_*` tương ứng loại node cụ thể (double dispatch)."""
        raise NotImplementedError

    @abstractmethod
    def children(self) -> tuple[Node, ...]:
        """Các node con trực tiếp; rỗng cho leaf (`Constant`/`Field`)."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Constant(Node):
    """Literal số (window int hoặc threshold float) — leaf, không có con."""

    value: float

    def accept(self, v: NodeVisitor[T]) -> T:
        return v.visit_constant(self)

    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class Field(Node):
    """Tham chiếu tới một cột dữ liệu thị trường theo tên (vd "close", "volume") — leaf."""

    name: str

    def accept(self, v: NodeVisitor[T]) -> T:
        return v.visit_field(self)

    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class Call(Node):
    """Lời gọi operator/hàm: `op` phải tồn tại trong OperatorRegistry; `args` định vị,
    có thể trộn sub-expression (Field/Call) và literal (Constant)."""

    op: str
    args: tuple[Node, ...]

    def accept(self, v: NodeVisitor[T]) -> T:
        return v.visit_call(self)

    def children(self) -> tuple[Node, ...]:
        return self.args
```

- [ ] **Step 4: Chạy test — xanh**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_ast.py -v
```
Expected: PASS (7 test).

- [ ] **Step 5: Commit**

```bash
git add src/lang/__init__.py src/lang/ast.py tests/unit/test_lang_ast.py
git commit -m "feat(lang): AST sealed hierarchy Constant/Field/Call + NodeVisitor"
```

---

### Task 1.3: Registry (`src/lang/registry.py`)

**Files:**
- Create: `src/lang/registry.py`
- Test: `tests/unit/test_lang_registry.py`

**Interfaces:**
- Consumes: không (module độc lập trong `src/lang`).
- Produces: `ArgKind` (enum: `PANEL, WINDOW, SCALAR, GROUP`), `OpCategory` (enum:
  `ARITHMETIC, CROSS_SECTIONAL, TIME_SERIES, GROUP, NEUTRALIZATION, CONDITIONAL, SCALING`),
  `OperatorSpec(name, category, signature, impl, bounded, depth_cost=1, gp_usable=True,
  window_choices=(5,10,20,60,120), commutative=False)` (frozen, slots),
  `OperatorRegistry` (`register(spec)`, `get(name)->OperatorSpec` raise `KeyError` nếu
  thiếu, `by_category(c)->list[OperatorSpec]`, `gp_function_set()->list[OperatorSpec]` chỉ
  trả `gp_usable=True`), decorator `register(name, category, signature, bounded,
  **kwargs)` đăng ký vào registry toàn cục `REGISTRY` và trả lại hàm gốc (để file
  `operators_local/*.py` dùng `@register(...)` lên hàm impl ở Phase 2), hàm
  `default_registry() -> OperatorRegistry` trả `REGISTRY` toàn cục đã có 5 op tối thiểu của
  Phase 1 (`rank, ts_mean, add, subtract, multiply, divide`) đăng ký sẵn (impl placeholder
  raise `NotImplementedError`).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_lang_registry.py
"""Test OperatorRegistry: đăng ký, lookup, lọc theo category/gp_usable."""

from __future__ import annotations

import pytest

from src.lang.registry import (
    ArgKind,
    OpCategory,
    OperatorRegistry,
    OperatorSpec,
    default_registry,
)


def _placeholder(*_args: object) -> object:
    raise NotImplementedError("placeholder test impl")


def test_register_and_get_roundtrip():
    reg = OperatorRegistry()
    spec = OperatorSpec(
        name="rank",
        category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,),
        impl=_placeholder,
        bounded=True,
    )
    reg.register(spec)
    assert reg.get("rank") is spec


def test_get_unknown_op_raises_keyerror():
    reg = OperatorRegistry()
    with pytest.raises(KeyError):
        reg.get("not_an_op")


def test_by_category_filters():
    reg = OperatorRegistry()
    rank_spec = OperatorSpec(
        name="rank", category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,), impl=_placeholder, bounded=True,
    )
    ts_mean_spec = OperatorSpec(
        name="ts_mean", category=OpCategory.TIME_SERIES,
        signature=(ArgKind.PANEL, ArgKind.WINDOW), impl=_placeholder, bounded=False,
    )
    reg.register(rank_spec)
    reg.register(ts_mean_spec)
    assert reg.by_category(OpCategory.CROSS_SECTIONAL) == [rank_spec]
    assert reg.by_category(OpCategory.TIME_SERIES) == [ts_mean_spec]


def test_gp_function_set_excludes_non_gp_usable():
    reg = OperatorRegistry()
    core = OperatorSpec(
        name="rank", category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,), impl=_placeholder, bounded=True, gp_usable=True,
    )
    wrapper = OperatorSpec(
        name="group_neutralize", category=OpCategory.GROUP,
        signature=(ArgKind.PANEL, ArgKind.GROUP), impl=_placeholder, bounded=False,
        gp_usable=False,
    )
    reg.register(core)
    reg.register(wrapper)
    fn_set = reg.gp_function_set()
    assert core in fn_set
    assert wrapper not in fn_set


def test_operator_spec_is_frozen():
    spec = OperatorSpec(
        name="rank", category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,), impl=_placeholder, bounded=True,
    )
    with pytest.raises(AttributeError):
        spec.name = "other"  # type: ignore[misc]


def test_default_registry_has_minimal_phase1_ops():
    reg = default_registry()
    for name in ("rank", "ts_mean", "add", "subtract", "multiply", "divide"):
        spec = reg.get(name)
        assert spec.name == name
        with pytest.raises(NotImplementedError):
            spec.impl()


def test_default_registry_arithmetic_ops_are_panel_panel_binary():
    reg = default_registry()
    for name in ("add", "subtract", "multiply", "divide"):
        spec = reg.get(name)
        assert spec.signature == (ArgKind.PANEL, ArgKind.PANEL)


def test_default_registry_add_is_commutative_others_not():
    reg = default_registry()
    assert reg.get("add").commutative is True
    assert reg.get("subtract").commutative is False
    assert reg.get("divide").commutative is False
```

- [ ] **Step 2: Chạy test — đỏ**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_registry.py -v
```
Expected: FAIL `ModuleNotFoundError: src.lang.registry`.

- [ ] **Step 3: Tạo file**

```python
# src/lang/registry.py
"""OperatorRegistry — nguồn sự thật duy nhất về operator FASTEXPR-subset.

Parser (Phase 1) dùng registry để validate operator tồn tại + arity. Evaluator (Phase 2)
dùng để dispatch impl thật. GP (Phase 7) dùng `gp_function_set()` để xây function set
(chỉ operator lõi, loại các wrapper config như group_neutralize/scale).

Ranh giới Phase 1: registry chỉ đăng ký SPEC (khai báo), `impl` của operator còn thiếu
trong Phase 1 là placeholder raise NotImplementedError — Phase 2 mới nạp impl thật qua
decorator `@register(...)` đặt lên hàm trong `src/operators_local/*.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ArgKind(Enum):
    """Loại đối số dương vị trí của một operator."""

    PANEL = auto()  # sub-expression bay hơi thành (T, N)
    WINDOW = auto()  # số nguyên dương lookback (vd 5, 10, 20)
    SCALAR = auto()  # literal float (ngưỡng, hệ số scale...)
    GROUP = auto()  # tên group key (vd "sector")


class OpCategory(Enum):
    """Phân nhóm operator — dùng để lọc function set GP và áp luật stage-separation."""

    ARITHMETIC = auto()
    CROSS_SECTIONAL = auto()
    TIME_SERIES = auto()
    GROUP = auto()
    NEUTRALIZATION = auto()
    CONDITIONAL = auto()
    SCALING = auto()


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Khai báo đầy đủ một operator: tên, nhóm, chữ ký đối số, impl, và các cờ cho GP/gate."""

    name: str
    category: OpCategory
    signature: tuple[ArgKind, ...]
    impl: Callable[..., Any]
    bounded: bool
    depth_cost: int = 1
    gp_usable: bool = True
    window_choices: tuple[int, ...] = (5, 10, 20, 60, 120)
    commutative: bool = False


class OperatorRegistry:
    """Bảng tra operator theo tên; nguồn sự thật cho parser/evaluator/GP."""

    def __init__(self) -> None:
        self._ops: dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        """Đăng ký một OperatorSpec; ghi đè nếu tên đã tồn tại (cho phép redefinition test)."""
        self._ops[spec.name] = spec

    def get(self, name: str) -> OperatorSpec:
        """Trả OperatorSpec theo tên; raise KeyError với thông điệp rõ nếu không tồn tại."""
        try:
            return self._ops[name]
        except KeyError as exc:
            raise KeyError(f"operator không tồn tại trong registry: {name!r}") from exc

    def by_category(self, c: OpCategory) -> list[OperatorSpec]:
        """Mọi OperatorSpec thuộc category `c`, theo thứ tự đăng ký."""
        return [spec for spec in self._ops.values() if spec.category is c]

    def gp_function_set(self) -> list[OperatorSpec]:
        """Operator lõi dùng được cho GP (gp_usable=True) — loại wrapper config."""
        return [spec for spec in self._ops.values() if spec.gp_usable]


REGISTRY = OperatorRegistry()


def register(
    name: str,
    category: OpCategory,
    signature: tuple[ArgKind, ...],
    bounded: bool,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: đăng ký hàm bên dưới làm impl của operator `name` vào REGISTRY toàn cục,
    trả lại hàm gốc không đổi (để vẫn gọi/test được trực tiếp)."""

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY.register(
            OperatorSpec(
                name=name, category=category, signature=signature,
                impl=fn, bounded=bounded, **kwargs,
            )
        )
        return fn

    return _wrap


def _not_implemented(*_args: Any, **_kwargs: Any) -> Any:
    """Impl placeholder cho operator Phase 1 chưa có logic thật (đó là việc của Phase 2)."""
    raise NotImplementedError("Impl operator thuộc Phase 2 (Operator Engine)")


def _register_phase1_minimal_ops() -> None:
    """Đăng ký tập operator tối thiểu cần để Parser (Phase 1) validate được arity/tồn tại.

    Đây KHÔNG phải danh sách operator đầy đủ của MiniBrain — chỉ đủ cho test parse của
    Phase 1 (rank/ts_mean/4 phép số học nhị phân). Phase 2 đăng ký toàn bộ operator thật
    qua `src/operators_local/*.py` (ghi đè placeholder này bằng impl thật cùng tên).
    """
    REGISTRY.register(OperatorSpec(
        name="rank", category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,), impl=_not_implemented, bounded=True,
    ))
    REGISTRY.register(OperatorSpec(
        name="ts_mean", category=OpCategory.TIME_SERIES,
        signature=(ArgKind.PANEL, ArgKind.WINDOW), impl=_not_implemented, bounded=False,
    ))
    REGISTRY.register(OperatorSpec(
        name="add", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_not_implemented, bounded=False,
        commutative=True,
    ))
    REGISTRY.register(OperatorSpec(
        name="subtract", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_not_implemented, bounded=False,
        commutative=False,
    ))
    REGISTRY.register(OperatorSpec(
        name="multiply", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_not_implemented, bounded=False,
        commutative=True,
    ))
    REGISTRY.register(OperatorSpec(
        name="divide", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_not_implemented, bounded=False,
        commutative=False,
    ))


_register_phase1_minimal_ops()


def default_registry() -> OperatorRegistry:
    """Registry toàn cục với tập operator tối thiểu Phase 1 đã đăng ký sẵn."""
    return REGISTRY
```

> Lưu ý: `field` import từ `dataclasses` không được dùng trong file trên — KHÔNG import nó
> (tránh lỗi ruff unused-import). `OperatorSpec` chỉ cần `dataclass` thường.

- [ ] **Step 4: Chạy test — xanh**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_registry.py -v
```
Expected: PASS (8 test).

- [ ] **Step 5: Commit**

```bash
git add src/lang/registry.py tests/unit/test_lang_registry.py
git commit -m "feat(lang): OperatorRegistry + ArgKind/OpCategory + 6 op tối thiểu Phase 1"
```

---

### Task 1.4: Grammar (`src/lang/grammar.lark`)

**Files:**
- Create: `src/lang/grammar.lark`
- Test: `tests/unit/test_lang_grammar.py`

**Interfaces:**
- Produces: file grammar Lark độc lập (không phải module Python) — test load bằng
  `lark.Lark(grammar_text, parser="lalr")` và `.parse(expr)` trả `lark.Tree` thô (chưa
  transform sang AST — đó là Task 1.5).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_lang_grammar.py
"""Test grammar.lark load được và parse cú pháp FASTEXPR-subset thành Tree thô."""

from __future__ import annotations

from pathlib import Path

import pytest
from lark import Lark
from lark.exceptions import UnexpectedInput

GRAMMAR_PATH = Path(__file__).resolve().parents[2] / "src" / "lang" / "grammar.lark"


@pytest.fixture(scope="module")
def lark_parser() -> Lark:
    text = GRAMMAR_PATH.read_text(encoding="utf-8")
    return Lark(text, parser="lalr", start="start")


def test_grammar_file_exists():
    assert GRAMMAR_PATH.is_file()


@pytest.mark.parametrize(
    "expr",
    [
        "close",
        "5",
        "5.5",
        "rank(close)",
        "ts_mean(close, 20)",
        "add(close, open)",
        "close + open",
        "close - open",
        "close * 2",
        "close / 2",
        "rank(ts_mean(close, 20))",
        "rank(close) + rank(open)",
    ],
)
def test_grammar_parses_valid_expressions(lark_parser: Lark, expr: str):
    tree = lark_parser.parse(expr)
    assert tree is not None


@pytest.mark.parametrize(
    "expr",
    [
        "",
        "rank(",
        "close +",
        "rank(close,)",
        "@close",
    ],
)
def test_grammar_rejects_invalid_syntax(lark_parser: Lark, expr: str):
    with pytest.raises(UnexpectedInput):
        lark_parser.parse(expr)
```

- [ ] **Step 2: Chạy test — đỏ**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_grammar.py -v
```
Expected: FAIL (file `grammar.lark` không tồn tại → `FileNotFoundError` trong fixture).

- [ ] **Step 3: Tạo file**

```lark
// src/lang/grammar.lark
// Grammar FASTEXPR-subset cho MiniBrain (Phase 1).
//
// Hỗ trợ: field (identifier không theo dấu '('), số (int/float), lời gọi hàm với
// danh sách args phân tách bởi ',', và 4 toán tử nhị phân + - * / với precedence chuẩn
// (* / cao hơn + -), đều left-associative. Không hỗ trợ unary minus ở Phase 1 (FASTEXPR
// thật biểu diễn qua subtract(0, x) hoặc multiply(x, -1) khi cần — giữ grammar tối giản).

start: expr

?expr: term
     | expr "+" term   -> add_expr
     | expr "-" term   -> sub_expr

?term: atom
     | term "*" atom    -> mul_expr
     | term "/" atom    -> div_expr

?atom: NUMBER           -> number_atom
     | call
     | field
     | "(" expr ")"

call: NAME "(" [expr ("," expr)*] ")"
field: NAME

NAME: /[A-Za-z_][A-Za-z0-9_]*/
NUMBER: /\d+(\.\d+)?/

%import common.WS
%ignore WS
```

- [ ] **Step 4: Chạy test — xanh**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_grammar.py -v
```
Expected: PASS (1 + 12 + 5 test). Nếu `rank(close,)` (trailing comma) **không** raise vì
grammar LALR chấp nhận nó do ambiguity — sửa rule `call` để bắt buộc không có comma cuối
(grammar trên dùng `[expr ("," expr)*]` nên không có trailing comma hợp lệ; nếu Lark vẫn
parse được, kiểm tra lại bằng cách thêm test riêng và siết rule trước khi qua Step 5).

- [ ] **Step 5: Commit**

```bash
git add src/lang/grammar.lark tests/unit/test_lang_grammar.py
git commit -m "feat(lang): grammar Lark FASTEXPR-subset (field/number/call/+-*/)"
```

---

### Task 1.5: Parser + transformer (`src/lang/parser.py`)

**Files:**
- Create: `src/lang/parser.py`
- Test: `tests/unit/test_lang_parser.py`

**Interfaces:**
- Consumes: `OperatorRegistry`/`default_registry()` (1.3), grammar (1.4), `Node/Constant/
  Field/Call` (1.2).
- Produces: `class ParseError(ValueError)`; `parse(text: str, registry: OperatorRegistry |
  None = None) -> Node` (raise `ParseError` rõ nguyên nhân cho cú pháp sai / operator không
  tồn tại / sai arity); chạy được `python -m src.lang.parser "<expr>"` (in ra
  `Serializer`-style repr tạm — Task 1.5 chỉ cần in `repr(node)`, vì `Serializer` thật chưa
  tồn tại tới Task 1.7; `__main__` sẽ được nâng cấp ở Task 1.7 sau khi có `Serializer`).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_lang_parser.py
"""Test parser: string FASTEXPR-subset -> AST, validate operator/arity qua registry."""

from __future__ import annotations

import subprocess
import sys

import pytest

from src.lang.ast import Call, Constant, Field
from src.lang.parser import ParseError, parse
from src.lang.registry import ArgKind, OpCategory, OperatorRegistry, OperatorSpec


def _placeholder(*_a: object) -> object:
    raise NotImplementedError


def _registry_with_rank_and_arith() -> OperatorRegistry:
    reg = OperatorRegistry()
    reg.register(OperatorSpec(
        name="rank", category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,), impl=_placeholder, bounded=True,
    ))
    reg.register(OperatorSpec(
        name="ts_mean", category=OpCategory.TIME_SERIES,
        signature=(ArgKind.PANEL, ArgKind.WINDOW), impl=_placeholder, bounded=False,
    ))
    reg.register(OperatorSpec(
        name="add", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_placeholder, bounded=False,
        commutative=True,
    ))
    return reg


def test_parse_field_leaf():
    node = parse("close", registry=_registry_with_rank_and_arith())
    assert node == Field("close")


def test_parse_number_leaf():
    node = parse("5", registry=_registry_with_rank_and_arith())
    assert node == Constant(5.0)


def test_parse_call_single_arg():
    node = parse("rank(close)", registry=_registry_with_rank_and_arith())
    assert node == Call(op="rank", args=(Field("close"),))


def test_parse_call_with_window():
    node = parse("ts_mean(close, 20)", registry=_registry_with_rank_and_arith())
    assert node == Call(op="ts_mean", args=(Field("close"), Constant(20.0)))


def test_parse_binary_plus_maps_to_add():
    node = parse("close + open", registry=_registry_with_rank_and_arith())
    assert node == Call(op="add", args=(Field("close"), Field("open")))


def test_parse_nested_call():
    node = parse("rank(ts_mean(close, 20))", registry=_registry_with_rank_and_arith())
    assert node == Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(20.0))),))


def test_parse_unknown_operator_raises_parse_error():
    with pytest.raises(ParseError, match="not_an_op"):
        parse("not_an_op(close)", registry=_registry_with_rank_and_arith())


def test_parse_wrong_arity_raises_parse_error():
    with pytest.raises(ParseError, match="arity"):
        parse("rank(close, open)", registry=_registry_with_rank_and_arith())


def test_parse_invalid_syntax_raises_parse_error():
    with pytest.raises(ParseError):
        parse("rank(", registry=_registry_with_rank_and_arith())


def test_parse_uses_default_registry_when_none_given():
    # default_registry() (Task 1.3) đã có "rank" đăng ký sẵn (impl placeholder Phase 1).
    node = parse("rank(close)")
    assert node == Call(op="rank", args=(Field("close"),))


def test_module_runs_as_main_and_prints_node():
    result = subprocess.run(
        [sys.executable, "-m", "src.lang.parser", "rank(close)"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "rank" in result.stdout
    assert "close" in result.stdout
```

- [ ] **Step 2: Chạy test — đỏ**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_parser.py -v
```
Expected: FAIL `ModuleNotFoundError: src.lang.parser`.

- [ ] **Step 3: Tạo file**

```python
# src/lang/parser.py
"""Parser FASTEXPR-subset: chuỗi -> AST (Node), validate operator/arity qua OperatorRegistry.

Dùng `lark` với grammar `grammar.lark` (Task 1.4) để parse cú pháp; một `lark.Transformer`
biến `lark.Tree` thành `Node` (Constant/Field/Call). Toán tử nhị phân `+ - * /` map sang
operator registry `add/subtract/multiply/divide` để mọi computation đi qua MỘT con đường
(Call) — không có node BinOp riêng.

Chạy độc lập: `python -m src.lang.parser "rank(close)"`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lark import Lark, Token, Transformer
from lark.exceptions import UnexpectedInput

from src.lang.ast import Call, Constant, Field, Node
from src.lang.registry import ArgKind, OperatorRegistry, default_registry

_GRAMMAR_PATH = Path(__file__).resolve().parent / "grammar.lark"
_GRAMMAR_TEXT = _GRAMMAR_PATH.read_text(encoding="utf-8")

_BINARY_OP_NAME = {
    "add_expr": "add",
    "sub_expr": "subtract",
    "mul_expr": "multiply",
    "div_expr": "divide",
}


class ParseError(ValueError):
    """Lỗi parse: cú pháp sai, operator không tồn tại, hoặc sai số lượng đối số."""


class _ToAst(Transformer[Token, Node]):
    """Biến lark.Tree (theo grammar.lark) thành cây Node, validate qua registry khi gặp Call."""

    def __init__(self, registry: OperatorRegistry) -> None:
        super().__init__()
        self._registry = registry

    def number_atom(self, children: list[Token]) -> Constant:
        return Constant(float(children[0]))

    def field(self, children: list[Token]) -> Field:
        return Field(str(children[0]))

    def call(self, children: list[object]) -> Call:
        name = str(children[0])
        args = tuple(c for c in children[1:] if isinstance(c, Node))
        self._validate(name, len(args))
        return Call(op=name, args=args)

    def add_expr(self, children: list[Node]) -> Call:
        return self._binary("add_expr", children)

    def sub_expr(self, children: list[Node]) -> Call:
        return self._binary("sub_expr", children)

    def mul_expr(self, children: list[Node]) -> Call:
        return self._binary("mul_expr", children)

    def div_expr(self, children: list[Node]) -> Call:
        return self._binary("div_expr", children)

    def _binary(self, rule_name: str, children: list[Node]) -> Call:
        op_name = _BINARY_OP_NAME[rule_name]
        left, right = children
        self._validate(op_name, 2)
        return Call(op=op_name, args=(left, right))

    def _validate(self, name: str, n_args: int) -> None:
        try:
            spec = self._registry.get(name)
        except KeyError as exc:
            raise ParseError(f"operator không tồn tại trong registry: {name!r}") from exc
        if len(spec.signature) != n_args:
            raise ParseError(
                f"sai arity cho operator {name!r}: cần {len(spec.signature)} đối số "
                f"({[k.name for k in spec.signature]}), nhận {n_args}"
            )


def _build_lark(start: str = "start") -> Lark:
    return Lark(_GRAMMAR_TEXT, parser="lalr", start=start)


_LARK = _build_lark()


def parse(text: str, registry: OperatorRegistry | None = None) -> Node:
    """Parse chuỗi FASTEXPR-subset thành AST; raise ParseError nếu cú pháp/operator/arity sai."""
    reg = registry if registry is not None else default_registry()
    try:
        tree = _LARK.parse(text)
    except UnexpectedInput as exc:
        raise ParseError(f"cú pháp không hợp lệ tại: {text!r} ({exc})") from exc
    result = _ToAst(reg).transform(tree)
    if not isinstance(result, Node):
        raise ParseError(f"không parse được thành AST hợp lệ: {text!r}")
    return result


def _arg_kind_label(kind: ArgKind) -> str:
    return kind.name


if __name__ == "__main__":
    expr = sys.argv[1] if len(sys.argv) > 1 else "rank(close)"
    print(repr(parse(expr)))
```

> Lưu ý: hàm `_arg_kind_label` không dùng tới nội bộ — sẽ gỡ nếu mypy/ruff không cần; thực
> tế: **không thêm hàm thừa**. Bỏ `_arg_kind_label` khỏi file thật (đã sửa ở bản trên — chỉ
> giữ nếu một test cần). Vì test trên không gọi `_arg_kind_label`, XÓA nó khỏi impl trước
> khi chạy ruff (không để dead code).

- [ ] **Step 4: Chạy test — xanh**

Trước khi chạy: xóa hàm `_arg_kind_label` (không dùng, dead code) khỏi `src/lang/parser.py`.

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_parser.py -v
```
Expected: PASS (11 test).

- [ ] **Step 5: Commit**

```bash
git add src/lang/parser.py tests/unit/test_lang_parser.py
git commit -m "feat(lang): parser FASTEXPR-subset string->AST validate qua registry"
```

---

### Task 1.6: `DepthVisitor` + `FieldCollector` (`src/lang/visitors.py`)

**Files:**
- Create: `src/lang/visitors.py`
- Test: `tests/unit/test_lang_visitors_depth_fields.py`

**Interfaces:**
- Consumes: `Node/Constant/Field/Call` (1.2).
- Produces: `class DepthVisitor` với `visit(node: Node) -> int` (alias gọi qua
  `node.accept(self)`); `class FieldCollector` với `visit(node: Node) -> set[str]`. Quy ước
  đếm depth: leaf (`Constant`/`Field`) có depth 1; `Call` có depth `1 + max(depth con,
  default=0)` — tức **đếm cả node Call wrapper** (đúng yêu cầu "DepthVisitor đếm cả
  wrapper").

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_lang_visitors_depth_fields.py
"""Test DepthVisitor (đếm cả wrapper) và FieldCollector."""

from __future__ import annotations

from src.lang.ast import Call, Constant, Field
from src.lang.visitors import DepthVisitor, FieldCollector


def test_depth_of_leaf_is_one():
    assert DepthVisitor().visit(Field("close")) == 1
    assert DepthVisitor().visit(Constant(5.0)) == 1


def test_depth_of_single_call_is_two():
    tree = Call(op="rank", args=(Field("close"),))
    assert DepthVisitor().visit(tree) == 2


def test_depth_counts_wrapper_call():
    # rank(ts_mean(close, 20)) -> rank(1) -> ts_mean(2) -> close/20(3) => depth 3
    tree = Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(20.0))),))
    assert DepthVisitor().visit(tree) == 3


def test_depth_takes_max_over_multiple_children():
    # add(close, ts_mean(close,20)) -> add(1) -> [close(2), ts_mean(2)->[close,20](3)]
    tree = Call(op="add", args=(
        Field("close"),
        Call(op="ts_mean", args=(Field("close"), Constant(20.0))),
    ))
    assert DepthVisitor().visit(tree) == 3


def test_field_collector_single_field():
    tree = Call(op="rank", args=(Field("close"),))
    assert FieldCollector().visit(tree) == {"close"}


def test_field_collector_multiple_distinct_fields_deduped():
    tree = Call(op="add", args=(Field("close"), Call(op="ts_mean", args=(Field("close"), Constant(20.0)))))
    assert FieldCollector().visit(tree) == {"close"}


def test_field_collector_no_fields_for_constants_only():
    tree = Call(op="add", args=(Constant(1.0), Constant(2.0)))
    assert FieldCollector().visit(tree) == set()


def test_field_collector_two_distinct_fields():
    tree = Call(op="add", args=(Field("close"), Field("open")))
    assert FieldCollector().visit(tree) == {"close", "open"}
```

- [ ] **Step 2: Chạy test — đỏ**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_visitors_depth_fields.py -v
```
Expected: FAIL `ModuleNotFoundError: src.lang.visitors`.

- [ ] **Step 3: Tạo file (phần Depth + FieldCollector; các visitor khác thêm ở Task 1.7–1.9)**

```python
# src/lang/visitors.py
"""Visitor cụ thể trên AST: DepthVisitor, FieldCollector, Serializer, CanonicalHasher,
ComplexityVisitor. Mỗi visitor một trách nhiệm (B4 design) — không tangle với evaluator.
"""

from __future__ import annotations

from src.lang.ast import Call, Constant, Field, Node, NodeVisitor


class DepthVisitor(NodeVisitor[int]):
    """Độ sâu tối đa của cây, ĐẾM CẢ wrapper Call (vd rank(...) tính 1 tầng độc lập với
    số args). Leaf có depth 1; Call có depth 1 + max(depth con, mặc định 0 nếu rỗng)."""

    def visit(self, node: Node) -> int:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> int:
        return 1

    def visit_field(self, node: Field) -> int:
        return 1

    def visit_call(self, node: Call) -> int:
        child_depths = [c.accept(self) for c in node.children()]
        return 1 + (max(child_depths) if child_depths else 0)


class FieldCollector(NodeVisitor["set[str]"]):
    """Tập tên field được tham chiếu trong cây — phục vụ validate field tồn tại và
    dead-field blacklist (Phase 0.7/Phase 5)."""

    def visit(self, node: Node) -> set[str]:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> set[str]:
        return set()

    def visit_field(self, node: Field) -> set[str]:
        return {node.name}

    def visit_call(self, node: Call) -> set[str]:
        result: set[str] = set()
        for c in node.children():
            result |= c.accept(self)
        return result
```

- [ ] **Step 4: Chạy test — xanh**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_visitors_depth_fields.py -v
```
Expected: PASS (8 test).

- [ ] **Step 5: Commit**

```bash
git add src/lang/visitors.py tests/unit/test_lang_visitors_depth_fields.py
git commit -m "feat(lang): DepthVisitor (đếm wrapper) + FieldCollector"
```

---

### Task 1.7: `Serializer` (`src/lang/visitors.py`) + nâng `parser.py` `__main__`

**Files:**
- Modify: `src/lang/visitors.py`
- Modify: `src/lang/parser.py` (nâng `__main__` dùng `Serializer` thay vì `repr`)
- Test: `tests/unit/test_lang_visitors_serializer.py`

**Interfaces:**
- Consumes: `Node/Constant/Field/Call`, `parse` (1.5).
- Produces: `class Serializer` với `visit(node: Node) -> str` render về FASTEXPR
  canonical: `Constant` → `repr(float)` rút gọn (vd `20.0` → `"20"` nếu là số nguyên,
  ngược lại giữ số thực — quy ước: in `int(value)` nếu `value.is_integer()` else
  `repr(value)`); `Field` → tên thẳng; `Call` → `"{op}({arg1}, {arg2}, ...)"` (luôn dạng
  hàm, KHÔNG render lại `add/subtract/multiply/divide` thành `+ - * /` — vì round-trip với
  parser chỉ cần `parse(serialize(node)) == node`, và `parse("add(a,b)")` hợp lệ theo
  grammar Task 1.4).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_lang_visitors_serializer.py
"""Test Serializer: AST -> string FASTEXPR canonical, round-trip với parser."""

from __future__ import annotations

from src.lang.ast import Call, Constant, Field
from src.lang.parser import parse
from src.lang.registry import ArgKind, OpCategory, OperatorRegistry, OperatorSpec
from src.lang.visitors import Serializer


def _placeholder(*_a: object) -> object:
    raise NotImplementedError


def _registry() -> OperatorRegistry:
    reg = OperatorRegistry()
    reg.register(OperatorSpec(
        name="rank", category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,), impl=_placeholder, bounded=True,
    ))
    reg.register(OperatorSpec(
        name="ts_mean", category=OpCategory.TIME_SERIES,
        signature=(ArgKind.PANEL, ArgKind.WINDOW), impl=_placeholder, bounded=False,
    ))
    reg.register(OperatorSpec(
        name="add", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_placeholder, bounded=False,
        commutative=True,
    ))
    return reg


def test_serialize_field():
    assert Serializer().visit(Field("close")) == "close"


def test_serialize_integer_constant_no_decimal():
    assert Serializer().visit(Constant(20.0)) == "20"


def test_serialize_fractional_constant_keeps_decimal():
    assert Serializer().visit(Constant(0.5)) == "0.5"


def test_serialize_call():
    tree = Call(op="ts_mean", args=(Field("close"), Constant(20.0)))
    assert Serializer().visit(tree) == "ts_mean(close, 20)"


def test_serialize_nested_call():
    tree = Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(20.0))),))
    assert Serializer().visit(tree) == "rank(ts_mean(close, 20))"


def test_round_trip_with_parser():
    reg = _registry()
    original = parse("rank(ts_mean(close, 20))", registry=reg)
    text = Serializer().visit(original)
    reparsed = parse(text, registry=reg)
    assert reparsed == original


def test_round_trip_preserves_binary_op_as_function_call():
    reg = _registry()
    original = parse("close + open" if False else "add(close, open)", registry=reg)
    text = Serializer().visit(original)
    assert text == "add(close, open)"
    assert parse(text, registry=reg) == original
```

- [ ] **Step 2: Chạy test — đỏ**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_visitors_serializer.py -v
```
Expected: FAIL `ImportError: cannot import name 'Serializer'`.

- [ ] **Step 3: Thêm `Serializer` vào `src/lang/visitors.py`**

Thêm vào cuối `src/lang/visitors.py`:

```python
class Serializer(NodeVisitor[str]):
    """AST -> chuỗi FASTEXPR canonical. Round-trip với parser:
    parse(Serializer().visit(node)) == node. Toán tử nhị phân luôn render dạng hàm
    (vd `add(a, b)`), không dạng infix — đơn giản hóa round-trip (grammar Task 1.4 chấp
    nhận cả hai dạng nhưng AST không phân biệt nguồn gốc cú pháp)."""

    def visit(self, node: Node) -> str:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> str:
        if node.value.is_integer():
            return str(int(node.value))
        return repr(node.value)

    def visit_field(self, node: Field) -> str:
        return node.name

    def visit_call(self, node: Call) -> str:
        args = ", ".join(c.accept(self) for c in node.children())
        return f"{node.op}({args})"
```

Sửa `src/lang/parser.py`: thay khối `if __name__ == "__main__":` để dùng `Serializer`
thay vì `repr`:

```python
if __name__ == "__main__":
    from src.lang.visitors import Serializer

    expr = sys.argv[1] if len(sys.argv) > 1 else "rank(close)"
    node = parse(expr)
    print(Serializer().visit(node))
```

- [ ] **Step 4: Chạy test — xanh**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_visitors_serializer.py tests/unit/test_lang_parser.py -v
```
Expected: PASS toàn bộ (lưu ý `test_module_runs_as_main_and_prints_node` ở Task 1.5 vẫn
PASS vì output vẫn chứa `"rank"` và `"close"`, giờ là `"rank(close)"` thay vì repr dataclass).

- [ ] **Step 5: Commit**

```bash
git add src/lang/visitors.py src/lang/parser.py tests/unit/test_lang_visitors_serializer.py
git commit -m "feat(lang): Serializer AST->FASTEXPR canonical, round-trip parser; CLI dùng Serializer"
```

---

### Task 1.8: `CanonicalHasher` (`src/lang/visitors.py`)

**Files:**
- Modify: `src/lang/visitors.py`
- Test: `tests/unit/test_lang_visitors_hasher.py`

**Interfaces:**
- Consumes: `Node/Constant/Field/Call`, `OperatorRegistry` (để biết `commutative` — cần
  registry vì hash phải sort args của operator commutative, ví dụ `add(a,b)` ==
  `add(b,a)`).
- Produces: `class CanonicalHasher` với `__init__(self, registry: OperatorRegistry |
  None = None)` (mặc định `default_registry()`), `visit(node: Node) -> str` trả hash
  `sha256` hex ổn định sau canonicalize: (a) literal `Constant` normalize qua
  `repr(float(value))` trước khi hash (loại phân biệt `5` vs `5.0` ở tầng input — cả hai
  đều là `float` trong AST nên đã tự nhiên giống nhau); (b) nếu `Call.op` có
  `registry.get(op).commutative is True`, sort các chuỗi-con-đã-hash của `args` trước khi
  ghép (đảm bảo `add(a,b)` và `add(b,a)` cho cùng hash); (c) op không tồn tại trong
  registry (an toàn fallback) coi như không commutative.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_lang_visitors_hasher.py
"""Test CanonicalHasher: hash ổn định, sort args commutative, normalize literal."""

from __future__ import annotations

from src.lang.ast import Call, Constant, Field
from src.lang.registry import ArgKind, OpCategory, OperatorRegistry, OperatorSpec
from src.lang.visitors import CanonicalHasher


def _placeholder(*_a: object) -> object:
    raise NotImplementedError


def _registry() -> OperatorRegistry:
    reg = OperatorRegistry()
    reg.register(OperatorSpec(
        name="add", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_placeholder, bounded=False,
        commutative=True,
    ))
    reg.register(OperatorSpec(
        name="subtract", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_placeholder, bounded=False,
        commutative=False,
    ))
    return reg


def test_hash_is_deterministic_for_same_tree():
    reg = _registry()
    tree = Call(op="add", args=(Field("close"), Field("open")))
    h1 = CanonicalHasher(reg).visit(tree)
    h2 = CanonicalHasher(reg).visit(tree)
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64  # sha256 hex digest


def test_hash_differs_for_different_trees():
    reg = _registry()
    t1 = Call(op="add", args=(Field("close"), Field("open")))
    t2 = Call(op="add", args=(Field("close"), Field("volume")))
    assert CanonicalHasher(reg).visit(t1) != CanonicalHasher(reg).visit(t2)


def test_hash_same_for_commutative_args_swapped():
    reg = _registry()
    t1 = Call(op="add", args=(Field("close"), Field("open")))
    t2 = Call(op="add", args=(Field("open"), Field("close")))
    assert CanonicalHasher(reg).visit(t1) == CanonicalHasher(reg).visit(t2)


def test_hash_differs_for_non_commutative_args_swapped():
    reg = _registry()
    t1 = Call(op="subtract", args=(Field("close"), Field("open")))
    t2 = Call(op="subtract", args=(Field("open"), Field("close")))
    assert CanonicalHasher(reg).visit(t1) != CanonicalHasher(reg).visit(t2)


def test_hash_normalizes_literal_representation():
    reg = _registry()
    # Constant sinh từ "20" hay từ float(20) đều cùng giá trị 20.0 -> cùng hash.
    t1 = Call(op="add", args=(Field("close"), Constant(20.0)))
    t2 = Call(op="add", args=(Field("close"), Constant(float("20"))))
    assert CanonicalHasher(reg).visit(t1) == CanonicalHasher(reg).visit(t2)


def test_hash_unknown_op_falls_back_non_commutative():
    reg = _registry()  # "mystery_op" không đăng ký
    t1 = Call(op="mystery_op", args=(Field("close"), Field("open")))
    t2 = Call(op="mystery_op", args=(Field("open"), Field("close")))
    assert CanonicalHasher(reg).visit(t1) != CanonicalHasher(reg).visit(t2)
```

- [ ] **Step 2: Chạy test — đỏ**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_visitors_hasher.py -v
```
Expected: FAIL `ImportError: cannot import name 'CanonicalHasher'`.

- [ ] **Step 3: Thêm `CanonicalHasher` vào `src/lang/visitors.py`**

Thêm import `hashlib` ở đầu file và thêm class ở cuối:

```python
# Thêm vào đầu file (sau "from __future__ import annotations"):
import hashlib

from src.lang.registry import OperatorRegistry, default_registry
```

```python
class CanonicalHasher(NodeVisitor[str]):
    """Hash sha256-hex ổn định sau canonicalize: literal normalize qua repr(float),
    args của operator commutative (theo registry) được sort trước khi ghép — đảm bảo
    add(a,b) và add(b,a) cho cùng hash. Dùng cho sub-expression cache, result cache,
    dedup quần thể GP (B12)."""

    def __init__(self, registry: OperatorRegistry | None = None) -> None:
        self._registry = registry if registry is not None else default_registry()

    def visit(self, node: Node) -> str:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> str:
        return self._digest(f"const:{repr(float(node.value))}")

    def visit_field(self, node: Field) -> str:
        return self._digest(f"field:{node.name}")

    def visit_call(self, node: Call) -> str:
        child_hashes = [c.accept(self) for c in node.children()]
        if self._is_commutative(node.op):
            child_hashes = sorted(child_hashes)
        return self._digest(f"call:{node.op}({','.join(child_hashes)})")

    def _is_commutative(self, op: str) -> bool:
        try:
            return self._registry.get(op).commutative
        except KeyError:
            return False

    @staticmethod
    def _digest(payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Chạy test — xanh**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_visitors_hasher.py -v
```
Expected: PASS (6 test).

- [ ] **Step 5: Commit**

```bash
git add src/lang/visitors.py tests/unit/test_lang_visitors_hasher.py
git commit -m "feat(lang): CanonicalHasher sha256 sort-commutative + normalize literal"
```

---

### Task 1.9: `ComplexityVisitor` (`src/lang/visitors.py`)

**Files:**
- Modify: `src/lang/visitors.py`
- Test: `tests/unit/test_lang_visitors_complexity.py`

**Interfaces:**
- Consumes: `Node/Constant/Field/Call`.
- Produces: `class ComplexityVisitor` với `visit(node: Node) -> int` = tổng số node trong
  cây (node count), tính cả leaf và Call (dùng cho GP anti-bloat penalty, Phase 7).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_lang_visitors_complexity.py
"""Test ComplexityVisitor: node count toàn cây (leaf + Call)."""

from __future__ import annotations

from src.lang.ast import Call, Constant, Field
from src.lang.visitors import ComplexityVisitor


def test_complexity_of_single_leaf_is_one():
    assert ComplexityVisitor().visit(Field("close")) == 1
    assert ComplexityVisitor().visit(Constant(5.0)) == 1


def test_complexity_counts_call_plus_children():
    tree = Call(op="rank", args=(Field("close"),))
    # rank + close = 2
    assert ComplexityVisitor().visit(tree) == 2


def test_complexity_of_nested_tree():
    # ts_mean(close, 20) -> ts_mean + close + 20 = 3 ; rank(...) -> rank + 3 = 4
    tree = Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(20.0))),))
    assert ComplexityVisitor().visit(tree) == 4


def test_complexity_of_binary_with_two_fields():
    tree = Call(op="add", args=(Field("close"), Field("open")))
    # add + close + open = 3
    assert ComplexityVisitor().visit(tree) == 3
```

- [ ] **Step 2: Chạy test — đỏ**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_visitors_complexity.py -v
```
Expected: FAIL `ImportError: cannot import name 'ComplexityVisitor'`.

- [ ] **Step 3: Thêm `ComplexityVisitor` vào cuối `src/lang/visitors.py`**

```python
class ComplexityVisitor(NodeVisitor[int]):
    """Số node toàn cây (leaf + Call) — proxy độ phức tạp cho GP anti-bloat penalty
    (Phase 7, FitnessVector.complexity_penalty)."""

    def visit(self, node: Node) -> int:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> int:
        return 1

    def visit_field(self, node: Field) -> int:
        return 1

    def visit_call(self, node: Call) -> int:
        return 1 + sum(c.accept(self) for c in node.children())
```

- [ ] **Step 4: Chạy test — xanh**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_lang_visitors_complexity.py -v
```
Expected: PASS (4 test).

- [ ] **Step 5: Commit**

```bash
git add src/lang/visitors.py tests/unit/test_lang_visitors_complexity.py
git commit -m "feat(lang): ComplexityVisitor node-count cho GP anti-bloat"
```

---

### Task 1.10: Migrate caller của `ast_utils` sang AST mới + xóa `ast_utils.py`

**Files:**
- Modify: `src/simulation/pre_filter.py`, `src/simulation/simulator.py`,
  `src/scoring/complexity.py`, `src/decorrelation/zoo.py`, `src/generation/novel_ideas.py`,
  `src/decorrelation/similarity.py`, `src/generation/local_select.py`,
  `src/llm/generator.py`, `src/llm/expr_synth.py`
- Delete: `src/generation/ast_utils.py`, `tests/test_ast_utils.py`
- Test: chạy lại toàn bộ test cũ của các module trên (không file test mới riêng — đây là
  refactor giữ hành vi, TDD ở đây = "test cũ phải tiếp tục xanh", không viết test mới).

**Interfaces:**
- Cũ (`src/generation/ast_utils.py`): `Leaf(value)`, `Node(op, children)`,
  `parse_expression(str) -> Node|Leaf`, `to_expression(node) -> str`,
  `tree_depth(node) -> int`, `node_count(node) -> int`, `all_subtrees(node) -> list`,
  `iter_leaves(node) -> Iterator[Leaf]`.
- Mới (mapping 1:1 sang `src/lang/{ast,parser,visitors}.py`):
  - `parse_expression(s)` → `src.lang.parser.parse(s)` (trả `Node` mới, không còn
    `Leaf`/`Node` cũ).
  - `to_expression(node)` → `src.lang.visitors.Serializer().visit(node)`.
  - `tree_depth(node)` → `src.lang.visitors.DepthVisitor().visit(node)`.
  - `node_count(node)` → `src.lang.visitors.ComplexityVisitor().visit(node)`.
  - `all_subtrees(node)` → hàm thuần mới `src.lang.visitors.all_subtrees(node: Node) ->
    list[Node]` (chưa có — thêm vào `visitors.py`, không phải visitor class vì chỉ là
    duyệt cây thường, không cần dispatch).
  - `iter_leaves(node)` → hàm thuần mới `src.lang.visitors.iter_leaves(node: Node) ->
    Iterator[Constant | Field]`.
  - `Leaf(value)` cũ gộp 2 vai (field tên chuỗi VÀ literal số) → AST mới tách riêng
    `Field(name: str)` và `Constant(value: float)`. Caller cũ check
    `isinstance(x, Leaf)` cần đổi thành `isinstance(x, (Field, Constant))` hoặc tách rõ
    theo ngữ cảnh (xem từng file dưới).

> **Vì sao gộp 1 task không TDD-step-by-step như các task trên:** đây là refactor cơ học
> (thay import + sửa call site), rủi ro là hành vi cũ bị vỡ — "đỏ" ở đây tự nhiên là
> **chạy test cũ trước khi sửa và thấy chúng xanh (baseline)**, sửa code, rồi chạy lại xác
> nhận vẫn xanh. KHÔNG bỏ qua bước review từng file.

- [ ] **Step 1: Baseline — xác nhận test cũ đang xanh trước khi đổi gì**

```bash
venv/Scripts/python.exe -m pytest tests/test_ast_utils.py tests/ -k "pre_filter or simulator or complexity or zoo or novel_ideas or similarity or local_select or generator or expr_synth" -v
```
Expected: PASS toàn bộ (baseline trước refactor). Ghi lại số lượng test PASS để so sánh sau.

- [ ] **Step 2: Thêm `all_subtrees`/`iter_leaves` vào `src/lang/visitors.py`**

Thêm vào `src/lang/visitors.py` (cuối file, sau `ComplexityVisitor`):

```python
from collections.abc import Iterator


def all_subtrees(node: Node) -> list[Node]:
    """Mọi sub-node của cây (gồm cả leaf và chính `node`) — duyệt pre-order.
    Dùng cho điểm crossover/mutation của GP (Phase 7); tương đương `ast_utils.all_subtrees`
    cũ nhưng trên AST mới (Constant/Field/Call)."""
    result: list[Node] = [node]
    for child in node.children():
        result.extend(all_subtrees(child))
    return result


def iter_leaves(node: Node) -> Iterator[Constant | Field]:
    """Duyệt mọi leaf (Constant hoặc Field) của cây, theo thứ tự trái-phải."""
    if isinstance(node, (Constant, Field)):
        yield node
    else:
        for child in node.children():
            yield from iter_leaves(child)
```

> Đặt `from collections.abc import Iterator` ở đầu file cùng nhóm import khác (không lặp
> import giữa file) — di chuyển dòng `from collections.abc import Iterator` lên đầu
> `src/lang/visitors.py` cùng `import hashlib` khi áp dụng patch thật.

- [ ] **Step 3: Đọc kỹ 9 file caller, sửa từng file (import + call site)**

Với mỗi file, đổi import:

```python
# CŨ (xóa)
from src.generation.ast_utils import Leaf, Node, parse_expression, to_expression, ...

# MỚI (thêm, chỉ những gì thực sự dùng trong file đó)
from src.lang.ast import Constant, Field, Node
from src.lang.parser import parse as parse_expression
from src.lang.visitors import (
    CanonicalHasher, ComplexityVisitor, DepthVisitor, Serializer,
    all_subtrees, iter_leaves,
)
```

Quy tắc sửa call site theo từng API cũ dùng trong file (đọc file thật trước khi sửa, vì
mapping `isinstance(x, Leaf)` phụ thuộc ngữ cảnh — file dùng `Leaf` để biểu diễn field
tên (so `isinstance(x, str)`-like) hay literal số):

- `to_expression(node)` → `Serializer().visit(node)`.
- `tree_depth(node)` → `DepthVisitor().visit(node)`.
- `node_count(node)` → `ComplexityVisitor().visit(node)`.
- `all_subtrees(node)` → `all_subtrees(node)` (hàm mới, cùng tên, import từ
  `src.lang.visitors`).
- `iter_leaves(node)` → `iter_leaves(node)` (hàm mới cùng tên).
- `isinstance(x, Leaf)` (đang dùng để phân biệt leaf vs node có con) →
  `isinstance(x, (Constant, Field))`.
- `Leaf(value)` khi `value` là tên field (chuỗi không phải số) → `Field(value)`.
- `Leaf(value)` khi `value` là số → `Constant(float(value))`.
- `leaf.value` khi đã biết là field → `field_node.name`; khi đã biết là số →
  `const_node.value`.
- `Node(op, children)` khi cần TỰ XÂY cây (không phải parse từ string) → `Call(op=op,
  args=tuple(children))`.

Áp dụng cho từng file — đọc nguyên file trước khi sửa (`Read` rồi `Edit`), theo thứ tự
phụ thuộc tăng dần (file không phụ thuộc file khác trong danh sách trước):

1. `src/scoring/complexity.py` — dùng `Leaf, iter_leaves, parse_expression, tree_depth`.
2. `src/decorrelation/zoo.py` — dùng `parse_expression`.
3. `src/decorrelation/similarity.py` — dùng tổ hợp (đọc import block để biết chính xác).
4. `src/generation/novel_ideas.py` — dùng `Leaf, iter_leaves, parse_expression`.
5. `src/generation/local_select.py` — dùng `parse_expression`.
6. `src/llm/generator.py` — dùng `iter_leaves, parse_expression`.
7. `src/llm/expr_synth.py` — dùng `Leaf, Node, parse_expression, to_expression`.
8. `src/simulation/pre_filter.py` — đọc import block (`from ... import (` nhiều dòng).
9. `src/simulation/simulator.py` — dùng `Leaf, Node, all_subtrees, parse_expression`.

Với mỗi file: sửa import + mọi call site liên quan, sau đó chạy ngay test riêng của module
đó (nếu có file test riêng, vd `tests/test_scoring_complexity.py`) trước khi sang file
tiếp theo — tránh dồn lỗi.

- [ ] **Step 4: Chạy full test cũ liên quan — phải xanh như baseline Step 1**

```bash
venv/Scripts/python.exe -m pytest tests/ -k "pre_filter or simulator or complexity or zoo or novel_ideas or similarity or local_select or generator or expr_synth" -v
```
Expected: PASS, **cùng số lượng test PASS như Step 1** (không test nào bị skip/xóa ngầm).

- [ ] **Step 5: Chạy TOÀN BỘ test suite (không chỉ phần liên quan) để chắc không vỡ chỗ khác**

```bash
venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: PASS toàn bộ (trừ test thuộc `tests/test_ast_utils.py` sẽ bị xóa ở Step 6 — nếu
còn tồn tại lúc này nó PHẢI vẫn PASS vì `ast_utils.py` chưa xóa).

- [ ] **Step 6: Xóa `ast_utils.py` + test cũ của nó (chỉ sau khi Step 5 xanh hoàn toàn)**

```bash
git rm src/generation/ast_utils.py tests/test_ast_utils.py
venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: PASS toàn bộ (không còn `ModuleNotFoundError` nào do file đã xóa — vì Step 3 đã
gỡ hết import `ast_utils` khỏi 9 caller).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(lang): migrate 9 caller ast_utils sang src/lang/ + xóa ast_utils.py"
```

---

### Task 1.11: Review + gate + merge

**Files:** —

- [ ] **Step 1: Full test suite**

```bash
venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: PASS toàn bộ, không lỗi/skip ngoài ý muốn.

- [ ] **Step 2: ruff**

```bash
venv/Scripts/python.exe -m ruff check src/lang tests/unit/test_lang_ast.py tests/unit/test_lang_registry.py tests/unit/test_lang_grammar.py tests/unit/test_lang_parser.py tests/unit/test_lang_visitors_depth_fields.py tests/unit/test_lang_visitors_serializer.py tests/unit/test_lang_visitors_hasher.py tests/unit/test_lang_visitors_complexity.py
```
Expected: "All checks passed!" — sửa mọi lỗi (unused import, line length...) trước khi tiếp.

- [ ] **Step 3: mypy --strict cho module mới**

```bash
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/lang
```
Expected: "Success: no issues found". Lỗi thường gặp cần xử lý:
- `NodeVisitor[T]` generic Protocol dùng làm base class cho visitor cụ thể
  (`DepthVisitor(NodeVisitor[int])`) — mypy strict có thể đòi implement đủ 3 method; đã có.
- `lark.Transformer[Token, Node]` generic — nếu `lark` thiếu stub strict, thêm
  `# type: ignore[misc]` CÓ chú thích lý do tại đúng dòng `class _ToAst(...)`, không tắt
  strict toàn file.
- `OperatorSpec.impl: Callable[..., Any]` — `Any` ở đây là chủ ý (impl ký hiệu đa dạng theo
  operator), không phải lỗi cần sửa.

- [ ] **Step 4: Cập nhật `PROGRESS.md`** (skill `session-journal`) — append entry Phase 1:
  done; quyết định ranh giới registry skeleton; 9 file migrate khỏi `ast_utils`; next =
  Phase 2 (Operator Engine).

- [ ] **Step 5: Merge + push**

```bash
git checkout main
git merge --no-ff phase-1-parser -m "merge: Phase 1 — parser (AST + registry + grammar + visitors)"
git push origin main
```

**DoD Phase 1:** `parse(str)->Node` chạy CLI được; 5 visitor đúng hành vi (Depth đếm
wrapper, FieldCollector dedup, Serializer round-trip, CanonicalHasher sort-commutative +
normalize literal, ComplexityVisitor node-count); registry skeleton với 6 op tối thiểu;
`ast_utils.py` + test cũ đã xóa, 9 caller migrate xong, full suite cũ xanh; ruff + mypy
--strict clean trên `src/lang`.

---

## Sub-agent

Task 1.1–1.4 tuần tự (mỗi task phụ thuộc task trước: nhánh → registry cần ast.py → grammar
độc lập nhưng parser cần cả registry+grammar). Task 1.6–1.9 đều sửa cùng file
`src/lang/visitors.py` nên **không song song hoá bằng nhiều sub-agent ghi đồng thời** —
chạy tuần tự trong 1 sub-agent (hoặc 1 sub-agent/task nhưng nối tiếp, review diff trước khi
sang task kế để tránh conflict ghi đè cùng file). Task 1.10 là refactor rủi ro cao (9 file
thật) — 1 sub-agent xuyên suốt, đọc từng file trước khi sửa, không đoán API. Task 1.11 cuối
cùng, sau khi 1.10 merge xong.

---

## Self-review

**Spec coverage (so với master plan Phase 1 + B4/B5):**
- 1.1 AST nodes → Task 1.2 ✔ (Node ABC + Constant/Field/Call frozen+slots + NodeVisitor Protocol)
- 1.2 Registry → Task 1.3 ✔ (ArgKind/OpCategory enum, OperatorSpec, OperatorRegistry, @register, gp_function_set)
- 1.3 Grammar → Task 1.4 ✔ (field, number, call(args), + - * /)
- 1.4 Parser + transformer → Task 1.5 ✔ (parse(str)->Node, lỗi rõ op/arity/cú pháp, `python -m src.lang.parser` chạy)
- 1.5 DepthVisitor → Task 1.6 ✔ (đếm cả wrapper)
- 1.6 FieldCollector → Task 1.6 ✔
- 1.7 Serializer → Task 1.7 ✔ (round-trip với parser)
- 1.8 CanonicalHasher → Task 1.8 ✔ (sort commutative + normalize literal)
- 1.9 ComplexityVisitor → Task 1.9 ✔ (node count)
- 1.10 Migrate + xóa ast_utils → Task 1.10 ✔ (9 caller thật — đã grep xác nhận, không chỉ
  2 file nêu trong master plan; mapping API 1:1 ghi rõ)

**Placeholder scan:** không "TBD"/"tương tự task N" nào trong code — mọi step có code đầy
đủ. `_not_implemented`/`NotImplementedError` trong registry là CHỦ Ý theo đúng ranh giới
Phase 1 (ghi rõ trong "Phạm vi Phase 1 đã chốt"), không phải placeholder bỏ quên — Task
1.11 DoD xác nhận registry skeleton là kết quả mong đợi, không phải nợ kỹ thuật ẩn.

**Type consistency:** `Node/Constant/Field/Call` định nghĩa 1 lần ở `ast.py`, mọi file sau
(`registry.py` chỉ dùng tên operator dạng `str`, không import `ast.py`; `parser.py`,
`visitors.py`) import đúng cùng class, không định nghĩa lại. `OperatorRegistry.get`
signature nhất quán `(name: str) -> OperatorSpec` xuyên suốt Task 1.3/1.5/1.8.
`Serializer`/`DepthVisitor`/`FieldCollector`/`CanonicalHasher`/`ComplexityVisitor` đều
implement đúng `NodeVisitor[T]` Protocol (3 method `visit_constant/visit_field/visit_call`)
và thêm `.visit(node)` convenience method gọi `node.accept(self)` — quy ước đồng nhất giữa
5 visitor để caller (Task 1.10 và Phase 2+) gọi giống nhau `XxxVisitor().visit(node)`.

**Rủi ro còn mở (không thuộc phạm vi Phase 1, ghi để Phase 2 biết):** validate "field có
tồn tại trong `MarketData.available_fields()`" CHƯA làm ở Phase 1 (chỉ validate cú
pháp/operator/arity) — `Field("bất_kỳ_tên_gì")` parse được dù field không tồn tại trong dữ
liệu thật; đây thuộc Phase 2 (Evaluator cần `MarketData` thật để biết field nào hợp lệ).
