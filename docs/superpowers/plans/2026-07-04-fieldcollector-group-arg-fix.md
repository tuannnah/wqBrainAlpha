# FieldCollector nhầm tham số GROUP thành field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `FieldCollector` không được coi tham số GROUP của operator (vd `sector` trong
`group_neutralize(x, sector)`) là field dữ liệu — chỉ thu thập field ở đúng vị trí
`ArgKind.PANEL`, khớp semantics `registry.get(op).signature`.

**Architecture:** Thêm tham số `registry: OperatorRegistry` BẮT BUỘC vào
`FieldCollector.__init__` (giống hệt `Evaluator(ctx)` đã làm đúng từ trước). `visit_call` tra
`spec = self.registry.get(node.op)`, zip `node.args` với `spec.signature`, chỉ đệ quy vào con có
`ArgKind.PANEL`. Vì đây là thay đổi phá vỡ chữ ký constructor (không có default), MỌI call site
`FieldCollector()` trong repo phải cập nhật CÙNG LÚC trong 1 task/1 commit — không thể tách nhỏ
mà vẫn giữ repo chạy được giữa các commit (một task dở dang = build vỡ).

**Tech Stack:** Python 3.12, pytest. Không thêm dependency mới.

## Global Constraints

- `FieldCollector.__init__(self, registry: OperatorRegistry)` — KHÔNG có default (khác thiết kế
  round-robin trước đó vốn ưu tiên default giữ hành vi cũ; ở đây spec đã chốt "registry bắt buộc"
  để không tái diễn bug ở call site mới trong tương lai).
- KHÔNG xóa/sửa `_GROUPING_FIELDS` ở `src/scoring/power_pool.py` hay
  `src/scoring/dataset_usage.py` — đây là quy tắc nghiệp vụ WQ Brain thật (field vừa có thể là
  field dữ liệu vừa là group-key), độc lập với bug AST này. Chỉ thêm tham số registry.
- Code/comment/commit message bằng tiếng Việt có dấu đầy đủ, khớp văn phong file gốc.
- Test/lệnh chạy qua venv của project: `./venv/Scripts/python.exe -m pytest ...` (python hệ
  thống thiếu dependency `lark`/`psycopg`).
- TDD bắt buộc: viết test trước, xác nhận FAIL đúng lý do, rồi mới sửa code.
- Spec gốc: `docs/superpowers/specs/2026-07-04-fieldcollector-group-arg-design.md`.

---

## Task 1: `FieldCollector` registry-aware + cập nhật toàn bộ call site (1 commit)

**Files:**
- Modify: `src/lang/visitors.py` (class `FieldCollector`, dòng 32-49)
- Modify: `src/pipeline/runner.py:64` (bug thật)
- Modify: `src/backtest/gate.py:51` (bug thật)
- Modify: `src/gp/engine.py:149,199` (bug thật — đây là nơi `GateEvaluator` thật sự chấm
  `fields_ok` cho từng cá thể GP trong `_evaluate_individual`/`_persist`, KHÔNG chỉ là cập nhật
  máy móc — sửa xong sẽ đổi trực tiếp pass/fail của cá thể GP)
- Modify: `src/scoring/power_pool.py:11-12,34` (thêm import + registry, giữ nguyên
  `_GROUPING_FIELDS`)
- Modify: `src/scoring/dataset_usage.py:9-10,23,50` (thêm import + registry, giữ nguyên
  `_GROUPING_FIELDS`)
- Modify: `src/scoring/genius_report.py:9-10,55,71` (thêm import + registry)
- Modify: `scripts/gen_groundtruth.py:27,139` (thêm import + registry)
- Modify: `tests/unit/test_lang_visitors_depth_fields.py` (4 chỗ gọi `FieldCollector()`)
- Modify: `tests/integration/test_storage_minibrain_integration.py:15,31,60` (thêm import +
  registry)
- Modify: `tests/integration/test_metrics_gates.py:39` (đã có sẵn import `default_registry`,
  chỉ sửa lời gọi)
- Test: tất cả các file test kể trên + test mới thêm vào
  `tests/unit/test_lang_visitors_depth_fields.py`

**Interfaces:**
- Consumes: `OperatorRegistry.get(name) -> OperatorSpec` (đã có, `src/lang/registry.py:66-71`),
  `OperatorSpec.signature: tuple[ArgKind, ...]` (đã có), `default_registry() -> OperatorRegistry`
  (đã có, `src/lang/registry.py:152-154`).
- Produces: `FieldCollector(registry: OperatorRegistry)` — chữ ký MỚI, mọi call site trong repo
  (kể cả code chưa viết sau này) phải truyền registry.

- [ ] **Step 1: Viết test tái hiện đúng bug (RED)**

Mở `tests/unit/test_lang_visitors_depth_fields.py`, thêm import và test mới vào cuối file:

```python
from src.lang.parser import parse
from src.lang.registry import default_registry


def test_field_collector_bo_qua_tham_so_group_cua_group_neutralize():
    """Bug thật: group_neutralize(x, sector) có tham số 2 là GROUP (tên nhóm), không phải
    field dữ liệu -- FieldCollector KHÔNG được coi 'sector' là field."""
    import src.operators_local  # noqa: F401  (side-effect: nạp group_neutralize vào registry)
    node = parse("group_neutralize(close, sector)")
    assert FieldCollector(default_registry()).visit(node) == {"close"}
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_lang_visitors_depth_fields.py -k group_neutralize -v`
Expected: `TypeError: FieldCollector() takes no arguments` (hoặc tương tự — constructor hiện tại
không nhận `registry`).

- [ ] **Step 3: Sửa `FieldCollector` trong `src/lang/visitors.py`**

Thay thế class `FieldCollector` (dòng 32-49 hiện tại):

```python
class FieldCollector(NodeVisitor["set[str]"]):
    """Tập tên field được tham chiếu trong cây — phục vụ validate field tồn tại và
    dead-field blacklist (Phase 0.7/Phase 5). Chỉ thu thập field ở đúng vị trí ArgKind.PANEL
    của mỗi operator (tín hiệu thật) -- bỏ qua WINDOW/SCALAR/GROUP (literal, không phải field
    tham chiếu, vd tên group "sector" trong group_neutralize(x, sector)). Khớp semantics
    Evaluator.visit_call (src/engine/evaluator.py) -- registry bắt buộc để tra signature."""

    def __init__(self, registry: OperatorRegistry) -> None:
        self.registry = registry

    def visit(self, node: Node) -> set[str]:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> set[str]:
        return set()

    def visit_field(self, node: Field) -> set[str]:
        return {node.name}

    def visit_call(self, node: Call) -> set[str]:
        spec = self.registry.get(node.op)
        result: set[str] = set()
        for arg, kind in zip(node.args, spec.signature, strict=True):
            if kind is ArgKind.PANEL:
                result |= arg.accept(self)
        return result
```

Thêm `ArgKind` vào import ở đầu `src/lang/visitors.py` (dòng 11 hiện tại đã có
`from src.lang.registry import OperatorRegistry, default_registry` — sửa thành):

```python
from src.lang.registry import ArgKind, OperatorRegistry, default_registry
```

- [ ] **Step 4: Chạy lại test bug-repro, xác nhận PASS**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_lang_visitors_depth_fields.py -k group_neutralize -v`
Expected: 1 passed.

- [ ] **Step 5: Sửa 4 test cũ của `FieldCollector` trong cùng file (giờ sẽ FAIL vì thiếu
  registry)**

Trong `tests/unit/test_lang_visitors_depth_fields.py`, sửa 4 dòng:

```python
def test_field_collector_single_field():
    tree = Call(op="rank", args=(Field("close"),))
    assert FieldCollector(default_registry()).visit(tree) == {"close"}


def test_field_collector_multiple_distinct_fields_deduped():
    tree = Call(op="add", args=(Field("close"), Call(op="ts_mean", args=(Field("close"), Constant(20.0)))))
    assert FieldCollector(default_registry()).visit(tree) == {"close"}


def test_field_collector_no_fields_for_constants_only():
    tree = Call(op="add", args=(Constant(1.0), Constant(2.0)))
    assert FieldCollector(default_registry()).visit(tree) == set()


def test_field_collector_two_distinct_fields():
    tree = Call(op="add", args=(Field("close"), Field("open")))
    assert FieldCollector(default_registry()).visit(tree) == {"close", "open"}
```

(4 op dùng ở đây — `rank`, `add`, `ts_mean` — đều có sẵn trong registry Phase1 tối thiểu với
signature toàn `ArgKind.PANEL`/`WINDOW`, không có `GROUP`, nên kết quả không đổi.)

- [ ] **Step 6: Chạy toàn bộ file, xác nhận PASS hết**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_lang_visitors_depth_fields.py -v`
Expected: 9 passed (4 cũ + 4 DepthVisitor không đổi + 1 test mới).

- [ ] **Step 7: Sửa 2 call site bug thật — `src/pipeline/runner.py` và `src/backtest/gate.py`**

Trong `src/pipeline/runner.py`, dòng 64, sửa:
```python
    fields = FieldCollector(default_registry()).visit(node)
```
(file đã import `default_registry` sẵn ở dòng 23, dùng ở dòng 68 — không cần thêm import.)

Trong `src/backtest/gate.py`, dòng 51, sửa tương tự:
```python
    fields = FieldCollector(default_registry()).visit(node)
```
(file đã import `default_registry` sẵn ở dòng 24 — không cần thêm import.)

- [ ] **Step 8: Sửa `src/gp/engine.py` (2 chỗ, dòng 149 và 199) — dùng `self.registry` có sẵn**

Dòng 149:
```python
        fields = ind.expr.accept(FieldCollector(self.registry))
```

Dòng 199:
```python
        fields = ind.expr.accept(FieldCollector(self.registry))
```

- [ ] **Step 9: Sửa `src/scoring/power_pool.py` (thêm import + registry, GIỮ NGUYÊN
  `_GROUPING_FIELDS`)**

Sửa import (dòng 11):
```python
from src.lang.parser import parse_expression
from src.lang.registry import default_registry
from src.lang.visitors import FieldCollector, OperatorCollector
```

Sửa dòng 34 (bên trong `count_operators_fields`):
```python
    fields = FieldCollector(default_registry()).visit(node) - _GROUPING_FIELDS
```

- [ ] **Step 10: Sửa `src/scoring/dataset_usage.py` (thêm import + registry, GIỮ NGUYÊN
  `_GROUPING_FIELDS`)**

Sửa import (dòng 9):
```python
from src.lang.parser import parse_expression
from src.lang.registry import default_registry
from src.lang.visitors import FieldCollector, OperatorCollector
```

Sửa dòng 23 (trong `dataset_of_alpha`) và dòng 50 (trong `datasets_used`) — cả hai đều đổi từ
`FieldCollector().visit(node)` thành:
```python
    fields = FieldCollector(default_registry()).visit(node)
```

- [ ] **Step 11: Sửa `src/scoring/genius_report.py` (thêm import + registry)**

Sửa import (dòng 9):
```python
from src.lang.parser import parse_expression
from src.lang.registry import default_registry
from src.lang.visitors import FieldCollector, OperatorCollector
```

Dòng 55 (trong `average_distinct_fields_per_alpha`):
```python
    counts = [len(FieldCollector(default_registry()).visit(parse_expression(e))) for e in exprs]
```

Dòng 71 (trong `total_distinct_fields`):
```python
        all_fields |= FieldCollector(default_registry()).visit(parse_expression(e))
```

- [ ] **Step 12: Sửa `scripts/gen_groundtruth.py` (thêm import + registry)**

Sửa import (dòng 27, sau dòng `from src.lang.parser import parse_expression  # noqa: E402`):
```python
from src.lang.parser import parse_expression  # noqa: E402
from src.lang.registry import default_registry  # noqa: E402
from src.lang.visitors import DepthVisitor, FieldCollector  # noqa: E402
```

Dòng 139:
```python
    field_collector = FieldCollector(default_registry())
```

- [ ] **Step 13: Sửa `tests/integration/test_storage_minibrain_integration.py` (thêm import +
  registry)**

Sửa import (dòng 15):
```python
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import CanonicalHasher, ComplexityVisitor, DepthVisitor, FieldCollector
```

Dòng 31 (trong `test_parse_visit_upsert_cache_roundtrip_with_real_ast`) và dòng 60 (trong
`test_failed_expression_recorded_with_reasons_not_cached`) — cả hai đang là
`fields = node.accept(FieldCollector())`, đổi thành:
```python
    fields = node.accept(FieldCollector(default_registry()))
```

- [ ] **Step 14: Sửa `tests/integration/test_metrics_gates.py:39` (đã có sẵn import)**

Dòng 39:
```python
    fields = FieldCollector(default_registry()).visit(node)
```

- [ ] **Step 15: Chạy toàn bộ test suite, xác nhận PASS hết**

Run: `./venv/Scripts/python.exe -m pytest -q`
Expected: tất cả PASS trừ `tests/test_db_postgres.py::test_make_engine_postgres_backend` (lỗi
môi trường có sẵn, thiếu `psycopg`, không liên quan). Nếu thấy `TypeError: FieldCollector()
missing 1 required positional argument` ở bất kỳ đâu khác — nghĩa là còn sót call site, tìm bằng
`grep -rn "FieldCollector()" --include=*.py .` (không kể thư mục `docs/`) và sửa nốt theo đúng
khuôn mẫu ở Step 9-14.

- [ ] **Step 16: Commit toàn bộ (1 commit duy nhất — không tách, vì tách sẽ để lại commit
  trung gian làm vỡ build)**

```bash
git add src/lang/visitors.py src/pipeline/runner.py src/backtest/gate.py src/gp/engine.py \
  src/scoring/power_pool.py src/scoring/dataset_usage.py src/scoring/genius_report.py \
  scripts/gen_groundtruth.py tests/unit/test_lang_visitors_depth_fields.py \
  tests/integration/test_storage_minibrain_integration.py tests/integration/test_metrics_gates.py
git commit -m "$(cat <<'EOF'
fix(lang): FieldCollector bo qua tham so GROUP, khong con nham la field

group_neutralize(x, sector) co tham so 2 la ArgKind.GROUP (ten nhom),
duoc bieu dien bang cung kieu AST node Field nhu field du lieu that --
FieldCollector duyet mu ca tham so nay, khien "sector" bi tinh la field
can ton tai trong panel, lam fields_ok=False oan cho moi cong thuc dung
group_neutralize/group_backfill. Sua theo dung khuon mau
Evaluator.visit_call da lam dung tu truoc: dung registry.signature loc
ArgKind.PANEL truoc khi thu thap field. registry gio la tham so bat
buoc cua FieldCollector (giong Evaluator(ctx)) -- da cap nhat het cac
call site trong repo trong cung 1 commit de khong de lai trang thai
build vo giua chung.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Xác minh hồi quy + xác nhận bug thật đã hết (không commit)

**Files:** không sửa file nào — chỉ chạy kiểm chứng.

**Interfaces:**
- Consumes: toàn bộ thay đổi Task 1.

- [ ] **Step 1: Xác nhận bug thật đã hết qua `score_local_gate` (đúng đường bug đã quan sát khi
  chạy thật mục 5)**

Run:
```bash
cd D:\wq\WorldQuant-Brain-Alpha
./venv/Scripts/python.exe -c "
import numpy as np
import src.operators_local
from src.backtest.config import PortfolioConfig
from src.backtest.gate import score_local_gate
from src.data.market_panel import MarketData

dates = (np.datetime64('2020-01-01') + np.arange(60)).astype('datetime64[D]')
assets = np.array(['A', 'B', 'C'])
close = np.random.default_rng(0).normal(100, 5, size=(60, 3))
universe = np.ones((60, 3), dtype=bool)
returns = np.zeros((60, 3))
sector = np.tile([0, 0, 1], (60, 1))
data = MarketData(dates=dates, assets=assets, fields={'close': close},
    universe=universe, returns=returns, groups={'sector': sector})
cfg = PortfolioConfig(delay=1)
verdict = score_local_gate('group_neutralize(rank(close), sector)', cfg, data)
print('passed:', verdict.passed, '| reason:', verdict.reason)
assert 'fields_ok' not in verdict.reason, 'BUG VAN CON: fields_ok van bi tinh sai'
print('XAC NHAN: fields_ok khong con bi tinh sai cho group_neutralize(..., sector)')
"
```
Expected: dòng cuối in ra `XAC NHAN: ...` — nếu gate vẫn fail thì phải là vì lý do KHÁC
`fields_ok` (vd sharpe/turnover thấp trên data giả lập ngẫu nhiên nhỏ — chấp nhận được, vì mục
đích chỉ là xác nhận `fields_ok` không còn sai).

- [ ] **Step 2: Không commit gì ở Task này** (chỉ là bước xác nhận).

## Ghi chú cho người review / thực thi plan

- Task 1 là MỘT khối không tách được (chữ ký constructor thay đổi phá vỡ mọi call site cùng
  lúc) — không dispatch song song, không tách thành nhiều task nhỏ hơn.
- Không đụng `_GROUPING_FIELDS` ở `power_pool.py`/`dataset_usage.py` — đúng như phạm vi đã chốt
  trong spec.
