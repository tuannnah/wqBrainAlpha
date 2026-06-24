# Phase 7 — GP Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (khuyến nghị) hoặc superpowers:executing-plans để thực thi từng task. Task 7.1
> (`Individual`) làm trước tiên (mọi thứ khác bọc quanh nó). Task 7.2–7.6
> (FitnessVector/Seeds/Init/Variation/Selection) **song song được sau 7.1** — không phụ
> thuộc lẫn nhau về import (chỉ phụ thuộc `Individual` + AST + registry + Phase 4/5/6).
> Task 7.7 (`GPEngine`) phụ thuộc TẤT CẢ 7.1–7.6. Task 7.8 (tích hợp loop + xóa template)
> luôn cuối cùng, sau khi 7.7 chạy được. Task 7.9 (review/merge/push) luôn cuối cùng tuyệt
> đối.

**Goal:** Dựng tầng Genetic Programming của MiniBrain — sinh, lai ghép, đột biến, chọn lọc
quần thể **biểu thức signal core** (không phải toàn bộ alpha có config) trên typed AST
(Phase 1), correlation-aware từ ngày đầu (pool + population), persist mọi outcome (pass và
fail) vào DB (Phase 5), evaluate bằng Phase 2 (Evaluator) + Phase 3 (Backtester) + Phase 4
(MetricsCalculator/GateEvaluator) + Phase 6 (PoolCorrelation). Đây là phase **XL** cuối
cùng trước short-list/CLI (Phase 8) — biến MiniBrain từ "chấm điểm 1 alpha viết tay" thành
"tự sinh hàng nghìn alpha/ngày được xếp hạng và khử tương quan". Sau khi engine chạy được,
xóa hẳn cơ chế sinh template cũ (`src/generation/template.py`) — GP thay thế nó hoàn toàn
cho vai trò "sinh biến thể có cấu trúc".

**Architecture:** Package mới `src/gp/{individual,fitness_vec,seeds,init,variation,
selection,engine}.py`. Theo dependency rule B1 (master plan): `src/gp` được phép import
`src/lang` (AST/registry/visitors), `src/engine` (Evaluator), `src/backtest`
(PortfolioBuilder/Backtester/MetricsCalculator/GateEvaluator/PoolCorrelation),
`src/storage` (AlphaRepository — GP là tầng "app", được phép phụ thuộc storage, khác với
`lang/operators_local/engine/backtest` là tầng thấp hơn không được phụ thuộc storage/gp).
`src/gp/seeds.py` thêm được phép import `src/generation/{families,novel_ideas}.py` (đọc
template kinh điển làm hạt giống) và **tùy chọn** `src/llm/{hypothesis,translator}.py`
(sinh hạt giống mới qua LLM) — đây là chiều phụ thuộc MỚI (`gp` → `llm`), không vi phạm B1
vì B1 chỉ cấm `lang/operators_local/engine/backtest` phụ thuộc `gp/storage/llm`; không nói
gì về `gp` phụ thuộc `llm` (hợp lý: GP là tầng tìm kiếm, được phép dùng LLM như một nguồn
hạt giống, miễn `llm` không phụ thuộc ngược lại `gp`). `src/gp/engine.py` là điểm tích hợp
duy nhất gọi seeds→init→variation→selection→eval theo vòng lặp tiến hóa.

**Tech Stack:** Python 3.12, numpy, joblib (parallel eval — THÊM vào `requirements.txt`
nếu chưa có, kiểm tra trước), pytest, ruff, mypy --strict. Không thêm `deap`/`numba`/`ray`
(quyết định A5 master spec — hand-rolled GP đủ kiểm soát cho stage separation + typed
crossover; chỉ đổi sang `deap` nếu sau này hand-rolled selector là bottleneck đo được).

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

## Pre-condition (đọc trước khi bắt đầu) — Phase 4/4.5/5/6 là PLAN, có thể chưa merge

Tại thời điểm viết plan này, `src/backtest/`, `src/engine/`, `src/operators_local/` **chưa
tồn tại** trong repo thật (chỉ tồn tại dưới dạng plan step-by-step ở
`docs/superpowers/plans/2026-06-24-phase-{2,3,4,4.5,5,6}-*.md`). Trước khi bắt đầu Task 7.1,
chạy:

```bash
venv/Scripts/python.exe -c "
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor, ComplexityVisitor, CanonicalHasher, all_subtrees
from src.engine.evaluator import Evaluator, EvalContext
from src.backtest.config import PortfolioConfig
from src.backtest.portfolio import PortfolioBuilder
from src.backtest.backtester import Backtester
from src.backtest.metrics_local import AlphaMetrics, MetricsCalculator
from src.backtest.gates import GateEvaluator, GateVerdict
from src.backtest.pool_corr import PoolCorrelation
from src.storage.repository import AlphaRepository
print('phase 1-6 ok')
"
```

- Nếu in `phase 1-6 ok` → tiếp tục bình thường, dùng chữ ký THẬT đọc trực tiếp từ các file
  đó (không suy diễn từ plan — implementation thật có thể lệch nhỏ so với plan nếu người
  thực thi đã tự quyết định khác đi, đúng tinh thần "ghi rõ quyết định tại chỗ" của các plan
  trước).
- Nếu `ModuleNotFoundError` ở bất kỳ import nào → Phase tương ứng chưa merge vào `main`.
  Task 7.1 (`Individual`, chỉ phụ thuộc `src.lang`) **không bị khoá** — làm trước, không
  chờ. Task 7.2 (`FitnessVector`) cần `AlphaMetrics`/`PoolCorrelation` thật để viết hàm dựng
  — nếu thiếu, implement `FitnessVector` (dataclass thuần) độc lập trước, viết hàm dựng
  `from_metrics(...)` ở bước riêng khi Phase 4/6 sẵn sàng (xem Task 7.2 chi tiết). Task
  7.3–7.6 chỉ phụ thuộc `src.lang` + `Individual` (7.1) — không bị khoá bởi Phase 4/5/6.
  Task 7.7 (`GPEngine`) là nơi TẤT CẢ các phụ thuộc hội tụ — nếu Phase 4/5/6 chưa xong khi
  tới đây, DỪNG, báo cáo block, không tự chế stub Evaluator/Backtester/Repository (lấn phạm
  vi phase khác).

**Mâu thuẫn chữ ký `load_pool` giữa Phase 5 và Phase 6 — ĐỌC TRƯỚC Task 7.7:** Phase 5 plan
(`docs/superpowers/plans/2026-06-24-phase-5-database.md`, dòng ~599) định nghĩa
`load_pool(self) -> dict[int, npt.NDArray[np.float64]]` (chỉ pnl, không kèm dates). Phase 6
plan (dòng ~74, ~576) định nghĩa `load_pool(self) -> dict[int, tuple[Dates,
npt.NDArray[np.float64]]]` (kèm dates — cần để align candidate với từng pool alpha có thể
có lịch sử khác nhau). Đây là **xung đột chưa giải quyết giữa hai plan trước Phase 7**.
Task 7.7 (GPEngine) cần dates để gọi `PoolCorrelation.max_corr(candidate_pnl, dates)` đúng
hợp đồng Task 6.1. **Quyết định cho Phase 7:** khi thực thi, đọc chữ ký THẬT của
`AlphaRepository.load_pool` đã merge (lệnh kiểm tra ở trên); nếu nó trả `dict[int,
NDArray]` (không dates, theo Phase 5), GPEngine **giả định mọi pool entry dùng chung trục
ngày của `data.dates`** (hợp lý nếu Phase 5/6 thật cũng chạy trên cùng `MarketData` cố định
trong một run) và tự cặp `(data.dates, pnl)` trước khi đưa vào `PoolCorrelation(pool=...)`
— ghi rõ giả định này bằng comment tại điểm gọi trong `src/gp/engine.py`. Nếu nó trả
`dict[int, tuple[Dates, NDArray]]` (theo Phase 6), dùng trực tiếp. Không tự sửa
`repository.py` ở Phase 7 để né mâu thuẫn — đó là việc dọn dẹp liên-phase, báo cáo lại ở
Task 7.9 self-review.

```bash
venv/Scripts/python.exe -c "from config.thresholds import MAX_DEPTH, SELF_CORR_MAX, TURNOVER_BAND; print(MAX_DEPTH, SELF_CORR_MAX, TURNOVER_BAND)"
```
Expected: `7 0.7 (0.01, 0.7)`. Nếu lỗi, Phase 0 chưa merge — DỪNG.

```bash
venv/Scripts/python.exe -c "import joblib; print(joblib.__version__)"
```
Nếu `ModuleNotFoundError` → thêm `joblib` vào `requirements.txt` và `venv/Scripts/pip
install joblib` trước Task 7.7 (chỉ Task 7.7 dùng joblib cho parallel eval; 7.1–7.6 không
cần).

---

### Task 7.1: `Individual` (`src/gp/individual.py`)

**Files:**
- Create: `src/gp/__init__.py`
- Create: `src/gp/individual.py`
- Test: `tests/unit/test_gp_individual.py`

**Interfaces:**
- Consumes: `Node` (`src/lang/ast.py`, Phase 1), `CanonicalHasher`/`DepthVisitor`/
  `ComplexityVisitor` (`src/lang/visitors.py`, Phase 1).
- Produces:

```python
# src/gp/individual.py
@dataclass(slots=True)
class Individual:
    expr: Node                            # signal core — KHÔNG config wrapper
    fitness: FitnessVector | None = None  # None = chưa eval; set 1 lần, cache
    generation: int = 0                   # thế hệ sinh ra (0 = seed/init)

    def canonical_hash(self) -> str: ...      # CanonicalHasher (lazy, không cache field riêng — Node bất biến nên hash ổn định, cho phép gọi lại rẻ)
    def depth(self) -> int: ...               # DepthVisitor
    def complexity(self) -> int: ...           # ComplexityVisitor
    def is_evaluated(self) -> bool: ...        # fitness is not None
```

`Individual` **không** `frozen` (khác AST node) vì `fitness`/`generation` được set SAU khi
khởi tạo (kết quả của bước eval trong vòng lặp GP — Task 7.7), nhưng `expr` không bao giờ
bị mutate tại chỗ: biến đổi (Task 7.5) luôn tạo `Node` mới + `Individual` mới, không sửa
`expr` của individual cũ. `canonical_hash()`/`depth()`/`complexity()` gọi visitor trực tiếp
mỗi lần (không cache nội bộ ở Task 7.1 — nếu sau này đo thấy nóng, thêm `functools.cache`
trên `Node` là việc tối ưu riêng, không phải hợp đồng bắt buộc ở đây).

- [ ] **Step 1: Tạo nhánh từ main sạch**

```bash
git checkout main && git pull --ff-only 2>/dev/null; git checkout -b phase-7-gp-engine
git status
```
Expected: "On branch phase-7-gp-engine", working tree clean.

- [ ] **Step 2: Viết test đỏ**

```python
# tests/unit/test_gp_individual.py
"""Test Individual: bọc Node, lazy depth/complexity/hash qua visitor Phase 1,
fitness/generation mutable nhưng expr bất biến theo quy ước (không sửa tại chỗ)."""

from __future__ import annotations

from src.gp.individual import Individual
from src.lang.ast import Call, Constant, Field


def _alpha() -> Individual:
    expr = Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(5.0))),))
    return Individual(expr=expr)


def test_individual_starts_unevaluated():
    ind = _alpha()
    assert ind.is_evaluated() is False
    assert ind.fitness is None
    assert ind.generation == 0


def test_individual_depth_matches_visitor():
    ind = _alpha()
    assert ind.depth() == 3  # rank(ts_mean(close, 5)) -> rank>ts_mean>close = 3 tầng


def test_individual_complexity_counts_all_nodes():
    ind = _alpha()
    assert ind.complexity() == 4  # rank, ts_mean, close(field), 5(const)


def test_individual_canonical_hash_is_deterministic_and_matches_structurally_equal_tree():
    ind1 = _alpha()
    ind2 = _alpha()  # cây khác instance, cùng cấu trúc
    assert ind1.canonical_hash() == ind2.canonical_hash()


def test_individual_canonical_hash_differs_for_different_tree():
    ind1 = _alpha()
    other_expr = Call(op="rank", args=(Field("volume"),))
    ind2 = Individual(expr=other_expr)
    assert ind1.canonical_hash() != ind2.canonical_hash()


def test_setting_fitness_marks_evaluated_without_mutating_expr():
    ind = _alpha()
    original_expr = ind.expr
    ind.fitness = object()  # placeholder cho FitnessVector thật (Task 7.2) — chỉ test cờ is_evaluated
    assert ind.is_evaluated() is True
    assert ind.expr is original_expr  # expr không bị đổi khi set fitness
```

- [ ] **Step 3: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_individual.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.gp'`.

- [ ] **Step 4: Tạo `src/gp/__init__.py` + `src/gp/individual.py`**

```python
# src/gp/__init__.py
"""Tầng Genetic Programming MiniBrain (Phase 7, B13 master design): sinh, lai ghép, đột
biến, chọn lọc quần thể signal-core AST, correlation-aware (pool + population) từ ngày
đầu. Dependency rule B1: src/gp được phép import src.lang/src.engine/src.backtest/
src.storage/src.generation/src.llm (tầng "app", cao nhất trừ pipeline/Phase 8) — ngược lại
các tầng đó KHÔNG được import src.gp.
"""
```

```python
# src/gp/individual.py
"""Individual — bọc một Node (signal core AST, Phase 1) cùng FitnessVector đã cache (nếu
đã eval) và số thế hệ sinh ra. Đây là đơn vị quần thể của GPEngine (Task 7.7); mọi biến đổi
(crossover/mutation, Task 7.5) tạo Individual MỚI từ Node mới, không sửa expr tại chỗ —
giữ tính bất biến của AST (Phase 1, frozen+slots) lan ra cả tầng GP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.lang.ast import Node
from src.lang.visitors import CanonicalHasher, ComplexityVisitor, DepthVisitor

if TYPE_CHECKING:
    from src.gp.fitness_vec import FitnessVector


@dataclass(slots=True)
class Individual:
    """Một cá thể quần thể GP: signal core (`expr`) + fitness đã eval (nếu có)."""

    expr: Node
    fitness: "FitnessVector | None" = None
    generation: int = 0

    def canonical_hash(self) -> str:
        return CanonicalHasher().visit(self.expr)

    def depth(self) -> int:
        return DepthVisitor().visit(self.expr)

    def complexity(self) -> int:
        return ComplexityVisitor().visit(self.expr)

    def is_evaluated(self) -> bool:
        return self.fitness is not None
```

> Test 6 dùng `ind.fitness = object()` (placeholder) thay vì `FitnessVector` thật để Task
> 7.1 không phụ thuộc ngược vào Task 7.2 (chưa tồn tại lúc này) — type annotation
> `FitnessVector | None` dùng string-forward-ref qua `TYPE_CHECKING` để mypy hài lòng mà
> không tạo import cycle/phụ thuộc thực thi. Khi Task 7.2 xong, test thật sẽ dùng
> `FitnessVector` thật (xem Task 7.2).

- [ ] **Step 5: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_individual.py -v
```
Expected: PASS (6 test).

- [ ] **Step 6: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/gp/
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/gp/individual.py
```
Expected: cả hai sạch.

- [ ] **Step 7: Commit**

```bash
git add src/gp/__init__.py src/gp/individual.py tests/unit/test_gp_individual.py
git commit -m "feat(gp): Individual boc Node + fitness cache + generation"
```

---

### Task 7.2: `FitnessVector` (`src/gp/fitness_vec.py`)

**Files:**
- Create: `src/gp/fitness_vec.py`
- Test: `tests/unit/test_gp_fitness_vec.py`

**Interfaces:**
- Consumes: `AlphaMetrics` (`src/backtest/metrics_local.py`, Phase 4), `PoolCorrelation`
  (`src/backtest/pool_corr.py`, Phase 6), `config/thresholds.py` (`TURNOVER_BAND`,
  `PER_YEAR_SHARPE_MIN`).
- Produces:

```python
# src/gp/fitness_vec.py
@dataclass(frozen=True, slots=True)
class FitnessVector:
    sharpe_deflated: float       # sharpe - haircut đa kiểm định (xem công thức dưới)
    per_year_min_sharpe: float   # min(per_year_sharpe.values()) — regime robustness
    turnover_penalty: float      # khoảng cách NGOÀI TURNOVER_BAND, 0 nếu trong band
    complexity_penalty: float    # node count chuẩn hóa (anti-bloat)
    pool_corr_penalty: float     # max|rho| vs pool (đã từ PoolCorrelation.max_corr)
    pop_corr_penalty: float      # max|rho| vs phần còn lại quần thể hiện tại

def deflated_sharpe(sharpe: float, n_trials: int) -> float: ...
def from_metrics(
    m: AlphaMetrics, complexity: int, pool_corr: float, pop_corr: float, n_trials: int,
) -> FitnessVector: ...
```

**Công thức (ghi rõ vì B13 không cho số cụ thể — quyết định triển khai cho Phase 7):**

- `deflated_sharpe(sharpe, n_trials)`: haircut đa kiểm định đơn giản kiểu Bailey-López de
  Prado rút gọn — `sharpe - sqrt(2 * ln(max(n_trials, 1))) / sqrt(252)` (haircut tăng theo
  log số lần thử, xấp xỉ độ lệch kỳ vọng của max trong `n_trials` phép thử ngẫu nhiên trên
  252 quan sát/năm); `n_trials <= 1` → haircut = 0 (không phạt cá thể đầu tiên). Đây là
  **xấp xỉ tương đối** (đúng tinh thần "treat as relative ranking only", không phải công
  thức Brain công bố) — ghi rõ trong docstring.
- `per_year_min_sharpe`: `min(m.per_year_sharpe.values())` nếu dict không rỗng, else `0.0`.
- `turnover_penalty`: nếu `m.turnover` trong `TURNOVER_BAND` (đọc từ thresholds, không
  hardcode) → `0.0`; nếu thấp hơn cận dưới → `TURNOVER_BAND[0] - m.turnover`; nếu cao hơn
  cận trên → `m.turnover - TURNOVER_BAND[1]` (khoảng cách dương ngoài band, GP tối thiểu hóa
  giá trị này).
- `complexity_penalty`: `complexity / 50.0` (chuẩn hóa thô; 50 node ~ cây phức tạp vừa —
  hệ số cố định trong code, KHÔNG phải threshold submission nên không bắt buộc đặt ở
  `config/thresholds.py`, nhưng đặt một hằng module-level `_COMPLEXITY_NORM = 50.0` dễ
  chỉnh, không hardcode rải trong hàm).
- `pool_corr_penalty` = `pool_corr` (đã là `max|rho|` từ `PoolCorrelation.max_corr`,
  truyền vào nguyên giá trị — `from_metrics` không tự gọi `PoolCorrelation`, caller (Task
  7.7) tính trước rồi truyền vào, giữ `fitness_vec.py` không phụ thuộc trực tiếp
  `pool_corr.py` về runtime ngoài type hint không bắt buộc — thực tế import được vì cùng
  tầng `backtest`, nhưng tách rõ trách nhiệm: nơi NÀO tính corr không phải việc của
  `FitnessVector`).
- `pop_corr_penalty` = `pop_corr` (cùng logic, truyền vào từ Task 7.7 — so với quần thể
  hiện tại, không phải pool DB).
- **Hướng tối ưu:** GP (Task 7.6 selection) coi `sharpe_deflated` và `per_year_min_sharpe`
  là **maximize**; `turnover_penalty`, `complexity_penalty`, `pool_corr_penalty`,
  `pop_corr_penalty` là **minimize**. Ghi rõ trong docstring `FitnessVector` để Task 7.6
  không hiểu nhầm hướng.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_gp_fitness_vec.py
"""Test FitnessVector: deflated_sharpe haircut theo n_trials, from_metrics map đúng
AlphaMetrics + corr penalty + turnover band, hướng tối ưu nhất quán (max sharpe/min năm
tệ nhất, min mọi penalty)."""

from __future__ import annotations

import math

import pytest

from config.thresholds import TURNOVER_BAND
from src.backtest.metrics_local import AlphaMetrics
from src.gp.fitness_vec import FitnessVector, deflated_sharpe, from_metrics


def _metrics(**overrides) -> AlphaMetrics:
    base = dict(
        sharpe=1.5, annual_return=0.20, turnover=0.30, max_drawdown=0.10,
        fitness=2.0, per_year_sharpe={2021: 1.2, 2022: 0.5}, weight_concentration=0.05,
    )
    base.update(overrides)
    return AlphaMetrics(**base)


def test_deflated_sharpe_no_haircut_for_single_trial():
    assert deflated_sharpe(1.5, n_trials=1) == pytest.approx(1.5)
    assert deflated_sharpe(1.5, n_trials=0) == pytest.approx(1.5)


def test_deflated_sharpe_haircut_grows_with_trials():
    d10 = deflated_sharpe(1.5, n_trials=10)
    d1000 = deflated_sharpe(1.5, n_trials=1000)
    assert d10 < 1.5
    assert d1000 < d10  # nhiều lần thử hơn -> haircut nặng hơn


def test_deflated_sharpe_matches_formula():
    sharpe, n = 2.0, 100
    expected = sharpe - math.sqrt(2 * math.log(n)) / math.sqrt(252)
    assert deflated_sharpe(sharpe, n) == pytest.approx(expected)


def test_from_metrics_per_year_min_sharpe_is_worst_year():
    fv = from_metrics(_metrics(), complexity=10, pool_corr=0.1, pop_corr=0.05, n_trials=1)
    assert fv.per_year_min_sharpe == pytest.approx(0.5)


def test_from_metrics_empty_per_year_gives_zero():
    fv = from_metrics(
        _metrics(per_year_sharpe={}), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.per_year_min_sharpe == pytest.approx(0.0)


def test_from_metrics_turnover_inside_band_has_zero_penalty():
    mid = (TURNOVER_BAND[0] + TURNOVER_BAND[1]) / 2
    fv = from_metrics(
        _metrics(turnover=mid), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.turnover_penalty == pytest.approx(0.0)


def test_from_metrics_turnover_below_band_penalized_by_distance():
    too_low = TURNOVER_BAND[0] - 0.05
    fv = from_metrics(
        _metrics(turnover=too_low), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.turnover_penalty == pytest.approx(0.05, abs=1e-9)


def test_from_metrics_turnover_above_band_penalized_by_distance():
    too_high = TURNOVER_BAND[1] + 0.10
    fv = from_metrics(
        _metrics(turnover=too_high), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.turnover_penalty == pytest.approx(0.10, abs=1e-9)


def test_from_metrics_passes_through_corr_penalties_unchanged():
    fv = from_metrics(_metrics(), complexity=10, pool_corr=0.42, pop_corr=0.31, n_trials=1)
    assert fv.pool_corr_penalty == pytest.approx(0.42)
    assert fv.pop_corr_penalty == pytest.approx(0.31)


def test_from_metrics_complexity_penalty_scales_with_node_count():
    fv_small = from_metrics(_metrics(), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1)
    fv_large = from_metrics(_metrics(), complexity=100, pool_corr=0.0, pop_corr=0.0, n_trials=1)
    assert fv_large.complexity_penalty > fv_small.complexity_penalty


def test_fitness_vector_is_frozen_and_hashable():
    fv = from_metrics(_metrics(), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1)
    with pytest.raises(AttributeError):
        fv.sharpe_deflated = 99.0  # type: ignore[misc]
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_fitness_vec.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.gp.fitness_vec'`.

- [ ] **Step 3: Tạo `src/gp/fitness_vec.py`**

```python
# src/gp/fitness_vec.py
"""FitnessVector — 6 chiều multi-objective, correlation- va regime-aware (B13 master
design, Gap #4 R4). Huong toi uu: sharpe_deflated/per_year_min_sharpe MAXIMIZE; turnover_
penalty/complexity_penalty/pool_corr_penalty/pop_corr_penalty MINIMIZE. Khong tu goi
PoolCorrelation/MetricsCalculator o day -- caller (GPEngine, Task 7.7) tinh truoc roi
truyen vao, giu module nay thuan tinh toan, de test doc lap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.thresholds import TURNOVER_BAND
from src.backtest.metrics_local import AlphaMetrics

_COMPLEXITY_NORM = 50.0  # chuan hoa node-count tho; khong phai threshold submission


@dataclass(frozen=True, slots=True)
class FitnessVector:
    """6 chieu fitness GP. sharpe_deflated/per_year_min_sharpe: cang cao cang tot.
    turnover_penalty/complexity_penalty/pool_corr_penalty/pop_corr_penalty: cang thap
    cang tot (0 = ly tuong)."""

    sharpe_deflated: float
    per_year_min_sharpe: float
    turnover_penalty: float
    complexity_penalty: float
    pool_corr_penalty: float
    pop_corr_penalty: float


def deflated_sharpe(sharpe: float, n_trials: int) -> float:
    """Haircut da kiem dinh xap xi (Bailey-Lopez de Prado rut gon): tru do lech ky vong
    cua max trong n_trials phep thu ngau nhien tren 252 quan sat/nam. n_trials<=1 -> khong
    haircut (chua co nhieu lan thu de overfit). Day la XAP XI TUONG DOI, khong phai cong
    thuc Brain cong bo -- chi dung de xep hang noi bo GP, khong bao cao tuyet doi."""
    if n_trials <= 1:
        return sharpe
    haircut = math.sqrt(2 * math.log(n_trials)) / math.sqrt(252)
    return sharpe - haircut


def _turnover_penalty(turnover: float) -> float:
    lo, hi = TURNOVER_BAND
    if turnover < lo:
        return lo - turnover
    if turnover > hi:
        return turnover - hi
    return 0.0


def from_metrics(
    m: AlphaMetrics,
    complexity: int,
    pool_corr: float,
    pop_corr: float,
    n_trials: int,
) -> FitnessVector:
    """Dung FitnessVector tu AlphaMetrics (Phase 4) + corr da tinh san (Phase 6 / quan the
    hien tai) + so node (Phase 1 ComplexityVisitor) + so lan thu (cho deflation)."""
    per_year_min = min(m.per_year_sharpe.values()) if m.per_year_sharpe else 0.0
    return FitnessVector(
        sharpe_deflated=deflated_sharpe(m.sharpe, n_trials),
        per_year_min_sharpe=per_year_min,
        turnover_penalty=_turnover_penalty(m.turnover),
        complexity_penalty=complexity / _COMPLEXITY_NORM,
        pool_corr_penalty=pool_corr,
        pop_corr_penalty=pop_corr,
    )
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_fitness_vec.py -v
```
Expected: PASS (11 test).

- [ ] **Step 5: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/gp/fitness_vec.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/gp/fitness_vec.py
```
Expected: cả hai sạch.

- [ ] **Step 6: Commit**

```bash
git add src/gp/fitness_vec.py tests/unit/test_gp_fitness_vec.py
git commit -m "feat(gp): FitnessVector 6 chieu + deflated_sharpe haircut da kiem dinh"
```

---

### Task 7.3: Seeds (`src/gp/seeds.py`)

**Files:**
- Create: `src/gp/seeds.py`
- Test: `tests/unit/test_gp_seeds.py`

**Interfaces:**
- Consumes: `Candidate` (`src/generation/local_select.py` — đọc `.expression: str`),
  `families.py` (hàm sinh `list[Candidate]` theo họ kinh điển — kiểm tra tên hàm export
  thật bằng `grep -n "^def " src/generation/families.py` trước khi gọi, KHÔNG suy diễn tên
  từ plan này), `novel_ideas.py` (`NOVEL_ALPHAS: list[Candidate]` — hằng module-level, đã
  xác nhận tồn tại), `parse` (`src/lang/parser.py`, Phase 1), `Node` (Phase 1). **Tùy
  chọn** (chỉ nếu `with_llm=True`): `HypothesisGenerator.generate` (`src/llm/hypothesis.py`)
  + `AlphaTranslator.translate` (`src/llm/translator.py`) — cả hai cần client LLM thật
  (`deepseek`/CLI bridge), nên seed LLM **không** chạy trong test mặc định (cần fake/inject,
  xem Step test).
- Produces:

```python
# src/gp/seeds.py
def seed_cores_from_families() -> list[Node]: ...
def seed_cores_from_novel_ideas() -> list[Node]: ...
def seed_cores_from_llm(
    hypothesis_gen, translator, research_directions: list[str],
) -> list[Node]: ...
def all_seed_cores(
    *, with_llm: bool = False, hypothesis_gen=None, translator=None,
    research_directions: list[str] | None = None,
) -> list[Node]: ...
```

**Quyết định cốt lõi — "chỉ cores" (stage separation B13):** `families.py`/`novel_ideas.py`
sinh `Candidate.expression` là **string FASTEXPR đầy đủ** (đã quan sát: `Candidate` có
`overrides: dict` riêng cho `decay`/`truncation` — nghĩa là `expression` string CHÍNH NÓ
phải đã là core thuần, config tách riêng trong `overrides`, KHỚP đúng stage separation;
xác nhận lại bằng cách đọc 2-3 `expression` mẫu trong `novel_ideas.py` đã đọc ở bước research
— không thấy `group_neutralize`/`scale`/`ts_decay_linear` áp ngoài cùng biểu thức, chỉ
`group_neutralize` đôi khi xuất hiện NỘI BỘ biểu thức như 1 operator core hợp lệ theo
registry — `group_neutralize` có `gp_usable=False` (Phase 2 B5) nên các seed dùng nó
**không** đưa được vào function set GP tự do, nhưng VẪN parse được thành `Node` hợp lệ làm
seed tĩnh — GP sẽ không tái tạo lại `group_neutralize` qua mutation/crossover ngẫu nhiên
(đúng ý B13), nhưng cây seed ban đầu chứa nó là chấp nhận được vì seed là "điểm khởi đầu
kinh nghiệm", không phải sản phẩm của function set ngẫu nhiên). Mỗi `seed_cores_from_*`:
lấy `.expression` từ mọi `Candidate`, `parse()` thành `Node`, lọc bỏ string nào parse lỗi
(ghi log warning, không raise — một seed lỗi không được sập toàn bộ seeding), trả
`list[Node]` (KHÔNG bọc `Individual` — đó là việc của Task 7.4 `init.py` khi ghép seed vào
quần thể ban đầu).

`seed_cores_from_llm`: với mỗi `research_direction`, gọi
`hypothesis_gen.generate(direction)` → `Hypothesis`, rồi `translator.translate(hypothesis)`
→ `AlphaCandidate | None`; nếu `None` (translator từ chối/field không hợp lệ) → bỏ qua
hướng đó; nếu có, `parse(candidate.expression)` → `Node`, lỗi parse → bỏ qua + log. Hàm
này **không catch lỗi mạng/LLM** (để caller — Task 7.7/CLI — quyết định retry/timeout,
đúng phạm vi: seeds.py chỉ là "đường ống dữ liệu", không phải resilience layer).

`all_seed_cores`: nối `seed_cores_from_families() + seed_cores_from_novel_ideas()` luôn
(rẻ, không mạng); nếu `with_llm=True` thì nối thêm `seed_cores_from_llm(...)` (yêu cầu
`hypothesis_gen`/`translator`/`research_directions` không `None`, raise `ValueError` rõ
nếu thiếu — fail-fast hơn là âm thầm bỏ qua phần LLM khi caller tưởng đã bật).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_gp_seeds.py
"""Test seeds.py: seed cores tu families/novel_ideas parse thanh Node hop le, seed LLM
qua fake hypothesis_gen/translator (khong mang thuc), all_seed_cores gop dung + fail-fast
khi with_llm=True thieu dependency."""

from __future__ import annotations

import pytest

from src.gp.seeds import (
    all_seed_cores,
    seed_cores_from_families,
    seed_cores_from_llm,
    seed_cores_from_novel_ideas,
)
from src.lang.ast import Node


def test_seed_cores_from_novel_ideas_returns_parsed_nodes():
    nodes = seed_cores_from_novel_ideas()
    assert len(nodes) > 0
    assert all(isinstance(n, Node) for n in nodes)


def test_seed_cores_from_families_returns_parsed_nodes():
    nodes = seed_cores_from_families()
    assert len(nodes) > 0
    assert all(isinstance(n, Node) for n in nodes)


def test_seed_cores_from_llm_uses_injected_fakes_no_network():
    class _FakeHypothesis:
        def to_dict(self):
            return {}

    class _FakeHypothesisGen:
        def generate(self, direction, palette=None):
            return _FakeHypothesis()

    class _FakeCandidate:
        expression = "rank(close)"

    class _FakeTranslator:
        def translate(self, hypothesis):
            return _FakeCandidate()

    nodes = seed_cores_from_llm(
        _FakeHypothesisGen(), _FakeTranslator(), research_directions=["momentum mới"],
    )
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)


def test_seed_cores_from_llm_skips_none_candidate():
    class _FakeHypothesisGen:
        def generate(self, direction, palette=None):
            return object()

    class _FakeTranslatorRejects:
        def translate(self, hypothesis):
            return None  # translator từ chối

    nodes = seed_cores_from_llm(
        _FakeHypothesisGen(), _FakeTranslatorRejects(), research_directions=["x"],
    )
    assert nodes == []


def test_all_seed_cores_without_llm_combines_families_and_novel():
    nodes = all_seed_cores(with_llm=False)
    expected_min = len(seed_cores_from_families()) + len(seed_cores_from_novel_ideas())
    assert len(nodes) == expected_min


def test_all_seed_cores_with_llm_true_requires_dependencies():
    with pytest.raises(ValueError):
        all_seed_cores(with_llm=True)  # thiếu hypothesis_gen/translator/research_directions
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_seeds.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.gp.seeds'`.

- [ ] **Step 3: Đọc tên hàm export thật của `families.py` trước khi code**

```bash
grep -n "^def " src/generation/families.py
```
Dùng (các) tên hàm thật trả về `list[Candidate]` tổng hợp mọi họ (nếu có hàm gộp sẵn,
dùng nó; nếu chỉ có hàm riêng từng họ, gọi tất cả rồi nối list trong `seeds.py`) — viết
`seed_cores_from_families()` khớp API thật, không khớp tên giả định.

- [ ] **Step 4: Tạo `src/gp/seeds.py`**

```python
# src/gp/seeds.py
"""Hat giong GP: cores (signal thuan, khong config wrapper) tu 3 nguon -- families.py
(khung cong thuc kinh dien), novel_ideas.py (10 alpha dataset it nguoi khai thac), va tuy
chon LLM (hypothesis -> translator). Day la "ramped half-and-half + SEEDING" cua B13 --
GP ngau nhien thuan lang phi danh gia tren cay vo nghia; seed kinh nghiem huong tim kiem
toi cau truc co gia thuyet kinh te. CHI tra Node (chua boc Individual) -- init.py (Task
7.4) la noi ghep seed vao quan the ban dau.
"""

from __future__ import annotations

import logging

from src.lang.ast import Node
from src.lang.parser import ParseError, parse

logger = logging.getLogger(__name__)


def _parse_all(expressions: list[str], *, source: str) -> list[Node]:
    nodes: list[Node] = []
    for expr in expressions:
        try:
            nodes.append(parse(expr))
        except ParseError as exc:
            logger.warning("seed tu %s parse loi, bo qua: %r (%s)", source, expr, exc)
    return nodes


def seed_cores_from_families() -> list[Node]:
    """Seed tu cac khung cong thuc kinh dien (src/generation/families.py)."""
    from src.generation.families import generate_family_candidates  # tên hàm thật — xem Step 3

    candidates = generate_family_candidates()
    return _parse_all([c.expression for c in candidates], source="families")


def seed_cores_from_novel_ideas() -> list[Node]:
    """Seed tu 10 alpha dataset it nguoi khai thac (src/generation/novel_ideas.py)."""
    from src.generation.novel_ideas import NOVEL_ALPHAS

    return _parse_all([c.expression for c in NOVEL_ALPHAS], source="novel_ideas")


def seed_cores_from_llm(
    hypothesis_gen: object,
    translator: object,
    research_directions: list[str],
) -> list[Node]:
    """Seed tu LLM: hypothesis_gen.generate(direction) -> Hypothesis ->
    translator.translate(hypothesis) -> AlphaCandidate | None -> parse. Khong catch loi
    mang/LLM o day -- caller quyet dinh retry/timeout."""
    nodes: list[Node] = []
    for direction in research_directions:
        hypothesis = hypothesis_gen.generate(direction)  # type: ignore[attr-defined]
        candidate = translator.translate(hypothesis)  # type: ignore[attr-defined]
        if candidate is None:
            logger.info("LLM seed bi translator tu choi cho huong: %s", direction)
            continue
        try:
            nodes.append(parse(candidate.expression))
        except ParseError as exc:
            logger.warning("LLM seed parse loi, bo qua: %r (%s)", candidate.expression, exc)
    return nodes


def all_seed_cores(
    *,
    with_llm: bool = False,
    hypothesis_gen: object | None = None,
    translator: object | None = None,
    research_directions: list[str] | None = None,
) -> list[Node]:
    """Gop toan bo seed: families + novel_ideas luon chay (re, khong mang); LLM tuy chon,
    fail-fast neu with_llm=True ma thieu dependency (tranh am tham bo qua phan LLM khi
    caller tuong da bat)."""
    nodes = seed_cores_from_families() + seed_cores_from_novel_ideas()
    if with_llm:
        if hypothesis_gen is None or translator is None or not research_directions:
            raise ValueError(
                "with_llm=True can hypothesis_gen, translator, research_directions day du"
            )
        nodes += seed_cores_from_llm(hypothesis_gen, translator, research_directions)
    return nodes
```

> **Lưu ý tên hàm `generate_family_candidates`:** đây là TÊN GIẢ ĐỊNH — Step 3 yêu cầu đọc
> `grep -n "^def " src/generation/families.py` thật trước khi code bước này; nếu tên thật
> khác (ví dụ `build_candidates`, `generate_all`, hoặc chỉ có hàm riêng từng họ như
> `reversal_candidates()`/`momentum_candidates()`...), sửa `seed_cores_from_families()` cho
> khớp — đây là quyết định triển khai tại chỗ, ghi lại tên hàm thật đã dùng trong commit
> message.

- [ ] **Step 5: Chạy test — PASS (sau khi sửa tên hàm families theo Step 3/4 cho khớp thật)**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_seeds.py -v
```
Expected: PASS (6 test).

- [ ] **Step 6: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/gp/seeds.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/gp/seeds.py
```
Expected: cả hai sạch (chú ý `# type: ignore[attr-defined]` trên `hypothesis_gen`/
`translator` vì chữ ký `object` cố ý lỏng — nếu muốn strict hơn, định nghĩa `Protocol`
tối giản `class _HypothesisGenLike(Protocol): def generate(self, direction): ...` thay
`object`, loại bỏ `type: ignore`; làm bước này nếu mypy phàn nàn nhiều hơn dự kiến).

- [ ] **Step 7: Commit**

```bash
git add src/gp/seeds.py tests/unit/test_gp_seeds.py
git commit -m "feat(gp): seeds tu families+novel_ideas+LLM tuy chon -> Node cores"
```

---

### Task 7.4: Init (`src/gp/init.py`)

**Files:**
- Create: `src/gp/init.py`
- Test: `tests/unit/test_gp_init.py`

**Interfaces:**
- Consumes: `OperatorRegistry.gp_function_set()` (Phase 1, `src/lang/registry.py`),
  `Individual` (Task 7.1), `all_seed_cores`/seed lists (Task 7.3), `DepthVisitor` (Phase 1),
  `MAX_DEPTH` (`config/thresholds.py`).
- Produces:

```python
# src/gp/init.py
def ramped_half_and_half(
    registry: OperatorRegistry, rng: np.random.Generator, n: int,
    min_depth: int = 2, max_depth: int = MAX_DEPTH,
    fields: tuple[str, ...] = (...),  # field pool khả dụng (vd VERIFIED_FIELDS)
) -> list[Node]: ...

def random_tree(
    registry: OperatorRegistry, rng: np.random.Generator, depth: int,
    fields: tuple[str, ...], full: bool,
) -> Node: ...

def init_population(
    registry: OperatorRegistry, rng: np.random.Generator, population_size: int,
    seed_cores: list[Node], fields: tuple[str, ...], max_depth: int = MAX_DEPTH,
) -> list[Individual]: ...
```

**Thuật toán (ramped half-and-half kinh điển, B13):**

- `random_tree(..., full=True)`: mọi nhánh đi tới đúng `depth` (cây "full" — mỗi node
  trong không phải leaf cuối cùng đều là `Call`, chỉ ở tầng `depth` cuối mới chọn leaf
  `Field`/`Constant`). `full=False` ("grow"): ở mỗi tầng, chọn ngẫu nhiên giữa tiếp tục mở
  rộng (`Call`) hoặc dừng sớm thành leaf (xác suất tăng dần khi gần `depth` tối đa), tạo
  cây hình dạng đa dạng hơn full.
- Khi chọn `Call`, chỉ chọn operator từ `registry.gp_function_set()` (stage separation —
  loại config wrapper); với mỗi `ArgKind` trong `spec.signature`: `PANEL` → đệ quy
  `random_tree` ở `depth-1`; `WINDOW` → `Constant(float(rng.choice(spec.window_choices)))`;
  `SCALAR` → `Constant(float(rng.uniform(-3, 3)))` (hằng số biên độ hợp lý cho threshold/hệ
  số — không có spec chuẩn từ B13, chọn dải nhỏ tránh nổ số trong eval); `GROUP` → bỏ qua
  trong cây core tự do (operator dùng `ArgKind.GROUP` như `group_neutralize` có
  `gp_usable=False` nên không xuất hiện ở đây qua `gp_function_set()` — nếu tương lai có
  operator core dùng `GROUP`, raise `NotImplementedError` rõ ràng thay vì âm thầm sai).
  Khi chọn `Field`, chọn ngẫu nhiên từ `fields` (tham số bắt buộc — KHÔNG hardcode danh sách
  field trong `init.py`; caller — Task 7.7 — truyền field pool từ `MarketData.fields.keys()`
  hoặc `VERIFIED_FIELDS`).
- `ramped_half_and_half(n, min_depth, max_depth)`: chia `n` cá thể đều cho mỗi độ sâu trong
  `range(min_depth, max_depth+1)`; với mỗi độ sâu, nửa số cá thể dùng `full=True`, nửa dùng
  `full=False` (kinh điển Koza). Phần dư (n không chia hết) phân bổ vào độ sâu lớn nhất.
- `init_population(population_size, seed_cores, ...)`: nếu
  `len(seed_cores) >= population_size`, lấy `population_size` cá thể ĐẦU từ
  `seed_cores` (ưu tiên kinh nghiệm hơn ngẫu nhiên khi seed đủ nhiều — quyết định triển
  khai, ghi rõ trong docstring); nếu ít hơn, lấy TẤT CẢ seed + lấp đầy phần còn lại bằng
  `ramped_half_and_half(n=population_size - len(seed_cores), ...)`. Mọi `Node` (seed hoặc
  random) phải qua `DepthVisitor` kiểm tra `depth <= max_depth` trước khi bọc
  `Individual` — seed nào vượt `max_depth` (hiếm nhưng có thể nếu template families.py sâu
  hơn `MAX_DEPTH`) bị loại + log warning, KHÔNG crash toàn bộ init.
- Mọi randomness đi qua `rng: np.random.Generator` được **inject từ caller** (Task 7.7 tạo
  `np.random.default_rng(settings.global_seed)` rồi truyền xuống) — `init.py` không tự gọi
  `np.random.default_rng()` nội bộ (Determinism, Global Constraints).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_gp_init.py
"""Test init.py: random_tree full/grow dung depth, ramped_half_and_half da dang do sau,
init_population uu tien seed + lap day random, tat ca <= max_depth, deterministic theo rng
inject."""

from __future__ import annotations

import numpy as np

from src.gp.individual import Individual
from src.gp.init import init_population, ramped_half_and_half, random_tree
from src.lang.ast import Call, Field
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor

_FIELDS = ("close", "volume", "returns")


def test_random_tree_full_reaches_exact_depth():
    rng = np.random.default_rng(0)
    registry = default_registry()
    tree = random_tree(registry, rng, depth=3, fields=_FIELDS, full=True)
    assert DepthVisitor().visit(tree) == 3


def test_random_tree_grow_does_not_exceed_depth():
    rng = np.random.default_rng(1)
    registry = default_registry()
    tree = random_tree(registry, rng, depth=4, fields=_FIELDS, full=False)
    assert DepthVisitor().visit(tree) <= 4


def test_random_tree_depth_one_is_a_leaf():
    rng = np.random.default_rng(2)
    registry = default_registry()
    tree = random_tree(registry, rng, depth=1, fields=_FIELDS, full=True)
    assert DepthVisitor().visit(tree) == 1


def test_random_tree_is_deterministic_for_same_seed():
    registry = default_registry()
    tree_a = random_tree(registry, np.random.default_rng(42), depth=3, fields=_FIELDS, full=False)
    tree_b = random_tree(registry, np.random.default_rng(42), depth=3, fields=_FIELDS, full=False)
    from src.lang.visitors import Serializer
    assert Serializer().visit(tree_a) == Serializer().visit(tree_b)


def test_ramped_half_and_half_spans_multiple_depths():
    rng = np.random.default_rng(3)
    registry = default_registry()
    trees = ramped_half_and_half(registry, rng, n=20, min_depth=2, max_depth=5, fields=_FIELDS)
    depths = {DepthVisitor().visit(t) for t in trees}
    assert len(trees) == 20
    assert min(depths) >= 2
    assert max(depths) <= 5
    assert len(depths) > 1  # da dang do sau, khong don mot muc


def test_init_population_uses_all_seeds_when_fewer_than_population_size():
    seed = Call(op="rank", args=(Field("close"),))
    rng = np.random.default_rng(4)
    registry = default_registry()
    pop = init_population(
        registry, rng, population_size=10, seed_cores=[seed], fields=_FIELDS, max_depth=5,
    )
    assert len(pop) == 10
    assert all(isinstance(ind, Individual) for ind in pop)
    assert any(ind.expr == seed for ind in pop)  # seed nguyen ban co mat


def test_init_population_caps_seeds_when_more_than_population_size():
    seeds = [Call(op="rank", args=(Field(f),)) for f in _FIELDS]
    rng = np.random.default_rng(5)
    registry = default_registry()
    pop = init_population(
        registry, rng, population_size=2, seed_cores=seeds, fields=_FIELDS, max_depth=5,
    )
    assert len(pop) == 2


def test_init_population_all_individuals_within_max_depth():
    rng = np.random.default_rng(6)
    registry = default_registry()
    pop = init_population(
        registry, rng, population_size=15, seed_cores=[], fields=_FIELDS, max_depth=4,
    )
    assert all(ind.depth() <= 4 for ind in pop)
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_init.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.gp.init'`.

- [ ] **Step 3: Tạo `src/gp/init.py`**

```python
# src/gp/init.py
"""Ramped half-and-half + seeding (B13): khoi tao quan the GP ban dau. Function set CHI
tu registry.gp_function_set() (stage separation -- loai config wrapper). Moi randomness
qua rng inject (Determinism, Global Constraints) -- khong tu goi np.random.default_rng()
noi bo.
"""

from __future__ import annotations

import logging

import numpy as np

from src.gp.individual import Individual
from src.lang.ast import Call, Constant, Field, Node
from src.lang.registry import ArgKind, OperatorRegistry
from src.lang.visitors import DepthVisitor

logger = logging.getLogger(__name__)

_SCALAR_RANGE = (-3.0, 3.0)  # bien do hop ly cho threshold/he so trong cay seed ngau nhien


def random_tree(
    registry: OperatorRegistry,
    rng: np.random.Generator,
    depth: int,
    fields: tuple[str, ...],
    full: bool,
) -> Node:
    """Sinh 1 cay ngau nhien sau toi da `depth`. full=True: moi nhanh di toi dung depth
    (cay "full"). full=False ("grow"): dung som ngau nhien o moi tang."""
    if depth <= 1:
        return _random_leaf(rng, fields)

    stop_early = (not full) and rng.random() < (1.0 / depth)
    if stop_early:
        return _random_leaf(rng, fields)

    ops = registry.gp_function_set()
    if not ops:
        return _random_leaf(rng, fields)
    spec = ops[rng.integers(0, len(ops))]

    args: list[Node] = []
    for kind in spec.signature:
        match kind:
            case ArgKind.PANEL:
                args.append(random_tree(registry, rng, depth - 1, fields, full))
            case ArgKind.WINDOW:
                choice = spec.window_choices[rng.integers(0, len(spec.window_choices))]
                args.append(Constant(float(choice)))
            case ArgKind.SCALAR:
                args.append(Constant(float(rng.uniform(*_SCALAR_RANGE))))
            case ArgKind.GROUP:
                raise NotImplementedError(
                    f"operator core '{spec.name}' dung ArgKind.GROUP nhung init.py chua "
                    "ho tro sinh GROUP cho function set tu do (chi config wrapper nhu "
                    "group_neutralize moi dung GROUP va co gp_usable=False)"
                )
    return Call(op=spec.name, args=tuple(args))


def _random_leaf(rng: np.random.Generator, fields: tuple[str, ...]) -> Node:
    if rng.random() < 0.7:  # ưu tiên field hơn constant ở leaf (tín hiệu thật > số tay)
        return Field(fields[rng.integers(0, len(fields))])
    return Constant(float(rng.integers(2, 60)))  # constant kiểu window nhỏ, hợp lý làm leaf


def ramped_half_and_half(
    registry: OperatorRegistry,
    rng: np.random.Generator,
    n: int,
    min_depth: int,
    max_depth: int,
    fields: tuple[str, ...],
) -> list[Node]:
    """Chia n cay deu cho moi do sau trong [min_depth, max_depth], nua full nua grow moi
    do sau (Koza). Phan du don vao do sau lon nhat."""
    depths = list(range(min_depth, max_depth + 1))
    per_depth = n // len(depths)
    remainder = n - per_depth * len(depths)

    trees: list[Node] = []
    for i, depth in enumerate(depths):
        count = per_depth + (remainder if i == len(depths) - 1 else 0)
        half = count // 2
        for j in range(count):
            full = j < half
            trees.append(random_tree(registry, rng, depth, fields, full))
    return trees


def init_population(
    registry: OperatorRegistry,
    rng: np.random.Generator,
    population_size: int,
    seed_cores: list[Node],
    fields: tuple[str, ...],
    max_depth: int,
) -> list[Individual]:
    """Quan the ban dau: uu tien seed kinh nghiem, lap day phan con lai bang ramped
    half-and-half. Seed/cay vuot max_depth bi loai + log warning (khong crash)."""
    valid_seeds = [t for t in seed_cores if DepthVisitor().visit(t) <= max_depth]
    dropped = len(seed_cores) - len(valid_seeds)
    if dropped:
        logger.warning("init_population: bo qua %d seed vuot max_depth=%d", dropped, max_depth)

    if len(valid_seeds) >= population_size:
        chosen = valid_seeds[:population_size]
        return [Individual(expr=t) for t in chosen]

    remaining = population_size - len(valid_seeds)
    filler = ramped_half_and_half(registry, rng, remaining, min_depth=2, max_depth=max_depth, fields=fields)
    return [Individual(expr=t) for t in valid_seeds + filler]
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_init.py -v
```
Expected: PASS (7 test).

- [ ] **Step 5: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/gp/init.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/gp/init.py
```
Expected: cả hai sạch.

- [ ] **Step 6: Commit**

```bash
git add src/gp/init.py tests/unit/test_gp_init.py
git commit -m "feat(gp): ramped half-and-half + seeding, depth cap, rng inject"
```

---

### Task 7.5: Variation (`src/gp/variation.py`)

**Files:**
- Create: `src/gp/variation.py`
- Test: `tests/unit/test_gp_variation.py`

**Interfaces:**
- Consumes: `Node`/`Call`/`Field`/`Constant` (Phase 1), `all_subtrees` (Phase 1,
  `src/lang/visitors.py` — đã tồn tại sẵn, TÁI DÙNG không viết lại), `CanonicalHasher`
  (Phase 1, dedup), `DepthVisitor` (Phase 1), `OperatorRegistry`/`ArgKind` (Phase 1),
  `random_tree` (Task 7.4, cho subtree mutation), `MAX_DEPTH` (thresholds).
- Produces:

```python
# src/gp/variation.py
def crossover(
    a: Node, b: Node, rng: np.random.Generator, max_depth: int = MAX_DEPTH,
) -> tuple[Node, Node]: ...

def point_mutation(
    node: Node, registry: OperatorRegistry, rng: np.random.Generator,
    fields: tuple[str, ...],
) -> Node: ...

def subtree_mutation(
    node: Node, registry: OperatorRegistry, rng: np.random.Generator,
    fields: tuple[str, ...], max_depth: int = MAX_DEPTH,
) -> Node: ...

def hoist_mutation(node: Node, rng: np.random.Generator) -> Node: ...

def dedup_population(
    individuals: list[Individual], registry: OperatorRegistry | None = None,
) -> list[Individual]: ...
```

**Thuật toán (typed — chỉ thao tác trên subtree/operator tương thích `ArgKind`, B13):**

- `crossover(a, b, rng, max_depth)`: chọn ngẫu nhiên 1 subtree của `a` và 1 subtree của
  `b` (qua `all_subtrees`), tráo đổi. **Typed**: chỉ tráo nếu hai subtree được chọn nằm ở vị
  trí cùng `ArgKind` mong đợi trong cây cha của chúng — đơn giản hóa: vì AST của ta không
  gắn `ArgKind` trực tiếp trên node (đó là thuộc tính của *vị trí đối số* trong
  `OperatorSpec.signature` của node CHA), "tương thích" được định nghĩa là: **cả hai subtree
  được chọn đều phải là `PANEL`-compatible** (mọi `Node` không phải `Constant` đơn lẻ giữ
  vai trò literal — vì trong AST hiện tại không có node "thuần WINDOW/SCALAR" tách biệt,
  `Constant` đóng cả hai vai; quy ước: chỉ chọn điểm crossover từ tập subtree mà
  **gốc không phải `Constant` đang đứng ở vị trí WINDOW/SCALAR của cha** — kiểm tra bằng
  cách dò cây cha-con qua một lượt walk phụ). Triển khai cụ thể: viết hàm nội bộ
  `_panel_compatible_subtrees(node)` trả các subtree mà **vai trò của nó trong cây cha (nếu
  có cha) là `ArgKind.PANEL`** (root luôn hợp lệ vì root luôn đóng vai signal core, không
  phải tham số của ai). Sau khi tráo, validate: `DepthVisitor` của CẢ HAI cây kết quả phải
  `<= max_depth`; nếu vượt, retry tráo điểm khác (tối đa `_MAX_CROSSOVER_RETRIES = 10` lần),
  hết lượt mà vẫn vượt → trả `(a, b)` nguyên bản (an toàn, không tạo cây invalid — "validity
  repair" tối giản: lùi về không đổi gì thay vì cắt cây tùy tiện).
- `point_mutation(node, registry, rng, fields)`: chọn 1 subtree ngẫu nhiên (qua
  `all_subtrees`); nếu là `Call` → thay `op` bằng operator KHÁC cùng `signature` y hệt
  (cùng tuple `ArgKind`, tìm trong `registry.gp_function_set()`) — nếu không có operator
  nào khớp signature, giữ nguyên; nếu là `Field` → thay bằng field khác trong `fields`; nếu
  là `Constant` đứng ở vị trí `WINDOW` của cha → thay bằng `window_choices` khác của
  operator cha (cần biết cha — point mutation cần duyệt cây tìm (parent, index) của node
  được chọn, không chỉ subtree độc lập); nếu `Constant` ở vị trí `SCALAR` → perturb bằng
  `Constant(value + rng.normal(0, 0.5))`. Trả CÂY MỚI (không sửa `node` gốc — AST bất biến).
- `subtree_mutation(node, ..., max_depth)`: chọn 1 điểm ngẫu nhiên, thay subtree tại đó
  bằng `random_tree(registry, rng, depth=rng.integers(1, remaining_depth+1), fields, full=
  bool(rng.integers(0,2)))` với `remaining_depth = max_depth - depth_to_point` (đảm bảo cây
  kết quả không vượt `max_depth`). Trả cây mới.
- `hoist_mutation(node, rng)`: chọn 1 subtree KHÔNG PHẢI root (qua `all_subtrees`, loại
  `node` chính nó nếu có >1 subtree), trả NÓ làm cây mới — "nâng" một nhánh nhỏ lên làm cây
  toàn bộ, chống bloat (B13/R6) bằng cách rút ngắn cây định kỳ. Nếu cây chỉ có 1 node (leaf
  đơn), trả nguyên `node` (không có gì để hoist).
- `dedup_population(individuals, registry)`: nhóm theo `CanonicalHasher().visit(ind.expr)`
  (dùng `ind.canonical_hash()` có sẵn từ Task 7.1), giữ cá thể ĐẦU TIÊN mỗi nhóm hash, loại
  phần còn lại — trả `list[Individual]` đã khử trùng lặp cấu trúc, **giữ thứ tự xuất hiện**
  (ổn định, dùng cho test xác định).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_gp_variation.py
"""Test variation.py: crossover/mutation tao cay hop le (hoac giu nguyen khi khong sua
duoc an toan), depth cap duoc giu, dedup theo canonical hash."""

from __future__ import annotations

import numpy as np

from src.gp.individual import Individual
from src.gp.variation import (
    crossover,
    dedup_population,
    hoist_mutation,
    point_mutation,
    subtree_mutation,
)
from src.lang.ast import Call, Constant, Field, Node
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor, Serializer

_FIELDS = ("close", "volume", "returns")


def _tree_a() -> Node:
    return Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(5.0))),))


def _tree_b() -> Node:
    return Call(op="rank", args=(Call(op="ts_mean", args=(Field("volume"), Constant(10.0))),))


def test_crossover_respects_max_depth_on_both_children():
    rng = np.random.default_rng(0)
    a, b = crossover(_tree_a(), _tree_b(), rng, max_depth=7)
    assert DepthVisitor().visit(a) <= 7
    assert DepthVisitor().visit(b) <= 7


def test_crossover_is_deterministic_for_same_seed():
    a1, b1 = crossover(_tree_a(), _tree_b(), np.random.default_rng(5), max_depth=7)
    a2, b2 = crossover(_tree_a(), _tree_b(), np.random.default_rng(5), max_depth=7)
    assert Serializer().visit(a1) == Serializer().visit(a2)
    assert Serializer().visit(b1) == Serializer().visit(b2)


def test_point_mutation_changes_something_or_no_op_safely():
    rng = np.random.default_rng(1)
    registry = default_registry()
    mutated = point_mutation(_tree_a(), registry, rng, fields=_FIELDS)
    assert isinstance(mutated, Node)
    assert DepthVisitor().visit(mutated) >= 1


def test_point_mutation_does_not_mutate_input_in_place():
    original = _tree_a()
    serialized_before = Serializer().visit(original)
    point_mutation(original, default_registry(), np.random.default_rng(2), fields=_FIELDS)
    assert Serializer().visit(original) == serialized_before


def test_subtree_mutation_respects_max_depth():
    rng = np.random.default_rng(3)
    registry = default_registry()
    mutated = subtree_mutation(_tree_a(), registry, rng, fields=_FIELDS, max_depth=5)
    assert DepthVisitor().visit(mutated) <= 5


def test_hoist_mutation_shrinks_or_keeps_tree_depth():
    rng = np.random.default_rng(4)
    original = _tree_a()
    hoisted = hoist_mutation(original, rng)
    assert DepthVisitor().visit(hoisted) <= DepthVisitor().visit(original)


def test_hoist_mutation_on_single_leaf_returns_same_leaf():
    leaf = Field("close")
    hoisted = hoist_mutation(leaf, np.random.default_rng(6))
    assert hoisted == leaf


def test_dedup_population_removes_structural_duplicates_keeps_first():
    ind1 = Individual(expr=_tree_a())
    ind2 = Individual(expr=_tree_a())  # cùng cấu trúc, instance khác
    ind3 = Individual(expr=_tree_b())
    result = dedup_population([ind1, ind2, ind3])
    assert len(result) == 2
    assert result[0] is ind1
    assert result[1] is ind3
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_variation.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.gp.variation'`.

- [ ] **Step 3: Tạo `src/gp/variation.py`**

```python
# src/gp/variation.py
"""Typed crossover + point/subtree/hoist mutation (B13) tren AST Phase 1. "Typed" = chi
trao subtree dong vai PANEL trong cay cha (khong trao nham vao vi tri WINDOW/SCALAR cua
mot Constant), chi doi operator cung signature, chi doi window theo window_choices cua
chinh operator do. Validity repair toi gian: het luot retry van vuot max_depth -> giu
nguyen cay goc (an toan hon cat cay tuy tien). Dedup qua CanonicalHasher (Task 7.1).
"""

from __future__ import annotations

import numpy as np

from src.gp.individual import Individual
from src.gp.init import random_tree
from src.lang.ast import Call, Constant, Field, Node
from src.lang.registry import ArgKind, OperatorRegistry
from src.lang.visitors import DepthVisitor, all_subtrees

_MAX_CROSSOVER_RETRIES = 10


def _panel_compatible_subtrees(root: Node, registry: OperatorRegistry) -> list[Node]:
    """Subtree ma vai tro cua no trong cay cha (neu co) la ArgKind.PANEL. Root luon hop le
    (khong phai tham so cua ai)."""
    result: list[Node] = [root]

    def _walk(node: Node) -> None:
        if not isinstance(node, Call):
            return
        try:
            spec = registry.get(node.op)
        except KeyError:
            return
        for child, kind in zip(node.args, spec.signature):
            if kind is ArgKind.PANEL:
                result.append(child)
            _walk(child)

    _walk(root)
    return result


def _replace_subtree(root: Node, target: Node, replacement: Node) -> Node:
    """Tra cay moi voi `target` (theo identity) duoc thay bang `replacement`; neu
    root is target, tra replacement luon."""
    if root is target:
        return replacement
    if not isinstance(root, Call):
        return root
    new_args = tuple(_replace_subtree(c, target, replacement) for c in root.args)
    return Call(op=root.op, args=new_args)


def crossover(
    a: Node, b: Node, rng: np.random.Generator, max_depth: int,
) -> tuple[Node, Node]:
    from src.lang.registry import default_registry

    registry = default_registry()
    for _ in range(_MAX_CROSSOVER_RETRIES):
        points_a = _panel_compatible_subtrees(a, registry)
        points_b = _panel_compatible_subtrees(b, registry)
        pa = points_a[rng.integers(0, len(points_a))]
        pb = points_b[rng.integers(0, len(points_b))]

        new_a = _replace_subtree(a, pa, pb)
        new_b = _replace_subtree(b, pb, pa)
        if DepthVisitor().visit(new_a) <= max_depth and DepthVisitor().visit(new_b) <= max_depth:
            return new_a, new_b
    return a, b


def point_mutation(
    node: Node, registry: OperatorRegistry, rng: np.random.Generator, fields: tuple[str, ...],
) -> Node:
    targets = all_subtrees(node)
    target = targets[rng.integers(0, len(targets))]

    if isinstance(target, Field):
        replacement: Node = Field(fields[rng.integers(0, len(fields))])
        return _replace_subtree(node, target, replacement)

    if isinstance(target, Constant):
        replacement = Constant(float(target.value) + float(rng.normal(0, 0.5)))
        return _replace_subtree(node, target, replacement)

    # target é Call: đổi op sang operator khác cùng signature
    spec = registry.get(target.op)
    candidates = [
        s for s in registry.gp_function_set()
        if s.signature == spec.signature and s.name != spec.name
    ]
    if not candidates:
        return node
    new_op = candidates[rng.integers(0, len(candidates))]
    replacement = Call(op=new_op.name, args=target.args)
    return _replace_subtree(node, target, replacement)


def subtree_mutation(
    node: Node, registry: OperatorRegistry, rng: np.random.Generator,
    fields: tuple[str, ...], max_depth: int,
) -> Node:
    targets = all_subtrees(node)
    target = targets[rng.integers(0, len(targets))]
    target_depth = DepthVisitor().visit(target)
    full_depth = DepthVisitor().visit(node)
    remaining = max(1, max_depth - (full_depth - target_depth))
    new_subtree = random_tree(
        registry, rng, depth=int(rng.integers(1, remaining + 1)), fields=fields,
        full=bool(rng.integers(0, 2)),
    )
    return _replace_subtree(node, target, new_subtree)


def hoist_mutation(node: Node, rng: np.random.Generator) -> Node:
    candidates = [s for s in all_subtrees(node) if s is not node]
    if not candidates:
        return node
    return candidates[rng.integers(0, len(candidates))]


def dedup_population(
    individuals: list[Individual], registry: OperatorRegistry | None = None,
) -> list[Individual]:
    seen: set[str] = set()
    result: list[Individual] = []
    for ind in individuals:
        h = ind.canonical_hash()
        if h in seen:
            continue
        seen.add(h)
        result.append(ind)
    return result
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_variation.py -v
```
Expected: PASS (8 test).

- [ ] **Step 5: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/gp/variation.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/gp/variation.py
```
Expected: cả hai sạch (sửa `registry` param không dùng ở `dedup_population` nếu mypy/ruff
phàn nàn unused — giữ tham số cho khả năng mở rộng tương lai thì đổi tên thành `_registry`
hoặc xóa param nếu Global Constraints "không unused" áp dụng nghiêm cho cả tham số không
dùng; quyết định tại chỗ, ưu tiên xóa param nếu thực sự không cần).

- [ ] **Step 6: Commit**

```bash
git add src/gp/variation.py tests/unit/test_gp_variation.py
git commit -m "feat(gp): typed crossover + point/subtree/hoist mutation + dedup canonical-hash"
```

---

### Task 7.6: Selection (`src/gp/selection.py`)

**Files:**
- Create: `src/gp/selection.py`
- Test: `tests/unit/test_gp_selection.py`

**Interfaces:**
- Consumes: `Individual`/`FitnessVector` (Task 7.1/7.2).
- Produces:

```python
# src/gp/selection.py
def dominates(a: FitnessVector, b: FitnessVector) -> bool: ...
def fast_non_dominated_sort(individuals: list[Individual]) -> list[list[Individual]]: ...
def crowding_distance(front: list[Individual]) -> dict[int, float]: ...  # key = id(individual)
def nsga2_select(
    individuals: list[Individual], n_survivors: int, rng: np.random.Generator,
) -> list[Individual]: ...
```

**Thuật toán (NSGA-II kinh điển, Deb et al. 2002, áp lên `FitnessVector` 6 chiều — B13
correlation-aware multi-objective):**

- `dominates(a, b)`: `a` Pareto-dominates `b` nếu `a` không tệ hơn `b` ở MỌI chiều và tốt
  hơn ở ÍT NHẤT 1 chiều, theo hướng tối ưu đã ghi ở Task 7.2 (`sharpe_deflated`/
  `per_year_min_sharpe`: cao hơn = tốt hơn; 4 penalty còn lại: thấp hơn = tốt hơn). Cụ thể:
  chuyển mọi chiều về "thấp hơn = tốt hơn" bằng cách âm hóa 2 chiều maximize trước khi so
  sánh (`-sharpe_deflated`, `-per_year_min_sharpe`), rồi `dominates` chuẩn minimize trên 6
  số.
- `fast_non_dominated_sort`: thuật toán O(MN²) chuẩn — rank 0 = không bị ai dominate; rank
  k = chỉ bị dominate bởi cá thể rank < k. Trả `list[list[Individual]]` (mỗi front 1 list,
  front[0] là Pareto front tốt nhất).
- `crowding_distance(front)`: với MỖI chiều trong 6 chiều `FitnessVector` (chuẩn hóa hướng
  như trên), sort front theo chiều đó, cộng `(f[i+1] - f[i-1]) / (f_max - f_min)` vào
  distance của cá thể `i` (biên front = `inf`, luôn được giữ). Trả `dict[int, float]` khóa
  bằng `id(individual)` (Python object identity — `Individual` không hashable theo giá trị
  vì `slots=True` không `frozen`, dùng `id()` là cách chuẩn để map phụ trợ không sửa class).
- `nsga2_select(individuals, n_survivors, rng)`: chạy `fast_non_dominated_sort`; thêm toàn
  bộ front vào kết quả theo thứ tự rank tăng dần cho tới khi front tiếp theo làm vượt
  `n_survivors`; front "biên" (làm vượt) được sort theo `crowding_distance` GIẢM DẦN (đa
  dạng hơn ưu tiên hơn), lấy đủ số còn thiếu. `rng` dùng để tie-break ngẫu nhiên khi nhiều
  cá thể cùng crowding distance (tránh thiên vị thứ tự list) — **xác định** theo seed
  inject, không gọi `np.random` global.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_gp_selection.py
"""Test selection.py: dominates dung huong toi uu (sharpe max, penalty min),
fast_non_dominated_sort phan front dung, crowding_distance giu bien=inf, nsga2_select giu
dung so luong va uu tien front tot + da dang."""

from __future__ import annotations

import numpy as np

from src.gp.fitness_vec import FitnessVector
from src.gp.individual import Individual
from src.gp.selection import crowding_distance, dominates, fast_non_dominated_sort, nsga2_select
from src.lang.ast import Field


def _fv(sharpe=1.0, per_year=0.5, turn=0.0, complex_p=0.0, pool=0.0, pop=0.0) -> FitnessVector:
    return FitnessVector(
        sharpe_deflated=sharpe, per_year_min_sharpe=per_year, turnover_penalty=turn,
        complexity_penalty=complex_p, pool_corr_penalty=pool, pop_corr_penalty=pop,
    )


def _ind(fv: FitnessVector) -> Individual:
    ind = Individual(expr=Field("close"))
    ind.fitness = fv
    return ind


def test_dominates_higher_sharpe_lower_penalties_wins():
    better = _fv(sharpe=2.0, turn=0.0)
    worse = _fv(sharpe=1.0, turn=0.1)
    assert dominates(better, worse) is True
    assert dominates(worse, better) is False


def test_dominates_false_when_tradeoff_no_domination():
    a = _fv(sharpe=2.0, turn=0.2)  # sharpe cao hon nhung turnover penalty cao hon
    b = _fv(sharpe=1.0, turn=0.0)
    assert dominates(a, b) is False
    assert dominates(b, a) is False


def test_fast_non_dominated_sort_front_zero_is_non_dominated():
    pop = [_ind(_fv(sharpe=2.0)), _ind(_fv(sharpe=1.0)), _ind(_fv(sharpe=0.5))]
    fronts = fast_non_dominated_sort(pop)
    assert pop[0] in fronts[0]  # sharpe cao nhat, moi thu khac bang nhau -> khong bi dominate


def test_fast_non_dominated_sort_covers_all_individuals():
    pop = [_ind(_fv(sharpe=s)) for s in [2.0, 1.5, 1.0, 0.5]]
    fronts = fast_non_dominated_sort(pop)
    total = sum(len(f) for f in fronts)
    assert total == len(pop)


def test_crowding_distance_boundary_individuals_are_infinite():
    front = [_ind(_fv(sharpe=s)) for s in [0.0, 1.0, 2.0, 3.0]]
    dist = crowding_distance(front)
    sharpes = sorted(front, key=lambda i: i.fitness.sharpe_deflated)
    assert dist[id(sharpes[0])] == float("inf")
    assert dist[id(sharpes[-1])] == float("inf")


def test_nsga2_select_returns_exact_count():
    rng = np.random.default_rng(0)
    pop = [_ind(_fv(sharpe=s, turn=abs(s - 1))) for s in np.linspace(0, 3, 12)]
    survivors = nsga2_select(pop, n_survivors=5, rng=rng)
    assert len(survivors) == 5


def test_nsga2_select_prefers_better_front_over_worse():
    rng = np.random.default_rng(1)
    dominant = _ind(_fv(sharpe=5.0, turn=0.0))
    dominated = _ind(_fv(sharpe=0.1, turn=0.5))
    survivors = nsga2_select([dominant, dominated], n_survivors=1, rng=rng)
    assert survivors == [dominant]


def test_nsga2_select_is_deterministic_for_same_seed():
    pop = [_ind(_fv(sharpe=s, turn=abs(s - 1))) for s in np.linspace(0, 3, 10)]
    s1 = nsga2_select(pop, n_survivors=4, rng=np.random.default_rng(7))
    s2 = nsga2_select(pop, n_survivors=4, rng=np.random.default_rng(7))
    assert [id(x) for x in s1] == [id(x) for x in s2]
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_selection.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.gp.selection'`.

- [ ] **Step 3: Tạo `src/gp/selection.py`**

```python
# src/gp/selection.py
"""NSGA-II (Deb et al. 2002) tren FitnessVector 6 chieu -- correlation-aware multi-
objective selection (B13/R4): ngan quan the sup vao "ngan clone tuong quan cao" bang cach
giu Pareto front + crowding distance (uu tien da dang) thay vi chi sap theo 1 so Sharpe.
Huong toi uu: sharpe_deflated/per_year_min_sharpe MAXIMIZE; 4 penalty con lai MINIMIZE
(xem fitness_vec.py).
"""

from __future__ import annotations

import numpy as np

from src.gp.fitness_vec import FitnessVector
from src.gp.individual import Individual

_MAXIMIZE_FIELDS = ("sharpe_deflated", "per_year_min_sharpe")
_ALL_FIELDS = (
    "sharpe_deflated", "per_year_min_sharpe", "turnover_penalty",
    "complexity_penalty", "pool_corr_penalty", "pop_corr_penalty",
)


def _as_minimize_vector(fv: FitnessVector) -> tuple[float, ...]:
    """6 so, tat ca theo huong 'thap hon = tot hon' (am hoa 2 chieu maximize)."""
    values = []
    for name in _ALL_FIELDS:
        v = getattr(fv, name)
        values.append(-v if name in _MAXIMIZE_FIELDS else v)
    return tuple(values)


def dominates(a: FitnessVector, b: FitnessVector) -> bool:
    va, vb = _as_minimize_vector(a), _as_minimize_vector(b)
    not_worse_anywhere = all(x <= y for x, y in zip(va, vb))
    better_somewhere = any(x < y for x, y in zip(va, vb))
    return not_worse_anywhere and better_somewhere


def fast_non_dominated_sort(individuals: list[Individual]) -> list[list[Individual]]:
    n = len(individuals)
    dominated_count = [0] * n
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for i in range(n):
        fi = individuals[i].fitness
        assert fi is not None, "fast_non_dominated_sort yêu cầu mọi Individual đã eval"
        for j in range(n):
            if i == j:
                continue
            fj = individuals[j].fitness
            assert fj is not None
            if dominates(fi, fj):
                dominates_list[i].append(j)
            elif dominates(fj, fi):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front: list[int] = []
        for i in fronts[k]:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    fronts.pop()  # front rỗng cuối cùng (điều kiện dừng while)
    return [[individuals[i] for i in front] for front in fronts]


def crowding_distance(front: list[Individual]) -> dict[int, float]:
    distances: dict[int, float] = {id(ind): 0.0 for ind in front}
    if len(front) <= 2:
        for ind in front:
            distances[id(ind)] = float("inf")
        return distances

    for name in _ALL_FIELDS:
        sign = -1.0 if name in _MAXIMIZE_FIELDS else 1.0

        def _key(ind: Individual) -> float:
            assert ind.fitness is not None
            return sign * getattr(ind.fitness, name)

        ordered = sorted(front, key=_key)
        values = [_key(ind) for ind in ordered]
        span = values[-1] - values[0]
        distances[id(ordered[0])] = float("inf")
        distances[id(ordered[-1])] = float("inf")
        if span == 0:
            continue
        for i in range(1, len(ordered) - 1):
            distances[id(ordered[i])] += (values[i + 1] - values[i - 1]) / span

    return distances


def nsga2_select(
    individuals: list[Individual], n_survivors: int, rng: np.random.Generator,
) -> list[Individual]:
    fronts = fast_non_dominated_sort(individuals)
    survivors: list[Individual] = []

    for front in fronts:
        if len(survivors) + len(front) <= n_survivors:
            survivors.extend(front)
            continue

        remaining = n_survivors - len(survivors)
        if remaining <= 0:
            break
        distances = crowding_distance(front)
        # tie-break ngẫu nhiên (xác định theo rng) trước khi sort ổn định theo distance
        order = list(range(len(front)))
        rng.shuffle(order)
        shuffled = [front[i] for i in order]
        shuffled.sort(key=lambda ind: distances[id(ind)], reverse=True)
        survivors.extend(shuffled[:remaining])
        break

    return survivors
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_selection.py -v
```
Expected: PASS (8 test).

- [ ] **Step 5: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/gp/selection.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/gp/selection.py
```
Expected: cả hai sạch (nếu mypy phàn nàn `assert fi is not None` không đủ để narrow type
qua `getattr(fv, name)` động — cân nhắc thay `_as_minimize_vector`/vòng lặp `_ALL_FIELDS`
bằng truy cập field trực tiếp nếu strict mode không chấp nhận `getattr` động trên dataclass
frozen; nếu vậy, viết lại không dùng `getattr` mà liệt kê tay 6 field — quyết định tại chỗ
theo lỗi mypy thực tế).

- [ ] **Step 6: Commit**

```bash
git add src/gp/selection.py tests/unit/test_gp_selection.py
git commit -m "feat(gp): NSGA-II Pareto front + crowding distance, correlation-aware select"
```

---
