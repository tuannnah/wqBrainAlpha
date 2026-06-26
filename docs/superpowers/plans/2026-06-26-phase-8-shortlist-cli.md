# Phase 8 — Short-list + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (khuyến nghị) hoặc superpowers:executing-plans. Steps dùng checkbox (`- [ ]`). Task 8.1
> (`build_shortlist`) và 8.2 (`score_one`) độc lập — làm song song được. Task 8.3
> (`generate_many`) phụ thuộc 8.1+8.2. Task 8.4 (CLI) phụ thuộc 8.2+8.3. Task 8.5
> (review+merge+push) luôn cuối.

**Goal:** Dựng tầng orchestration cuối của MiniBrain — `score_one` (chấm 1 expr local, không
đốt sim) + `generate_many` (drive GPEngine rồi rút short-list rank+decorrelate pool-aware) +
2 lệnh CLI (`score-one`, nâng cấp `generate`) để chạy pipeline local từ PowerShell.

**Architecture:** `src/pipeline/shortlist.py` chứa rank+decorrelate thuần (không I/O).
`src/pipeline/runner.py` là orchestration network-agnostic (`score_one`, helper nội bộ
`_score_one_full` trả thêm pnl/dates để tránh backtest 2 lần, `generate_many`). `main.py` là
lớp mỏng gọi runner + in bằng `rich`. `src/pipeline` KHÔNG import `src.llm`/`src.generation`.

**Tech Stack:** Python 3.12, numpy, SQLAlchemy (qua `MiniBrainRepository`), Typer
(`typer.testing.CliRunner`), rich, pytest.

## Global Constraints

- Python 3.12; cú pháp hiện đại (`match`, `X | None`, `@dataclass(frozen=True, slots=True)`,
  `Protocol`).
- Full type hints; `mypy --strict --follow-imports=silent` clean trên file mới; `ruff`
  clean; không unused import.
- No look-ahead / no survivorship / delay-1 / stage separation (đã enforce ở Phase 0/2/3 —
  KHÔNG bypass).
- Thresholds chỉ ở `config/thresholds.py` (gate dùng `GateEvaluator` sẵn có — không hardcode).
- Determinism: randomness qua seed inject (GPEngine đã có `seed`).
- **TDD:** test trước, đỏ → code tối thiểu → xanh → commit.
- **Tiếng Việt giữ dấu đúng chính tả** trong mọi docstring/comment mới — KIỂM trước commit.
- Dependency rule B1: `src/pipeline` được phép import `src.lang/src.engine/src.backtest/
  src.data/src.storage`; được tiêu thụ bởi `main.py`; KHÔNG import `src.llm`/`src.generation`;
  KHÔNG import cứng `src.gp` (dùng Protocol structural cho `generate_many`).

## Pre-condition (chữ ký thật đã xác minh — KHÔNG đoán lại)

```python
# src/lang/parser.py
def parse(expr: str) -> Node: ...
class ParseError(Exception): ...
# src/lang/registry.py
def default_registry() -> OperatorRegistry: ...
# src/lang/visitors.py  (DÙNG .visit(node), KHÔNG .accept)
DepthVisitor().visit(node) -> int
FieldCollector().visit(node) -> set[str]
Serializer().visit(node) -> str
# src/engine/evaluator.py
@dataclass(frozen=True, slots=True)
class EvalContext: data; registry; cache  # cache: SubexprCache | None
class Evaluator: def __init__(self, ctx: EvalContext); def evaluate(self, node: Node) -> Panel
# src/backtest/portfolio.py
PortfolioBuilder().build(signal: Panel, cfg: PortfolioConfig, data: MarketData) -> Panel
# src/backtest/backtester.py
Backtester().run(weights: Panel, data: MarketData) -> BacktestResult  # .daily_pnl, .equity_curve
# src/backtest/metrics_local.py
MetricsCalculator().compute(bt: BacktestResult, data: MarketData) -> AlphaMetrics
AlphaMetrics(sharpe, annual_return, turnover, max_drawdown, fitness, per_year_sharpe, weight_concentration)
# src/backtest/gates.py
GateEvaluator().evaluate(m: AlphaMetrics, self_corr: float, depth: int, fields_ok: bool) -> GateVerdict
GateEvaluator().evaluate_with_pool(m, candidate_pnl, candidate_dates, pool_corr, depth, fields_ok) -> GateVerdict
GateVerdict(passed: bool, hard_failures: list[str], soft_scores: dict[str, float])
# src/backtest/pool_corr.py
PoolCorrelation(pool: dict[int, tuple[Dates, NDArray[float64]]]); .max_corr(pnl, dates) -> tuple[float, int|None]
# src/data/market_panel.py
MarketData.field_names() -> set[str]   # GỒM 'returns'
MarketData.dates: Dates
# src/data/adapters/parquet_source.py
ParquetSource(dir).load(start: str, end: str, universe: str = "TOP3000") -> MarketData
# src/gp/engine.py (Phase 7.7)
GPEngine(data, repo, config, registry, *, pop_size, n_generations, seed, ...); .run() -> GPRunResult
GPRunResult.final_population: list[Individual]   # Individual.expr: Node; .fitness: FitnessVector | None
# src/storage/repository.py
MiniBrainRepository(session_factory); .load_pool() -> dict[int, tuple[Dates, NDArray[float64]]]
# tests/conftest.py
fixture small_panel: MarketData, fields = {"close", "volume"}, returns phái sinh, T=120 N=30
```

## File Structure

- **Create** `src/pipeline/__init__.py` (rỗng — package marker; kiểm `Glob` trước khi tạo).
- **Create** `src/pipeline/shortlist.py` (~95 dòng): `ShortlistCandidate` + `build_shortlist`
  + helper `_pairwise_abs_rho`.
- **Create** `src/pipeline/runner.py` (~130 dòng): `_ScoreResult` + `_score_one_full` +
  `score_one` + `generate_many` + Protocol `_RunsGP`.
- **Create** `tests/unit/test_shortlist.py` (~110 dòng).
- **Create** `tests/unit/test_runner_score_one.py` (~75 dòng).
- **Create** `tests/unit/test_runner_generate_many.py` (~70 dòng).
- **Create** `tests/unit/test_cli_score_one_generate.py` (~90 dòng).
- **Modify** `main.py`: nâng cấp lệnh `generate` (~40 dòng) + thêm lệnh `score-one` (~45
  dòng) + helper `_portfolio_config_from_opts`.

---

### Task 8.1: `build_shortlist` — rank + decorrelate pool-aware

**Files:**
- Create: `src/pipeline/__init__.py`, `src/pipeline/shortlist.py`
- Test: `tests/unit/test_shortlist.py`

**Interfaces:**
- Consumes: `AlphaMetrics` (backtest.metrics_local), `PoolCorrelation` (backtest.pool_corr),
  `Dates` (local_types).
- Produces:
  ```python
  @dataclass(frozen=True, slots=True)
  class ShortlistCandidate:
      expr: str
      metrics: AlphaMetrics
      pnl: npt.NDArray[np.float64]
      dates: Dates
  def build_shortlist(candidates: list[ShortlistCandidate], top_k: int, max_corr: float,
                      pool_corr: PoolCorrelation | None = None) -> list[ShortlistCandidate]: ...
  ```

- [ ] **Step 1: Tạo package marker nếu chưa có**

```bash
venv/Scripts/python.exe -c "import os; print(os.path.exists('src/pipeline/__init__.py'))"
```
Nếu in `False`: tạo file rỗng `src/pipeline/__init__.py` (một dòng comment `# pipeline package`).

- [ ] **Step 2: Viết test đỏ `tests/unit/test_shortlist.py`**

```python
"""Test build_shortlist: rank theo fitness giảm dần + decorrelate (loại candidate có
max|rho| với cái đã chọn vượt ngưỡng), pool-aware qua PoolCorrelation."""

from __future__ import annotations

import numpy as np

from src.backtest.metrics_local import AlphaMetrics
from src.backtest.pool_corr import PoolCorrelation
from src.pipeline.shortlist import ShortlistCandidate, build_shortlist


def _dates(start: str, n: int) -> np.ndarray:
    return (np.datetime64(start) + np.arange(n)).astype("datetime64[D]")


def _metrics(fitness: float) -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.0, annual_return=0.1, turnover=0.3, max_drawdown=0.05,
        fitness=fitness, per_year_sharpe={2021: 1.0}, weight_concentration=0.05,
    )


def test_ranks_by_fitness_descending_when_uncorrelated() -> None:
    dates = _dates("2021-01-01", 20)
    rng = np.random.default_rng(0)
    low = ShortlistCandidate("low", _metrics(0.5), rng.normal(size=20), dates)
    high = ShortlistCandidate("high", _metrics(2.0), rng.normal(size=20), dates)
    out = build_shortlist([low, high], top_k=2, max_corr=0.7)
    assert [c.expr for c in out] == ["high", "low"]


def test_decorrelate_drops_high_correlation_pair() -> None:
    dates = _dates("2021-01-01", 20)
    base = np.linspace(0.01, 0.20, 20)
    a = ShortlistCandidate("a_best", _metrics(2.0), base.copy(), dates)
    b = ShortlistCandidate("b_dup", _metrics(1.5), base.copy() * 2.0, dates)
    c = ShortlistCandidate("c_diff", _metrics(1.0), -base.copy(), dates)
    out = build_shortlist([a, b, c], top_k=3, max_corr=0.7)
    names = [cand.expr for cand in out]
    assert "a_best" in names
    assert "b_dup" not in names  # rho=+1.0 với a_best
    assert "c_diff" not in names  # |rho|=1.0 (rho=-1) với a_best
    assert len(out) == 1


def test_respects_top_k_limit() -> None:
    dates = _dates("2021-01-01", 20)
    rng = np.random.default_rng(1)
    cands = [
        ShortlistCandidate(f"x{i}", _metrics(float(i)), rng.normal(size=20), dates)
        for i in range(5)
    ]
    out = build_shortlist(cands, top_k=2, max_corr=0.99)
    assert len(out) == 2
    assert out[0].expr == "x4"
    assert out[1].expr == "x3"


def test_pool_aware_drops_candidate_correlated_with_existing_pool() -> None:
    dates = _dates("2021-01-01", 20)
    pool_pnl = np.linspace(0.01, 0.20, 20)
    pool_corr = PoolCorrelation(pool={1: (dates, pool_pnl.copy())})
    dup = ShortlistCandidate("dup_pool", _metrics(2.0), pool_pnl.copy() * 3.0, dates)
    fresh = ShortlistCandidate("fresh", _metrics(1.0), -pool_pnl.copy(), dates)
    out = build_shortlist([dup, fresh], top_k=2, max_corr=0.7, pool_corr=pool_corr)
    names = [c.expr for c in out]
    assert "dup_pool" not in names
    assert "fresh" not in names  # |rho|=1.0 với pool


def test_empty_candidates_returns_empty_list() -> None:
    assert build_shortlist([], top_k=5, max_corr=0.7) == []


def test_does_not_mutate_input_list() -> None:
    dates = _dates("2021-01-01", 10)
    cands = [
        ShortlistCandidate("a", _metrics(1.0), np.ones(10), dates),
        ShortlistCandidate("b", _metrics(2.0), np.ones(10) * -1, dates),
    ]
    original = [c.expr for c in cands]
    build_shortlist(cands, top_k=2, max_corr=0.99)
    assert [c.expr for c in cands] == original
```

- [ ] **Step 3: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_shortlist.py -q
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.pipeline.shortlist'`.

- [ ] **Step 4: Viết `src/pipeline/shortlist.py`**

```python
"""Rank + decorrelate candidate đã có metrics/pnl → short-list cuối để sim Brain.

Đây là bước tổng hợp của pipeline: nhận candidate đã qua score_one, xếp hạng theo fitness,
rồi loại tuần tự candidate tương quan PnL quá cao với cái ĐÃ CHỌN (decorrelate nội bộ) VÀ với
pool đã pass (decorrelate pool-aware) — đúng nguyên tắc B9: PnL self-corr là nguyên nhân
reject hàng đầu, không phải AST-hash dedup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.backtest.metrics_local import AlphaMetrics
from src.backtest.pool_corr import PoolCorrelation
from src.local_types import Dates


@dataclass(frozen=True, slots=True)
class ShortlistCandidate:
    """Một candidate đã backtest xong: expr + metrics + PnL hằng ngày (để tính tương quan)."""

    expr: str
    metrics: AlphaMetrics
    pnl: npt.NDArray[np.float64]
    dates: Dates


def _pairwise_abs_rho(
    pnl_a: npt.NDArray[np.float64], dates_a: Dates,
    pnl_b: npt.NDArray[np.float64], dates_b: Dates,
) -> float | None:
    """Pearson |rho| trên giao ngày chung; None nếu thiếu điểm/phương sai bằng 0 (giống
    PoolCorrelation._pairwise_rho Phase 6) — KHÔNG bịa rho=0 giả."""
    common = np.intersect1d(dates_a, dates_b)
    if common.size < 2:
        return None
    idx_a = np.searchsorted(dates_a, common)
    idx_b = np.searchsorted(dates_b, common)
    x = pnl_a[idx_a]
    y = pnl_b[idx_b]
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return None
    x = x[finite]
    y = y[finite]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    rho = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(rho):
        return None
    return abs(rho)


def build_shortlist(
    candidates: list[ShortlistCandidate],
    top_k: int,
    max_corr: float,
    pool_corr: PoolCorrelation | None = None,
) -> list[ShortlistCandidate]:
    """Xếp hạng `candidates` theo `metrics.fitness` giảm dần, rồi quét tuần tự: giữ candidate
    nếu max|rho| với MỌI candidate đã giữ trước đó VÀ với pool (qua `pool_corr.max_corr` nếu
    có) đều < `max_corr`. Dừng khi đủ `top_k` hoặc hết. Không sửa đổi danh sách đầu vào."""
    ranked = sorted(candidates, key=lambda c: c.metrics.fitness, reverse=True)
    kept: list[ShortlistCandidate] = []
    for cand in ranked:
        if len(kept) >= top_k:
            break
        if pool_corr is not None:
            pool_rho, _worst = pool_corr.max_corr(cand.pnl, cand.dates)
            if abs(pool_rho) >= max_corr:
                continue
        too_correlated = False
        for chosen in kept:
            rho = _pairwise_abs_rho(cand.pnl, cand.dates, chosen.pnl, chosen.dates)
            if rho is not None and rho >= max_corr:
                too_correlated = True
                break
        if not too_correlated:
            kept.append(cand)
    return kept
```

- [ ] **Step 5: Chạy test — PASS (6 test)**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_shortlist.py -q
```

- [ ] **Step 6: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/pipeline/shortlist.py src/pipeline/__init__.py tests/unit/test_shortlist.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/pipeline/shortlist.py
```
Expected: sạch.

- [ ] **Step 7: Kiểm dấu tiếng Việt trong shortlist.py, rồi commit**

```bash
git add src/pipeline/__init__.py src/pipeline/shortlist.py tests/unit/test_shortlist.py
git commit -m "feat(pipeline): build_shortlist - rank fitness + decorrelate pool-aware"
```

---

### Task 8.2: `score_one` + `_score_one_full` (`src/pipeline/runner.py`)

**Files:**
- Create: `src/pipeline/runner.py`
- Test: `tests/unit/test_runner_score_one.py`

**Interfaces:**
- Consumes: `parse`/`ParseError`, `default_registry`, `EvalContext`/`Evaluator`,
  `PortfolioBuilder`, `Backtester`, `MetricsCalculator`/`AlphaMetrics`, `GateEvaluator`/
  `GateVerdict`, `PoolCorrelation`, `MarketData`, `DepthVisitor`/`FieldCollector`, `Dates`.
- Produces:
  ```python
  @dataclass(frozen=True, slots=True)
  class _ScoreResult:
      metrics: AlphaMetrics
      verdict: GateVerdict
      pnl: npt.NDArray[np.float64]   # rỗng nếu fail trước backtest
      dates: Dates
  def _score_one_full(expr, cfg, data, pool=None) -> _ScoreResult: ...
  def score_one(expr: str, cfg: PortfolioConfig, data: MarketData,
                pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] | None = None,
                ) -> tuple[AlphaMetrics, GateVerdict]: ...
  ```

- [ ] **Step 1: Viết test đỏ `tests/unit/test_runner_score_one.py`**

```python
"""Test score_one: parse→eval→backtest→metrics→gate trên small_panel, không mạng/sim Brain."""

from __future__ import annotations

import numpy as np

from src.backtest.config import PortfolioConfig
from src.backtest.gates import GateVerdict
from src.backtest.metrics_local import AlphaMetrics
from src.pipeline.runner import score_one


def test_valid_expression_returns_metrics_and_verdict(small_panel) -> None:  # noqa: ANN001
    metrics, verdict = score_one("close", PortfolioConfig(delay=1), small_panel)
    assert isinstance(metrics, AlphaMetrics)
    assert isinstance(verdict, GateVerdict)
    assert np.isfinite(metrics.sharpe)


def test_parse_error_returns_failing_verdict_not_exception(small_panel) -> None:  # noqa: ANN001
    metrics, verdict = score_one("not_a_real_op(close,", PortfolioConfig(), small_panel)
    assert verdict.passed is False
    assert any("parse" in f.lower() for f in verdict.hard_failures)
    assert metrics.sharpe == 0.0


def test_unknown_field_returns_failing_verdict(small_panel) -> None:  # noqa: ANN001
    metrics, verdict = score_one("totally_unknown_field_xyz", PortfolioConfig(), small_panel)
    assert verdict.passed is False


def test_pool_aware_metrics_unchanged_by_pool(small_panel) -> None:  # noqa: ANN001
    dates = small_panel.dates
    pool = {1: (dates, np.linspace(0.01, 0.10, len(dates)))}
    m_no, _ = score_one("close", PortfolioConfig(delay=1), small_panel, pool=None)
    m_pool, v_pool = score_one("close", PortfolioConfig(delay=1), small_panel, pool=pool)
    assert m_no == m_pool  # pool chỉ ảnh hưởng verdict.self_corr, không đổi AlphaMetrics
    assert isinstance(v_pool, GateVerdict)


def test_deterministic_same_inputs_same_output(small_panel) -> None:  # noqa: ANN001
    cfg = PortfolioConfig(delay=1)
    m1, v1 = score_one("close", cfg, small_panel)
    m2, v2 = score_one("close", cfg, small_panel)
    assert m1 == m2
    assert v1.passed == v2.passed
    assert v1.hard_failures == v2.hard_failures
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_runner_score_one.py -q
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.pipeline.runner'`.

- [ ] **Step 3: Viết `src/pipeline/runner.py` (phần score_one)**

```python
"""Lớp orchestration cuối của MiniBrain: score_one chấm 1 expr KHÔNG đốt sim; generate_many
drive GPEngine rồi rút short-list. Network-agnostic — nhận MarketData/GPEngine qua tham số
injected, test được bằng fake hoàn toàn. KHÔNG import src.llm/src.generation (dependency rule
B1); KHÔNG import cứng src.gp (generate_many dùng Protocol structural)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.backtest.backtester import Backtester
from src.backtest.config import PortfolioConfig
from src.backtest.gates import GateEvaluator, GateVerdict
from src.backtest.metrics_local import AlphaMetrics, MetricsCalculator
from src.backtest.pool_corr import PoolCorrelation
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import ParseError, parse
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor, FieldCollector
from src.local_types import Dates

_EMPTY_METRICS = AlphaMetrics(
    sharpe=0.0, annual_return=0.0, turnover=0.0, max_drawdown=0.0,
    fitness=0.0, per_year_sharpe={}, weight_concentration=0.0,
)


@dataclass(frozen=True, slots=True)
class _ScoreResult:
    """Kết quả đầy đủ của một lần chấm: metrics + verdict + PnL/dates (PnL rỗng nếu fail
    trước khi backtest chạy). Dùng nội bộ để generate_many lấy PnL không phải backtest lại."""

    metrics: AlphaMetrics
    verdict: GateVerdict
    pnl: npt.NDArray[np.float64]
    dates: Dates


def _score_one_full(
    expr: str,
    cfg: PortfolioConfig,
    data: MarketData,
    pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] | None = None,
) -> _ScoreResult:
    """parse → eval → portfolio → backtest → metrics → pool_corr → gate. Thuần local, tất
    định với cùng (expr, cfg, data, pool). Lỗi parse/eval → metrics rỗng + verdict fail có lý
    do rõ ràng (KHÔNG silent, KHÔNG bịa metrics) và PnL rỗng."""
    empty_pnl: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    try:
        node = parse(expr)
    except ParseError as exc:
        return _ScoreResult(
            _EMPTY_METRICS,
            GateVerdict(passed=False, hard_failures=[f"parse lỗi: {exc}"]),
            empty_pnl, data.dates,
        )

    fields = FieldCollector().visit(node)
    fields_ok = bool(fields) and fields.issubset(data.field_names())
    depth = DepthVisitor().visit(node)

    ctx = EvalContext(data=data, registry=default_registry(), cache=None)
    try:
        signal = Evaluator(ctx).evaluate(node)
    except (KeyError, ValueError) as exc:
        return _ScoreResult(
            _EMPTY_METRICS,
            GateVerdict(passed=False, hard_failures=[f"eval lỗi: {exc}"]),
            empty_pnl, data.dates,
        )

    if bool(np.all(np.isnan(signal))):
        return _ScoreResult(
            _EMPTY_METRICS,
            GateVerdict(passed=False, hard_failures=["signal toàn NaN — không dùng được"]),
            empty_pnl, data.dates,
        )

    weights = PortfolioBuilder().build(signal, cfg, data)
    bt = Backtester().run(weights, data)
    metrics = MetricsCalculator().compute(bt, data)

    if pool:
        verdict = GateEvaluator().evaluate_with_pool(
            metrics, candidate_pnl=bt.daily_pnl, candidate_dates=data.dates,
            pool_corr=PoolCorrelation(pool=pool), depth=depth, fields_ok=fields_ok,
        )
    else:
        verdict = GateEvaluator().evaluate(
            metrics, self_corr=0.0, depth=depth, fields_ok=fields_ok,
        )
    return _ScoreResult(metrics, verdict, bt.daily_pnl, data.dates)


def score_one(
    expr: str,
    cfg: PortfolioConfig,
    data: MarketData,
    pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] | None = None,
) -> tuple[AlphaMetrics, GateVerdict]:
    """Chấm 1 expr local (không đốt sim). Trả (AlphaMetrics, GateVerdict). Xem
    `_score_one_full` cho ngữ nghĩa lỗi/pool đầy đủ."""
    res = _score_one_full(expr, cfg, data, pool)
    return res.metrics, res.verdict
```

- [ ] **Step 4: Chạy test — PASS (5 test)**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_runner_score_one.py -q
```
Nếu `Evaluator.evaluate` ném exception KHÁC `KeyError`/`ValueError` cho field rác (đọc
traceback): mở rộng tuple `except` cho khớp loại lỗi thật — KHÔNG bắt `Exception` trần.

- [ ] **Step 5: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/pipeline/runner.py tests/unit/test_runner_score_one.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/pipeline/runner.py
```
Expected: sạch.

- [ ] **Step 6: Kiểm dấu tiếng Việt, rồi commit**

```bash
git add src/pipeline/runner.py tests/unit/test_runner_score_one.py
git commit -m "feat(pipeline): score_one - parse->eval->backtest->metrics->gate, khong dot sim"
```

---

### Task 8.3: `generate_many` — drive GPEngine, trả ranked short-list

**Files:**
- Modify: `src/pipeline/runner.py` (thêm import + Protocol + hàm ở cuối)
- Test: `tests/unit/test_runner_generate_many.py`

**Interfaces:**
- Consumes: `_score_one_full` (Task 8.2 cùng file), `build_shortlist`/`ShortlistCandidate`
  (Task 8.1), `Serializer` (lang.visitors), `PoolCorrelation`.
- Produces:
  ```python
  class _RunsGP(Protocol):
      def run(self) -> _GPRunResultLike: ...
  def generate_many(gp_engine: _RunsGP, cfg: PortfolioConfig, data: MarketData,
                    top_k: int, max_corr: float,
                    pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] | None = None,
                    ) -> list[ShortlistCandidate]: ...
  ```

- [ ] **Step 1: Viết test đỏ `tests/unit/test_runner_generate_many.py`**

```python
"""Test generate_many: drive (fake) GPEngine.run, score lại bằng _score_one_full, rồi
build_shortlist. Dùng fake GPEngine — KHÔNG chạy GP thực."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from src.backtest.config import PortfolioConfig
from src.lang.parser import parse
from src.pipeline.runner import generate_many


@dataclass
class _FakeFitness:
    sharpe_deflated: float = 1.0


@dataclass
class _FakeIndividual:
    expr: object  # Node thực từ parse()
    fitness: object | None = None
    generation: int = 0


@dataclass
class _FakeRunResult:
    final_population: list = dc_field(default_factory=list)


class _FakeGPEngine:
    """Fake GPEngine: run() trả GPRunResult-like với final_population gồm Individual có AST
    parse từ string cố định — không chạy GP thực (không cần registry/evaluator/...)."""

    def __init__(self, exprs: list[str]) -> None:
        self._exprs = exprs

    def run(self) -> _FakeRunResult:
        return _FakeRunResult(final_population=[
            _FakeIndividual(expr=parse(e), fitness=_FakeFitness()) for e in self._exprs
        ])


def test_generate_many_returns_shortlist_from_fake_gp(small_panel) -> None:  # noqa: ANN001
    engine = _FakeGPEngine(["close", "volume"])
    out = generate_many(
        gp_engine=engine, cfg=PortfolioConfig(delay=1), data=small_panel,
        top_k=5, max_corr=0.99,
    )
    assert len(out) <= 2
    assert all(c.expr in ("close", "volume") for c in out)


def test_generate_many_skips_individuals_with_no_fitness(small_panel) -> None:  # noqa: ANN001
    class _Partial(_FakeGPEngine):
        def run(self) -> _FakeRunResult:
            return _FakeRunResult(final_population=[
                _FakeIndividual(expr=parse("close"), fitness=_FakeFitness()),
                _FakeIndividual(expr=parse("volume"), fitness=None),  # chưa eval -> bỏ qua
            ])

    out = generate_many(
        gp_engine=_Partial([]), cfg=PortfolioConfig(delay=1), data=small_panel,
        top_k=5, max_corr=0.99,
    )
    assert [c.expr for c in out] == ["close"]


def test_generate_many_respects_top_k(small_panel) -> None:  # noqa: ANN001
    engine = _FakeGPEngine(["close", "volume"])
    out = generate_many(
        gp_engine=engine, cfg=PortfolioConfig(delay=1), data=small_panel,
        top_k=1, max_corr=0.99,
    )
    assert len(out) <= 1
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_runner_generate_many.py -q
```
Expected: FAIL `ImportError: cannot import name 'generate_many'`.

- [ ] **Step 3: Thêm import + Protocol + `generate_many` vào cuối `src/pipeline/runner.py`**

Thêm vào khối import đầu file:
```python
from typing import Protocol

from src.lang.visitors import DepthVisitor, FieldCollector, Serializer  # thêm Serializer
from src.pipeline.shortlist import ShortlistCandidate, build_shortlist
```
(gộp `Serializer` vào dòng import visitors đã có — KHÔNG tạo dòng trùng.)

Thêm cuối file:
```python
class _GPIndividualLike(Protocol):
    expr: object
    fitness: object | None


class _GPRunResultLike(Protocol):
    final_population: list[_GPIndividualLike]


class _RunsGP(Protocol):
    def run(self) -> _GPRunResultLike: ...


def generate_many(
    gp_engine: _RunsGP,
    cfg: PortfolioConfig,
    data: MarketData,
    top_k: int,
    max_corr: float,
    pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] | None = None,
) -> list[ShortlistCandidate]:
    """Chạy `gp_engine.run()` → final_population; với mỗi Individual đã eval (fitness không
    None), serialize AST → string, chấm lại qua `_score_one_full` (một nguồn AlphaMetrics +
    PnL duy nhất, KHÔNG backtest 2 lần), giữ cái pass gate, rồi `build_shortlist` top_k +
    decorrelate pool-aware. Individual fitness=None (chưa eval trong GP) bị bỏ qua."""
    result = gp_engine.run()
    serializer = Serializer()
    pool_corr = PoolCorrelation(pool=pool) if pool else None

    candidates: list[ShortlistCandidate] = []
    seen: set[str] = set()
    for ind in result.final_population:
        if ind.fitness is None:
            continue
        expr_str = serializer.visit(ind.expr)  # type: ignore[arg-type]
        if expr_str in seen:
            continue
        seen.add(expr_str)
        res = _score_one_full(expr_str, cfg, data, pool)
        if not res.verdict.passed:
            continue
        candidates.append(
            ShortlistCandidate(expr=expr_str, metrics=res.metrics, pnl=res.pnl, dates=res.dates)
        )
    return build_shortlist(candidates, top_k=top_k, max_corr=max_corr, pool_corr=pool_corr)
```

- [ ] **Step 4: Chạy test — PASS (3 test)**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_runner_generate_many.py -q
```

- [ ] **Step 5: Chạy lại toàn pipeline + ruff + mypy**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_shortlist.py tests/unit/test_runner_score_one.py tests/unit/test_runner_generate_many.py -q
venv/Scripts/python.exe -m ruff check src/pipeline/runner.py tests/unit/test_runner_generate_many.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/pipeline/runner.py
```
Expected: tất cả PASS + sạch. Nếu mypy than `serializer.visit(ind.expr)` do `ind.expr: object`:
giữ `# type: ignore[arg-type]` (Protocol structural cố ý lỏng kiểu để test bằng fake — ghi
chú đã có).

- [ ] **Step 6: Kiểm dấu tiếng Việt, rồi commit**

```bash
git add src/pipeline/runner.py tests/unit/test_runner_generate_many.py
git commit -m "feat(pipeline): generate_many - drive GPEngine roi rank+decorrelate shortlist"
```

---

### Task 8.4: CLI `score-one` + nâng cấp `generate` (`main.py`)

**Files:**
- Modify: `main.py`
- Test: `tests/unit/test_cli_score_one_generate.py`

**Interfaces:**
- Consumes: `score_one`/`generate_many` (runner), `PortfolioConfig`/`Neutralization`
  (backtest.config), `ParquetSource`, `GPEngine`, `default_registry`, `MiniBrainRepository`,
  `init_db`/`make_engine`/`make_session_factory` (đã import sẵn trong main.py),
  `app`/`console`/`_setup_logging`.
- Produces: lệnh `@app.command("score-one")` + nâng cấp `@app.command() generate` + helper
  module-level `_portfolio_config_from_opts(neutralization, decay, truncation, delay)`.

**Bước 0:** Lệnh `generate` hiện tại ở `main.py` (đã viết Phase 7.8) gọi `GPEngine.run()` trực
tiếp và in thống kê. Đọc lại nguyên hàm `generate` để thay phần in kết quả bằng short-list.
ParquetSource load + dựng GPEngine giữ nguyên pattern.

- [ ] **Step 1: Viết test đỏ `tests/unit/test_cli_score_one_generate.py`**

> Đọc `src/data/adapters/parquet_source.py` xem có method GHI panel (`save`/`write`) không.
> Nếu KHÔNG có, test ghi parquet trực tiếp bằng cách tái dùng layout `load()` mong đợi. Mẫu
> dưới dùng `_write_panel` helper gọi API thật — XÁC NHẬN tên ở Bước 0; nếu khác, sửa helper.

```python
"""Test CLI score-one/generate: dùng CliRunner, KHÔNG mạng/sim Brain. Ghi MarketData fake ra
parquet tạm để CLI đọc lại qua --market-data-dir."""

from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from main import app

runner = CliRunner()


def _write_panel(data_dir, panel) -> None:
    """Ghi small_panel ra layout ParquetSource.load() đọc được. XÁC NHẬN API thật ở Bước 0:
    nếu ParquetSource có classmethod ghi, gọi nó; nếu không, ghi parquet thủ công theo load()."""
    from src.data.adapters.parquet_source import ParquetSource  # noqa: F401
    # TODO-AT-EXEC: thay bằng lời gọi ghi thật sau khi đọc parquet_source.py.
    raise NotImplementedError


def test_score_one_missing_market_data_dir_fails_clearly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["score-one", "close", "--market-data-dir", str(tmp_path / "nope")],
    )
    assert result.exit_code == 1


def test_score_one_real_panel_prints_metrics(tmp_path, small_panel) -> None:  # noqa: ANN001
    data_dir = tmp_path / "panel"
    _write_panel(data_dir, small_panel)
    result = runner.invoke(app, ["score-one", "close", "--market-data-dir", str(data_dir)])
    assert result.exit_code == 0
    assert "sharpe" in result.stdout.lower()


def test_score_one_invalid_expr_exits_zero_prints_fail(tmp_path, small_panel) -> None:  # noqa: ANN001
    data_dir = tmp_path / "panel2"
    _write_panel(data_dir, small_panel)
    result = runner.invoke(
        app, ["score-one", "not_a_real_op(close,", "--market-data-dir", str(data_dir)],
    )
    assert result.exit_code == 0  # CLI không crash; in verdict fail
    assert "false" in result.stdout.lower() or "fail" in result.stdout.lower()


def test_generate_missing_market_data_dir_fails_clearly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["generate", "--market-data-dir", str(tmp_path / "nope"), "--count", "4"],
    )
    assert result.exit_code == 1
```

**Bước 0 (bắt buộc trước Step 2):** đọc `src/data/adapters/parquet_source.py` để hoàn thiện
`_write_panel`. Nếu không có method ghi, hiện thực `_write_panel` bằng pyarrow/pandas theo
đúng layout `load()` đọc (per-field parquet + dates/assets/universe). Sau đó xoá
`raise NotImplementedError`.

- [ ] **Step 2: Chạy test — FAIL** (`No command "score-one"` / `_write_panel` NotImplemented)

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_cli_score_one_generate.py -q
```

- [ ] **Step 3: Thêm helper `_portfolio_config_from_opts` + lệnh `score-one` vào `main.py`**

Thêm gần các lệnh khác (sau import top-level đã có; KHÔNG thêm import trùng):
```python
def _portfolio_config_from_opts(
    neutralization: str, decay: int, truncation: float, delay: int,
):
    """Dựng PortfolioConfig từ option CLI; neutralization là tên enum không phân biệt hoa."""
    from src.backtest.config import Neutralization, PortfolioConfig

    try:
        neut = Neutralization[neutralization.upper()]
    except KeyError as exc:
        console.print(
            f"[red]neutralization '{neutralization}' không hợp lệ. Chọn: "
            f"{', '.join(n.name for n in Neutralization)}[/red]"
        )
        raise typer.Exit(code=1) from exc
    return PortfolioConfig(
        neutralization=neut, decay=decay, truncation=truncation, scale_book=1.0, delay=delay,
    )


@app.command("score-one")
def score_one_cmd(
    expr: str = typer.Argument(..., help="Biểu thức FASTEXPR cần chấm (signal core)"),
    market_data_dir: str = typer.Option(..., help="Thư mục parquet MarketData (ParquetSource)"),
    universe: str = typer.Option("TOP3000", help="Universe panel"),
    neutralization: str = typer.Option("NONE", help="NONE/MARKET/SECTOR/INDUSTRY/SUBINDUSTRY"),
    decay: int = typer.Option(0, help="Decay (ngày)"),
    truncation: float = typer.Option(0.10, help="Truncation trọng số"),
    delay: int = typer.Option(1, help="Delay (delay-1 chuẩn)"),
    no_pool: bool = typer.Option(False, "--no-pool", help="Bỏ qua pool DB (self_corr=0)"),
) -> None:
    """Chấm 1 expression local (parse→eval→backtest→metrics→gate), KHÔNG đốt sim Brain. Nạp
    pool PnL từ DB hiện hành để gate self-correlation có nghĩa (trừ khi --no-pool)."""
    _setup_logging()

    import src.operators_local  # noqa: F401  (nạp 27 operator vào registry)
    from pathlib import Path

    from src.data.adapters.parquet_source import ParquetSource
    from src.pipeline.runner import score_one
    from src.storage.repository import MiniBrainRepository

    if not Path(market_data_dir).is_dir():
        console.print(f"[red]Không thấy thư mục MarketData: {market_data_dir}[/red]")
        raise typer.Exit(code=1)

    try:
        data = ParquetSource(market_data_dir).load("1900-01-01", "2999-12-31", universe)
    except (FileNotFoundError, AssertionError, OSError) as exc:
        console.print(f"[red]Không load được MarketData: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    cfg = _portfolio_config_from_opts(neutralization, decay, truncation, delay)

    pool = None
    if not no_pool:
        repo = MiniBrainRepository(make_session_factory(init_db(make_engine())))
        pool = repo.load_pool() or None

    metrics, verdict = score_one(expr, cfg, data, pool=pool)
    table = Table(title=f"score-one: {expr}")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("sharpe", f"{metrics.sharpe:.4f}")
    table.add_row("fitness", f"{metrics.fitness:.4f}")
    table.add_row("turnover", f"{metrics.turnover:.4f}")
    table.add_row("max_drawdown", f"{metrics.max_drawdown:.4f}")
    table.add_row("passed", str(verdict.passed))
    if verdict.hard_failures:
        table.add_row("fail", "; ".join(verdict.hard_failures))
    console.print(table)
```

- [ ] **Step 4: Nâng cấp lệnh `generate` — in short-list qua `generate_many`**

Thay phần TÍNH + IN của hàm `generate` hiện tại. Thêm option `--top-k`/`--max-corr`; sau khi
dựng `gp_engine` (giữ nguyên load data + repo + cfg), thay `result = gp_engine.run()` + in
thống kê bằng:
```python
    # (thêm vào chữ ký generate, sau seed:)
    #   top_k: int = typer.Option(10, help="Số alpha giữ trong short-list cuối"),
    #   max_corr: float = typer.Option(0.70, help="Ngưỡng |rho| decorrelate short-list"),
    #   neutralization/decay/truncation/delay như score-one (thay cfg cứng cũ)
    from src.pipeline.runner import generate_many

    pool = repo.load_pool() or None
    shortlist = generate_many(
        gp_engine=gp_engine, cfg=cfg, data=data, top_k=top_k, max_corr=max_corr, pool=pool,
    )
    table = Table(title=f"Short-list ({len(shortlist)} alpha, max_corr={max_corr})")
    table.add_column("#"); table.add_column("expr"); table.add_column("sharpe")
    table.add_column("fitness")
    for i, c in enumerate(shortlist, 1):
        table.add_row(str(i), c.expr, f"{c.metrics.sharpe:.3f}", f"{c.metrics.fitness:.3f}")
    console.print(table)
    console.print(f"[green]GP done[/green]: short-list {len(shortlist)} alpha (đã decorrelate).")
```
Đồng thời thay `cfg = PortfolioConfig(...)` cứng trong `generate` bằng
`cfg = _portfolio_config_from_opts(neutralization, decay, truncation, delay)`.

- [ ] **Step 5: Smoke test CLI parse (không chạy DB thật)**

```bash
venv/Scripts/python.exe main.py score-one --help
venv/Scripts/python.exe main.py generate --help
```
Expected: in usage có `--market-data-dir`, `--neutralization`, `--top-k`/`--max-corr`
(generate), không crash.

- [ ] **Step 6: Chạy test CLI — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_cli_score_one_generate.py -q
```
Expected: 4 PASS (sau khi `_write_panel` đã hiện thực ở Bước 0).

- [ ] **Step 7: ruff main.py (chỉ phần mới) + kiểm dấu tiếng Việt**

```bash
venv/Scripts/python.exe -m ruff check main.py
```
main.py có lỗi ruff TIỀN-TỒN (E402 import dòng 22-33, F841 dòng ~360 ở lệnh sweep-config) —
KHÔNG thuộc lệnh mới; document, không sửa. Nếu có lỗi MỚI trong score-one/generate: fix.

- [ ] **Step 8: Commit**

```bash
git add main.py tests/unit/test_cli_score_one_generate.py
git commit -m "feat(cli): score-one + generate in short-list qua generate_many (config flags)"
```

---

### Task 8.5: Review toàn Phase 8 + Merge + Push

**Files:** không tạo mới — review nhánh `phase-8-shortlist-cli`.

- [ ] **Step 1: Full suite + ruff + mypy Phase 8**

```bash
venv/Scripts/python.exe -m pytest tests/ -q
venv/Scripts/python.exe -m ruff check src/pipeline/
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/pipeline/
```
Expected: pytest PASS toàn bộ (trừ `test_db_postgres` psycopg tiền-tồn); ruff + mypy sạch
src/pipeline.

- [ ] **Step 2: Xác nhận `calibrate` đã wired (Task 8.5 plan gốc = no-op)**

```bash
grep -n "make_local_scorer\|CalibrationHarness" main.py
```
Expected: lệnh `calibrate` (`main.py:~1423`) đã dùng `make_local_scorer` + `CalibrationHarness`
trên `ParquetSource` thật → KHÔNG tạo lệnh mới. Ghi chú no-op trong journal.

- [ ] **Step 3: Self-review diff toàn nhánh**

```bash
git diff main...HEAD --stat
```
Kiểm tay:
- [ ] `src/pipeline/shortlist.py` + `runner.py` mới, mỗi file 1 responsibility, < 140 dòng.
- [ ] `src/pipeline` KHÔNG import `src.llm`/`src.generation`/`src.gp` (Protocol cho gp_engine).
- [ ] `score_one` lỗi parse/eval → verdict fail, KHÔNG exception nổi CLI.
- [ ] `_score_one_full` chống backtest 2 lần (generate_many không gọi Backtester trực tiếp).
- [ ] Tiếng Việt giữ dấu trong file mới.

- [ ] **Step 4: Merge --no-ff + push**

```bash
git checkout main
git pull --ff-only
git merge --no-ff phase-8-shortlist-cli -m "merge: Phase 8 - short-list + CLI (score-one + generate)"
git push origin main
```

- [ ] **Step 5: Cập nhật journal `skill/minibrain-skills-bundle/PROGRESS.md`**

Append Session entry + làm tươi Current state (Phase 8 hoàn tất → Next: Phase 9 hoặc kết
thúc MVP-to-scale). Commit + push:
```bash
git add skill/minibrain-skills-bundle/PROGRESS.md
git commit -m "docs(progress): Phase 8 hoan tat (short-list + CLI) - journal"
git push origin main
```

- [ ] **Step 6: Xóa nhánh local**

```bash
git branch -d phase-8-shortlist-cli
```

---

## Self-review

**Spec coverage:**
- [x] `build_shortlist` rank+decorrelate pool-aware — Task 8.1.
- [x] `score_one` parse→...→gate, lỗi → verdict fail — Task 8.2.
- [x] `_score_one_full` chống tính 2 lần — Task 8.2 (dùng ở 8.3).
- [x] `generate_many` drive GPEngine.run + shortlist — Task 8.3.
- [x] CLI `score-one` nạp pool không persist — Task 8.4 Step 3.
- [x] CLI `generate` nâng cấp + config flags — Task 8.4 Step 4.
- [x] `calibrate` no-op (đã wired) — Task 8.5 Step 2.
- [x] Review + merge + push + journal — Task 8.5.

**Placeholder scan:** Có MỘT chủ ý: `_write_panel` (Task 8.4 test) `raise NotImplementedError`
+ TODO-AT-EXEC vì phụ thuộc API ghi parquet thật (Bước 0 hiện thực). Mọi step khác có code cụ
thể.

**Type consistency:**
- `score_one(...) -> tuple[AlphaMetrics, GateVerdict]`; `_score_one_full(...) -> _ScoreResult`
  (metrics/verdict/pnl/dates) — nhất quán giữa Task 8.2 và 8.3.
- `generate_many(gp_engine, cfg, data, top_k, max_corr, pool=None) -> list[ShortlistCandidate]`
  — khớp chữ ký dùng ở CLI Task 8.4 (KHÔNG có tham số `generations`; engine giữ n_generations).
- `ShortlistCandidate(expr, metrics, pnl, dates)` — nhất quán Task 8.1/8.3.
- `build_shortlist(candidates, top_k, max_corr, pool_corr=None)` — Task 8.1 = lời gọi 8.3.

**Risks / gotchas:**
1. `_write_panel` phụ thuộc API ghi parquet (Bước 0 Task 8.4) — nếu ParquetSource chỉ đọc,
   ghi thủ công theo layout `load()`.
2. `Evaluator.evaluate` loại exception cho field rác (Task 8.2 Step 4) — mở rộng `except`
   theo traceback thật, không bắt `Exception` trần.
3. `generate` đã đổi nghĩa output (short-list thay vì thống kê thô) — người dùng quen lệnh cũ
   cần biết; ghi trong journal.
