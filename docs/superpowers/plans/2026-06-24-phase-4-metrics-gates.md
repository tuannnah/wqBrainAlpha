# Phase 4 — Metrics + Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (khuyến nghị) hoặc superpowers:executing-plans để thực thi từng task. Steps dùng checkbox
> (`- [ ]`) để theo dõi; chạy tuần tự — Task 4.1→4.2 (cùng file `metrics_local.py`, 4.2 nối
> tiếp 4.1) →4.3 (`gates.py`, phụ thuộc `AlphaMetrics`) →4.4 (`filter.evaluate_local`, phụ
> thuộc `GateEvaluator`) →4.5 (integration, phụ thuộc cả bốn) →4.6 (review+merge+push, luôn
> cuối).

**Goal:** Dựng tầng đo lường + cổng chặn local của MiniBrain — `AlphaMetrics` +
`MetricsCalculator` (sharpe/annual_return/turnover/max_drawdown/fitness/per_year_sharpe/
weight_concentration) tính từ `BacktestResult` (Phase 3) trên `MarketData` (Phase 0), và
`GateVerdict` + `GateEvaluator` (hard gates depth/fields/self_corr/concentration + soft
scores sharpe/fitness/turnover-band/per_year_min) với ngưỡng đọc duy nhất từ
`config/thresholds.py`. Thêm `evaluate_local(...)` vào `src/scoring/filter.py` hiện có (wrap
`GateEvaluator`, không phá filter cũ cho sim Brain). Chứng minh end-to-end
parse→eval→portfolio→backtest→metrics→gate trên `small_panel` thật.

**Architecture:** `src/backtest/metrics_local.py` và `src/backtest/gates.py` mới, cùng tầng
với `config.py`/`portfolio.py`/`backtester.py` (Phase 3) — **không** import `src/gp`,
`src/storage`, `src/llm` (dependency rule B1/master plan). `metrics_local.py` chỉ phụ thuộc
`src/backtest/backtester.py` (`BacktestResult`), `src/data/market_panel.py` (`MarketData`),
`config/thresholds.py`. `gates.py` chỉ phụ thuộc `metrics_local.py` và
`config/thresholds.py`. `src/scoring/filter.py` (tầng `scoring`, không phải `backtest`)
được phép import `src/backtest/gates.py` và `src/backtest/metrics_local.py` — hướng phụ
thuộc `scoring → backtest` một chiều, không ngược lại, giữ đúng quy tắc B1.

**Tên file lưu ý:** Phase 3 đã tạo `src/backtest/gate.py` (số ít, chứa `score_local_gate` —
gate tối thiểu MVP dùng tạm, ghi rõ trong code của Phase 3 là "sẽ mở rộng gọi
MetricsCalculator + GateEvaluator khi các thành phần đó tồn tại"). Phase 4 tạo
`src/backtest/gates.py` (số nhiều, file MỚI, KHÔNG đụng `gate.py`) chứa `GateVerdict` +
`GateEvaluator` — đây là cổng đầy đủ (Metrics + hard/soft) theo B8 master spec. Hai file
cùng tồn tại có chủ đích khác nhau: `gate.py` (Phase 3) là wrapper end-to-end cho
`RefinementLoop` (parse→eval→backtest→verdict tối giản), `gates.py` (Phase 4) là
`GateEvaluator` thuần (nhận `AlphaMetrics` đã tính sẵn, không tự parse/eval/backtest). Task
4.6 nối hai file: cập nhật `score_local_gate` trong `gate.py` để gọi
`MetricsCalculator.compute` + `GateEvaluator.evaluate` thay cho điều kiện tối thiểu cũ
("signal toàn-NaN" / "không sinh được pnl").

**Tech Stack:** Python 3.12, numpy, pytest, ruff, mypy --strict. Không thêm dependency mới.

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

## Pre-condition (đọc trước khi bắt đầu)

Phase 4 **consume** trực tiếp `BacktestResult` từ Phase 3 (`src/backtest/backtester.py`) và
`MarketData` từ Phase 0 (`src/data/market_panel.py`). Trước khi bắt đầu Task 4.1, xác nhận
hai file này tồn tại và có đúng chữ ký đã dùng trong plan này:

```bash
venv/Scripts/python.exe -c "from src.backtest.backtester import BacktestResult, Backtester; from src.data.market_panel import MarketData; print('ok')"
```

- Nếu `ok` → tiếp tục Task 4.1 như viết dưới.
- Nếu `ModuleNotFoundError` → Phase 3 chưa merge vào `main`. DỪNG, báo cáo block, không tự
  viết `BacktestResult`/`Backtester` tạm ở đây — đó là lấn phạm vi Phase 3.

`BacktestResult` (Phase 3, đã merge giả định) có chữ ký:
```python
@dataclass(frozen=True, slots=True)
class BacktestResult:
    daily_pnl: npt.NDArray[np.float64]      # (T,)
    equity_curve: npt.NDArray[np.float64]   # (T,)
    weights: Panel                          # (T, N), đã delay
```

`MarketData` (Phase 0, đã merge) có `dates: Dates`, `assets: Assets`,
`fields: dict[str, Panel]`, `universe: Mask`, `returns: Panel`, `groups: dict[str, ndarray]`,
method `field(name) -> Panel`, `years() -> dict[int, slice]` (slice hàng theo năm dương
lịch, dùng trực tiếp cho per-year Sharpe — không tự suy diễn năm từ `dates` ở
`metrics_local.py`, luôn gọi `data.years()`).

`config/thresholds.py` (Phase 0, đã merge) có các hằng số module-level:
`MAX_DEPTH: int = 7`, `SELF_CORR_MAX: float = 0.70`, `TURNOVER_FLOOR: float = 0.125`,
`WEIGHT_CONCENTRATION_CAP: float = 0.10`, `SHARPE_MIN: float = 1.0`,
`PER_YEAR_SHARPE_MIN: float = 0.0`, `TURNOVER_BAND: tuple[float, float] = (0.01, 0.70)`,
`CALIBRATION_RHO_BAR: float = 0.5`.

---

### Task 4.1: `AlphaMetrics` + `MetricsCalculator.compute` (sharpe/annual_return/turnover/max_drawdown/fitness)

**Files:**
- Create: `src/backtest/metrics_local.py`
- Test: `tests/unit/test_metrics_local.py`

**Interfaces:**
- Consumes: `BacktestResult` (Phase 3, `src/backtest/backtester.py`), `MarketData` (Phase 0,
  `src/data/market_panel.py`), `Panel` (Phase 0, `src/local_types.py`), `TURNOVER_FLOOR`
  (Phase 0, `config/thresholds.py`).
- Produces: `@dataclass(frozen=True, slots=True) class AlphaMetrics` với `sharpe: float`,
  `annual_return: float`, `turnover: float`, `max_drawdown: float`, `fitness: float`,
  `per_year_sharpe: dict[int, float]`, `weight_concentration: float`; `class
  MetricsCalculator` với `PERIODS_PER_YEAR: int = 252` (class attribute) và `def
  compute(self, bt: BacktestResult, data: MarketData) -> AlphaMetrics`.

Công thức (B8 master spec, **đọc `TURNOVER_FLOOR` từ `config/thresholds.py`, KHÔNG
hardcode `0.125`**):

- `sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(252)` trên các giá trị `daily_pnl` hữu
  hạn (loại NaN trước khi tính mean/std); nếu std == 0 hoặc < 2 giá trị hữu hạn → `sharpe =
  0.0` (tránh chia 0 / NaN lan).
- `annual_return = mean(daily_pnl) * 252` (annualized simple return của pnl hàng ngày, theo
  đúng ghi chú B8 "dùng annualized returns, không dùng CAGR" — Gap #7 master spec).
- `turnover = mean_t sum_i |w_t - w_{t-1}|` trên `bt.weights` (T,N); hàng đầu tiên không có
  `w_{t-1}` → bỏ qua khỏi mean (không coi là 0 cũng không NaN-lan); cell ngoài universe ở
  CẢ hai hàng `t` và `t-1` → góp 0 vào sum (không phải NaN) vì `|NaN - NaN|` phải được mask
  trước, không để lan NaN vào turnover toàn cục.
- `max_drawdown = max_t (running_max(equity_curve)[t] - equity_curve[t])` trên
  `bt.equity_curve` (giá trị dương, 0 nếu equity không giảm bao giờ).
- `fitness = sharpe * sqrt(abs(annual_return) / max(turnover, TURNOVER_FLOOR))` — đọc
  `TURNOVER_FLOOR` từ `config/thresholds.py` (import `from config.thresholds import
  TURNOVER_FLOOR`), không hardcode số.
- `weight_concentration` và `per_year_sharpe` đặt là Task 4.2 (cùng file, cùng class, nối
  tiếp ngay sau — không tách commit khỏi Task 4.1 nếu cùng PR review, nhưng test/step viết
  tách bạch để review từng phần rõ).

- [ ] **Step 1: Tạo nhánh từ main sạch**

```bash
git checkout main && git pull --ff-only 2>/dev/null; git checkout -b phase-4-metrics-gates
git status
```
Expected: "On branch phase-4-metrics-gates", working tree clean.

- [ ] **Step 2: Viết test đỏ cho sharpe/annual_return/turnover/max_drawdown/fitness**

```python
# tests/unit/test_metrics_local.py
"""Test AlphaMetrics + MetricsCalculator.compute: sharpe/annual_return/turnover/
max_drawdown/fitness trên BacktestResult biết trước, tính tay đối chiếu."""

from __future__ import annotations

import numpy as np
import pytest

from config.thresholds import TURNOVER_FLOOR
from src.backtest.backtester import BacktestResult
from src.backtest.metrics_local import AlphaMetrics, MetricsCalculator
from src.data.market_panel import MarketData


def _panel_3d_2n() -> MarketData:
    t, n = 3, 2
    dates = (np.datetime64("2021-01-01") + np.arange(t)).astype("datetime64[D]")
    assets = np.array(["A", "B"], dtype=np.str_)
    universe = np.ones((t, n), dtype=bool)
    returns = np.zeros((t, n))
    groups = {"sector": np.zeros((t, n), dtype=np.int64)}
    return MarketData(dates=dates, assets=assets, fields={}, universe=universe,
                      returns=returns, groups=groups)


def test_sharpe_matches_hand_calculation():
    data = _panel_3d_2n()
    daily_pnl = np.array([0.01, -0.005, 0.02])
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=np.zeros((3, 2)))
    m = MetricsCalculator().compute(bt, data)
    expected = daily_pnl.mean() / daily_pnl.std(ddof=0) * np.sqrt(252)
    assert isinstance(m, AlphaMetrics)
    assert np.isclose(m.sharpe, expected)


def test_sharpe_zero_when_std_is_zero():
    data = _panel_3d_2n()
    daily_pnl = np.array([0.01, 0.01, 0.01])
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=np.zeros((3, 2)))
    m = MetricsCalculator().compute(bt, data)
    assert m.sharpe == 0.0


def test_annual_return_is_mean_pnl_times_252():
    data = _panel_3d_2n()
    daily_pnl = np.array([0.001, 0.002, 0.0])
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=np.zeros((3, 2)))
    m = MetricsCalculator().compute(bt, data)
    assert np.isclose(m.annual_return, daily_pnl.mean() * 252)


def test_turnover_matches_hand_calculation():
    data = _panel_3d_2n()
    daily_pnl = np.zeros(3)
    weights = np.array([[0.5, -0.5], [0.3, -0.3], [0.5, -0.5]])
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=weights)
    m = MetricsCalculator().compute(bt, data)
    # |w1-w0| = |0.3-0.5|+|-0.3+0.5| = 0.4 ; |w2-w1| = |0.5-0.3|+|-0.5+0.3| = 0.4
    expected_turnover = np.mean([0.4, 0.4])
    assert np.isclose(m.turnover, expected_turnover)


def test_max_drawdown_matches_hand_calculation():
    data = _panel_3d_2n()
    daily_pnl = np.array([0.10, -0.20, 0.05])  # equity: 0.10, -0.10, -0.05
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=np.zeros((3, 2)))
    m = MetricsCalculator().compute(bt, data)
    # running_max: 0.10, 0.10, 0.10 ; drawdown: 0, 0.20, 0.15 -> max = 0.20
    assert np.isclose(m.max_drawdown, 0.20)


def test_fitness_uses_turnover_floor_from_config_not_hardcoded():
    data = _panel_3d_2n()
    daily_pnl = np.array([0.01, 0.01, 0.01])
    weights = np.zeros((3, 2))  # turnover = 0.0 -> phải dùng floor
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=weights)
    m = MetricsCalculator().compute(bt, data)
    assert m.turnover == 0.0
    expected_fitness = m.sharpe * np.sqrt(abs(m.annual_return) / TURNOVER_FLOOR)
    assert np.isclose(m.fitness, expected_fitness)


def test_alpha_metrics_is_frozen():
    m = AlphaMetrics(sharpe=1.0, annual_return=0.1, turnover=0.1, max_drawdown=0.05,
                     fitness=1.0, per_year_sharpe={}, weight_concentration=0.1)
    with pytest.raises(AttributeError):
        m.sharpe = 2.0  # type: ignore[misc]
```

- [ ] **Step 3: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_metrics_local.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.backtest.metrics_local'`.

- [ ] **Step 4: Tạo `src/backtest/metrics_local.py` (phần Task 4.1 — chưa có per_year_sharpe/weight_concentration thật, đặt placeholder kiểu đúng để Task 4.2 hoàn thiện ngay sau)**

```python
# src/backtest/metrics_local.py
"""AlphaMetrics + MetricsCalculator — đo lường BacktestResult (B8 master spec).

fitness dùng TURNOVER_FLOOR từ config/thresholds.py (Gap #7/R9: ngưỡng chỉ ở MỘT nơi,
không hardcode ở call site). annual_return dùng annualized simple return (KHÔNG CAGR —
đúng sửa Gap #7). per_year_sharpe (Task 4.2) dùng data.years() — regime robustness
first-class, không phải số phụ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.thresholds import TURNOVER_FLOOR
from src.backtest.backtester import BacktestResult
from src.data.market_panel import MarketData


@dataclass(frozen=True, slots=True)
class AlphaMetrics:
    sharpe: float
    annual_return: float
    turnover: float
    max_drawdown: float
    fitness: float
    per_year_sharpe: dict[int, float]
    weight_concentration: float


class MetricsCalculator:
    """Tính AlphaMetrics từ BacktestResult + MarketData. Stateless, an toàn dùng lại."""

    PERIODS_PER_YEAR: int = 252

    def compute(self, bt: BacktestResult, data: MarketData) -> AlphaMetrics:
        sharpe = self._sharpe(bt.daily_pnl)
        annual_return = self._annual_return(bt.daily_pnl)
        turnover = self._turnover(bt.weights)
        max_drawdown = self._max_drawdown(bt.equity_curve)
        fitness = sharpe * np.sqrt(abs(annual_return) / max(turnover, TURNOVER_FLOOR))
        per_year_sharpe = self._per_year_sharpe(bt.daily_pnl, data)
        weight_concentration = self._weight_concentration(bt.weights)
        return AlphaMetrics(
            sharpe=sharpe, annual_return=annual_return, turnover=turnover,
            max_drawdown=max_drawdown, fitness=float(fitness),
            per_year_sharpe=per_year_sharpe, weight_concentration=weight_concentration,
        )

    def _sharpe(self, daily_pnl: np.ndarray) -> float:
        valid = daily_pnl[np.isfinite(daily_pnl)]
        if valid.size < 2:
            return 0.0
        std = valid.std(ddof=0)
        if std == 0.0:
            return 0.0
        return float(valid.mean() / std * np.sqrt(self.PERIODS_PER_YEAR))

    def _annual_return(self, daily_pnl: np.ndarray) -> float:
        valid = daily_pnl[np.isfinite(daily_pnl)]
        if valid.size == 0:
            return 0.0
        return float(valid.mean() * self.PERIODS_PER_YEAR)

    def _turnover(self, weights: np.ndarray) -> float:
        if weights.shape[0] < 2:
            return 0.0
        prev = weights[:-1]
        curr = weights[1:]
        diff = np.abs(curr - prev)
        both_nan = np.isnan(prev) & np.isnan(curr)
        diff = np.where(both_nan, 0.0, diff)
        with np.errstate(invalid="ignore"):
            per_day = np.nansum(diff, axis=1)
        valid_rows = ~np.all(np.isnan(prev) | np.isnan(curr), axis=1)
        if not valid_rows.any():
            return 0.0
        return float(per_day[valid_rows].mean())

    def _max_drawdown(self, equity_curve: np.ndarray) -> float:
        if equity_curve.size == 0:
            return 0.0
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = running_max - equity_curve
        return float(np.max(drawdown))

    def _per_year_sharpe(self, daily_pnl: np.ndarray, data: MarketData) -> dict[int, float]:
        out: dict[int, float] = {}
        for year, sl in data.years().items():
            out[year] = self._sharpe(daily_pnl[sl])
        return out

    def _weight_concentration(self, weights: np.ndarray) -> float:
        if weights.size == 0:
            return 0.0
        gross = np.nansum(np.abs(weights), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            max_abs = np.nanmax(np.abs(weights), axis=1)
            share = np.where(gross > 0, max_abs / gross, 0.0)
        finite_share = share[np.isfinite(share)]
        if finite_share.size == 0:
            return 0.0
        return float(np.max(finite_share))
```

- [ ] **Step 5: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_metrics_local.py -v
```
Expected: PASS (7 test).

- [ ] **Step 6: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/backtest/metrics_local.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/backtest/metrics_local.py
```
Expected: cả hai sạch.

- [ ] **Step 7: Commit**

```bash
git add src/backtest/metrics_local.py tests/unit/test_metrics_local.py
git commit -m "feat(backtest): AlphaMetrics + MetricsCalculator (sharpe/annual_return/turnover/max_dd/fitness)"
```

---

### Task 4.2: per-year Sharpe + weight_concentration — test riêng đối chiếu tay (regime robustness first-class)

**Files:**
- Modify: `src/backtest/metrics_local.py` (đã có `_per_year_sharpe`/`_weight_concentration`
  từ Task 4.1 — task này CHỈ thêm test xác nhận hành vi đúng theo `data.years()` thật và các
  case biên; không sửa lại implementation trừ khi test phát hiện lỗi).
- Modify: `tests/unit/test_metrics_local.py`

**Interfaces:**
- Consumes: `data.years() -> dict[int, slice]` (Phase 0, `src/data/market_panel.py`).
- Produces: xác nhận `AlphaMetrics.per_year_sharpe: dict[int, float]` đúng theo từng năm
  trong `data.years()`, và `AlphaMetrics.weight_concentration: float` đúng là max share của
  1 mã trên gross book tại NGÀY tệ nhất (không phải trung bình toàn kỳ).

- [ ] **Step 1: Viết test đỏ cho per_year_sharpe đa năm + weight_concentration biên**

```python
# tests/unit/test_metrics_local.py — THÊM các hàm test dưới vào cuối file đã tạo ở Task 4.1
import numpy as np

from src.backtest.backtester import BacktestResult
from src.backtest.metrics_local import MetricsCalculator
from src.data.market_panel import MarketData


def _two_year_panel() -> MarketData:
    """4 ngày: 2 ngày năm 2021, 2 ngày năm 2022."""
    t, n = 4, 2
    dates = np.array(
        ["2021-12-30", "2021-12-31", "2022-01-01", "2022-01-02"], dtype="datetime64[D]"
    )
    assets = np.array(["A", "B"], dtype=np.str_)
    universe = np.ones((t, n), dtype=bool)
    returns = np.zeros((t, n))
    groups = {"sector": np.zeros((t, n), dtype=np.int64)}
    return MarketData(dates=dates, assets=assets, fields={}, universe=universe,
                      returns=returns, groups=groups)


def test_per_year_sharpe_splits_by_data_years():
    data = _two_year_panel()
    daily_pnl = np.array([0.01, 0.01, -0.02, -0.02])  # 2021 toàn lãi, 2022 toàn lỗ
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=np.zeros((4, 2)))
    m = MetricsCalculator().compute(bt, data)
    years = data.years()
    assert set(m.per_year_sharpe) == set(years)
    for year, sl in years.items():
        expected = MetricsCalculator()._sharpe(daily_pnl[sl])
        assert np.isclose(m.per_year_sharpe[year], expected)
    # 2021 toàn lãi đều -> std=0 -> sharpe=0.0 theo quy ước; 2022 cũng vậy
    assert m.per_year_sharpe[2021] == 0.0
    assert m.per_year_sharpe[2022] == 0.0


def test_weight_concentration_is_worst_day_max_name_share():
    data = _two_year_panel()
    daily_pnl = np.zeros(4)
    # ngày 0: cân bằng 50/50 ; ngày 2: mã A chiếm 90% book -> concentration phải bắt ngày này
    weights = np.array([
        [0.5, -0.5],
        [0.5, -0.5],
        [0.9, -0.1],
        [0.5, -0.5],
    ])
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=weights)
    m = MetricsCalculator().compute(bt, data)
    assert np.isclose(m.weight_concentration, 0.9)


def test_weight_concentration_zero_when_all_weights_nan():
    data = _two_year_panel()
    daily_pnl = np.zeros(4)
    weights = np.full((4, 2), np.nan)
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=weights)
    m = MetricsCalculator().compute(bt, data)
    assert m.weight_concentration == 0.0
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL nếu implementation Task 4.1 sai biên, hoặc PASS ngay nếu đúng (chạy để xác nhận, không giả định)**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_metrics_local.py -v
```
Expected: nếu implementation Task 4.1 đúng như viết, PASS toàn bộ (10 test cộng dồn). Nếu
`test_weight_concentration_is_worst_day_max_name_share` FAIL, kiểm tra lại
`_weight_concentration` — phải là `max` theo NGÀY (`np.max` trên vector `share` theo `t`),
không phải `mean` theo ngày.

- [ ] **Step 3: Nếu FAIL, sửa `_weight_concentration`/`_per_year_sharpe` trong `src/backtest/metrics_local.py` cho khớp, chạy lại đến PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_metrics_local.py -v
```
Expected: PASS (10 test).

- [ ] **Step 4: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/backtest/metrics_local.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/backtest/metrics_local.py
```
Expected: sạch.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/metrics_local.py tests/unit/test_metrics_local.py
git commit -m "test(backtest): per_year_sharpe đa năm + weight_concentration ngày tệ nhất"
```

---

### Task 4.3: `GateVerdict` + `GateEvaluator` (`src/backtest/gates.py`)

**Files:**
- Create: `src/backtest/gates.py`
- Test: `tests/unit/test_gates.py`

**Interfaces:**
- Consumes: `AlphaMetrics` (Task 4.1, `src/backtest/metrics_local.py`); `MAX_DEPTH`,
  `SELF_CORR_MAX`, `WEIGHT_CONCENTRATION_CAP`, `SHARPE_MIN`, `PER_YEAR_SHARPE_MIN`,
  `TURNOVER_BAND` (Phase 0, `config/thresholds.py`).
- Produces: `@dataclass(frozen=True, slots=True) class GateVerdict` với `passed: bool`,
  `hard_failures: list[str]`, `soft_scores: dict[str, float]`; `class GateEvaluator` với
  `def evaluate(self, m: AlphaMetrics, self_corr: float, depth: int, fields_ok: bool) ->
  GateVerdict`.

**Hard gates** (binary, mỗi cái fail thêm 1 string mô tả vào `hard_failures`; `passed =
len(hard_failures) == 0`):
1. `depth <= MAX_DEPTH` — fail → `"depth {depth} > MAX_DEPTH {MAX_DEPTH}"`.
2. `fields_ok is True` — fail → `"fields_ok=False (field không hợp lệ)"`.
3. `abs(self_corr) < SELF_CORR_MAX` — fail → `"self_corr {self_corr:.3f} >= SELF_CORR_MAX
   {SELF_CORR_MAX}"` (chú ý `<`, KHÔNG `<=` — đúng B8 "self_corr >= ngưỡng → hard fail" nghĩa
   là pass khi strict-less-than).
4. `m.weight_concentration <= WEIGHT_CONCENTRATION_CAP` — fail →
   `"weight_concentration {m.weight_concentration:.3f} > WEIGHT_CONCENTRATION_CAP
   {WEIGHT_CONCENTRATION_CAP}"`.

**Soft scores** (luôn tính, không chặn `passed`, dict trả về để xếp hạng/GP dùng sau —
Phase 7):
- `"sharpe": m.sharpe`.
- `"fitness": m.fitness`.
- `"turnover_band": 1.0` nếu `TURNOVER_BAND[0] <= m.turnover <= TURNOVER_BAND[1]` else
  khoảng cách âm tới biên gần nhất: `-(TURNOVER_BAND[0] - m.turnover)` nếu dưới sàn,
  `-(m.turnover - TURNOVER_BAND[1])` nếu trên trần (giá trị âm = lệch band, 1.0 = trong
  band — soft score càng cao càng tốt, đúng quy ước "tradable in search" của B8).
- `"per_year_min"`: `min(m.per_year_sharpe.values())` nếu `per_year_sharpe` không rỗng, else
  `0.0` — KHÔNG so sánh với `PER_YEAR_SHARPE_MIN` ở đây (đó là điểm thô để GP/ranking dùng
  sau; so sánh ngưỡng cụ thể là quyết định của caller, không phải của `GateEvaluator` vì B8
  liệt kê `per_year-Sharpe min` trong soft scores, không trong hard gates).

Ghi chú quan trọng: `SHARPE_MIN` và `PER_YEAR_SHARPE_MIN` từ `config/thresholds.py` được
**đọc nhưng không dùng để hard-fail** trong `evaluate` — chúng tồn tại cho caller (Task 4.4
`evaluate_local`, hoặc Phase 7 GP) tự quyết định lọc thêm trên `soft_scores`, đúng tách bạch
hard-gate-vs-soft-score B8 ("Soft scores (tradable in search)"). KHÔNG import
`SHARPE_MIN`/`PER_YEAR_SHARPE_MIN` vào `gates.py` nếu không dùng — tránh unused import (ruff
fail). Nếu Task 4.4 cần dùng `SHARPE_MIN` để quyết định pass/fail ở tầng `filter.py`, import
ở đó, không ở `gates.py`.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_gates.py
"""Test GateVerdict + GateEvaluator: hard gates (depth/fields/self_corr/concentration)
tách bạch khỏi soft scores (sharpe/fitness/turnover_band/per_year_min)."""

from __future__ import annotations

from config.thresholds import MAX_DEPTH, SELF_CORR_MAX, TURNOVER_BAND, WEIGHT_CONCENTRATION_CAP
from src.backtest.gates import GateEvaluator, GateVerdict
from src.backtest.metrics_local import AlphaMetrics


def _good_metrics() -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.5, annual_return=0.20, turnover=0.30, max_drawdown=0.10,
        fitness=2.0, per_year_sharpe={2021: 1.2, 2022: 0.8},
        weight_concentration=WEIGHT_CONCENTRATION_CAP / 2,
    )


def test_all_pass_when_within_every_hard_limit():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert isinstance(verdict, GateVerdict)
    assert verdict.passed is True
    assert verdict.hard_failures == []


def test_depth_over_cap_is_hard_failure():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=MAX_DEPTH + 1, fields_ok=True)
    assert verdict.passed is False
    assert any("depth" in f for f in verdict.hard_failures)


def test_fields_not_ok_is_hard_failure():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=False)
    assert verdict.passed is False
    assert any("fields_ok" in f for f in verdict.hard_failures)


def test_self_corr_at_or_above_max_is_hard_failure():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(m, self_corr=SELF_CORR_MAX, depth=3, fields_ok=True)
    assert verdict.passed is False
    assert any("self_corr" in f for f in verdict.hard_failures)


def test_self_corr_just_below_max_passes_that_gate():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(
        m, self_corr=SELF_CORR_MAX - 0.01, depth=3, fields_ok=True
    )
    assert not any("self_corr" in f for f in verdict.hard_failures)


def test_weight_concentration_over_cap_is_hard_failure():
    m = AlphaMetrics(
        sharpe=1.5, annual_return=0.2, turnover=0.3, max_drawdown=0.1, fitness=2.0,
        per_year_sharpe={}, weight_concentration=WEIGHT_CONCENTRATION_CAP + 0.01,
    )
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.passed is False
    assert any("weight_concentration" in f for f in verdict.hard_failures)


def test_multiple_hard_failures_all_recorded():
    m = AlphaMetrics(
        sharpe=0.0, annual_return=0.0, turnover=0.0, max_drawdown=0.0, fitness=0.0,
        per_year_sharpe={}, weight_concentration=1.0,
    )
    verdict = GateEvaluator().evaluate(
        m, self_corr=0.99, depth=MAX_DEPTH + 5, fields_ok=False
    )
    assert verdict.passed is False
    assert len(verdict.hard_failures) == 4  # depth + fields + self_corr + concentration


def test_soft_scores_contain_sharpe_fitness_turnover_band_per_year_min():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.soft_scores["sharpe"] == m.sharpe
    assert verdict.soft_scores["fitness"] == m.fitness
    assert verdict.soft_scores["turnover_band"] == 1.0  # 0.30 trong TURNOVER_BAND
    assert verdict.soft_scores["per_year_min"] == min(m.per_year_sharpe.values())


def test_turnover_band_score_negative_when_below_floor():
    m = AlphaMetrics(
        sharpe=1.0, annual_return=0.1, turnover=TURNOVER_BAND[0] - 0.005, max_drawdown=0.1,
        fitness=1.0, per_year_sharpe={2021: 0.5}, weight_concentration=0.05,
    )
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.soft_scores["turnover_band"] < 0.0


def test_turnover_band_score_negative_when_above_ceiling():
    m = AlphaMetrics(
        sharpe=1.0, annual_return=0.1, turnover=TURNOVER_BAND[1] + 0.05, max_drawdown=0.1,
        fitness=1.0, per_year_sharpe={2021: 0.5}, weight_concentration=0.05,
    )
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.soft_scores["turnover_band"] < 0.0


def test_per_year_min_zero_when_per_year_sharpe_empty():
    m = AlphaMetrics(
        sharpe=1.0, annual_return=0.1, turnover=0.3, max_drawdown=0.1, fitness=1.0,
        per_year_sharpe={}, weight_concentration=0.05,
    )
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.soft_scores["per_year_min"] == 0.0


def test_gate_verdict_is_frozen():
    import pytest
    verdict = GateVerdict(passed=True, hard_failures=[], soft_scores={})
    with pytest.raises(AttributeError):
        verdict.passed = False  # type: ignore[misc]
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gates.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.backtest.gates'`.

- [ ] **Step 3: Tạo `src/backtest/gates.py`**

```python
# src/backtest/gates.py
"""GateVerdict + GateEvaluator — hard gates (chặn) tách bạch khỏi soft scores (xếp hạng).

Ngưỡng CHỈ đọc từ config/thresholds.py (Gap #7/R9 master spec) — không hardcode số ở đây.
Hard gates: depth<=MAX_DEPTH, fields_ok, self_corr<SELF_CORR_MAX (strict), weight_
concentration<=WEIGHT_CONCENTRATION_CAP. Soft scores (B8: "tradable in search", không chặn
passed): sharpe, fitness, turnover-band, per_year_min — caller (filter.evaluate_local,
GP fitness Phase 7) tự quyết định ngưỡng thêm trên các điểm này.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.thresholds import MAX_DEPTH, SELF_CORR_MAX, TURNOVER_BAND, WEIGHT_CONCENTRATION_CAP
from src.backtest.metrics_local import AlphaMetrics


@dataclass(frozen=True, slots=True)
class GateVerdict:
    passed: bool
    hard_failures: list[str] = field(default_factory=list)
    soft_scores: dict[str, float] = field(default_factory=dict)


class GateEvaluator:
    """Đánh giá AlphaMetrics đã tính sẵn (không tự parse/eval/backtest — đó là việc của
    src/backtest/gate.py ở tầng wrapper end-to-end, Task 4.6 sẽ nối hai lớp này)."""

    def evaluate(
        self, m: AlphaMetrics, self_corr: float, depth: int, fields_ok: bool
    ) -> GateVerdict:
        hard_failures: list[str] = []

        if depth > MAX_DEPTH:
            hard_failures.append(f"depth {depth} > MAX_DEPTH {MAX_DEPTH}")
        if not fields_ok:
            hard_failures.append("fields_ok=False (field không hợp lệ)")
        if abs(self_corr) >= SELF_CORR_MAX:
            hard_failures.append(
                f"self_corr {self_corr:.3f} >= SELF_CORR_MAX {SELF_CORR_MAX}"
            )
        if m.weight_concentration > WEIGHT_CONCENTRATION_CAP:
            hard_failures.append(
                f"weight_concentration {m.weight_concentration:.3f} > "
                f"WEIGHT_CONCENTRATION_CAP {WEIGHT_CONCENTRATION_CAP}"
            )

        soft_scores = {
            "sharpe": m.sharpe,
            "fitness": m.fitness,
            "turnover_band": self._turnover_band_score(m.turnover),
            "per_year_min": min(m.per_year_sharpe.values()) if m.per_year_sharpe else 0.0,
        }

        return GateVerdict(
            passed=len(hard_failures) == 0,
            hard_failures=hard_failures,
            soft_scores=soft_scores,
        )

    def _turnover_band_score(self, turnover: float) -> float:
        lo, hi = TURNOVER_BAND
        if lo <= turnover <= hi:
            return 1.0
        if turnover < lo:
            return -(lo - turnover)
        return -(turnover - hi)
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gates.py -v
```
Expected: PASS (13 test).

- [ ] **Step 5: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/backtest/gates.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/backtest/gates.py
```
Expected: cả hai sạch.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/gates.py tests/unit/test_gates.py
git commit -m "feat(backtest): GateVerdict + GateEvaluator (hard gates tách bạch soft scores)"
```

---

### Task 4.4: `evaluate_local` trong `src/scoring/filter.py`

**Files:**
- Modify: `src/scoring/filter.py` (đọc toàn văn trước khi sửa — file hiện có
  `FilterThresholds`, `passes`, `blocking_dimensions` cho filter sim-Brain cũ dùng
  `ScoreVector`/`normalize` từ `src/scoring/metrics.py`; **không sửa/xoá các hàm này**, chỉ
  THÊM hàm mới `evaluate_local` ở cuối file).
- Test: `tests/unit/test_filter_evaluate_local.py`

**Interfaces:**
- Consumes: `AlphaMetrics` (Task 4.1, `src/backtest/metrics_local.py`), `GateEvaluator`,
  `GateVerdict` (Task 4.3, `src/backtest/gates.py`).
- Produces: `def evaluate_local(metrics: AlphaMetrics, self_corr: float, depth: int,
  fields_ok: bool) -> GateVerdict` — wrap mỏng `GateEvaluator().evaluate(...)`, là điểm vào
  duy nhất cho local gate đầy đủ dùng trong loop/CLI (khác `passes`/`blocking_dimensions` —
  hai cái đó dành cho kết quả sim Brain thật qua `ScoreVector`, không liên quan
  `AlphaMetrics` local).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_filter_evaluate_local.py
"""Test evaluate_local trong src/scoring/filter.py: wrap GateEvaluator, không đụng
passes/blocking_dimensions (filter sim-Brain cũ)."""

from __future__ import annotations

from src.backtest.gates import GateVerdict
from src.backtest.metrics_local import AlphaMetrics
from src.scoring.filter import evaluate_local, passes  # noqa: F401  (passes vẫn import được, không bị xoá)


def _passing_metrics() -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.5, annual_return=0.2, turnover=0.3, max_drawdown=0.1, fitness=2.0,
        per_year_sharpe={2021: 1.0}, weight_concentration=0.05,
    )


def test_evaluate_local_returns_gate_verdict():
    verdict = evaluate_local(_passing_metrics(), self_corr=0.1, depth=3, fields_ok=True)
    assert isinstance(verdict, GateVerdict)
    assert verdict.passed is True


def test_evaluate_local_hard_fail_propagates_reason():
    m = _passing_metrics()
    verdict = evaluate_local(m, self_corr=0.99, depth=3, fields_ok=True)
    assert verdict.passed is False
    assert any("self_corr" in r for r in verdict.hard_failures)


def test_legacy_passes_function_still_importable_and_unmodified_signature():
    # đảm bảo evaluate_local KHÔNG phá filter cũ — passes() vẫn nhận (source, thresholds)
    import inspect
    sig = inspect.signature(passes)
    assert list(sig.parameters) == ["source", "thresholds"]
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_filter_evaluate_local.py -v
```
Expected: FAIL `ImportError: cannot import name 'evaluate_local' from 'src.scoring.filter'`.

- [ ] **Step 3: Thêm `evaluate_local` vào cuối `src/scoring/filter.py`**

```python
# THÊM vào cuối src/scoring/filter.py (sau blocking_dimensions, không sửa gì phía trên)

from src.backtest.gates import GateEvaluator, GateVerdict
from src.backtest.metrics_local import AlphaMetrics


def evaluate_local(
    metrics: AlphaMetrics, self_corr: float, depth: int, fields_ok: bool
) -> GateVerdict:
    """Cổng local đầy đủ (Phase 4, B8 master spec) — wrap GateEvaluator cho loop/CLI dùng.

    Khác `passes`/`blocking_dimensions` ở trên: hai hàm đó chấm điểm KẾT QUẢ SIM BRAIN THẬT
    (qua `ScoreVector`/`normalize`); `evaluate_local` chấm điểm `AlphaMetrics` tính LOCAL
    (Phase 3/4, không tốn quota sim) — dùng trước khi quyết định có đáng đốt sim hay không.
    """
    return GateEvaluator().evaluate(metrics, self_corr=self_corr, depth=depth, fields_ok=fields_ok)
```

> Lưu ý import vị trí đầu file: nếu codebase project (ruff/isort) yêu cầu import ở đầu
> module, di chuyển 2 dòng `from src.backtest...` lên khối import đầu file (cùng nhóm với
> `from src.scoring.metrics import normalize` đã có) khi áp dụng patch thật — viết ở đây vào
> cuối file chỉ để rõ ràng đây là bổ sung, không phải patch toàn file. Chạy `ruff check
> --fix` ở Step 5 sẽ tự sắp xếp lại nếu cần.

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_filter_evaluate_local.py -v
```
Expected: PASS (3 test).

- [ ] **Step 5: Chạy lại toàn bộ test cũ của `filter.py` để xác nhận không phá filter sim-Brain**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_filter.py tests/unit/test_filter_evaluate_local.py -v
venv/Scripts/python.exe -m ruff check --fix src/scoring/filter.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/scoring/filter.py
```
Expected: tất cả PASS; ruff/mypy sạch. Nếu `tests/unit/test_filter.py` không tồn tại với tên
đó, tìm test file thật của `passes`/`blocking_dimensions` bằng
`venv/Scripts/python.exe -m pytest --collect-only -q | grep -i filter` và chạy đúng tên tìm
được — không bỏ qua bước xác nhận filter cũ còn xanh.

- [ ] **Step 6: Commit**

```bash
git add src/scoring/filter.py tests/unit/test_filter_evaluate_local.py
git commit -m "feat(scoring): evaluate_local wrap GateEvaluator, giữ nguyên filter sim-Brain cũ"
```

---

### Task 4.5: Integration parse→eval→portfolio→backtest→metrics→gate trên `small_panel`

**Files:**
- Create: `tests/integration/test_metrics_gates.py`

**Interfaces:**
- Consumes: `parse` (Phase 1, `src/lang/parser.py`), `Evaluator`/`EvalContext` (Phase 2,
  `src/engine/evaluator.py`), `PortfolioConfig`/`PortfolioBuilder` (Phase 3,
  `src/backtest/config.py`/`portfolio.py`), `Backtester`/`BacktestResult` (Phase 3,
  `src/backtest/backtester.py`), `MetricsCalculator`/`AlphaMetrics` (Task 4.1),
  `GateEvaluator`/`GateVerdict` (Task 4.3), `evaluate_local` (Task 4.4,
  `src/scoring/filter.py`), fixture `small_panel` (Phase 0, `tests/conftest.py`).
- Produces: không có module mới — 1 test integration chứng minh end-to-end thật, không mock.

> **Trước khi viết test:** chạy lại lệnh pre-condition đầu file plan để xác nhận
> `Backtester`/`BacktestResult` còn đúng chữ ký, và xác nhận thêm
> `from src.lang.parser import parse; from src.engine.evaluator import Evaluator,
> EvalContext; from src.backtest.config import PortfolioConfig; from src.backtest.portfolio
> import PortfolioBuilder` import được — nếu Phase 1/2/3 chưa merge đủ, DỪNG và báo cáo
> block (không tự viết tạm các thành phần này, đúng nguyên tắc Phase 3 plan đã đặt ra).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/integration/test_metrics_gates.py
"""Integration Phase 4: parse -> eval -> portfolio -> backtest -> metrics -> gate,
end-to-end thật trên fixture small_panel (Phase 0), không mock bất cứ thành phần nào."""

from __future__ import annotations

import numpy as np

from src.backtest.backtester import Backtester
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.gates import GateEvaluator, GateVerdict
from src.backtest.metrics_local import AlphaMetrics, MetricsCalculator
from src.backtest.portfolio import PortfolioBuilder
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import parse
from src.lang.visitors import DepthVisitor, FieldCollector
from src.scoring.filter import evaluate_local


def test_handwritten_alpha_end_to_end_metrics_and_gate(small_panel):
    # Biểu thức dùng field có sẵn trong small_panel (xem Phase 0 conftest fields=close/volume);
    # nếu parser/registry báo field không hợp lệ, đổi sang biểu thức tương đương hợp lệ trên
    # field thật của small_panel và ghi rõ trong docstring test (quyết định triển khai tại
    # chỗ, không phải thay đổi hợp đồng — giống ghi chú Task 3.4 Phase 3 plan).
    expr = "rank(ts_mean(close, 5))"
    node = parse(expr)

    depth = DepthVisitor().visit(node) if hasattr(DepthVisitor(), "visit") else DepthVisitor().compute(node)
    fields = FieldCollector().visit(node) if hasattr(FieldCollector(), "visit") else FieldCollector().compute(node)
    fields_ok = fields.issubset(set(small_panel.fields.keys()))

    ctx = EvalContext(data=small_panel, registry=None, cache=None)
    signal = Evaluator(ctx).evaluate(node)
    assert signal.shape == (len(small_panel.dates), len(small_panel.assets))

    cfg = PortfolioConfig(neutralization=Neutralization.SECTOR, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    weights = PortfolioBuilder().build(signal, cfg, small_panel)
    bt = Backtester().run(weights, small_panel)

    metrics = MetricsCalculator().compute(bt, small_panel)
    assert isinstance(metrics, AlphaMetrics)
    assert np.isfinite(metrics.sharpe)
    assert metrics.per_year_sharpe  # small_panel multi-day -> ít nhất 1 năm

    verdict = evaluate_local(metrics, self_corr=0.0, depth=depth, fields_ok=fields_ok)
    assert isinstance(verdict, GateVerdict)
    # Không assert verdict.passed is True cứng — alpha viết tay trên data nhỏ có thể không
    # đạt sharpe/turnover thật; assert ĐÚNG HÀNH VI: verdict luôn có cả hard_failures (list)
    # và soft_scores (dict) đầy đủ 4 khoá, bất kể pass/fail.
    assert isinstance(verdict.hard_failures, list)
    assert set(verdict.soft_scores) == {"sharpe", "fitness", "turnover_band", "per_year_min"}
    print(
        f"[Phase4 demo] sharpe={metrics.sharpe:.3f} fitness={metrics.fitness:.3f} "
        f"turnover={metrics.turnover:.3f} concentration={metrics.weight_concentration:.3f} "
        f"gate_passed={verdict.passed} hard_failures={verdict.hard_failures}"
    )
```

> **Lưu ý API `DepthVisitor`/`FieldCollector`:** Phase 1 plan (1.5/1.6) không xác định tên
> method gọi (`.visit(node)` vs `.compute(node)` vs `node.accept(visitor)`) — kiểm tra
> `src/lang/visitors.py` thật trước khi viết Step 1 thật (không chạy mù theo đoạn code trên
> nguyên văn). Sửa lời gọi cho khớp API thật đã merge (ví dụ nếu visitor dùng
> `NodeVisitor.accept`, gọi `node.accept(DepthVisitor())` thay vì `DepthVisitor().visit(node)`)
> — đây là điều chỉnh tại chỗ bắt buộc, không phải thay đổi hợp đồng của Task 4.5.

- [ ] **Step 2: Chạy test — sửa lời gọi API thật (xem lưu ý trên) đến khi PASS**

```bash
venv/Scripts/python.exe -m pytest tests/integration/test_metrics_gates.py -v -s
```
Expected: PASS (1 test), in ra dòng `[Phase4 demo] sharpe=... fitness=... turnover=...
concentration=... gate_passed=... hard_failures=...` — copy dòng này vào báo cáo Task 4.6.

- [ ] **Step 3: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check tests/integration/test_metrics_gates.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent tests/integration/test_metrics_gates.py
```
Expected: sạch.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_metrics_gates.py
git commit -m "test(backtest): integration parse->eval->portfolio->backtest->metrics->gate"
```

---

### Task 4.6: Wire `score_local_gate` (Phase 3) gọi `MetricsCalculator`+`GateEvaluator` thật + Review + Merge + Push

**Files:**
- Modify: `src/backtest/gate.py` (Phase 3 — đọc toàn văn trước khi sửa; hiện chỉ kiểm "signal
  không toàn-NaN" + "có pnl hữu hạn", đúng như ghi chú để lại từ Phase 3: "Phase 4 sẽ mở
  rộng score_local_gate gọi thêm MetricsCalculator + GateEvaluator khi các thành phần đó tồn
  tại").
- Modify: `tests/unit/test_backtest_gate.py` (test cũ của Phase 3 — thêm test mới, KHÔNG xoá
  test cũ trừ khi hành vi cũ bị thay thế hoàn toàn bởi hành vi mới tương đương).
- Modify (nếu cần): `docs/superpowers/plans/PROGRESS.md` hoặc file journal tương đương của
  repo (kiểm tra tên file journal thật dùng cho các phase trước — đọc cách Phase 0-3 đã ghi
  journal, dùng đúng convention đó, không tạo file journal mới khác tên).

**Interfaces:**
- Consumes: `MetricsCalculator` (Task 4.1), `GateEvaluator`/`GateVerdict` (Task 4.3),
  `Backtester`/`BacktestResult` (Phase 3), `PortfolioBuilder`/`PortfolioConfig` (Phase 3),
  `Evaluator`/`EvalContext`/`parse` (Phase 1/2), `DepthVisitor`/`FieldCollector` (Phase 1).
- Produces: `score_local_gate(expr: str, cfg: PortfolioConfig, data: MarketData) ->
  LocalGateVerdict` (chữ ký GIỮ NGUYÊN như Phase 3 — không phá API loop đang dùng) nhưng nội
  dung `passed` giờ dựa trên `GateEvaluator.evaluate(...).passed` thật (depth/fields/
  self_corr/concentration + ngưỡng sharpe tối thiểu), không còn chỉ "có pnl hữu hạn".

- [ ] **Step 1: Đọc toàn văn `src/backtest/gate.py` hiện tại (Phase 3) để xác nhận điểm sửa chính xác**

```bash
venv/Scripts/python.exe -c "import inspect, src.backtest.gate as g; print(inspect.getsource(g))"
```
Ghi lại dòng chính xác của hàm `score_local_gate` trong file thật (số dòng có thể khác bản
nháp Phase 3 plan nếu đã chỉnh sửa trong review) trước khi viết patch ở Step 3.

- [ ] **Step 2: Viết test đỏ — `score_local_gate` phải fail khi self_corr cao (hành vi MỚI Phase 4, Phase 3 cũ không kiểm self_corr)**

```python
# THÊM vào tests/unit/test_backtest_gate.py (giữ nguyên 3 test cũ của Phase 3 ở trên)
from src.backtest.config import PortfolioConfig
from src.backtest.gate import score_local_gate


def test_score_local_gate_fails_when_self_corr_too_high(small_panel, monkeypatch):
    # self_corr cao phải chặn pass dù expr hợp lệ và sinh pnl được — hành vi Phase 4 MỚI,
    # Phase 3 cũ KHÔNG có tham số self_corr nên test này xác nhận chữ ký đã mở rộng.
    verdict = score_local_gate(
        "close", PortfolioConfig(delay=1), small_panel, self_corr=0.99,
    )
    assert verdict.passed is False
    assert "self_corr" in verdict.reason.lower()


def test_score_local_gate_passes_with_low_self_corr_and_valid_expression(small_panel):
    verdict = score_local_gate(
        "close", PortfolioConfig(delay=1), small_panel, self_corr=0.0,
    )
    # Không assert cứng passed=True (sharpe trên data thật có thể thấp) — assert reason
    # KHÔNG còn là lý do tối thiểu cũ ("no_pnl"/"signal toàn NaN") khi expr hợp lệ.
    assert verdict.reason not in {"signal toàn NaN — không có giá trị dùng được"}
```

- [ ] **Step 3: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_backtest_gate.py -v
```
Expected: FAIL — `score_local_gate() got an unexpected keyword argument 'self_corr'` (chữ ký
cũ Phase 3 chưa có tham số này).

- [ ] **Step 4: Sửa `src/backtest/gate.py` — thêm tham số `self_corr`, gọi `MetricsCalculator` + `GateEvaluator` thật**

```python
# src/backtest/gate.py — patch hàm score_local_gate (giữ nguyên import parse/Evaluator/
# EvalContext/Backtester/PortfolioBuilder/PortfolioConfig/MarketData đã có từ Phase 3;
# THÊM import mới ở đầu file)

from src.backtest.gates import GateEvaluator
from src.backtest.metrics_local import MetricsCalculator
from src.lang.visitors import DepthVisitor, FieldCollector  # API thật: kiểm tra Step 1 trước khi dùng


def score_local_gate(
    expr: str, cfg: PortfolioConfig, data: MarketData, self_corr: float = 0.0,
) -> LocalGateVerdict:
    try:
        node = parse(expr)
    except ParseError as exc:
        return LocalGateVerdict(False, f"parse lỗi: {exc}")

    fields = FieldCollector().visit(node)  # CHỈNH theo API thật xác nhận ở Task 4.5/Step 1
    fields_ok = fields.issubset(set(data.fields.keys()))
    depth = DepthVisitor().visit(node)  # CHỈNH theo API thật xác nhận ở Task 4.5/Step 1

    ctx = EvalContext(data=data, registry=None, cache=None)
    try:
        signal = Evaluator(ctx).evaluate(node)
    except (KeyError, ValueError) as exc:
        return LocalGateVerdict(False, f"eval lỗi: {exc}")

    if np.all(np.isnan(signal)):
        return LocalGateVerdict(False, "signal toàn NaN — không có giá trị dùng được")

    weights = PortfolioBuilder().build(signal, cfg, data)
    result = Backtester().run(weights, data)
    if not np.isfinite(result.daily_pnl).any():
        return LocalGateVerdict(False, "không sinh được pnl hữu hạn")

    metrics = MetricsCalculator().compute(result, data)
    verdict = GateEvaluator().evaluate(
        metrics, self_corr=self_corr, depth=depth, fields_ok=fields_ok,
    )
    if not verdict.passed:
        return LocalGateVerdict(False, f"gate hard fail: {'; '.join(verdict.hard_failures)}")
    return LocalGateVerdict(True, "ok")
```

> **Quan trọng:** xác nhận tên method `FieldCollector`/`DepthVisitor` thật (Step 1 của Task
> 4.5 đã ghi chú lưu ý này) — nếu khác `.visit(node)`, sửa cả hai lời gọi ở đây cho khớp.
> Không để code mẫu nguyên văn nếu API thật khác.

- [ ] **Step 5: Chạy test — PASS (toàn bộ `test_backtest_gate.py`, kể cả 3 test cũ Phase 3)**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_backtest_gate.py -v
```
Expected: PASS (5 test: 3 cũ Phase 3 + 2 mới Phase 4).

- [ ] **Step 6: Chạy lại toàn bộ test loop Phase 3 (`test_loop_local_gate.py`) để xác nhận không phá hợp đồng `local_gate_fn`**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_loop_local_gate.py -v
```
Expected: PASS. Nếu `RefinementLoop` gọi `local_gate_fn(expr, cfg, data)` (3 vị trí, không
truyền `self_corr`), `score_local_gate` với `self_corr: float = 0.0` mặc định vẫn tương
thích ngược (gate self_corr coi như 0.0 khi loop chưa wire `PoolCorrelation` — đúng, vì
`PoolCorrelation` chỉ xuất hiện ở Phase 6; ghi chú rõ điều này trong code Step 4 nếu chưa có).

- [ ] **Step 7: ruff + mypy toàn bộ `src/backtest/`**

```bash
venv/Scripts/python.exe -m ruff check src/backtest/ src/scoring/filter.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/backtest/ src/scoring/filter.py
```
Expected: cả hai sạch trên toàn package.

- [ ] **Step 8: Chạy full test suite của nhánh để xác nhận xanh toàn bộ trước khi merge**

```bash
venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: PASS toàn bộ (không có test nào của Phase 0-3 bị phá bởi thay đổi Phase 4).

- [ ] **Step 9: Commit**

```bash
git add src/backtest/gate.py tests/unit/test_backtest_gate.py
git commit -m "feat(backtest): score_local_gate dùng MetricsCalculator+GateEvaluator thật (nối Phase 3->4)"
```

- [ ] **Step 10: Review cuối — tổng hợp kết quả cho user**

In lại dòng `[Phase4 demo] sharpe=... fitness=... turnover=... concentration=...
gate_passed=...` từ Task 4.5/Step 2, kết quả Step 8 (full suite), và danh sách file mới/sửa
(`src/backtest/metrics_local.py`, `src/backtest/gates.py`, `src/backtest/gate.py` (sửa),
`src/scoring/filter.py` (thêm `evaluate_local`), 4 test file mới + 1 test file Phase 3 mở
rộng) — ghi vào journal phase (đúng convention `PROGRESS.md`/tương đương đã xác nhận ở Step
1 mục Files).

- [ ] **Step 11: Merge vào main + push**

```bash
git checkout main
git pull --ff-only
git merge --no-ff phase-4-metrics-gates -m "Merge phase-4-metrics-gates: AlphaMetrics + GateEvaluator + evaluate_local"
venv/Scripts/python.exe -m pytest tests/ -v
git push origin main
```
Expected: merge sạch (không conflict), full suite PASS trên `main` sau merge, push thành
công.

---

## Self-review (spec coverage)

- `AlphaMetrics`/`MetricsCalculator` (sharpe/annual_return/turnover/max_dd/fitness) — B8 →
  Task 4.1. ✔
- `fitness` dùng `max(turnover, TURNOVER_FLOOR)` đọc từ `config/thresholds.py` (không
  hardcode `0.125`) — Correctness brief → Task 4.1 (`from config.thresholds import
  TURNOVER_FLOOR`). ✔
- per-year Sharpe first-class (dùng `data.years()`, không phải số phụ) — Correctness brief
  + B8 → Task 4.2. ✔
- `weight_concentration` = max single-name |weight| share tại ngày tệ nhất — B8 → Task 4.1 +
  Task 4.2 (test biên `test_weight_concentration_is_worst_day_max_name_share`). ✔
- `GateVerdict`/`GateEvaluator` với hard gates (depth/fields/self_corr/concentration) tách
  bạch soft scores (sharpe/fitness/turnover-band/per_year_min) — B8 → Task 4.3. ✔
- Ngưỡng chỉ từ `config/thresholds.py`, không hardcode ở call site — Global Constraints +
  Correctness brief → Task 4.1 (`TURNOVER_FLOOR`), Task 4.3 (`MAX_DEPTH`, `SELF_CORR_MAX`,
  `WEIGHT_CONCENTRATION_CAP`, `TURNOVER_BAND`). ✔
- `src/scoring/filter.py` thêm `evaluate_local`, KHÔNG phá filter sim-Brain cũ (`passes`,
  `blocking_dimensions`) — Phạm vi brief → Task 4.4 (test xác nhận `passes` còn nguyên chữ
  ký, file cũ chạy xanh). ✔
- Integration end-to-end parse→eval→portfolio→backtest→metrics→gate trên `small_panel` —
  Phạm vi brief (B8 + master plan 4.5) → Task 4.5. ✔
- Nối `score_local_gate` (Phase 3, gate tối thiểu) sang dùng `MetricsCalculator`+
  `GateEvaluator` thật — ghi chú để lại từ Phase 3 plan ("Phase 4 sẽ mở rộng score_local_gate
  ... khi các thành phần đó tồn tại") → Task 4.6. ✔ (Không có trong yêu cầu liệt kê tường
  minh của brief Phase 4 gốc, nhưng bắt buộc để Phase 4 thực sự "consumed" bởi loop — nếu
  không nối, `AlphaMetrics`/`GateEvaluator` mới chỉ là code chết không ai gọi trong pipeline
  thật. Quyết định bổ sung Task 4.6 vào plan để giữ tính "Produces ... cho P4.5/P6/P7" của
  master plan có nghĩa — P4.5 calibration cần `score_local_gate` đã dùng `MetricsCalculator`
  thật để re-score so sánh Brain, không phải gate tối thiểu Phase 3.)
- TDD mỗi task: test đỏ (code thật, không placeholder) → FAIL → impl thật → PASS → commit —
  toàn bộ 6 task đều theo cấu trúc này. ✔
- mypy --strict / ruff clean mỗi task — Step riêng trong từng task + Step 7/8 tổng ở Task
  4.6. ✔
- Python 3.12 type hints, tiếng Việt giữ dấu trong docstring/comment — áp dụng xuyên suốt
  code mẫu mọi task. ✔

**Placeholder scan:** không có "TBD"/"implement later"/"add appropriate" trong các bước
code — mọi step code có nội dung đầy đủ chạy được. Hai điểm "kiểm tra API thật trước khi
dùng" (`DepthVisitor`/`FieldCollector` method name ở Task 4.5/4.6) là **rủi ro tích hợp có
thật** (Phase 1 plan không cố định tên method `.visit` vs `.accept`), không phải placeholder
che giấu thiếu thiết kế — đã ghi rõ hành động cụ thể (đọc `src/lang/visitors.py` thật, sửa
lời gọi) thay vì để mơ hồ.

**Type consistency:** `AlphaMetrics` (Task 4.1) → dùng nguyên trong `GateEvaluator.evaluate`
(Task 4.3), `evaluate_local` (Task 4.4), integration (Task 4.5) — cùng field name xuyên suốt
(`sharpe`, `annual_return`, `turnover`, `max_drawdown`, `fitness`, `per_year_sharpe`,
`weight_concentration`). `GateVerdict` (Task 4.3) → dùng nguyên trong `evaluate_local` (Task
4.4) và integration (Task 4.5) — field name `passed`/`hard_failures`/`soft_scores` nhất
quán. `score_local_gate` (Task 4.6) giữ chữ ký cũ Phase 3 cộng `self_corr: float = 0.0` —
không đổi tên tham số đã có (`expr`, `cfg`, `data`), tương thích ngược với
`RefinementLoop._evaluate` đang gọi 3 vị trí.
