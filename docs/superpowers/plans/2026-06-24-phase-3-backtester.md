# Phase 3 — Backtester (MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (khuyến nghị) hoặc superpowers:executing-plans để thực thi từng task. Mỗi step dùng
> checkbox (`- [ ]`); chạy tuần tự — Task 3.1→3.2→3.3 phụ thuộc chuỗi (config → builder →
> backtester), Task 3.4 (integration MVP) phụ thuộc cả ba, Task 3.5 (gỡ đường cũ) độc lập về
> mã nguồn nhưng nên làm SAU 3.1–3.4 vì `score_local_gate` cần `PortfolioBuilder`/
> `Backtester` thật để tính được gì đó (không phải stub). Task 3.6 là review cuối, luôn cuối
> cùng.

**Goal:** Dựng tầng backtest tối thiểu của MiniBrain — `PortfolioConfig` (stage-separation
config), `PortfolioBuilder` (signal → weights qua decay/neutralize/truncate/scale/delay),
`Backtester` (weights → daily PnL/equity, delay-1) — và chứng minh end-to-end trên dữ liệu
thật (`small_panel`): 1 alpha viết tay → parse → eval → build → backtest → đọc được equity
curve + Sharpe sơ bộ. Đây là **MVP milestone của toàn dự án MiniBrain** (Part E, master
spec) — dừng, demo, review trước khi sang Phase 4. Đồng thời gỡ đường cũ (D9): biến
`score_local_gate(expr, cfg)` thành cổng **bắt buộc** trong `RefinementLoop._evaluate`
trước mọi `simulator.simulate(...)`, để local hard-fail không còn đốt sim Brain.

**Architecture:** `src/backtest/{config,portfolio,backtester}.py` mới, **không** import
`src/gp`, `src/storage`, `src/llm` (dependency rule B1/master plan) — chỉ phụ thuộc
`src/local_types.py` và `src/data/market_panel.py` (đã có từ Phase 0). `score_local_gate`
đặt tại `src/backtest/gate.py` (không phải `src/pipeline/` — package đó chưa tồn tại và sẽ
chỉ xuất hiện ở Phase 8; đặt trong `src/backtest` giữ nó cùng tầng với
config/portfolio/backtester mà nó điều phối, và tránh tạo thêm package mới chỉ cho 1 hàm).
`score_local_gate` là điểm **duy nhất** mà `src/llm/loop.py` được phép import từ tầng
MiniBrain — giữ hướng phụ thuộc một chiều (`llm` → `backtest`, không ngược lại).

**Tech Stack:** Python 3.12, numpy, pytest, ruff, mypy --strict. Không thêm dependency mới
(Phase 3 không cần `lark`/`pyarrow` trực tiếp — đã có từ Phase 0/1).

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

Phase 3 **consume** trực tiếp `Evaluator`/`parse` từ Phase 1+2 (`src/lang/parser.py`,
`src/engine/evaluator.py`, `src/lang/registry.py`, `src/operators_local/*`). Tại thời điểm
viết plan này, `src/lang/` đang được dựng (Phase 1 plan tồn tại,
`docs/superpowers/plans/2026-06-24-phase-1-parser.md`) nhưng `src/engine/` và
`src/operators_local/` **chưa tồn tại** trong repo. Người thực thi Task 3.4 (Integration
MVP) **phải kiểm tra trước** bằng:

```bash
venv/Scripts/python.exe -c "from src.lang.parser import parse; from src.engine.evaluator import Evaluator, EvalContext; print('ok')"
```

- Nếu lệnh trên chạy `ok` → Phase 1+2 đã xong, tiếp tục Task 3.4 như viết dưới.
- Nếu lỗi `ModuleNotFoundError` → Phase 1 và/hoặc Phase 2 chưa merge vào `main`. Task 3.1–3.3
  (config/portfolio/backtester) **không phụ thuộc** Evaluator nên vẫn làm được độc lập (chỉ
  cần `MarketData`, numpy thuần) — chỉ Task 3.4 (integration thật) và Task 3.5 (gate dùng
  trong loop) bị khoá. Trong trường hợp này: hoàn thành 3.1–3.3, dừng lại, báo cáo block,
  đợi Phase 1+2 merge rồi tiếp 3.4–3.6. KHÔNG viết Evaluator/operators tạm trong Phase 3 —
  đó là lấn phạm vi phase khác.

---

### Task 3.1: `PortfolioConfig` (`src/backtest/config.py`)

**Files:**
- Create: `src/backtest/__init__.py`
- Create: `src/backtest/config.py`
- Test: `tests/unit/test_backtest_config.py`

**Interfaces:**
- Consumes: không (module độc lập, chỉ stdlib `enum`/`dataclasses`).
- Produces: `class Neutralization(Enum)` với 5 giá trị `NONE, MARKET, SECTOR, INDUSTRY,
  SUBINDUSTRY`; `@dataclass(frozen=True, slots=True) class PortfolioConfig` với field
  `neutralization: Neutralization = Neutralization.SECTOR`, `decay: int = 0`,
  `truncation: float = 0.10`, `scale_book: float = 1.0`, `delay: int = 1`.

- [ ] **Step 1: Tạo nhánh từ main sạch**

```bash
git checkout main && git pull --ff-only 2>/dev/null; git checkout -b phase-3-backtester
git status
```
Expected: "On branch phase-3-backtester", working tree clean.

- [ ] **Step 2: Viết test đỏ**

```python
# tests/unit/test_backtest_config.py
"""Test PortfolioConfig: stage-separation config (neut/decay/trunc/scale/delay)."""

from __future__ import annotations

import pytest

from src.backtest.config import Neutralization, PortfolioConfig


def test_neutralization_has_five_members():
    names = {m.name for m in Neutralization}
    assert names == {"NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"}


def test_default_config_matches_master_spec():
    cfg = PortfolioConfig()
    assert cfg.neutralization is Neutralization.SECTOR
    assert cfg.decay == 0
    assert cfg.truncation == pytest.approx(0.10)
    assert cfg.scale_book == pytest.approx(1.0)
    assert cfg.delay == 1


def test_config_is_frozen():
    cfg = PortfolioConfig()
    with pytest.raises(AttributeError):
        cfg.decay = 5  # type: ignore[misc]


def test_config_is_hashable_for_cache_key():
    cfg1 = PortfolioConfig(decay=10)
    cfg2 = PortfolioConfig(decay=10)
    assert hash(cfg1) == hash(cfg2)
    assert cfg1 == cfg2


def test_config_custom_values():
    cfg = PortfolioConfig(
        neutralization=Neutralization.MARKET, decay=5, truncation=0.05,
        scale_book=2.0, delay=2,
    )
    assert cfg.neutralization is Neutralization.MARKET
    assert cfg.decay == 5
    assert cfg.truncation == pytest.approx(0.05)
    assert cfg.scale_book == pytest.approx(2.0)
    assert cfg.delay == 2
```

- [ ] **Step 3: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_backtest_config.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.backtest'`.

- [ ] **Step 4: Tạo `src/backtest/__init__.py` + `src/backtest/config.py`**

```python
# src/backtest/__init__.py
"""Tầng backtest local MiniBrain: config/portfolio/backtester (+ metrics/pool_corr ở
Phase 4/6). Dependency rule (master plan B1): src/backtest KHÔNG import src/gp,
src/storage, src/llm. `gate.py` (Task 3.5) là điểm duy nhất src/llm được phép gọi vào.
"""
```

```python
# src/backtest/config.py
"""PortfolioConfig — tách "cấu hình" khỏi "tín hiệu" (stage separation, Gap #8 master spec).

Expression (AST từ src/lang) chỉ là signal core; mọi WQ "settings" (neutralization, decay,
truncation, scale, delay) sống ở đây và được áp ở tầng portfolio (Task 3.2), KHÔNG trong
cây AST. Lý do: search GP (Phase 7) tìm core trong ngân sách độ sâu ≈7 — trộn config vào
cây AST lãng phí depth và làm nhiễu attribution (alpha tốt vì core hay vì config?).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Neutralization(Enum):
    """Cách trừ trung bình cross-sectional khi build weights (Task 3.2)."""

    NONE = auto()
    MARKET = auto()
    SECTOR = auto()
    INDUSTRY = auto()
    SUBINDUSTRY = auto()


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    """Toàn bộ tham số "config stage" của một alpha — tách khỏi expression.

    `decay`: window ts_decay_linear áp lên signal trước neutralize; 0 = tắt (signal nhanh,
    turnover là alpha, không nên decay). `truncation`: cap |w_i| theo tỉ lệ book (gate tập
    trung). `scale_book`: tổng |w| sau scale (1.0 = dollar-neutral chuẩn long-short).
    `delay`: weight tại t áp cho return tại t+delay (delay-1 mặc định, đúng convention WQ
    Delay-1)."""

    neutralization: Neutralization = Neutralization.SECTOR
    decay: int = 0
    truncation: float = 0.10
    scale_book: float = 1.0
    delay: int = 1
```

- [ ] **Step 5: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_backtest_config.py -v
```
Expected: PASS (5 test).

- [ ] **Step 6: ruff + mypy nhanh trên file mới**

```bash
venv/Scripts/python.exe -m ruff check src/backtest/
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/backtest/config.py
```
Expected: cả hai sạch (no output / "Success").

- [ ] **Step 7: Commit**

```bash
git add src/backtest/__init__.py src/backtest/config.py tests/unit/test_backtest_config.py
git commit -m "feat(backtest): PortfolioConfig + Neutralization (stage separation)"
```

---

### Task 3.2: `PortfolioBuilder` (`src/backtest/portfolio.py`)

**Files:**
- Create: `src/backtest/portfolio.py`
- Test: `tests/unit/test_backtest_portfolio.py`

**Interfaces:**
- Consumes: `PortfolioConfig`/`Neutralization` (3.1), `MarketData` (Phase 0,
  `src/data/market_panel.py`), `Panel`/`Mask` (Phase 0, `src/local_types.py`).
- Produces: `class PortfolioBuilder` với `build(self, signal: Panel, cfg: PortfolioConfig,
  data: MarketData) -> Panel` trả weights `(T, N)`.

Thứ tự pipeline bên trong `build` (đúng B7 master spec, **không đảo**):

1. **decay** (tuỳ chọn, bỏ qua nếu `cfg.decay == 0`): trung bình trọng số tuyến tính giảm
   dần trên trailing `cfg.decay` ngày của `signal` (decay_linear đơn giản, không cần
   `ts_decay_linear` operator của Phase 2 — đây là phép decay tại tầng config, độc lập
   khỏi registry operator).
2. **neutralize**: trừ trung bình cross-sectional theo từng ngày, chỉ trên cell in-universe.
   `NONE` → giữ nguyên; `MARKET` → trừ mean toàn universe; `SECTOR/INDUSTRY/SUBINDUSTRY` →
   trừ mean theo group tương ứng đọc từ `data.groups["sector"]` (Phase 0 fixture chỉ có
   `"sector"` — `INDUSTRY`/`SUBINDUSTRY` dùng cùng cơ chế group nhưng sẽ NaN/lỗi rõ nếu
   `data.groups` thiếu key tương ứng; Task 3.2 chỉ cần hỗ trợ đúng key tồn tại, raise
   `KeyError` rõ ràng cho key thiếu — không silently fallback).
3. **truncate**: cap `|w_i|` tại `cfg.truncation` (tỉ lệ trên tổng book tại thời điểm đó),
   rồi renormalize lại tổng `|w|` về giá trị trước truncate (truncate không được thay đổi
   tổng exposure, chỉ phân bổ lại).
4. **scale**: `w /= nansum(|w|)` theo từng ngày (an toàn chia 0 → ngày đó toàn NaN), rồi
   `w *= cfg.scale_book` → dollar-neutral (long+short cân bằng vì bước neutralize đã trừ
   mean trước khi tới đây).
5. **delay**: dịch toàn bộ ma trận weights xuống `cfg.delay` dòng (`w_delayed[t] =
   w[t-cfg.delay]`; `cfg.delay` dòng đầu = NaN — chưa có weight nào áp được).

Chỉ tính trên cell `data.universe == True`; cell ngoài universe luôn NaN ở mọi bước.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_backtest_portfolio.py
"""Test PortfolioBuilder.build: decay -> neutralize -> truncate -> scale -> delay."""

from __future__ import annotations

import numpy as np
import pytest

from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData


def _tiny_panel() -> MarketData:
    """4 ngày x 4 mã, universe đủ, 2 sector {0,0,1,1} — dễ tính tay."""
    t, n = 4, 4
    dates = (np.datetime64("2021-01-01") + np.arange(t)).astype("datetime64[D]")
    assets = np.array(["A", "B", "C", "D"], dtype=np.str_)
    universe = np.ones((t, n), dtype=bool)
    returns = np.full((t, n), 0.01)
    sector = np.tile(np.array([0, 0, 1, 1]), (t, 1)).astype(np.int64)
    return MarketData(
        dates=dates, assets=assets, fields={}, universe=universe,
        returns=returns, groups={"sector": sector},
    )


def test_neutralize_none_keeps_signal_then_scale_dollar_neutral():
    data = _tiny_panel()
    signal = np.array([[1.0, -1.0, 2.0, -2.0]] * 4)
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, truncation=1.0, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data)
    # scale: w /= sum(|w|) -> [1,-1,2,-2]/6 ; sum(|w|) per day == scale_book(1.0)
    row0 = w[0]
    assert np.isclose(np.nansum(np.abs(row0)), 1.0)
    assert np.isclose(row0[0] / row0[2], 0.5)  # tỉ lệ tương đối giữ nguyên


def test_neutralize_market_demeans_cross_sectionally():
    data = _tiny_panel()
    signal = np.array([[1.0, 2.0, 3.0, 4.0]] * 4)
    cfg = PortfolioConfig(neutralization=Neutralization.MARKET, truncation=1.0, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data)
    # sau demean mean=0 -> cross-section sum trước scale = 0 -> vẫn 0 sau scale (chia hằng số)
    pre_scale_proxy = w[0] * np.nansum(np.abs(w[0]))  # phục hồi tỉ lệ tương đối
    assert np.isclose(np.nansum(pre_scale_proxy), 0.0, atol=1e-9)


def test_neutralize_sector_demeans_within_group():
    data = _tiny_panel()
    signal = np.array([[1.0, 3.0, 10.0, 20.0]] * 4)  # sector0={1,3} sector1={10,20}
    cfg = PortfolioConfig(neutralization=Neutralization.SECTOR, truncation=1.0, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data)
    ratio = w[0]
    # sau demean trong sector: sector0 -> [-1,1]; sector1 -> [-5,5] -> tỉ lệ A:B = -1:1
    assert np.isclose(ratio[0] / ratio[1], -1.0)
    assert np.isclose(ratio[2] / ratio[3], -1.0)


def test_truncate_caps_per_name_weight_and_renormalizes():
    data = _tiny_panel()
    # 1 mã áp đảo -> sau neutralize(NONE)+scale, truncate phải cap rồi renorm giữ tổng
    signal = np.array([[100.0, -1.0, 1.0, -100.0]] * 4)
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, truncation=0.10, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data)
    assert np.all(np.abs(w[0]) <= 0.10 + 1e-9)
    assert np.isclose(np.nansum(np.abs(w[0])), 1.0)  # renormalize giữ scale_book=1.0


def test_scale_book_sets_total_gross_exposure():
    data = _tiny_panel()
    signal = np.array([[1.0, -1.0, 2.0, -2.0]] * 4)
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, truncation=1.0,
                          scale_book=2.0, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data)
    assert np.isclose(np.nansum(np.abs(w[0])), 2.0)


def test_delay_shifts_weights_down_by_delay_rows():
    data = _tiny_panel()
    signal = np.array([[1.0, -1.0, 2.0, -2.0]] * 4)
    cfg_no_delay = PortfolioConfig(neutralization=Neutralization.NONE, truncation=1.0, delay=0)
    cfg_delay1 = PortfolioConfig(neutralization=Neutralization.NONE, truncation=1.0, delay=1)
    w0 = PortfolioBuilder().build(signal, cfg_no_delay, data)
    w1 = PortfolioBuilder().build(signal, cfg_delay1, data)
    assert np.all(np.isnan(w1[0]))  # ngày đầu chưa có weight để delay vào
    np.testing.assert_allclose(w1[1], w0[0])
    np.testing.assert_allclose(w1[2], w0[1])


def test_out_of_universe_cells_stay_nan():
    data = _tiny_panel()
    universe = data.universe.copy()
    universe[:, 3] = False  # mã D ngoài universe toàn bộ
    data2 = MarketData(dates=data.dates, assets=data.assets, fields=data.fields,
                        universe=universe, returns=data.returns, groups=data.groups)
    signal = np.array([[1.0, -1.0, 2.0, -2.0]] * 4)
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, truncation=1.0, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data2)
    assert np.all(np.isnan(w[:, 3]))


def test_unknown_group_key_raises_keyerror():
    data = _tiny_panel()  # chỉ có groups["sector"], không có "industry"
    signal = np.ones((4, 4))
    cfg = PortfolioConfig(neutralization=Neutralization.INDUSTRY, delay=0)
    with pytest.raises(KeyError):
        PortfolioBuilder().build(signal, cfg, data)
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_backtest_portfolio.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.backtest.portfolio'`.

- [ ] **Step 3: Tạo `src/backtest/portfolio.py`**

```python
# src/backtest/portfolio.py
"""PortfolioBuilder: signal (T,N) -> weights (T,N) qua 5 bước cấu hình (B7 master spec).

Thứ tự CỐ ĐỊNH: decay -> neutralize -> truncate -> scale -> delay. Mỗi bước chỉ tác động
cell in-universe; cell ngoài universe luôn NaN xuyên suốt (no-survivorship, B3 invariant).
"""

from __future__ import annotations

import numpy as np

from src.backtest.config import Neutralization, PortfolioConfig
from src.data.market_panel import MarketData
from src.local_types import Panel

_GROUP_KEY = {
    Neutralization.SECTOR: "sector",
    Neutralization.INDUSTRY: "industry",
    Neutralization.SUBINDUSTRY: "subindustry",
}


class PortfolioBuilder:
    """Áp `PortfolioConfig` lên một signal thô để ra weights tradable."""

    def build(self, signal: Panel, cfg: PortfolioConfig, data: MarketData) -> Panel:
        masked = np.where(data.universe, signal, np.nan)
        decayed = self._decay(masked, cfg.decay)
        neutralized = self._neutralize(decayed, cfg.neutralization, data)
        truncated = self._truncate(neutralized, cfg.truncation)
        scaled = self._scale(truncated, cfg.scale_book)
        return self._delay(scaled, cfg.delay)

    def _decay(self, signal: Panel, window: int) -> Panel:
        """Trung bình trọng số tuyến tính giảm dần trên trailing `window` ngày.
        window<=0 hoặc 1 -> không đổi (decay tắt)."""
        if window <= 1:
            return signal
        t = signal.shape[0]
        out = np.full_like(signal, np.nan)
        weights = np.arange(1, window + 1, dtype=np.float64)  # xa nhất=1 ... gần nhất=window
        for row in range(t):
            lo = max(0, row - window + 1)
            chunk = signal[lo : row + 1]
            w = weights[-(chunk.shape[0]) :]
            with np.errstate(invalid="ignore"):
                num = np.nansum(chunk * w[:, None], axis=0)
                valid_mask = ~np.isnan(chunk)
                denom = np.where(valid_mask, w[:, None], 0.0).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[row] = np.where(denom > 0, num / denom, np.nan)
        return out

    def _neutralize(self, signal: Panel, kind: Neutralization, data: MarketData) -> Panel:
        if kind is Neutralization.NONE:
            return signal
        if kind is Neutralization.MARKET:
            row_mean = np.nanmean(signal, axis=1, keepdims=True)
            return signal - row_mean
        group_key = _GROUP_KEY[kind]
        groups = data.groups[group_key]  # raise KeyError nếu thiếu — đúng hợp đồng
        out = np.full_like(signal, np.nan)
        t = signal.shape[0]
        for row in range(t):
            row_signal = signal[row]
            row_groups = groups[row]
            for g in np.unique(row_groups):
                idx = row_groups == g
                vals = row_signal[idx]
                if np.all(np.isnan(vals)):
                    continue
                gmean = np.nanmean(vals)
                out[row, idx] = vals - gmean
        return out

    def _truncate(self, signal: Panel, cap: float) -> Panel:
        if cap <= 0:
            return signal
        gross = np.nansum(np.abs(signal), axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cap_abs = np.where(gross > 0, cap * gross, np.inf)
        capped = np.clip(signal, -cap_abs, cap_abs)
        capped = np.where(np.isnan(signal), np.nan, capped)
        new_gross = np.nansum(np.abs(capped), axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale_back = np.where(new_gross > 0, gross / new_gross, 1.0)
        return capped * scale_back

    def _scale(self, signal: Panel, scale_book: float) -> Panel:
        gross = np.nansum(np.abs(signal), axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized = np.where(gross > 0, signal / gross, np.nan)
        return normalized * scale_book

    def _delay(self, signal: Panel, delay: int) -> Panel:
        if delay <= 0:
            return signal
        out = np.full_like(signal, np.nan)
        out[delay:] = signal[:-delay]
        return out
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_backtest_portfolio.py -v
```
Expected: PASS (8 test). Nếu `test_truncate_caps_per_name_weight_and_renormalizes` fail vì
sai số nổi (`np.clip` + renorm), kiểm tra lại `cap_abs` tính trên `gross` TRƯỚC truncate
(đúng impl trên) — không tính trên gross sau.

- [ ] **Step 5: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/backtest/portfolio.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/backtest/portfolio.py
```
Expected: sạch.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/portfolio.py tests/unit/test_backtest_portfolio.py
git commit -m "feat(backtest): PortfolioBuilder decay->neutralize->truncate->scale->delay"
```

---

### Task 3.3: `Backtester` (`src/backtest/backtester.py`)

**Files:**
- Create: `src/backtest/backtester.py`
- Test: `tests/unit/test_backtester.py`

**Interfaces:**
- Consumes: `Panel` (Phase 0), `MarketData` (Phase 0).
- Produces: `@dataclass(frozen=True, slots=True) class BacktestResult` với `daily_pnl:
  npt.NDArray[np.float64]` (T,), `equity_curve: npt.NDArray[np.float64]` (T,), `weights:
  Panel` (T,N); `class Backtester` với `run(self, weights: Panel, data: MarketData) ->
  BacktestResult`.

Công thức (B7/Global Constraints, **delay đã áp ở PortfolioBuilder.build, KHÔNG áp lại ở
đây**): `pnl_t = nansum(weights[t] * data.returns[t])` theo trục asset — `weights` truyền
vào `run` đã là weights ĐÃ DELAY (đầu ra của `PortfolioBuilder.build`), nên công thức chỉ
cần nhân element-wise cùng dòng `t`, không tự dịch thêm. `equity_curve = cumsum(daily_pnl)`
(NaN ở các dòng đầu nếu weights NaN do delay → coi `nan_to_num` về 0 cho riêng phép cumsum
để equity không bị NaN lan toàn bộ về sau, nhưng `daily_pnl` thô giữ NaN nguyên — test phải
phân biệt rõ hai mảng).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_backtester.py
"""Test Backtester.run: weights (đã delay) + returns -> daily_pnl + equity_curve."""

from __future__ import annotations

import numpy as np

from src.backtest.backtester import Backtester, BacktestResult
from src.data.market_panel import MarketData


def _panel_with_known_returns() -> MarketData:
    t, n = 3, 2
    dates = (np.datetime64("2021-01-01") + np.arange(t)).astype("datetime64[D]")
    assets = np.array(["A", "B"], dtype=np.str_)
    universe = np.ones((t, n), dtype=bool)
    returns = np.array([[0.01, -0.02], [0.02, 0.01], [-0.01, 0.0]])
    groups = {"sector": np.zeros((t, n), dtype=np.int64)}
    return MarketData(dates=dates, assets=assets, fields={}, universe=universe,
                      returns=returns, groups=groups)


def test_pnl_is_dot_product_of_weights_and_returns_same_row():
    data = _panel_with_known_returns()
    weights = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
    result = Backtester().run(weights, data)
    expected_pnl = np.array([
        0.5 * 0.01 + (-0.5) * (-0.02),
        0.5 * 0.02 + (-0.5) * 0.01,
        0.5 * (-0.01) + (-0.5) * 0.0,
    ])
    np.testing.assert_allclose(result.daily_pnl, expected_pnl)


def test_equity_curve_is_cumsum_of_pnl():
    data = _panel_with_known_returns()
    weights = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
    result = Backtester().run(weights, data)
    np.testing.assert_allclose(result.equity_curve, np.cumsum(result.daily_pnl))


def test_first_row_nan_weights_from_delay_give_zero_pnl_not_nan_propagated_to_equity():
    data = _panel_with_known_returns()
    weights = np.array([[np.nan, np.nan], [0.5, -0.5], [0.5, -0.5]])
    result = Backtester().run(weights, data)
    assert result.daily_pnl[0] == 0.0  # nansum trên toàn-NaN row -> 0.0, không NaN
    assert not np.isnan(result.equity_curve).any()


def test_result_stores_weights_passed_in():
    data = _panel_with_known_returns()
    weights = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
    result = Backtester().run(weights, data)
    np.testing.assert_allclose(result.weights, weights)


def test_backtest_result_is_frozen():
    data = _panel_with_known_returns()
    weights = np.zeros((3, 2))
    result = Backtester().run(weights, data)
    import pytest
    with pytest.raises(AttributeError):
        result.daily_pnl = np.zeros(3)  # type: ignore[misc]


def test_out_of_universe_cells_excluded_from_pnl():
    data = _panel_with_known_returns()
    universe = data.universe.copy()
    universe[1, 0] = False  # mã A ngày 2 ngoài universe
    data2 = MarketData(dates=data.dates, assets=data.assets, fields=data.fields,
                        universe=universe, returns=data.returns, groups=data.groups)
    weights = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
    # weights[1,0] phải bị mask NaN bởi caller (PortfolioBuilder) trước khi tới Backtester;
    # Backtester tự mask lại theo universe để an toàn dù caller quên.
    result = Backtester().run(weights, data2)
    expected_day1 = -0.5 * 0.01  # chỉ còn cạnh B (asset A bị loại khỏi universe)
    assert np.isclose(result.daily_pnl[1], expected_day1)
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_backtester.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.backtest.backtester'`.

- [ ] **Step 3: Tạo `src/backtest/backtester.py`**

```python
# src/backtest/backtester.py
"""Backtester: weights (đã delay bởi PortfolioBuilder) + returns -> daily PnL + equity.

Delay-1 KHÔNG được áp lại ở đây — `weights` truyền vào `run` là đầu ra của
`PortfolioBuilder.build` (đã dịch `cfg.delay` dòng). Công thức: pnl_t = nansum(w_t * ret_t)
theo trục asset, chỉ trên cell in-universe (an toàn double-mask dù caller đã mask).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.data.market_panel import MarketData
from src.local_types import Panel


@dataclass(frozen=True, slots=True)
class BacktestResult:
    daily_pnl: npt.NDArray[np.float64]  # (T,)
    equity_curve: npt.NDArray[np.float64]  # (T,)
    weights: Panel  # (T, N), đã delay


class Backtester:
    """Chạy backtest delay-1 (hoặc delay tuỳ ý đã áp sẵn trong `weights`)."""

    def run(self, weights: Panel, data: MarketData) -> BacktestResult:
        masked_weights = np.where(data.universe, weights, np.nan)
        contrib = masked_weights * data.returns
        with np.errstate(invalid="ignore"):
            daily_pnl = np.nansum(contrib, axis=1)
        # Ngày toàn-NaN (vd do delay ở đầu chuỗi) -> nansum trả 0.0 (đúng ngữ nghĩa numpy),
        # giữ nguyên — không có pnl phát sinh là hợp lý cho ngày chưa có weight.
        equity_curve = np.cumsum(daily_pnl)
        return BacktestResult(
            daily_pnl=daily_pnl, equity_curve=equity_curve, weights=weights,
        )
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_backtester.py -v
```
Expected: PASS (6 test).

- [ ] **Step 5: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/backtest/backtester.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/backtest/backtester.py
```
Expected: sạch.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/backtester.py tests/unit/test_backtester.py
git commit -m "feat(backtest): Backtester delay-1 weights*returns -> pnl + equity"
```

---

### Task 3.4: Integration MVP — equity curve thật trên `small_panel`

**Files:**
- Create: `tests/integration/test_backtest_mvp.py`

**Interfaces:**
- Consumes: `parse` (`src/lang/parser.py`, Phase 1), `Evaluator`/`EvalContext`
  (`src/engine/evaluator.py`, Phase 2), `PortfolioConfig`/`PortfolioBuilder` (3.1/3.2),
  `Backtester`/`BacktestResult` (3.3), fixture `small_panel` (`tests/conftest.py`, Phase 0).
- Produces: không có module mới — chỉ 1 test integration là **bằng chứng MVP** (parse→eval→
  build→backtest end-to-end thật, không mock).

> **Trước khi viết test:** chạy lệnh pre-condition ở đầu file plan này
> (`from src.lang.parser import parse; from src.engine.evaluator import Evaluator,
> EvalContext`). Nếu lỗi import, DỪNG — Phase 1/2 chưa xong, không tự chế Evaluator tạm ở
> đây. Báo cáo trạng thái và chờ Phase 1/2 merge.

- [ ] **Step 1: Viết test đỏ (alpha viết tay đơn giản, parse được theo grammar Phase 1)**

```python
# tests/integration/test_backtest_mvp.py
"""MVP end-to-end: parse(alpha viết tay) -> eval -> build portfolio -> backtest ->
Sharpe sơ bộ + equity curve, trên dữ liệu thật (fixture small_panel).

Đây là MILESTONE MVP của toàn dự án MiniBrain (Part E master spec): chứng minh đường ống
parse->eval->backtest chạy thông trên một alpha thật, không mock bất cứ thành phần nào.
"""

from __future__ import annotations

import numpy as np

from src.backtest.backtester import Backtester
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.portfolio import PortfolioBuilder
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import parse


def _rough_sharpe(daily_pnl: np.ndarray) -> float:
    valid = daily_pnl[~np.isnan(daily_pnl)]
    if valid.std(ddof=0) == 0 or valid.size < 2:
        return 0.0
    return float(valid.mean() / valid.std(ddof=0) * np.sqrt(252))


def test_handwritten_alpha_runs_end_to_end_and_produces_equity_curve(small_panel):
    expr = "rank(ts_mean(divide(subtract(close, open), open), 5))"
    # small_panel (Phase 0 fixture) chỉ có field "close"/"volume" -> dùng alpha thay thế
    # với field thực có sẵn nếu "open" không tồn tại (xem Step 1b dưới nếu prefilter/parse
    # báo field không hợp lệ).
    node = parse(expr)
    ctx = EvalContext(data=small_panel, registry=None, cache=None)  # registry mặc định nếu None
    signal = Evaluator(ctx).evaluate(node)
    assert signal.shape == (len(small_panel.dates), len(small_panel.assets))

    cfg = PortfolioConfig(neutralization=Neutralization.SECTOR, decay=0,
                          truncation=0.10, scale_book=1.0, delay=1)
    weights = PortfolioBuilder().build(signal, cfg, small_panel)
    result = Backtester().run(weights, small_panel)

    assert result.equity_curve.shape == (len(small_panel.dates),)
    assert not np.isnan(result.equity_curve).any()
    sharpe = _rough_sharpe(result.daily_pnl)
    assert np.isfinite(sharpe)
    print(f"[MVP demo] equity_curve[-1]={result.equity_curve[-1]:.4f} sharpe~{sharpe:.3f}")
```

> **Lưu ý field:** fixture `small_panel` (Phase 0, `tests/conftest.py`) hiện chỉ có
> `fields={"close", "volume"}` — KHÔNG có `"open"`. Nếu `parse`/`Evaluator` raise lỗi field
> không tồn tại với alpha mẫu trên, đổi sang alpha tương đương chỉ dùng field có sẵn, ví dụ:
> `"rank(ts_mean(divide(subtract(close, ts_delay(close, 1)), ts_delay(close, 1)), 5))"`
> (return 1 ngày làm "open" giả lập) — **viết lại test với biểu thức nào parse+eval thành
> công thật trên `small_panel`**, không để alpha mẫu là placeholder không chạy được. Đây là
> quyết định triển khai tại chỗ, không phải thay đổi hợp đồng — ghi lại biểu thức cuối dùng
> trong docstring test.

- [ ] **Step 2: Chạy test — FAIL hoặc lỗi field (xem ghi chú trên), điều chỉnh alpha cho khớp field thật**

```bash
venv/Scripts/python.exe -m pytest tests/integration/test_backtest_mvp.py -v
```
Expected ban đầu: FAIL (do field "open" không tồn tại trong `small_panel`, hoặc do
`EvalContext`/`Evaluator` có chữ ký khác — kiểm tra `src/engine/evaluator.py` thật và sửa
lời gọi cho khớp signature thực tế đã merge, không suy diễn).

- [ ] **Step 3: Sửa alpha string + lời gọi `EvalContext`/`Evaluator` cho khớp API thật, chạy lại đến PASS**

```bash
venv/Scripts/python.exe -m pytest tests/integration/test_backtest_mvp.py -v
```
Expected: PASS (1 test), có in ra dòng `[MVP demo] equity_curve[-1]=... sharpe~...` —
**đây chính là bằng chứng demo MVP**, copy dòng output vào báo cáo review (Task 3.6).

- [ ] **Step 4: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check tests/integration/test_backtest_mvp.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent tests/integration/test_backtest_mvp.py
```
Expected: sạch (test file cũng phải type-check sạch theo Global Constraints).

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_backtest_mvp.py
git commit -m "test(backtest): MVP integration parse->eval->portfolio->backtest trên small_panel"
```

- [ ] **Step 6: DỪNG — demo cho user**

In lại output `[MVP demo] equity_curve[-1]=... sharpe~...` cho user xem trực tiếp (chạy lại
`pytest tests/integration/test_backtest_mvp.py -v -s` để thấy `print`). Đây là **MVP
milestone** theo Part E master spec — không tự ý tiếp tục sang Task 3.5 mà không xác nhận
việc demo này đã được nhìn thấy/ghi nhận (ghi vào PROGRESS.md ở Task 3.6 là đủ xác nhận
cho agentic worker chạy không giám sát; nếu có user tương tác trực tiếp, dừng và hỏi).

---

### Task 3.5: Gỡ đường cũ (D9) — `score_local_gate` thành cổng bắt buộc trong `RefinementLoop`

**Files:**
- Create: `src/backtest/gate.py`
- Test: `tests/unit/test_backtest_gate.py`
- Modify: `src/llm/loop.py`
- Test: `tests/unit/test_loop_local_gate.py`

**Bước 0 — BẮT BUỘC đọc trước khi sửa `loop.py`:** đã đọc toàn văn `src/llm/loop.py` để
viết plan này (565 dòng). Điểm chèn xác định chính xác là trong `RefinementLoop._evaluate`
(khoảng dòng 227–286 ở bản hiện tại), theo thứ tự hiện có:

```
prefilter.check(expr)                          # dòng 232-235 — GIỮ NGUYÊN (cú pháp rẻ)
originality gate (zoo)                          # dòng 237-246 — GIỮ NGUYÊN
cache lookup (get_cached_simulation)            # dòng 248-255 — GIỮ NGUYÊN (trả sớm nếu hit)
aligner gate                                    # dòng 257-267 — GIỮ NGUYÊN
if sims_used >= max_simulations: return None    # dòng 269-270
result = self.simulator.simulate(...)           # dòng 272   <-- ĐIỂM CHÈN: NGAY TRƯỚC dòng này
```

`score_local_gate` phải chèn **sau** dòng 267 (sau aligner gate, vì aligner là tín hiệu LLM
rẻ hơn backtest local) và **trước** dòng 269 (trước khi kiểm `sims_used`/gọi `simulate`).
Lý do thứ tự: prefilter/originality/cache/aligner đều rẻ hơn một backtest local đầy đủ
(parse+eval+portfolio+backtest); chạy chúng trước để fail nhanh, KHÔNG tốn thời gian build
backtest cho candidate đã bị loại bởi gate rẻ hơn. `score_local_gate` fail → `return None`
ngay (giống các gate khác), ghi `record_failure(expr, "local_gate_fail", reason, "llm")`,
**không** gọi `simulator.simulate` (không đốt sim) và **không** tăng `self.sims_used`.

**Interfaces:**
- `src/backtest/gate.py` Consumes: `parse` (Phase 1), `Evaluator`/`EvalContext` (Phase 2),
  `PortfolioBuilder`/`Backtester`/`PortfolioConfig` (3.1–3.3), `MarketData` (Phase 0).
  Produces: `@dataclass(frozen=True, slots=True) class LocalGateVerdict` với `passed: bool`,
  `reason: str`; `def score_local_gate(expr: str, cfg: PortfolioConfig, data: MarketData) ->
  LocalGateVerdict` — parse expr; lỗi parse → `passed=False, reason="parse: <msg>"`; eval →
  build → backtest; nếu daily_pnl toàn NaN/rỗng → `passed=False, reason="no_pnl"`; tạm thời
  (Phase 3 MVP, **trước khi có MetricsCalculator ở Phase 4**) chỉ gate tối thiểu: parse
  thành công + signal eval không toàn-NaN + ít nhất 1 ngày pnl hữu hạn → `passed=True`.
  **Ghi rõ trong code**: đây là gate tối thiểu placeholder-đúng-nghĩa-hẹp (không phải
  placeholder rỗng) — Phase 4 sẽ mở rộng `score_local_gate` gọi thêm `MetricsCalculator` +
  `GateEvaluator` (hard gates depth/self_corr/concentration) khi các thành phần đó tồn tại;
  Phase 3 chỉ đảm bảo expr "evaluable và sinh được pnl", chặn các biểu thức vô nghĩa
  (parse lỗi, field sai, signal toàn NaN) trước khi đốt sim — đúng đúng phạm vi D9 cho
  Phase 3 (gate KHÔNG cần đầy đủ Sharpe/turnover/concentration ở bước này).
- `src/llm/loop.py` Consumes (mới): `score_local_gate`, `PortfolioConfig`, `MarketData` từ
  `src/backtest`. Loop nhận thêm tham số constructor `local_gate_fn=score_local_gate`
  (inject được — test dùng fake) và `market_data: MarketData | None = None`,
  `local_gate_cfg: PortfolioConfig | None = None`. Nếu `market_data is None` (chưa có data
  thật wired ở Phase 3 — `market_fetch` full pipeline là việc của CLI/Phase 8), `score_local_
  gate` được **bỏ qua** (log warning rõ "local gate tắt: thiếu market_data") để không phá
  vỡ hành vi hiện tại của tool khi chạy thật chưa có data — nhưng **mặc định khi
  `market_data` có giá trị, gate là bắt buộc, không tuỳ chọn** (đúng yêu cầu D9: "cổng bắt
  buộc" áp dụng khi local stack đã sẵn sàng; cờ tắt chỉ tồn tại cho trường hợp chưa wire
  data, không phải để bypass gate khi đã có data).

- [ ] **Step 1: Viết test đỏ cho `score_local_gate`**

```python
# tests/unit/test_backtest_gate.py
"""Test score_local_gate: cổng local tối thiểu Phase 3 (parse + eval + pnl sinh được)."""

from __future__ import annotations

from src.backtest.config import PortfolioConfig
from src.backtest.gate import LocalGateVerdict, score_local_gate


def test_valid_simple_expression_passes(small_panel):
    verdict = score_local_gate("close", PortfolioConfig(delay=1), small_panel)
    assert isinstance(verdict, LocalGateVerdict)
    assert verdict.passed is True


def test_parse_error_fails_with_reason(small_panel):
    verdict = score_local_gate("not_a_real_op(close,", PortfolioConfig(), small_panel)
    assert verdict.passed is False
    assert "parse" in verdict.reason.lower()


def test_unknown_field_fails(small_panel):
    verdict = score_local_gate("totally_unknown_field_xyz", PortfolioConfig(), small_panel)
    assert verdict.passed is False
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_backtest_gate.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.backtest.gate'`.

- [ ] **Step 3: Tạo `src/backtest/gate.py`**

```python
# src/backtest/gate.py
"""score_local_gate — cổng local BẮT BUỘC trước khi đốt sim Brain (D9, gỡ đường cũ).

Phase 3 MVP: gate tối thiểu — expr phải parse được, eval ra signal không toàn-NaN, và
backtest sinh được ít nhất 1 ngày pnl hữu hạn. KHÔNG còn đủ Sharpe/turnover/concentration —
đó là việc của Phase 4 (MetricsCalculator + GateEvaluator), sẽ mở rộng hàm này khi có. Đây
là điểm DUY NHẤT src/llm được phép import từ tầng backtest (dependency rule một chiều).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backtest.backtester import Backtester
from src.backtest.config import PortfolioConfig
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import ParseError, parse


@dataclass(frozen=True, slots=True)
class LocalGateVerdict:
    passed: bool
    reason: str


def score_local_gate(expr: str, cfg: PortfolioConfig, data: MarketData) -> LocalGateVerdict:
    try:
        node = parse(expr)
    except ParseError as exc:
        return LocalGateVerdict(False, f"parse lỗi: {exc}")

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

    return LocalGateVerdict(True, "ok")
```

> Lưu ý chữ ký `EvalContext(data=..., registry=None, cache=None)`: kiểm tra `registry=None`
> có được `Evaluator`/`EvalContext` (Phase 2) chấp nhận là "dùng default_registry()" hay
> không bằng cách đọc `src/engine/evaluator.py` thật trước khi code bước này. Nếu
> `EvalContext` đòi `registry: OperatorRegistry` không-optional, sửa thành
> `default_registry()` (import từ `src.lang.registry`) — KHÔNG để `None` gây lỗi runtime.

- [ ] **Step 4: Chạy test — PASS (sửa theo API thật của EvalContext/Evaluator nếu khác giả định)**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_backtest_gate.py -v
```
Expected: PASS (3 test).

- [ ] **Step 5: ruff + mypy cho `gate.py`**

```bash
venv/Scripts/python.exe -m ruff check src/backtest/gate.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/backtest/gate.py
```
Expected: sạch.

- [ ] **Step 6: Commit `gate.py`**

```bash
git add src/backtest/gate.py tests/unit/test_backtest_gate.py
git commit -m "feat(backtest): score_local_gate — cổng local tối thiểu trước sim (D9)"
```

- [ ] **Step 7: Viết test đỏ cho loop — local gate fail thì KHÔNG gọi simulate (fake, không mạng thật)**

```python
# tests/unit/test_loop_local_gate.py
"""Test RefinementLoop._evaluate: score_local_gate fail -> bỏ candidate, KHÔNG gọi
simulator.simulate (không đốt sim). Toàn bộ phụ thuộc là fake/monkeypatch — không gọi
mạng/Brain thật.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.backtest.config import PortfolioConfig
from src.backtest.gate import LocalGateVerdict
from src.llm.loop import RefinementLoop
from src.simulation.config import SimConfig


@dataclass
class _FakeCandidate:
    expression: str

    class hypothesis:
        @staticmethod
        def to_dict():
            return {}

    description: str = "fake"


class _FakePrefilter:
    def check(self, expr):
        return True, "ok"


class _FakeSimulator:
    def __init__(self):
        self.calls = 0

    def simulate(self, expr, settings):
        self.calls += 1
        raise AssertionError("simulate() KHÔNG được gọi khi local gate fail")


class _FakeRepo:
    def __init__(self):
        self.failures = []

    def record_failure(self, expr, kind, reason, source):
        self.failures.append((expr, kind, reason))

    def get_cached_simulation(self, expr, config_key=None):
        return None

    def save_alpha(self, *a, **kw):
        return "fake-alpha-id"

    def save_simulation(self, *a, **kw):
        return None

    def recent_failures(self, n):
        return self.failures[:n]


def _make_loop(local_gate_fn, market_data=object()):
    sim_config = SimConfig.default(region="USA", universe="TOP3000", delay=1)
    return RefinementLoop(
        hypothesis_gen=None, translator=None, refiner=None,
        simulator=_FakeSimulator(), prefilter=_FakePrefilter(), repo=_FakeRepo(),
        region="USA", universe="TOP3000", delay=1, sim_config=sim_config,
        max_simulations=10,
        local_gate_fn=local_gate_fn, market_data=market_data,
        local_gate_cfg=PortfolioConfig(),
    )


def test_local_gate_fail_blocks_simulate_and_records_failure():
    def fake_gate(expr, cfg, data):
        return LocalGateVerdict(False, "fake fail reason")

    loop = _make_loop(fake_gate)
    cand = _FakeCandidate(expression="rank(close)")
    ev = loop._evaluate(cand, parent_id=None)

    assert ev is None
    assert loop.simulator.calls == 0
    assert any(kind == "local_gate_fail" for _, kind, _ in loop.repo.failures)


def test_local_gate_pass_allows_simulate_to_be_called():
    def fake_gate(expr, cfg, data):
        return LocalGateVerdict(True, "ok")

    # Simulator fake ở đây PHẢI thực sự cho phép gọi (không assert chặn) để xác nhận
    # nhánh pass đi xuống simulate như cũ.
    class _AllowSimulator:
        def __init__(self):
            self.calls = 0

        def simulate(self, expr, settings):
            self.calls += 1
            raise RuntimeError("dừng sớm có chủ đích — chỉ cần xác nhận ĐÃ gọi simulate")

    loop = _make_loop(fake_gate)
    loop.simulator = _AllowSimulator()
    cand = _FakeCandidate(expression="rank(close)")
    with pytest.raises(RuntimeError, match="dừng sớm"):
        loop._evaluate(cand, parent_id=None)
    assert loop.simulator.calls == 1


def test_local_gate_skipped_when_market_data_is_none():
    """market_data=None (chưa wire data thật) -> gate bị bỏ qua, hành vi cũ giữ nguyên."""
    gate_called = []

    def fake_gate(expr, cfg, data):
        gate_called.append(expr)
        return LocalGateVerdict(False, "should not matter")

    class _AllowSimulator:
        def __init__(self):
            self.calls = 0

        def simulate(self, expr, settings):
            self.calls += 1
            raise RuntimeError("dừng sớm có chủ đích")

    loop = _make_loop(fake_gate, market_data=None)
    loop.simulator = _AllowSimulator()
    cand = _FakeCandidate(expression="rank(close)")
    with pytest.raises(RuntimeError, match="dừng sớm"):
        loop._evaluate(cand, parent_id=None)
    assert gate_called == []  # gate không được gọi khi market_data=None
    assert loop.simulator.calls == 1
```

- [ ] **Step 8: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_loop_local_gate.py -v
```
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'local_gate_fn'`
(constructor `RefinementLoop` chưa có tham số mới).

- [ ] **Step 9: Sửa `src/llm/loop.py`**

Sửa `__init__` — thêm tham số mới vào cuối danh sách (sau `config_tuner`, giữ thứ tự cũ
không đổi để không phá lời gọi vị trí hiện có nào còn sót, dù constructor này chủ yếu được
gọi bằng keyword arguments trong codebase — kiểm tra `Grep "RefinementLoop("` trước khi sửa
để chắc không có lời gọi positional dài sẽ lệch). Thêm:

```python
        config_tuner=None,
        local_gate_fn=None,
        market_data=None,
        local_gate_cfg: PortfolioConfig | None = None,
    ):
        ...
        self.config_tuner = config_tuner
        # Local pre-filter BẮT BUỘC trước simulate (D9 — gỡ đường cũ "LLM->sim trực tiếp").
        # market_data=None -> gate bị bỏ qua (chưa wire data thật, Phase 3 MVP); có
        # market_data -> MỌI candidate phải pass local_gate_fn trước khi tốn sim.
        self.local_gate_fn = local_gate_fn or score_local_gate
        self.market_data = market_data
        self.local_gate_cfg = local_gate_cfg or PortfolioConfig()
        self.sims_used = 0
        self.zoo_added = 0
```

Thêm import ở đầu file (cùng nhóm import `src.simulation.config`):

```python
from src.backtest.config import PortfolioConfig
from src.backtest.gate import score_local_gate
```

Chèn gate trong `_evaluate`, NGAY TRƯỚC dòng `if self.sims_used >= self.max_simulations:`
(giữ nguyên mọi dòng trước/sau, chỉ chèn block mới):

```python
        # Local gate BẮT BUỘC (D9): chỉ chạy khi đã có market_data thật wire vào loop.
        # Local hard-fail -> bỏ NGAY, không tăng sims_used, không gọi simulator.
        if self.market_data is not None:
            verdict = self.local_gate_fn(expr, self.local_gate_cfg, self.market_data)
            if not verdict.passed:
                self.repo.record_failure(expr, "local_gate_fail", verdict.reason, "llm")
                return None

        if self.sims_used >= self.max_simulations:
            return None  # hết trần sim, không gọi WQ thêm
```

- [ ] **Step 10: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_loop_local_gate.py -v
```
Expected: PASS (3 test).

- [ ] **Step 11: Chạy lại TOÀN BỘ test cũ liên quan `loop.py` để xác nhận không phá hành vi cũ**

```bash
venv/Scripts/python.exe -m pytest tests/ -k "loop" -v
```
Expected: PASS toàn bộ — đặc biệt mọi test cũ của `RefinementLoop` (không có `market_data`
truyền vào → gate bỏ qua → hành vi y như trước khi sửa).

- [ ] **Step 12: ruff + mypy cho `loop.py`**

```bash
venv/Scripts/python.exe -m ruff check src/llm/loop.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/llm/loop.py
```
Expected: sạch. Nếu mypy phàn nàn `local_gate_fn=None` kiểu `Callable | None` gán vào field
không optional, thêm type hint rõ:
`local_gate_fn: Callable[[str, PortfolioConfig, MarketData], LocalGateVerdict] | None = None`
trong signature `__init__` và field tương ứng.

- [ ] **Step 13: Commit**

```bash
git add src/llm/loop.py tests/unit/test_loop_local_gate.py
git commit -m "refactor(llm): score_local_gate thành cổng bắt buộc trước simulate (D9)"
```

---

### Task 3.6: Review cuối Phase 3 + PROGRESS.md + merge + push

**Files:**
- Modify: `PROGRESS.md` (tạo nếu chưa có ở root)

- [ ] **Step 1: Chạy toàn bộ test suite**

```bash
venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: PASS toàn bộ (test cũ + test Phase 0–3 mới). Không skip, không xfail bất ngờ.

- [ ] **Step 2: ruff toàn repo**

```bash
venv/Scripts/python.exe -m ruff check .
```
Expected: sạch (0 lỗi). Nếu có lỗi ở file ngoài phạm vi Phase 3 đã tồn tại từ trước, KHÔNG
tự sửa lan man — chỉ đảm bảo file Phase 3 tạo/sửa sạch và không làm tăng số lỗi tổng so với
trước khi bắt đầu phase (so sánh bằng `git stash` + chạy lại nếu cần xác minh).

- [ ] **Step 3: mypy strict toàn repo**

```bash
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent .
```
Expected: sạch cho mọi file Phase 3 (`src/backtest/*.py`). Lỗi mypy tiền-tồn ở module khác
(nếu có) ghi nhận riêng, không thuộc trách nhiệm phase này để sửa — nhưng phải không tăng
thêm lỗi mới do thay đổi của Phase 3.

- [ ] **Step 4: Chạy lại riêng integration MVP với output để chốt demo**

```bash
venv/Scripts/python.exe -m pytest tests/integration/test_backtest_mvp.py -v -s
```
Expected: PASS, dòng `[MVP demo] equity_curve[-1]=... sharpe~...` hiển thị — copy số liệu
này vào `PROGRESS.md`.

- [ ] **Step 5: Cập nhật `PROGRESS.md`**

Append (tạo file với header nếu chưa tồn tại) một mục mới:

```markdown
## Phase 3 — Backtester (MVP) — xong (nhánh `phase-3-backtester`)

- `src/backtest/config.py`: `PortfolioConfig` + `Neutralization` (stage separation).
- `src/backtest/portfolio.py`: `PortfolioBuilder.build` decay->neutralize->truncate->scale->delay.
- `src/backtest/backtester.py`: `Backtester.run` delay-1 `pnl_t=nansum(w_t*ret_t)` (delay đã
  áp ở portfolio).
- `src/backtest/gate.py`: `score_local_gate` — cổng local tối thiểu Phase 3 (parse+eval+pnl
  hữu hạn); Phase 4 sẽ mở rộng với MetricsCalculator/GateEvaluator đầy đủ.
- **MVP demo:** parse("<biểu thức thật đã dùng ở Task 3.4>") -> eval -> backtest trên
  `small_panel` -> equity_curve[-1]=<giá trị>, sharpe~<giá trị>.
- **D9 (gỡ đường cũ):** `RefinementLoop._evaluate` chèn `score_local_gate` BẮT BUỘC trước
  `simulator.simulate` khi `market_data` được wire; `market_data=None` (chưa wire ở Phase 3)
  -> gate bỏ qua, hành vi cũ giữ nguyên (không phá test/luồng hiện có).
- Rủi ro mở: `score_local_gate` Phase 3 CHƯA có metrics/self-corr/concentration đầy đủ —
  KHÔNG dùng làm gate cuối cùng cho production, chỉ chặn expr vô nghĩa. Phase 4 phải nối
  tiếp trước khi coi gate là "đủ mạnh" để thay hoàn toàn vai trò prefilter cũ.
- Test: `tests/unit/test_backtest_config.py`, `test_backtest_portfolio.py`,
  `test_backtester.py`, `test_backtest_gate.py`, `test_loop_local_gate.py`,
  `tests/integration/test_backtest_mvp.py`.
```

```bash
git add PROGRESS.md
git commit -m "docs: cập nhật PROGRESS.md cho Phase 3 (Backtester MVP)"
```

- [ ] **Step 6: Merge vào `main` + push**

```bash
git checkout main
git pull --ff-only
git merge --no-ff phase-3-backtester -m "Merge phase-3-backtester: PortfolioBuilder + Backtester (MVP) + gỡ đường cũ (D9)"
venv/Scripts/python.exe -m pytest tests/ -v
git push origin main
```
Expected: merge sạch (không conflict), test PASS lại trên `main` sau merge, push thành công.

- [ ] **Step 7: Self-review cuối (đối chiếu lại spec)**

Đối chiếu danh sách sau, đánh dấu từng mục:

- [ ] `PortfolioConfig` đúng 5 field + default đúng B7 (neut=SECTOR, decay=0, trunc=0.10,
  scale=1.0, delay=1).
- [ ] `PortfolioBuilder.build` đúng thứ tự 5 bước, không đảo; chỉ tính trên cell in-universe.
- [ ] `Backtester.run` không tự áp delay lần 2 (delay đã ở portfolio).
- [ ] MVP integration test chạy thật trên `small_panel`, không mock Evaluator/parser.
- [ ] `score_local_gate` chèn đúng vị trí trong `_evaluate` (sau aligner, trước
  sims_used-check/simulate) — đối chiếu lại dòng thật trong `src/llm/loop.py` sau khi sửa,
  không chỉ tin vào kế hoạch.
- [ ] Test loop dùng fake/monkeypatch, không gọi mạng/Brain thật (`_FakeSimulator`,
  `_FakeRepo` — không import `src.data.client` hay tương đương).
- [ ] Test cũ của `RefinementLoop` (trước Phase 3) vẫn PASS không sửa đổi giả định
  (`market_data=None` mặc định bảo toàn hành vi).
- [ ] mypy --strict + ruff sạch trên toàn bộ file Phase 3.
- [ ] `PROGRESS.md` có số liệu demo thật (không bịa số).
- [ ] Không tạo thêm package `src/pipeline/` (chưa đến Phase 8) — `gate.py` nằm trong
  `src/backtest/`.

Nếu bất kỳ mục nào KHÔNG đạt, không coi Phase 3 là xong — sửa và lặp lại review trước khi
báo cáo hoàn thành.
