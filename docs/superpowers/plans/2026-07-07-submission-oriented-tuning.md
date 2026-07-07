# Submission-oriented Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Nâng chất lượng alpha của vòng kín hướng tới chuẩn nộp — sửa neutralization cho price/volume, gate turnover, nhắm Power Pool, proxy robustness sub-universe.

**Architecture:** Dồn vào `LocalTuner` (sweep neutralization; ràng buộc turnover) + `LocalTunerRefiner` (map neutralization Brain; cờ Power Pool; gate sub-universe) + đổi default neutralization closed-loop.

**Tech Stack:** Python 3.12, numpy, pytest. Đã có LocalTuner/LocalTunerRefiner (spec 1).

## Global Constraints
- Code/comment tiếng Việt CÓ DẤU đầy đủ; commit SUBJECT ASCII (quy ước repo).
- TDD: test đỏ trước, mỗi task ≥1 commit.
- Không mạng/LLM trong test.
- Sweep neutralization CHỈ {MARKET, SECTOR} (panel local chỉ có group sector).
- Test: `venv/Scripts/python.exe -m pytest <path> -q`
- Commit trailer (2 dòng cuối):
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01V7aUj8puXBx9HtUftNWGL9

---

### Task 1: LocalTuner sweep neutralization {MARKET, SECTOR}

**Files:**
- Modify: `src/backtest/local_tuner.py`
- Test: `tests/unit/test_local_tuner_neut.py`

**Interfaces:**
- Consumes: `Neutralization` (`src/backtest/config.py`), `tune`/`TuneResult` (spec 1).
- Produces: `tune` giờ quét thêm neutralization; `TuneResult.best_config.neutralization` là giá trị thắng.

- [ ] **Step 1: Test đỏ**

```python
# tests/unit/test_local_tuner_neut.py
from __future__ import annotations

import src.operators_local  # noqa: F401
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.local_tuner import tune


def _cfg():
    return PortfolioConfig(neutralization=Neutralization.MARKET, decay=4, truncation=0.08)


def test_tune_quet_neutralization_chon_sector():
    # eval_fn cho điểm cao khi neutralization=SECTOR -> phải được chọn.
    def eval_fn(node, config):
        return 2.0 if config.neutralization is Neutralization.SECTOR else 0.5

    res = tune("rank(close)", _cfg(), data=None, budget=60, eval_fn=eval_fn)
    assert res.best_config.neutralization is Neutralization.SECTOR
    assert res.local_sharpe == 2.0
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_local_tuner_neut.py -q`
Expected: FAIL (tune chưa quét neutralization -> giữ MARKET).

- [ ] **Step 3: Implement**

Trong `src/backtest/local_tuner.py`, thêm import + hằng và mở rộng Giai đoạn 2:

```python
from src.backtest.config import Neutralization, PortfolioConfig
```
(thêm `Neutralization` vào dòng import PortfolioConfig sẵn có)

Thêm hằng cạnh `_DECAYS`/`_TRUNCS`:
```python
# Chỉ MARKET/SECTOR: docs khuyến nghị cho price/volume + eval local được (panel có group sector).
_NEUTS = (Neutralization.MARKET, Neutralization.SECTOR)
```

Sửa `phase1_cap` để chừa chỗ cho lưới config lớn hơn:
```python
    phase1_cap = max(1, budget - len(_DECAYS) * len(_TRUNCS) * len(_NEUTS))
```

Thay vòng Giai đoạn 2 (decay × truncation) bằng (decay × truncation × neutralization):
```python
    # Giai đoạn 2: quét config (decay x truncation x neutralization) quanh biểu thức tốt nhất.
    for neut in _NEUTS:
        for d in _DECAYS:
            for t in _TRUNCS:
                if evals >= budget:
                    break
                cfg = replace(base_config, decay=d, truncation=t, neutralization=neut)
                if cfg == best_config:
                    continue
                s, m = score(best_node, cfg)
                evals += 1
                if s > best:
                    best, best_config, best_metrics = s, cfg, m
```

- [ ] **Step 4: Chạy — xanh**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_local_tuner_neut.py tests/unit/test_local_tuner_tune.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/local_tuner.py tests/unit/test_local_tuner_neut.py
git commit -m "$(printf 'feat(tuner): sweep neutralization MARKET/SECTOR cho price-volume\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01V7aUj8puXBx9HtUftNWGL9')"
```

---

### Task 2: LocalTuner gate turnover (loại config TO > 0.70)

**Files:**
- Modify: `src/backtest/local_tuner.py`
- Test: `tests/unit/test_local_tuner_turnover.py`

**Interfaces:**
- Produces: `tune(..., max_turnover: float = 0.70)`; config có local turnover > max_turnover không bao giờ được chọn (điểm −inf).

- [ ] **Step 1: Test đỏ** (monkeypatch `local_metrics` để kịch bản hóa turnover/sharpe)

```python
# tests/unit/test_local_tuner_turnover.py
from __future__ import annotations

import src.operators_local  # noqa: F401
from src.backtest import local_tuner
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.local_tuner import tune
from src.backtest.metrics_local import AlphaMetrics


def _metrics(sharpe, turnover):
    return AlphaMetrics(
        sharpe=sharpe, annual_return=0.2, turnover=turnover, max_drawdown=0.1,
        fitness=1.2, per_year_sharpe={2020: sharpe}, weight_concentration=0.05,
    )


def test_tune_loai_config_turnover_qua_cao(monkeypatch):
    # decay=2 -> TO 0.9 (quá cao) Sharpe 3.0; các config khác -> TO 0.3 Sharpe 1.5.
    # Dù Sharpe 3.0 cao hơn, config TO>0.70 phải bị loại -> winner có TO<=0.70.
    def fake_metrics(node, config, data, registry):
        if config.decay == 2:
            return _metrics(3.0, 0.9)
        return _metrics(1.5, 0.3)

    monkeypatch.setattr(local_tuner, "local_metrics", fake_metrics)
    base = PortfolioConfig(neutralization=Neutralization.MARKET, decay=4, truncation=0.08)
    res = tune("rank(close)", base, data=object(), budget=80)
    assert res.local_sharpe == 1.5          # không phải 3.0 (config TO cao bị loại)
    assert res.best_config.decay != 2
    assert res.local_metrics.turnover <= 0.70
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_local_tuner_turnover.py -q`
Expected: FAIL (chưa gate turnover -> chọn Sharpe 3.0 / TO 0.9).

- [ ] **Step 3: Implement**

Thêm hằng cạnh `_NEUTS`:
```python
_MAX_TURNOVER = 0.70  # Brain đòi 1%-70%; config vượt trần là rác chắc chắn fail.
```

Sửa chữ ký `tune` thêm `max_turnover: float = _MAX_TURNOVER` (sau `budget`), và trong nhánh real-path của `score()` (khi `eval_fn is None`), loại config quá turnover:
```python
            m = local_metrics(node, config, data, registry)
            if m is None:
                return float("-inf"), None
            if m.turnover is not None and m.turnover > max_turnover:
                return float("-inf"), m   # vượt trần turnover -> loại (giữ metrics để báo cáo)
            s = m.sharpe
            return (float(s) if s is not None and np.isfinite(s) else float("-inf")), m
```

- [ ] **Step 4: Chạy — xanh**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_local_tuner_turnover.py tests/unit/test_local_tuner_tune.py tests/unit/test_local_tuner_neut.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/local_tuner.py tests/unit/test_local_tuner_turnover.py
git commit -m "$(printf 'feat(tuner): gate turnover >0.70 (Brain doi 1-70%%)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01V7aUj8puXBx9HtUftNWGL9')"
```

---

### Task 3: Cờ Power Pool eligibility trên IdeaOutcome + refiner tính + log

**Files:**
- Modify: `src/pipeline/closed_loop.py` (thêm field `power_pool_eligible` vào `IdeaOutcome`; log khi trúng)
- Modify: `src/app/closed_loop_adapters.py` (refiner tính cờ)
- Test: `tests/unit/test_power_pool_flag.py`

**Interfaces:**
- Produces: `IdeaOutcome.power_pool_eligible: bool = False`; helper `is_power_pool(expr, sharpe, self_corr, registry) -> bool` trong `closed_loop_adapters.py`.
- Tiêu chí: unique operator ≤ 8, unique field (trừ grouping) ≤ 3, sharpe ≥ 1.0, self_corr là None HOẶC ≤ 0.5.

- [ ] **Step 1: Test đỏ**

```python
# tests/unit/test_power_pool_flag.py
from __future__ import annotations

import src.operators_local  # noqa: F401
from src.app.closed_loop_adapters import is_power_pool
from src.lang.registry import default_registry


def test_power_pool_dat_khi_don_gian_va_sharpe_du():
    reg = default_registry()
    assert is_power_pool("rank(ts_delta(close, 5))", 1.2, 0.3, reg) is True


def test_power_pool_khong_dat_khi_sharpe_thap():
    reg = default_registry()
    assert is_power_pool("rank(ts_delta(close, 5))", 0.8, 0.3, reg) is False


def test_power_pool_khong_dat_khi_self_corr_cao():
    reg = default_registry()
    assert is_power_pool("rank(ts_delta(close, 5))", 1.5, 0.6, reg) is False


def test_power_pool_khong_dat_khi_qua_nhieu_field():
    reg = default_registry()
    # 4 field khác nhau > 3
    expr = "add(add(close, open), add(high, low))"
    assert is_power_pool(expr, 1.5, 0.1, reg) is False
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_power_pool_flag.py -q`
Expected: FAIL (`is_power_pool` chưa tồn tại).

- [ ] **Step 3: Implement helper (`src/app/closed_loop_adapters.py`)**

Thêm import + hàm (đặt cạnh `LocalTunerRefiner`):
```python
from src.lang.visitors import FieldCollector, OperatorCollector

# Grouping field không tính vào giới hạn "3 field dữ liệu" của Power Pool.
_POWER_POOL_GROUPS = frozenset(
    {"country", "exchange", "market", "sector", "industry", "subindustry", "currency"}
)


def is_power_pool(expr: str, sharpe, self_corr, registry) -> bool:
    """Đủ tiêu chí Power Pool: ≤8 operator, ≤3 field (trừ grouping), Sharpe≥1.0, self_corr≤0.5."""
    if sharpe is None or sharpe < 1.0:
        return False
    if self_corr is not None and abs(self_corr) > 0.5:
        return False
    node = parse(expr)
    n_ops = len(OperatorCollector().visit(node))
    n_fields = len(FieldCollector(registry).visit(node) - _POWER_POOL_GROUPS)
    return n_ops <= 8 and n_fields <= 3
```

- [ ] **Step 4: `IdeaOutcome` + tính cờ trong refiner + log**

Trong `src/pipeline/closed_loop.py`, thêm field cuối `IdeaOutcome` (default để tương thích ngược):
```python
    power_pool_eligible: bool = False
```

Trong `src/app/closed_loop_adapters.py::LocalTunerRefiner.refine_and_sim`, ở nhánh trả outcome sau sim (khi có metric Brain), tính cờ và đưa vào `IdeaOutcome(...)`:
```python
        registry = self.registry or default_registry()
        power_pool = passed and is_power_pool(tr.best_expr, result.sharpe, self_corr, registry)
        return IdeaOutcome(
            expr=tr.best_expr, canonical_hash=canonical_hash, passed=passed,
            wq_alpha_id=result.alpha_id, sharpe=result.sharpe, fitness=result.fitness,
            turnover=result.turnover, self_corr=self_corr, sims_used=1,
            stop_reason="local_tuned", power_pool_eligible=power_pool,
        )
```
(import `default_registry` nếu chưa có: `from src.lang.registry import default_registry`.)

Trong `src/pipeline/closed_loop.py::ClosedLoop.run`, sau dòng log kết quả 1 ý tưởng, thêm:
```python
                if getattr(outcome, "power_pool_eligible", False):
                    logger.info("   ⭐ Power Pool eligible (Sharpe≥1.0, ≤8 op, ≤3 field, self_corr≤0.5)")
```

- [ ] **Step 5: Chạy — xanh + hồi quy**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_power_pool_flag.py tests/unit/test_local_tuner_refiner.py tests/unit/test_closed_loop_local_refiner.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/closed_loop.py src/app/closed_loop_adapters.py tests/unit/test_power_pool_flag.py
git commit -m "$(printf 'feat(closed-loop): co Power Pool eligibility (Sharpe>=1.0, <=8 op, <=3 field)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01V7aUj8puXBx9HtUftNWGL9')"
```

---

### Task 4: Proxy robustness sub-universe (gate trên winner trong refiner)

**Files:**
- Create: `src/backtest/sub_universe.py`
- Modify: `src/app/closed_loop_adapters.py` (gate winner trước sim)
- Test: `tests/unit/test_sub_universe.py`

**Interfaces:**
- Produces: `sub_universe_ok(node, config, data, registry, *, full_sharpe, frac=0.5) -> bool` — True nếu `sub_sharpe ≥ 0.75·√frac·full_sharpe` (frac = tỉ lệ sub/univ). full_sharpe ≤ 0 -> True (không phạt alpha âm ở đây; turnover/floor lo).

- [ ] **Step 1: Test đỏ**

```python
# tests/unit/test_sub_universe.py
from __future__ import annotations

import numpy as np

import src.operators_local  # noqa: F401
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.sub_universe import sub_universe_ok
from src.data.market_panel import MarketData
from src.lang.parser import parse
from src.lang.registry import default_registry


def _panel(t=80, n=12, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.arange("2020-01-01", "2021-06-01", dtype="datetime64[D]")[:t].astype("datetime64[ns]")
    close = 100 + np.cumsum(rng.normal(0, 1, (t, n)), axis=0)
    fields = {k: close.copy() for k in ("close", "open", "high", "low", "vwap")}
    fields["volume"] = np.abs(rng.normal(1e6, 2e5, (t, n)))
    return MarketData(
        dates=dates, assets=np.array([f"S{i}" for i in range(n)]), fields=fields,
        universe=np.ones((t, n), dtype=bool),
        returns=np.vstack([np.zeros((1, n)), np.diff(close, axis=0) / close[:-1]]),
        groups={"sector": (np.arange(n) % 3).reshape(1, n).repeat(t, axis=0)},
    )


def test_sub_universe_ok_tra_bool_khong_sap():
    data = _panel()
    node = parse("rank(ts_delta(close, 5))")
    cfg = PortfolioConfig(neutralization=Neutralization.MARKET, decay=4, truncation=0.08)
    out = sub_universe_ok(node, cfg, data, default_registry(), full_sharpe=1.0, frac=0.5)
    assert isinstance(out, bool)


def test_sub_universe_full_sharpe_khong_duong_thi_pass():
    data = _panel()
    node = parse("rank(ts_delta(close, 5))")
    cfg = PortfolioConfig(neutralization=Neutralization.MARKET, decay=4, truncation=0.08)
    assert sub_universe_ok(node, cfg, data, default_registry(), full_sharpe=-0.5, frac=0.5) is True
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_sub_universe.py -q`
Expected: FAIL (module chưa có).

- [ ] **Step 3: Implement `src/backtest/sub_universe.py`**

```python
"""Proxy robustness sub-universe (local): kiểm alpha còn giữ Sharpe khi giới hạn về nhóm mã
thanh khoản nhất — bắt chước sub-universe test của Brain (sub_sharpe ≥ 0.75·√(sub/univ)·sharpe).
Local không có nhiều universe; xấp xỉ bằng top `frac` mã theo thanh khoản (mean(volume*close))."""

from __future__ import annotations

import dataclasses

import numpy as np

from src.backtest.config import PortfolioConfig
from src.lang.ast import Node
from src.lang.registry import OperatorRegistry
from src.local_types import Panel


def _sub_universe_mask(data, frac: float) -> Panel:
    """Mask (T,N) bool: giữ top `frac` mã theo thanh khoản trung bình (volume*close)."""
    close = data.field("close")
    volume = data.field("volume")
    with np.errstate(invalid="ignore"):
        liq = np.nanmean(volume * close, axis=0)  # (N,)
    n = liq.shape[0]
    keep = max(1, int(round(n * frac)))
    order = np.argsort(-np.nan_to_num(liq, nan=-np.inf))
    top = set(order[:keep].tolist())
    col = np.array([i in top for i in range(n)], dtype=bool)
    return data.universe & col[None, :]


def sub_universe_ok(
    node: Node, config: PortfolioConfig, data, registry: OperatorRegistry, *,
    full_sharpe: float, frac: float = 0.5,
) -> bool:
    """True nếu Sharpe trên sub-universe đạt ngưỡng 0.75·√frac·full_sharpe. full_sharpe ≤ 0 -> True."""
    if full_sharpe is None or full_sharpe <= 0:
        return True
    from src.backtest.backtester import Backtester
    from src.backtest.metrics_local import MetricsCalculator
    from src.backtest.portfolio import PortfolioBuilder
    from src.engine.evaluator import EvalContext, Evaluator

    try:
        sub_data = dataclasses.replace(data, universe=_sub_universe_mask(data, frac))
        signal = Evaluator(EvalContext(data=sub_data, registry=registry, cache=None)).evaluate(node)
        if np.all(np.isnan(signal)):
            return False
        weights = PortfolioBuilder().build(signal, config, sub_data)
        result = Backtester().run(weights, sub_data)
        if not np.isfinite(result.daily_pnl).any():
            return False
        sub_sharpe = MetricsCalculator().compute(result, sub_data).sharpe
    except (KeyError, ValueError, ZeroDivisionError):
        return False
    if sub_sharpe is None or not np.isfinite(sub_sharpe):
        return False
    return sub_sharpe >= 0.75 * (frac ** 0.5) * full_sharpe
```

(Lưu ý: `MarketData` là frozen dataclass slots -> `dataclasses.replace(data, universe=...)` tạo bản mới hợp lệ; các mảng khác giữ nguyên.)

- [ ] **Step 4: Gate trong refiner (`src/app/closed_loop_adapters.py`)**

Trong `refine_and_sim`, SAU khi qua sàn turnover/floor và TRƯỚC khi sim Brain, thêm gate sub-universe trên winner:
```python
        from src.backtest.sub_universe import sub_universe_ok

        registry = self.registry or default_registry()
        if tr.local_metrics is not None and not sub_universe_ok(
            parse(tr.best_expr), tr.best_config, self.data, registry,
            full_sharpe=tr.local_metrics.sharpe,
        ):
            return IdeaOutcome(
                expr=tr.best_expr, canonical_hash=canonical_hash, passed=False,
                wq_alpha_id=None, sharpe=None, fitness=None, turnover=None,
                self_corr=None, sims_used=0, stop_reason="sub_universe",
            )
```
(Đặt sau khối `if tr.local_sharpe < self.min_local_sharpe:` return, trước khi dựng `sim_cfg`.)

- [ ] **Step 5: Chạy — xanh + hồi quy**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_sub_universe.py tests/unit/test_local_tuner_refiner.py tests/unit/test_closed_loop_local_refiner.py -q`
Expected: PASS. (Nếu test refiner cũ giả `data=object()` vỡ vì gate sub-universe eval thật, cho các test đó truyền `local_metrics=None` trong fake_tune HOẶC set data là panel nhỏ — sửa fixture tối thiểu, ghi rõ trong report.)

- [ ] **Step 6: Commit**

```bash
git add src/backtest/sub_universe.py src/app/closed_loop_adapters.py tests/unit/test_sub_universe.py
git commit -m "$(printf 'feat(gate): proxy robustness sub-universe truoc sim Brain\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01V7aUj8puXBx9HtUftNWGL9')"
```

---

### Task 5: Đổi default neutralization closed-loop -> MARKET + tích hợp

**Files:**
- Modify: `main.py` (`_run_closed_loop_session`/`closed_loop_cmd`/`_menu_auto_sim` default neutralization; refiner map neutralization Brain từ best_config)
- Modify: `src/app/closed_loop_adapters.py` (`sim_cfg` dùng `tr.best_config.neutralization.name`)
- Test: `tests/unit/test_closed_loop_local_refiner.py` (bổ sung: neutralization Brain = giá trị sweep)

**Interfaces:**
- Consumes: Task 1 (`TuneResult.best_config.neutralization`).

- [ ] **Step 1: Test đỏ (refiner áp neutralization đã tune vào SimConfig)**

Thêm vào `tests/unit/test_local_tuner_refiner.py`:
```python
def test_refiner_ap_neutralization_da_tune_vao_sim():
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.backtest.local_tuner import TuneResult

    seen = {}

    class _Sim:
        def simulate(self, expr, settings=None):
            seen["neut"] = settings.get("neutralization")
            return _passed_result(expr, settings)

    def fake_tune(expr, cfg, data, **kw):
        return TuneResult(
            best_expr="rank(ts_delta(close, 20))",
            best_config=PortfolioConfig(neutralization=Neutralization.SECTOR, decay=3, truncation=0.02),
            local_sharpe=1.6, local_metrics=None,
        )

    r = LocalTunerRefiner(
        simulator=_Sim(), repo=_Repo(), data=object(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(), tune_fn=fake_tune,
    )
    r.refine_and_sim(_cand())
    assert seen["neut"] == "SECTOR"   # neutralization tune -> áp vào Brain sim
```
(`local_metrics=None` để bỏ qua gate sub-universe Task 4 trong test này.)

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_local_tuner_refiner.py::test_refiner_ap_neutralization_da_tune_vao_sim -q`
Expected: FAIL (sim_cfg chưa override neutralization -> giữ SUBINDUSTRY default).

- [ ] **Step 3: Refiner áp neutralization (`src/app/closed_loop_adapters.py`)**

Sửa chỗ dựng `sim_cfg`:
```python
        sim_cfg = self.sim_config.with_overrides(
            decay=tr.best_config.decay, truncation=tr.best_config.truncation,
            neutralization=tr.best_config.neutralization.name,
        )
```

- [ ] **Step 4: Đổi default neutralization closed-loop (`main.py`)**

Trong `_run_closed_loop_session` đổi `neutralization: str = "SUBINDUSTRY"` -> `neutralization: str = "MARKET"`; trong `closed_loop_cmd` đổi `typer.Option("SUBINDUSTRY")` -> `typer.Option("MARKET", help="neutralization khoi diem (sweep se chon MARKET/SECTOR)")` nếu có option; `_menu_auto_sim` không cần đổi (dùng default hàm).

- [ ] **Step 5: Chạy — xanh + ast.parse + full suite**

Run: `venv/Scripts/python.exe -c "import ast; ast.parse(open('main.py',encoding='utf-8').read())" && venv/Scripts/python.exe -m pytest tests/unit/test_local_tuner_refiner.py tests/unit/test_closed_loop_local_refiner.py tests/ -q -k "closed_loop or tuner or power_pool or sub_universe or neut or turnover or menu"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add main.py src/app/closed_loop_adapters.py tests/unit/test_local_tuner_refiner.py
git commit -m "$(printf 'feat(closed-loop): default neutralization MARKET + ap neut da tune vao Brain sim\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01V7aUj8puXBx9HtUftNWGL9')"
```

---

## Xác minh cuối (sau 5 task)
- [ ] Full suite: `venv/Scripts/python.exe -m pytest -q` (chấp nhận 1 fail psycopg).
- [ ] Live (cần session Brain): `venv/Scripts/python.exe -u main.py closed-loop --market-data-dir data/market_yf --refiner local --max-ideas 8` — quan sát Sharpe (kỳ vọng cao hơn nhờ MARKET/SECTOR neut), TO trong dải, và có dòng "⭐ Power Pool eligible" không.

## Self-review coverage
Lever A → Task 1 (sweep) + Task 5 (map Brain + default). Lever B → Task 2. Lever C → Task 3. Lever D → Task 4. Không sweep INDUSTRY/SUBINDUSTRY (Global Constraints). Tương thích ngược: IdeaOutcome field mới có default; tune param mới có default.
