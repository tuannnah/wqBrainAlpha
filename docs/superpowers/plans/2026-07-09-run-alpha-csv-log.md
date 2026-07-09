# Run Alpha CSV Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mỗi phiên Auto SIM tự log tất cả ý tưởng (công thức + setting; Sharpe/fitness chỉ cho cái đạt) ra một file CSV riêng theo timestamp, để người dùng soi độ lặp công thức giữa các lần chạy.

**Architecture:** Thêm 2 field optional vào `IdeaOutcome` (settings + source) do refiner điền khi có sim; module ghi CSV `RunAlphaLogger`; `ClosedLoop` nhận logger optional gọi mỗi ý tưởng; `main` tạo logger per-run + truyền qua `build_closed_loop`.

**Tech Stack:** Python 3, csv (stdlib), pytest. KHÔNG thêm dependency (dùng .csv, không openpyxl).

## Global Constraints

- Code/comment/commit message: tiếng Việt CÓ ĐỦ DẤU (không bỏ dấu trong comment/docstring).
- TDD: đỏ→xanh, mỗi task ≥1 commit.
- File CSV mỗi phiên: `logs/alphas_<YYYY-MM-DD_HHMM>.csv`. Ghi header 1 lần, append + flush TỪNG dòng (Ctrl+C/hết quota vẫn giữ dữ liệu).
- Cột (đúng thứ tự): `#, status, source, expression, region, universe, delay, neutralization, decay, truncation, sharpe, fitness, turnover, self_corr, power_pool, wq_alpha_id, sims, stop_reason`.
- Quy tắc điền:
  - passed=True → điền đủ metrics.
  - passed=False (failed/error/gated) → `sharpe` và `fitness` để TRỐNG (chuỗi rỗng); các cột khác vẫn điền nếu có.
  - Không có `sim_settings` (ý tưởng bị gate 0 sim) → region/universe/delay/neutralization/decay/truncation để trống.
  - Giá trị None → ô trống.
- Test runner: `./venv/Scripts/python.exe -m pytest` (python hệ thống thiếu lark).

## File Structure

- `src/pipeline/closed_loop.py` — thêm field `sim_settings`/`source` vào `IdeaOutcome`; `ClosedLoop` nhận `alpha_logger`, gọi trong `run()` (Task 1 + Task 3).
- `src/app/closed_loop_adapters.py` — `_finalize` điền `sim_settings`/`source`; `build_closed_loop` truyền `alpha_logger` (Task 1 + Task 4).
- `src/reporting/run_alpha_log.py` — MỚI: `RunAlphaLogger` (Task 2).
- `main.py` — `_run_closed_loop_session` tạo logger per-run + truyền vào (Task 4).
- Test: `tests/unit/test_run_alpha_log.py` (mới), `tests/unit/test_closed_loop.py`, `tests/unit/test_power_pool_flag.py`.

---

### Task 1: IdeaOutcome mang sim_settings + source; refiner điền

**Files:**
- Modify: `src/pipeline/closed_loop.py:38-54` (dataclass IdeaOutcome)
- Modify: `src/app/closed_loop_adapters.py:194-199` (IdeaOutcome trong `_finalize`)
- Test: `tests/unit/test_power_pool_flag.py`

**Interfaces:**
- Produces: `IdeaOutcome(..., sim_settings: dict | None = None, source: str | None = None)`. `_finalize` trả outcome có `sim_settings=sim_cfg.to_settings()` và `source=<source param>`.

- [ ] **Step 1: Viết test đỏ**

Thêm vào `tests/unit/test_power_pool_flag.py`:

```python
def test_sim_direct_gan_sim_settings_va_source(monkeypatch):
    """Nhánh alt-data (_sim_direct) phải điền sim_settings (dict settings Brain) + source='alt_data'
    vào IdeaOutcome để logger CSV ghi được setting thật."""
    monkeypatch.setattr("src.backtest.sub_universe.sub_universe_ok", lambda *a, **kw: True)

    class _SimGia2:
        def simulate(self, expr, settings=None):
            return SimulationResult(
                expression=expr, alpha_id="wq-s", status="passed",
                sharpe=1.2, fitness=1.1, turnover=0.3, drawdown=0.1, raw={},
            )

    r = LocalTunerRefiner(
        simulator=_SimGia2(), repo=_RepoGia(), data=object(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(),
    )
    monkeypatch.setattr(r, "_is_alt_data", lambda expr: True)
    out = r.refine_and_sim(_cand_gia("ts_backfill(implied_volatility_call_30, 22)"))
    assert out.source == "alt_data"
    assert isinstance(out.sim_settings, dict)
    assert out.sim_settings["region"] == "USA"
    assert "neutralization" in out.sim_settings
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_power_pool_flag.py -k gan_sim_settings -q`
Expected: FAIL (`AttributeError: 'IdeaOutcome' object has no attribute 'source'`).

- [ ] **Step 3a: Thêm field vào IdeaOutcome** — `src/pipeline/closed_loop.py`, sau dòng `power_pool_eligible: bool = False`:

```python
    # Settings Brain thật đã dùng khi sim (dict từ SimConfig.to_settings) — None nếu ý tưởng bị
    # gate local chặn (0 sim). Nguồn ý tưởng (alt_data/gp_local_tuner...) để soi độ lặp công thức.
    sim_settings: dict | None = None
    source: str | None = None
```

- [ ] **Step 3b: Điền trong `_finalize`** — `src/app/closed_loop_adapters.py`, trong `return IdeaOutcome(...)` của `_finalize` (khối kết thúc bằng `power_pool_eligible=power_pool,`), thêm 2 dòng:

```python
            stop_reason=stop_reason, power_pool_eligible=power_pool,
            sim_settings=sim_cfg.to_settings(), source=source,
        )
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_power_pool_flag.py -q`
Expected: PASS (gồm test cũ — field mới có default nên không phá nơi tạo IdeaOutcome khác).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/closed_loop.py src/app/closed_loop_adapters.py tests/unit/test_power_pool_flag.py
git commit -m "feat(closed-loop): IdeaOutcome mang sim_settings + source cho log CSV"
```

---

### Task 2: RunAlphaLogger — ghi CSV per-run

**Files:**
- Create: `src/reporting/run_alpha_log.py`
- Create: `src/reporting/__init__.py` (nếu chưa có)
- Test: `tests/unit/test_run_alpha_log.py`

**Interfaces:**
- Consumes: `IdeaOutcome` (Task 1).
- Produces: `RunAlphaLogger(path: str | Path)` với `.log(index: int, outcome) -> None`. Tạo file + ghi header ngay khi khởi tạo; mỗi `.log` append + flush 1 dòng.

- [ ] **Step 1: Viết test đỏ**

Tạo `tests/unit/test_run_alpha_log.py`:

```python
from __future__ import annotations

import csv
from dataclasses import dataclass

from src.reporting.run_alpha_log import COLUMNS, RunAlphaLogger


@dataclass
class _FakeOutcome:
    expr: str
    passed: bool
    sharpe: float | None = None
    fitness: float | None = None
    turnover: float | None = None
    self_corr: float | None = None
    sims_used: int = 1
    stop_reason: str = ""
    power_pool_eligible: bool = False
    wq_alpha_id: str | None = None
    sim_settings: dict | None = None
    source: str | None = None


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def test_header_ghi_ngay_khi_khoi_tao(tmp_path):
    p = tmp_path / "a.csv"
    RunAlphaLogger(p)
    rows = _read(p)
    assert rows[0] == COLUMNS


def test_passed_ghi_du_metrics(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(1, _FakeOutcome(
        expr="rank(close)", passed=True, sharpe=1.5, fitness=1.1, turnover=0.3,
        self_corr=0.2, power_pool_eligible=True, wq_alpha_id="wq1", source="alt_data",
        sim_settings={"region": "USA", "universe": "TOP1000", "delay": 1,
                      "neutralization": "STATISTICAL", "decay": 3, "truncation": 0.02},
    ))
    rows = _read(p)
    d = dict(zip(rows[0], rows[1]))
    assert d["status"] == "passed"
    assert d["sharpe"] == "1.5" and d["fitness"] == "1.1"
    assert d["universe"] == "TOP1000" and d["neutralization"] == "STATISTICAL"
    assert d["source"] == "alt_data" and d["expression"] == "rank(close)"


def test_failed_de_trong_sharpe_fitness_nhung_giu_setting(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(2, _FakeOutcome(
        expr="rank(open)", passed=False, sharpe=1.04, fitness=0.71, turnover=0.28,
        stop_reason="alt_data_direct", source="alt_data",
        sim_settings={"region": "USA", "universe": "TOP1000", "delay": 1,
                      "neutralization": "CROWDING", "decay": 0, "truncation": 0.08},
    ))
    d = dict(zip(*_read(p)))
    assert d["status"] == "failed"
    assert d["sharpe"] == "" and d["fitness"] == ""       # để trống theo yêu cầu
    assert d["turnover"] == "0.28"                         # cột khác vẫn giữ
    assert d["neutralization"] == "CROWDING"


def test_gated_0sim_khong_co_setting(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(3, _FakeOutcome(
        expr="multiply(-1, ts_mean(close, 5))", passed=False, sims_used=0,
        stop_reason="local_floor", sim_settings=None, source=None,
    ))
    d = dict(zip(*_read(p)))
    assert d["status"] == "failed"
    assert d["universe"] == "" and d["neutralization"] == ""
    assert d["expression"] == "multiply(-1, ts_mean(close, 5))"
    assert d["stop_reason"] == "local_floor"


def test_append_nhieu_dong(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(1, _FakeOutcome(expr="a", passed=True, sharpe=1.0))
    lg.log(2, _FakeOutcome(expr="b", passed=False))
    rows = _read(p)
    assert len(rows) == 3  # header + 2
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_run_alpha_log.py -q`
Expected: FAIL (`ModuleNotFoundError: src.reporting.run_alpha_log`).

- [ ] **Step 3a: Tạo `src/reporting/__init__.py`** (rỗng, nếu chưa tồn tại):

```python
```

- [ ] **Step 3b: Tạo `src/reporting/run_alpha_log.py`**

```python
"""Ghi log mỗi phiên Auto SIM ra CSV: một dòng/ý tưởng (công thức + setting; Sharpe/fitness chỉ
cho ý tưởng ĐẠT) để người dùng soi độ lặp công thức giữa các lần chạy. Mỗi phiên một file
`logs/alphas_<timestamp>.csv`; append + flush từng dòng để Ctrl+C/hết quota vẫn giữ dữ liệu."""

from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = [
    "#", "status", "source", "expression", "region", "universe", "delay",
    "neutralization", "decay", "truncation", "sharpe", "fitness", "turnover",
    "self_corr", "power_pool", "wq_alpha_id", "sims", "stop_reason",
]


def _s(v) -> str:
    """None -> ô trống; còn lại -> str."""
    return "" if v is None else str(v)


class RunAlphaLogger:
    """Mở CSV per-run + ghi header ngay; `.log(index, outcome)` append 1 dòng và flush."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8-sig")
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNS)
        self._f.flush()

    def log(self, index: int, outcome) -> None:
        s = outcome.sim_settings or {}
        passed = bool(outcome.passed)
        status = "passed" if passed else ("error" if outcome.stop_reason == "error" else "failed")
        # passed -> điền Sharpe/fitness; không đạt -> để trống theo yêu cầu người dùng.
        sharpe = _s(outcome.sharpe) if passed else ""
        fitness = _s(outcome.fitness) if passed else ""
        self._w.writerow([
            index, status, _s(outcome.source), _s(outcome.expr),
            _s(s.get("region")), _s(s.get("universe")), _s(s.get("delay")),
            _s(s.get("neutralization")), _s(s.get("decay")), _s(s.get("truncation")),
            sharpe, fitness, _s(outcome.turnover), _s(outcome.self_corr),
            _s(getattr(outcome, "power_pool_eligible", False)),
            _s(outcome.wq_alpha_id), _s(outcome.sims_used), _s(outcome.stop_reason),
        ])
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_run_alpha_log.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reporting/__init__.py src/reporting/run_alpha_log.py tests/unit/test_run_alpha_log.py
git commit -m "feat(reporting): RunAlphaLogger ghi cong thuc+setting ra CSV per-run"
```

---

### Task 3: ClosedLoop gọi alpha_logger mỗi ý tưởng

**Files:**
- Modify: `src/pipeline/closed_loop.py:116-133` (`ClosedLoop.__init__`), `:183-197` (trong `run()`)
- Test: `tests/unit/test_closed_loop.py`

**Interfaces:**
- Consumes: `RunAlphaLogger.log` (Task 2).
- Produces: `ClosedLoop(..., alpha_logger=None)`; mỗi ý tưởng xử lý (có outcome) gọi `alpha_logger.log(ideas_tried, outcome)`. None → bỏ qua.

- [ ] **Step 1: Viết test đỏ**

Thêm vào `tests/unit/test_closed_loop.py` (dùng fake sẵn có trong file; nếu chưa có refiner/idea_source fake, tạo tối thiểu như dưới):

```python
def test_closed_loop_goi_alpha_logger_moi_y_tuong():
    """ClosedLoop gọi alpha_logger.log cho mỗi ý tưởng có outcome (index tăng dần)."""
    from src.pipeline.closed_loop import ClosedLoop, IdeaOutcome
    from src.pipeline.shortlist import ShortlistCandidate
    import numpy as np

    cand = ShortlistCandidate(
        expr="rank(close)", metrics=None, pnl=np.zeros(2),
        dates=np.arange("2020-01-01", "2020-01-03", dtype="datetime64[D]"),
    )

    class _Src:
        def __init__(self):
            self.done = False
        def next_batch(self):
            if self.done:
                return []
            self.done = True
            return [cand]

    class _Ref:
        def refine_and_sim(self, c):
            return IdeaOutcome(
                expr=c.expr, canonical_hash="h", passed=True, wq_alpha_id="wq",
                sharpe=1.5, fitness=1.1, turnover=0.3, self_corr=0.2, sims_used=1,
                stop_reason="ok",
            )

    class _Repo:
        def avoided_exprs(self): return set()
        def record_brain_sim(self, **kw): return None

    logged = []

    class _Logger:
        def log(self, index, outcome):
            logged.append((index, outcome.expr))

    cl = ClosedLoop(_Src(), _Ref(), _Repo(), max_ideas=1, alpha_logger=_Logger())
    cl.run()
    assert logged == [(1, "rank(close)")]
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop.py -k alpha_logger -q`
Expected: FAIL (`__init__() got an unexpected keyword argument 'alpha_logger'`).

- [ ] **Step 3a: Thêm tham số `alpha_logger`** — `src/pipeline/closed_loop.py`, trong `ClosedLoop.__init__`, thêm sau `calibration_tracker: CalibrationTracker | None = None,`:

```python
        calibration_tracker: CalibrationTracker | None = None,
        alpha_logger=None,
    ) -> None:
```

và gán (sau `self.calibration_tracker = calibration_tracker`):

```python
        self.alpha_logger = alpha_logger
```

- [ ] **Step 3b: Gọi logger trong `run()`** — sau `ideas_tried += 1` (ngay dưới `record_brain_sim(...)`), thêm:

```python
                ideas_tried += 1
                if self.alpha_logger is not None:
                    self.alpha_logger.log(ideas_tried, outcome)
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop.py -q`
Expected: PASS (gồm test cũ).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/closed_loop.py tests/unit/test_closed_loop.py
git commit -m "feat(closed-loop): ClosedLoop goi alpha_logger moi y tuong"
```

---

### Task 4: Wiring — build_closed_loop truyền qua + main tạo logger per-run

**Files:**
- Modify: `src/app/closed_loop_adapters.py:376-411` (`build_closed_loop`)
- Modify: `main.py` (`_run_closed_loop_session` — tạo logger + truyền vào `build_closed_loop`)
- Test: `tests/unit/test_run_alpha_log.py` (thêm test đặt tên file per-run)

**Interfaces:**
- Consumes: `RunAlphaLogger` (Task 2), `ClosedLoop(alpha_logger=)` (Task 3).
- Produces: `build_closed_loop(..., alpha_logger=None)` chuyển tiếp vào `ClosedLoop`; helper `run_log_path(now=None) -> Path` trả `logs/alphas_<YYYY-MM-DD_HHMM>.csv`.

- [ ] **Step 1: Viết test đỏ** (đặt tên file per-run) — thêm vào `tests/unit/test_run_alpha_log.py`:

```python
from datetime import datetime

from src.reporting.run_alpha_log import run_log_path


def test_run_log_path_theo_timestamp():
    p = run_log_path(datetime(2026, 7, 9, 16, 20))
    assert p.name == "alphas_2026-07-09_1620.csv"
    assert p.parent.name == "logs"
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_run_alpha_log.py -k run_log_path -q`
Expected: FAIL (`ImportError: run_log_path`).

- [ ] **Step 3a: Thêm `run_log_path` vào `src/reporting/run_alpha_log.py`** (sau `COLUMNS`):

```python
from datetime import datetime


def run_log_path(now: datetime | None = None, log_dir: str | Path = "logs") -> Path:
    """Đường dẫn file log per-run: <log_dir>/alphas_<YYYY-MM-DD_HHMM>.csv."""
    ts = (now or datetime.now()).strftime("%Y-%m-%d_%H%M")
    return Path(log_dir) / f"alphas_{ts}.csv"
```

(Gộp import `datetime` lên đầu file cùng `csv`/`pathlib` nếu muốn — miễn hợp lệ.)

- [ ] **Step 3b: `build_closed_loop` chuyển tiếp** — `src/app/closed_loop_adapters.py`, thêm `alpha_logger=None` vào chữ ký `build_closed_loop(...)` (cạnh `include_alt_data: bool = False,`) và truyền vào `ClosedLoop(...)`:

```python
    return ClosedLoop(
        idea_source=idea_source, refiner=refiner, repo=repo,  # type: ignore[arg-type]
        region=region, universe=universe, max_ideas=max_ideas,
        calibration_tracker=tracker, alpha_logger=alpha_logger,
    )
```

- [ ] **Step 3c: `main._run_closed_loop_session` tạo logger + truyền** — trong `main.py`, ngay trước `cl = build_closed_loop(`, thêm:

```python
    from src.reporting.run_alpha_log import RunAlphaLogger, run_log_path

    _log_path = run_log_path()
    _alpha_logger = RunAlphaLogger(_log_path)
    console.print(f"[cyan]📄 Log công thức alpha phiên này: {_log_path}[/cyan]")
```

và thêm `alpha_logger=_alpha_logger,` vào lời gọi `build_closed_loop(...)`.

- [ ] **Step 4: Chạy test + smoke import** — kỳ vọng PASS

Run:
```bash
./venv/Scripts/python.exe -m pytest tests/unit/test_run_alpha_log.py -q && ./venv/Scripts/python.exe -c "import main; print('import OK')"
```
Expected: PASS + `import OK`.

- [ ] **Step 5: Commit**

```bash
git add src/reporting/run_alpha_log.py src/app/closed_loop_adapters.py main.py tests/unit/test_run_alpha_log.py
git commit -m "feat(closed-loop): main tao RunAlphaLogger per-run + build_closed_loop chuyen tiep"
```

---

### Task 5: Regression

- [ ] **Step 1: Chạy suite liên quan + toàn bộ**

Run:
```bash
./venv/Scripts/python.exe -m pytest tests/unit/test_run_alpha_log.py tests/unit/test_closed_loop.py tests/unit/test_power_pool_flag.py -q
./venv/Scripts/python.exe -m pytest -q
```
Expected: suite liên quan PASS; toàn bộ PASS trừ `tests/test_db_postgres.py` (fail có sẵn: thiếu `psycopg`, không liên quan).

- [ ] **Step 2: Commit nếu có sửa vặt regression**

```bash
git add -A && git commit -m "test: xanh suite run alpha csv log"
```

---

## Self-Review

**Spec coverage:**
- IdeaOutcome mang settings+source → Task 1 ✅
- Ghi CSV (header, passed đủ metrics, non-pass trống sharpe/fitness, gated trống setting, append+flush) → Task 2 ✅
- ClosedLoop gọi logger mỗi ý tưởng (gồm gated vì mọi outcome đều qua đây) → Task 3 ✅
- File per-run theo timestamp + wiring main → Task 4 ✅
- Regression → Task 5 ✅

**Placeholder scan:** không có TBD; mọi step có code/command.

**Type consistency:** `sim_settings: dict | None`/`source: str | None` nhất quán Task 1→2→3; `RunAlphaLogger`/`run_log_path`/`COLUMNS` khớp Task 2↔4; `alpha_logger` khớp Task 3↔4.
