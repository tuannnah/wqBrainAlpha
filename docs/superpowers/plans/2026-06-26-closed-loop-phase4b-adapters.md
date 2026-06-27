# Closed-Loop Phase 4B — Adapters (`GPIdeaSource` + `RefinementLoopRefiner`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development hoặc
> superpowers:executing-plans. Phase 4B của feature "Vòng kín AI + MiniBrain" (spec
> `docs/superpowers/specs/2026-06-26-ai-minibrain-closed-loop-design.md`). Phase 1-3 + 4A
> (`run_from_seed`) ĐÃ xong trên nhánh `closed-loop-integration`.

**Goal:** Hai adapter ở tầng composition (`src/app/`, được phép phụ thuộc gp+llm+simulation):
`GPIdeaSource` (bọc `generate_many` với seed tăng dần → `_GeneratesIdeas` của ClosedLoop) và
`RefinementLoopRefiner` (bọc `RefinementLoop.run_from_seed` → `IdeaOutcome` của ClosedLoop).
Kèm mở rộng `LoopResult` mang đủ metric Brain để map. Sau 4B, `ClosedLoop` ráp được với
thành phần thật; 4C chỉ còn menu + `.env` + chạy + feedback a/d.

**Architecture:** `LoopResult` thêm 4 field (best_passed/best_alpha_id/best_metrics/
best_self_corr) populate ở `_refine_loop`. `src/app/closed_loop_adapters.py` chứa 2 adapter —
đây là composition layer (KHÔNG nằm trong `src/pipeline`, nên được phép import `src.gp`/
`src.llm`/`src.simulation`/`src.lang`); test bằng fake + small_panel.

**Tech Stack:** Python 3.12, pytest. Sửa `src/llm/loop.py`; tạo `src/app/`.

## Global Constraints

- Python 3.12; full type hints trên code mới; `ruff` clean; không unused import.
- **REGRESSION CỨNG (Task 1):** 5 file test loop + `test_loop_run_from_seed.py` phải xanh sau
  khi thêm field LoopResult.
- `src/pipeline/closed_loop.py` KHÔNG đổi (adapters ở `src/app/`, không phá B1 của pipeline).
- **Tiếng Việt giữ dấu đúng chính tả** trong docstring/comment mới.
- TDD: test trước (đỏ) → code (xanh) → commit. Mỗi task = 1 commit.

## Pre-condition (chữ ký thật đã xác minh)

```python
# src/llm/loop.py (sau 4A)
@dataclass
class LoopResult:  # best_candidate, best_vector, history, zoo_added, failures, sims_used, stop_reason
class RefinementLoop:
    def run_from_seed(self, expression: str, on_progress=None) -> LoopResult: ...
    def _refine_loop(self, best_cand, best_ev, research_direction, current_config, history, emit) -> LoopResult:
        # return cuối có `best_ev` trong scope: best_ev.passed(bool), best_ev.alpha_id(str|None),
        # best_ev.metrics(dict keys: sharpe/fitness/turnover/returns/drawdown/margin), best_ev.pool_corr(float|None)

# src/pipeline/closed_loop.py (Phase 2)
@dataclass(frozen=True, slots=True)
class IdeaOutcome:  # expr, canonical_hash, passed, wq_alpha_id, sharpe, fitness, turnover,
                    # self_corr, sims_used, stop_reason
class QuotaExhausted(Exception): ...

# src/pipeline/shortlist.py (Phase 8)
@dataclass(frozen=True, slots=True)
class ShortlistCandidate:  # expr, metrics, pnl, dates

# src/pipeline/runner.py (Phase 8)
def generate_many(gp_engine, cfg, data, top_k, max_corr, pool=None) -> list[ShortlistCandidate]:
    # gp_engine.run() -> re-score -> build_shortlist. gp_engine = GPEngine (Phase 7).

# src/gp/engine.py: GPEngine(data, repo, config, registry, *, pop_size, n_generations, seed)
# src/lang/parser.py: parse(expr)->Node ; src/lang/visitors.py: CanonicalHasher().visit(node)->str
```

## File Structure

- **Modify** `src/llm/loop.py` (~6 dòng): thêm 4 field LoopResult + populate ở `_refine_loop`.
- **Create** `src/app/__init__.py` (rỗng) + `src/app/closed_loop_adapters.py` (~70 dòng).
- **Modify** `tests/unit/test_loop_run_from_seed.py` (~20 dòng): test field mới.
- **Create** `tests/unit/test_closed_loop_adapters.py` (~120 dòng).

---

### Task 1: Mở rộng `LoopResult` mang metric Brain (best_passed/alpha_id/metrics/self_corr)

**Files:**
- Modify: `src/llm/loop.py`
- Test: `tests/unit/test_loop_run_from_seed.py`

**Interfaces:**
- Produces: `LoopResult` thêm `best_passed: bool = False`, `best_alpha_id: str | None = None`,
  `best_metrics: dict = field(default_factory=dict)`, `best_self_corr: float | None = None`.

- [ ] **Step 1: Viết test đỏ (thêm vào `tests/unit/test_loop_run_from_seed.py`)**

```python
def test_run_from_seed_loopresult_carries_brain_metrics() -> None:
    """LoopResult sau run_from_seed mang best_passed + best_metrics (sharpe/fitness/turnover)
    + best_alpha_id, để adapter map sang IdeaOutcome. Dùng fake simulator pass có metric."""
    loop = _make_loop(...)  # fake simulator trả SimulationResult passed, sharpe/fitness/turnover
    result = loop.run_from_seed("rank(close)")
    assert result.best_passed is True
    assert "sharpe" in result.best_metrics
    # alpha_id do repo.save_alpha trả (fake repo) — không None khi đã sim
    assert result.best_alpha_id is not None
```
> Ở fake `_make_loop`, đảm bảo fake simulator trả `SimulationResult(status="passed", ...)` có
> sharpe/fitness > sàn hard_filter để `passed=True`. Tái dùng pattern test 4A.

- [ ] **Step 2: Chạy test — FAIL** (`AttributeError: 'LoopResult' object has no attribute 'best_passed'`)

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_loop_run_from_seed.py::test_run_from_seed_loopresult_carries_brain_metrics -q
```

- [ ] **Step 3: Thêm 4 field vào `LoopResult` + populate ở `_refine_loop`**

Trong `@dataclass class LoopResult` (sau `stop_reason`):
```python
    best_passed: bool = False
    best_alpha_id: str | None = None
    best_metrics: dict = field(default_factory=dict)
    best_self_corr: float | None = None
```
Trong `_refine_loop`, ở `return LoopResult(...)` cuối (chỗ `best_ev` còn trong scope), thêm:
```python
            best_passed=best_ev.passed,
            best_alpha_id=best_ev.alpha_id,
            best_metrics=dict(best_ev.metrics),
            best_self_corr=best_ev.pool_corr,
```
(Các return `no_seed` giữ default — best_ev None nên không có metric, đúng.)

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_loop_run_from_seed.py -q
```

- [ ] **Step 5: REGRESSION (gate cứng)**

```bash
venv/Scripts/python.exe -m pytest tests/test_loop.py tests/test_loop_referee.py tests/test_loop_reseed.py tests/test_loop_seed.py tests/unit/test_loop_local_gate.py -q
```
Expected: tất cả PASS (thêm field default không phá construction cũ).

- [ ] **Step 6: ruff + kiểm dấu + commit**

```bash
venv/Scripts/python.exe -m ruff check src/llm/loop.py tests/unit/test_loop_run_from_seed.py
git add src/llm/loop.py tests/unit/test_loop_run_from_seed.py
git commit -m "feat(llm): LoopResult mang metric Brain (best_passed/alpha_id/metrics/self_corr)"
```

---

### Task 2: Adapters `GPIdeaSource` + `RefinementLoopRefiner` (`src/app/closed_loop_adapters.py`)

**Files:**
- Create: `src/app/__init__.py`, `src/app/closed_loop_adapters.py`
- Test: `tests/unit/test_closed_loop_adapters.py`

**Interfaces:**
- Consumes: `generate_many` (runner), `ShortlistCandidate` (shortlist), `IdeaOutcome`
  (closed_loop), `GPEngine` (gp.engine), `parse`/`CanonicalHasher` (lang), `LoopResult`
  (llm.loop) — chỉ để hiểu kiểu; adapter nhận `loop` đã dựng.
- Produces:
  ```python
  class RefinementLoopRefiner:
      def __init__(self, loop) -> None: ...   # loop: RefinementLoop (có .run_from_seed)
      def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome: ...

  class GPIdeaSource:
      def __init__(self, data, repo, config, registry, *, pop_size: int = 30,
                   n_generations: int = 3, base_seed: int = 42, top_k: int = 10,
                   max_corr: float = 0.70) -> None: ...
      def next_batch(self) -> list[ShortlistCandidate]: ...   # GPEngine seed mới mỗi lần
  ```

- [ ] **Step 1: Viết test đỏ `tests/unit/test_closed_loop_adapters.py`**

```python
"""Test adapter vòng kín: RefinementLoopRefiner map LoopResult->IdeaOutcome; GPIdeaSource bọc
generate_many với seed tăng dần. RefinementLoopRefiner test bằng fake loop (không AI/sim thật);
GPIdeaSource test trên small_panel + DB in-memory."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.operators_local  # noqa: F401
from src.app.closed_loop_adapters import GPIdeaSource, RefinementLoopRefiner
from src.backtest.config import Neutralization, PortfolioConfig
from src.lang.registry import default_registry
from src.pipeline.closed_loop import IdeaOutcome
from src.pipeline.shortlist import ShortlistCandidate
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


class _FakeLoopResult:
    def __init__(self) -> None:
        self.best_candidate = type("C", (), {"expression": "rank(close)"})()
        self.best_passed = True
        self.best_alpha_id = "WQ42"
        self.best_metrics = {"sharpe": 1.6, "fitness": 1.3, "turnover": 0.2}
        self.best_self_corr = 0.35
        self.sims_used = 3
        self.stop_reason = "patience"


class _FakeLoop:
    def __init__(self) -> None:
        self.seeds: list[str] = []

    def run_from_seed(self, expression: str, on_progress=None) -> _FakeLoopResult:
        self.seeds.append(expression)
        return _FakeLoopResult()


def _cand(expr: str) -> ShortlistCandidate:
    from src.backtest.metrics_local import AlphaMetrics
    m = AlphaMetrics(sharpe=1.0, annual_return=0.1, turnover=0.3, max_drawdown=0.05,
                     fitness=1.0, per_year_sharpe={2021: 1.0}, weight_concentration=0.05)
    d = (np.datetime64("2021-01-01") + np.arange(5)).astype("datetime64[D]")
    return ShortlistCandidate(expr=expr, metrics=m, pnl=np.ones(5), dates=d)


def test_refiner_maps_loopresult_to_ideaoutcome() -> None:
    refiner = RefinementLoopRefiner(_FakeLoop())
    outcome = refiner.refine_and_sim(_cand("rank(close)"))
    assert isinstance(outcome, IdeaOutcome)
    assert outcome.passed is True
    assert outcome.wq_alpha_id == "WQ42"
    assert outcome.sharpe == 1.6
    assert outcome.fitness == 1.3
    assert outcome.turnover == 0.2
    assert outcome.self_corr == 0.35
    assert outcome.sims_used == 3
    assert outcome.stop_reason == "patience"
    assert outcome.canonical_hash  # tính được từ expr (parse+CanonicalHasher), không rỗng


def test_refiner_seeds_loop_with_candidate_expr() -> None:
    loop = _FakeLoop()
    RefinementLoopRefiner(loop).refine_and_sim(_cand("ts_mean(close, 5)"))
    assert loop.seeds == ["ts_mean(close, 5)"]  # seed loop bằng đúng expr candidate


@pytest.fixture
def repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    return MiniBrainRepository(sessionmaker(bind=engine, future=True, expire_on_commit=False))


def test_gp_idea_source_yields_candidates_and_advances_seed(small_panel, repo) -> None:  # noqa: ANN001
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    src = GPIdeaSource(small_panel, repo, cfg, default_registry(),
                       pop_size=6, n_generations=0, base_seed=42, top_k=5, max_corr=0.99)
    b1 = src.next_batch()
    b2 = src.next_batch()
    assert all(isinstance(c, ShortlistCandidate) for c in b1)
    assert isinstance(b2, list)  # batch 2 dùng seed khác (42 -> 43), không crash
```

- [ ] **Step 2: Chạy test — FAIL** (`ModuleNotFoundError: src.app.closed_loop_adapters`)

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop_adapters.py -q
```

- [ ] **Step 3: Tạo `src/app/__init__.py` (rỗng) + `src/app/closed_loop_adapters.py`**

```python
# src/app/__init__.py
# Composition layer — được phép phụ thuộc gp/llm/simulation/pipeline.
```

```python
# src/app/closed_loop_adapters.py
"""Adapter nối thành phần thật vào ClosedLoop (Phase 2). Tầng composition: được phép import
src.gp/src.llm/src.pipeline/src.lang (khác src/pipeline vốn cấm src.llm/src.gp theo B1).

- RefinementLoopRefiner: bọc RefinementLoop.run_from_seed (4A) → IdeaOutcome.
- GPIdeaSource: bọc generate_many (Phase 8) với seed GPEngine tăng dần → nguồn ý tưởng."""

from __future__ import annotations

from src.gp.engine import GPEngine
from src.lang.parser import parse
from src.lang.visitors import CanonicalHasher
from src.pipeline.closed_loop import IdeaOutcome
from src.pipeline.runner import generate_many
from src.pipeline.shortlist import ShortlistCandidate


class RefinementLoopRefiner:
    """Bọc RefinementLoop: refine+sim một core (qua run_from_seed) → IdeaOutcome cho ClosedLoop."""

    def __init__(self, loop: object) -> None:
        self.loop = loop

    def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome:
        result = self.loop.run_from_seed(candidate.expr)  # type: ignore[attr-defined]
        best = result.best_candidate
        expr = best.expression if best is not None else candidate.expr
        canonical_hash = CanonicalHasher().visit(parse(expr))
        m = result.best_metrics or {}
        return IdeaOutcome(
            expr=expr, canonical_hash=canonical_hash, passed=bool(result.best_passed),
            wq_alpha_id=result.best_alpha_id, sharpe=m.get("sharpe"), fitness=m.get("fitness"),
            turnover=m.get("turnover"), self_corr=result.best_self_corr,
            sims_used=result.sims_used, stop_reason=result.stop_reason,
        )


class GPIdeaSource:
    """Nguồn ý tưởng cho ClosedLoop: mỗi next_batch() chạy GPEngine với seed MỚI (tăng dần để
    đa dạng) rồi rút short-list qua generate_many. Pool decorrelate lấy từ repo.load_pool()."""

    def __init__(
        self, data: object, repo: object, config: object, registry: object, *,
        pop_size: int = 30, n_generations: int = 3, base_seed: int = 42,
        top_k: int = 10, max_corr: float = 0.70,
    ) -> None:
        self.data = data
        self.repo = repo
        self.config = config
        self.registry = registry
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.base_seed = base_seed
        self.top_k = top_k
        self.max_corr = max_corr
        self._batch = 0

    def next_batch(self) -> list[ShortlistCandidate]:
        seed = self.base_seed + self._batch
        self._batch += 1
        engine = GPEngine(
            data=self.data, repo=self.repo, config=self.config, registry=self.registry,
            pop_size=self.pop_size, n_generations=self.n_generations, seed=seed,
        )
        pool = self.repo.load_pool() or None  # type: ignore[attr-defined]
        return generate_many(
            gp_engine=engine, cfg=self.config, data=self.data, top_k=self.top_k,
            max_corr=self.max_corr, pool=pool,
        )
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop_adapters.py -q
```
Expected: PASS (GPIdeaSource test có thể chậm vài giây do chạy GP thật trên small_panel).

- [ ] **Step 5: ruff + mypy + kiểm dấu tiếng Việt**

```bash
venv/Scripts/python.exe -m ruff check src/app/ tests/unit/test_closed_loop_adapters.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/app/closed_loop_adapters.py
```
Expected: ruff sạch. mypy: `loop: object`/`data: object` + `# type: ignore[attr-defined]` là
cách giữ adapter không phụ thuộc kiểu cứng RefinementLoop (test bằng fake); nếu mypy than
`m.get`/`result.*` do `object`, giữ `# type: ignore` hoặc dùng `Any`. Sạch trên phần khai báo.

- [ ] **Step 6: Commit**

```bash
git add src/app/__init__.py src/app/closed_loop_adapters.py tests/unit/test_closed_loop_adapters.py
git commit -m "feat(app): GPIdeaSource + RefinementLoopRefiner adapters cho ClosedLoop"
```

---

## Self-review

**Spec coverage (4B scope):**
- [x] LoopResult mang metric Brain để map — Task 1.
- [x] `RefinementLoopRefiner` map LoopResult→IdeaOutcome (bọc run_from_seed) — Task 2.
- [x] `GPIdeaSource` bọc generate_many seed tăng dần — Task 2.
- [~] QuotaExhausted detection (Brain hết quota) — 4C (cần tín hiệu Simulator thật; run_from_seed
  hiện nuốt lỗi sim → quota nhận biết ở tầng wiring 4C).
- [~] Menu + .env + feedback a/d — 4C.

**Placeholder scan:** Có chủ ý `_make_loop(...)` trong test Task 1 (phụ thuộc fake pattern 4A —
đọc test_loop_run_from_seed.py hiện có để tái dùng). Phần khác có code cụ thể.

**Type consistency:**
- `IdeaOutcome(expr, canonical_hash, passed, wq_alpha_id, sharpe, fitness, turnover,
  self_corr, sims_used, stop_reason)` — khớp Phase 2 def + mapping Task 2.
- `LoopResult` thêm field default — khớp populate `_refine_loop` + test Task 1.
- `RefinementLoopRefiner.refine_and_sim(candidate) -> IdeaOutcome` + `GPIdeaSource.next_batch()
  -> list[ShortlistCandidate]` — khớp Protocol `_RefinesIdea`/`_GeneratesIdeas` của ClosedLoop.
- `generate_many(gp_engine, cfg, data, top_k, max_corr, pool)` — khớp Phase 8.

**Risks / gotchas:**
1. `GPIdeaSource.next_batch()` chạy GPEngine thật mỗi lần — đắt; pop_size/n_generations nhỏ cho
   test. Trong vận hành thật, mỗi batch là một thế hệ ý tưởng mới (đúng "chạy đến hết quota").
2. `RefinementLoopRefiner` chưa ném `QuotaExhausted` — 4C wiring sẽ thêm bắt tín hiệu quota từ
   Simulator/client (run_from_seed hiện trả LoopResult kể cả khi sim lỗi).
3. canonical_hash tính qua parse(expr) — expr từ best_candidate.expression luôn parse được
   (đã qua prefilter/sim); nếu parse lỗi bất ngờ, để exception nổi (4C bọc) — chấp nhận v1.
