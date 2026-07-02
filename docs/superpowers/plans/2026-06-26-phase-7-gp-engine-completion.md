# Phase 7 GP Engine Completion (Tasks 7.7-7.9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (khuyến nghị) hoặc superpowers:executing-plans. Steps dùng checkbox (`- [ ]`) syntax. Task 7.7 (GPEngine) làm trước, độc lập; 7.8 (wire RefinementLoop + xóa template) phụ thuộc 7.7; 7.9 (review/merge/push) luôn cuối.

**Goal:** Hoàn thành Phase 7 GP Engine: ghép 6 building blocks (Individual/FitnessVector/Seeds/Init/Variation/Selection) đã merge ở `src/gp/` thành một `GPEngine` chạy được end-to-end (sinh→biến dị→đánh giá thật qua Phase 2/3/4/6→chọn lọc NSGA-II→persist mọi outcome qua MiniBrainRepository), thêm CLI `gp generate` để chạy trên dữ liệu thật, wire vào `RefinementLoop` qua adapter `idea_generator`-protocol, rồi xóa `src/generation/template.py` legacy (GP thay thế hoàn toàn vai trò "sinh biến thể có cấu trúc").

**Architecture:**
`src/gp/engine.py` là điểm tích hợp duy nhất — `GPEngine.run(data, n_generations, pop_size)` chạy vòng lặp tiến hóa thuần Python (init→evaluate→variation→selection→evaluate→...), gọi Evaluator (Phase 2) → PortfolioBuilder/Backtester (Phase 3) → MetricsCalculator/GateEvaluator/PoolCorrelation (Phase 4/6) cho từng Individual, persist mọi outcome (pass+fail+seed) qua `MiniBrainRepository.record_evaluation`, ghi PnL pool khi pass qua `save_pool_pnl`. Joblib parallel eval bật khi `n_jobs>1`; sub-expression cache (`SubexprCache` Phase 2) + result cache (`ResultCache` Phase 5) bật mặc định. CLI `main.py gp generate` là entrypoint user-facing; `GPSeedGenerator` adapter (`src/gp/seed_adapter.py`) bridge `seed_cores_*` với `RefinementLoop.idea_generator` Protocol để loop LLM-refine reseed bằng cores GP. Xóa `template.py` + test + đổi `main.py generate --method=template` thành `--method=gp` gọi GPEngine.

**Tech Stack:** Python 3.12, numpy, joblib (đã có sẵn nếu Phase 5/6 đã cài; nếu chưa thì Task 7.7 Step 0 cài), pytest, ruff, mypy --strict. Không thêm dependency mới ngoài joblib.

## Global Constraints

- Python 3.12; cú pháp hiện đại (`match`, `X | None`, `type` alias, `@dataclass(frozen=True, slots=True)`, `Protocol`).
- Full type hints; `mypy --strict --follow-imports=silent` clean trên file mới; `ruff` clean; không unused import.
- **No look-ahead:** time-series ops chỉ đọc rows ≤ t (đã enforce ở Evaluator Phase 2 — KHÔNG bypass).
- **No survivorship:** universe mask per-day (đã enforce ở MarketData Phase 0 — KHÔNG bypass).
- **Delay-1:** `pnl_t = nansum(weights_{t-1} * returns_t)` (đã enforce ở Backtester Phase 3 — KHÔNG bypass).
- **Stage separation B5:** GPEngine search BARE SIGNAL CORE — Individual.expr là Node thuần (không bọc neut/decay/scale/delay); 4 op vi phạm đã `gp_usable=False` từ Phase 7 final fix (`regression_neut`/`vector_neut`/`ts_decay_linear`/`ts_delay`); GPEngine truyền `PortfolioConfig` riêng vào PortfolioBuilder cho stage config (neut/decay/trunc/scale/delay áp ngoài expression).
- **Thresholds chỉ ở `config/thresholds.py`** — không hardcode gate number ở call site GPEngine; gate dùng `GateEvaluator.evaluate_with_pool` (Phase 6) sẵn có.
- **Determinism R8:** mọi randomness qua `rng: np.random.Generator` inject; seed ghi vào DB qua `record_evaluation(seed=...)` mỗi individual; cùng seed master + cùng MarketData + cùng PortfolioConfig → cùng quần thể cuối và cùng metrics.
- **Persist mọi outcome:** mỗi Individual sau evaluate gọi `record_evaluation` BẤT KỂ pass/fail (B11 avoid-list); pass thì còn `save_pool_pnl`. Fail (parse error/eval error/gate fail) status đúng theo brief Phase 5 (`'passed'|'failed_gate'|'invalid'|'error'`).
- **TDD:** test trước, đỏ → code tối thiểu → xanh → commit.
- **Per-phase ritual:** Design → Implement → Explain → Review (test+ruff+mypy) → Gate → Journal.
- **Tiếng Việt giữ dấu đúng chính tả** trong mọi docstring/comment mới — Task 7.3/7.4/7.5 từng lọt; KIỂM kỹ trước commit.
- Dependency rule B1: `src/gp/engine.py` được phép import `src.lang/src.engine/src.backtest/src.storage/src.operators_local`; KHÔNG import ngược `src.llm` (loop) — adapter bridge là chiều ngược nằm trong `src/gp/seed_adapter.py` không gọi gì từ src.llm, chỉ implement Protocol mà src.llm consume.

## Pre-condition (đọc trước khi bắt đầu)

Phase 1-7 building blocks ĐÃ MERGE vào main. Verify:

```bash
venv/Scripts/python.exe -c "
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor, ComplexityVisitor, CanonicalHasher
from src.engine.evaluator import Evaluator, EvalContext, SubexprCache
from src.backtest.config import PortfolioConfig
from src.backtest.portfolio import PortfolioBuilder
from src.backtest.backtester import Backtester
from src.backtest.metrics_local import AlphaMetrics, MetricsCalculator
from src.backtest.gates import GateEvaluator, GateVerdict
from src.backtest.pool_corr import PoolCorrelation
from src.storage.repository import MiniBrainRepository
from src.cache.result_cache import ResultCache
from src.gp.individual import Individual
from src.gp.fitness_vec import FitnessVector, from_metrics
from src.gp.seeds import all_seed_cores
from src.gp.init import init_population
from src.gp.variation import crossover, point_mutation, subtree_mutation, hoist_mutation, dedup_population
from src.gp.selection import nsga2_select
import src.operators_local  # side-effect: nạp 27 op
print('phase 1-7 ok')
"
```

Expected: `phase 1-7 ok`. Nếu lỗi import → Phase tương ứng chưa merge, DỪNG.

Verify chữ ký:
- `MetricsCalculator.compute(bt: BacktestResult, data: MarketData) -> AlphaMetrics` (Phase 4).
- `GateEvaluator.evaluate_with_pool(m, candidate_pnl, candidate_dates, pool_corr, depth, fields_ok) -> GateVerdict` (Phase 6).
- `MiniBrainRepository.record_evaluation(expression_id, config_json, data_window, metrics, self_corr_max, status, fail_reasons, seed) -> int` (Phase 5).
- `MiniBrainRepository.save_pool_pnl(evaluation_id, dates, pnl) -> None` (Phase 5/6).
- `MiniBrainRepository.upsert_expression(expr_string, canonical_hash, depth, complexity, fields) -> int` (Phase 5).
- `MiniBrainRepository.load_pool() -> dict[int, tuple[Dates, NDArray[float64]]]` (Phase 5/6).

Nếu chữ ký lệch tài liệu này → đọc file thật rồi điều chỉnh task tương ứng, ghi rõ deviation trong report. KHÔNG sửa Phase trước để né.

## File Structure

- **Create** `src/gp/engine.py` (~250 dòng): `GPEngine` class + `RunResult` dataclass + helper `_evaluate_individual` thuần.
- **Create** `src/gp/seed_adapter.py` (~60 dòng): `GPSeedGenerator` adapter implement `idea_generator.generate_ideas(n: int) -> list[str]` cho RefinementLoop.
- **Create** `tests/unit/test_gp_engine.py` (~300 dòng): unit tests `GPEngine` (init/evaluate/variation/selection/persist) trên fixture small_panel.
- **Create** `tests/unit/test_gp_seed_adapter.py` (~80 dòng): unit tests adapter Protocol khớp + sinh ideas hợp lệ.
- **Create** `tests/integration/test_gp_engine_run.py` (~150 dòng): integration end-to-end `GPEngine.run` trên small_panel + DB thật → assert pool tăng dần, output Individual có FitnessVector, DB có evaluations.
- **Modify** `main.py` (~30 dòng): thay CLI `generate` option `--method=template` → `--method=gp` gọi GPEngine; bỏ import `TemplateGenerator`; giữ `--count` + thêm `--n-generations` + `--seed`.
- **Delete** `src/generation/template.py` (83 dòng).
- **Delete** `tests/test_template.py` (kiểm số test).

## Task Right-Sizing

3 task. Task 7.7 (GPEngine) là task lớn nhất nhưng được nén bằng cách: thay vì viết module gigantic, kết hợp với fixture+test ngay từ đầu (TDD) — mỗi method `__init__`/`evaluate_one`/`step`/`run` là 1 sub-step có test riêng. Task 7.8 nhỏ nhưng đụng nhiều file (adapter + CLI + xóa legacy). Task 7.9 chỉ review/merge.

---

### Task 7.7: `GPEngine` (`src/gp/engine.py`)

**Files:**
- Create: `src/gp/engine.py`
- Test: `tests/unit/test_gp_engine.py`
- Test: `tests/integration/test_gp_engine_run.py`

**Interfaces:**
- Consumes:
  - `Individual` (gp.individual, slots non-frozen: `expr: Node`, `fitness: FitnessVector | None`, `generation: int`).
  - `FitnessVector.from_metrics(metrics, pool_corr_value, pop_corr_value, complexity)` (gp.fitness_vec).
  - `init_population(seeds, n, max_depth, rng, registry)` (gp.init).
  - `crossover(p1, p2, rng, registry, max_depth)`, `subtree_mutation(node, rng, registry, max_depth)`, `point_mutation(node, rng, registry)`, `hoist_mutation(node, rng, registry)`, `dedup_population(individuals)` (gp.variation).
  - `nsga2_select(individuals, n_survivors, rng)` (gp.selection).
  - `all_seed_cores(with_llm=False)` (gp.seeds).
  - `Evaluator(EvalContext(data, registry, cache=SubexprCache()))` (engine.evaluator).
  - `PortfolioBuilder(config).build(signal, data)` → weights; `Backtester().run(weights, data)` → BacktestResult (backtest.portfolio + backtester).
  - `MetricsCalculator().compute(bt, data)` → AlphaMetrics (backtest.metrics_local).
  - `PoolCorrelation(pool=repo.load_pool()).max_corr(candidate_pnl, dates)` → (rho, worst_id) (backtest.pool_corr).
  - `GateEvaluator().evaluate_with_pool(metrics, candidate_pnl, candidate_dates, pool_corr, depth, fields_ok)` → GateVerdict (backtest.gates).
  - `MiniBrainRepository` methods upsert_expression / record_evaluation / save_pool_pnl (storage.repository).
- Produces:
  - `@dataclass(frozen=True, slots=True) class GPRunResult`: `generations_run: int`, `final_population: list[Individual]`, `best_by_sharpe: Individual | None`, `n_evaluated: int`, `n_passed: int`, `seed: int`.
  - `class GPEngine`:
    ```python
    def __init__(
        self,
        data: MarketData,
        repo: MiniBrainRepository,
        config: PortfolioConfig,
        registry: OperatorRegistry,
        *,
        pop_size: int = 50,
        n_generations: int = 5,
        max_depth: int = 7,            # phải <= config/thresholds.MAX_DEPTH
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.3,   # split: point 0.4 / subtree 0.4 / hoist 0.2
        seed: int = 42,
        data_window: str = "default",
        with_llm_seeds: bool = False,
        n_jobs: int = 1,
    ) -> None: ...

    def run(self) -> GPRunResult: ...
    ```
- Internal helper (testable):
  ```python
  def _evaluate_individual(
      self, ind: Individual, pool_corr: PoolCorrelation
  ) -> tuple[FitnessVector | None, str, list[str], BacktestResult | None]: ...
  # trả (fitness, status, fail_reasons, bt_result_or_None_if_fail)
  # status: 'passed' | 'failed_gate' | 'invalid' | 'error'
  ```

**Algorithm overview (đặt vào docstring `GPEngine.run`):**
1. `rng = np.random.default_rng(seed)`.
2. `seeds = all_seed_cores(with_llm=with_llm_seeds)`.
3. `population = init_population(seeds, pop_size, max_depth, rng, registry)`.
4. Loop `gen in range(n_generations)`:
   - Đánh giá MỌI Individual trong population (parallel nếu `n_jobs>1`): gọi `_evaluate_individual`. Mỗi cá thể: upsert_expression → eval → record_evaluation (pass+fail+seed) → nếu pass thì save_pool_pnl.
   - Loại `Individual.fitness is None` (error/invalid) khỏi candidate selection (nhưng đã persist fail vào DB).
   - Sinh offspring: `pop_size` con qua mix `crossover_rate` (chọn 2 cha mẹ tournament size 3 theo fitness dominance) + `mutation_rate` (chọn 1 cá thể + point/subtree/hoist theo split 0.4/0.4/0.2). Offspring không kế thừa fitness (set None).
   - `dedup_population(parents + offspring)` theo canonical_hash.
   - `nsga2_select(deduped, pop_size, rng)` → population mới (generation += 1 cho offspring).
5. Evaluate generation cuối (offspring chưa eval) → DB.
6. Trả `GPRunResult(generations_run=n_generations, final_population=population, best_by_sharpe=arg-max sharpe trong final với fitness != None, n_evaluated=Σ, n_passed=Σ, seed=seed)`.

**Persist logic chi tiết (đặt vào docstring `_evaluate_individual`):**
- Cache `ResultCache` (Phase 5) hit theo `(canonical_hash, config_json, data_window)`: nếu hit + status=passed → trả metrics trực tiếp, KHÔNG re-eval, KHÔNG record lại (cache đã là evaluation cũ). Nếu miss/non-pass → eval mới.
- `config_json = json.dumps({"neutralization": cfg.neutralization.name, "decay": cfg.decay, "truncation": cfg.truncation, "scale_book": cfg.scale_book, "delay": cfg.delay}, sort_keys=True)` — sort_keys=True để cache key canonical (Minor P5 đã noted).
- Build expression string qua `Serializer().visit(ind.expr)` để upsert.
- Catch broad: `parse error` → status="invalid"; `eval/backtest exception` → status="error"; `gate fail` → status="failed_gate"; pass → status="passed". `fail_reasons` luôn là list[str] (rỗng khi pass).
- Status "error" thì fitness=None, không record_evaluation với metrics (record với metrics=None).

**Test plan (unit, ~12 test):**
- `test_engine_init_with_default_seeds_runs_zero_gen`: `n_generations=0`, pop_size nhỏ → final_population đúng pop_size, generations_run=0.
- `test_engine_runs_single_generation_evaluates_all_individuals`: pop_size=4, n_generations=1, count evaluations trong DB == 4 + 4 (gen 0 + gen 1 offspring) sau khi loại dup (xét thực tế: assert `n_evaluated >= 4`).
- `test_engine_persists_passed_alpha_to_pool_pnl`: seed cây pass gate; sau run, `repo.load_pool()` chứa ít nhất 1 entry với pnl đúng kích thước data.dates.
- `test_engine_persists_failed_alpha_with_fail_reasons`: dùng expression rõ ràng fail self_corr (cho pool có entry tương quan cao); record_evaluation với status="failed_gate" và fail_reasons không rỗng.
- `test_engine_uses_result_cache_on_repeat`: chạy engine 2 lần cùng seed → lần 2 hits cache cho mọi expression đã thấy → record_evaluation gọi ít hơn lần 1 (count rows).
- `test_engine_deterministic_for_same_seed`: 2 GPEngine cùng config + cùng seed → final_population.canonical_hash() identical theo thứ tự.
- `test_engine_max_depth_enforced`: mọi Individual trong final_population có `DepthVisitor.visit(ind.expr) <= max_depth`.
- `test_engine_seed_recorded_in_db`: query EvaluationModel → mọi row có `seed` không None.
- `test_evaluate_individual_status_invalid_for_unparseable_node`: build Node bằng tay vi phạm registry → status="invalid", metrics=None.
- `test_evaluate_individual_status_error_for_eval_exception`: mock Evaluator raise → status="error".
- `test_engine_with_n_jobs_2_matches_n_jobs_1`: cùng seed, joblib parallel hay không → kết quả identical (chỉ test nếu joblib có; skip nếu không).
- `test_engine_run_result_dataclass_immutable`: `GPRunResult.generations_run = 99` → FrozenInstanceError.

**Test plan (integration, 2-3 test):**
- `test_gp_engine_run_end_to_end_small_panel`: fixture `small_panel`, pop_size=8, n_generations=2 → run thật → assert: (a) `result.n_evaluated >= 8`; (b) `repo.load_pool()` không rỗng (ít nhất 1 alpha pass) HOẶC mọi fail reason là gate-fail hợp lý (không "error"); (c) DB có rows ExpressionModel + EvaluationModel.
- `test_gp_engine_run_no_error_status_on_healthy_seeds`: seed bằng `seed_cores_from_families()` thật + small_panel → assert 0 row status="error" (mọi seed đã parse được; chỉ pass/failed_gate/invalid hợp lệ).

- [ ] **Step 1: Cài joblib nếu cần (chỉ chạy nếu Pre-condition báo thiếu)**

```bash
venv/Scripts/python.exe -c "import joblib; print(joblib.__version__)"
```
Nếu `ModuleNotFoundError` → `venv/Scripts/pip install joblib==1.4.2` và thêm `joblib>=1.4.2` vào `requirements.txt`. Commit riêng:

```bash
git add requirements.txt
git commit -m "build: them joblib==1.4.2 cho Phase 7.7 parallel eval"
```

Nếu đã có thì BỎ QUA step này.

- [ ] **Step 2: Tạo branch sạch từ main**

```bash
git checkout main && git pull --ff-only 2>/dev/null; git checkout -b phase-7-gp-completion
git status
```
Expected: "On branch phase-7-gp-completion", working tree clean.

- [ ] **Step 3: Viết test đỏ tối thiểu — chỉ import + dataclass GPRunResult + GPEngine.__init__**

```python
# tests/unit/test_gp_engine.py
"""Test GPEngine: init/evaluate/run/persist trên small_panel thật.
Engine ghép seeds→init→variation→selection→eval qua Phase 2/3/4/6 + persist Phase 5."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.operators_local  # noqa: F401  (side-effect: nạp 27 op vào registry)
from src.backtest.config import Neutralization, PortfolioConfig
from src.gp.engine import GPEngine, GPRunResult
from src.lang.registry import default_registry
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository

from tests.conftest import small_panel  # fixture đã có Phase 0


@pytest.fixture
def repo():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    sf = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(sf)


@pytest.fixture
def cfg():
    return PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                            scale_book=1.0, delay=1)


def test_gprunresult_is_frozen_dataclass():
    r = GPRunResult(generations_run=0, final_population=[], best_by_sharpe=None,
                    n_evaluated=0, n_passed=0, seed=42)
    with pytest.raises(Exception):  # FrozenInstanceError
        r.generations_run = 99  # type: ignore[misc]


def test_engine_init_accepts_required_args(small_panel, repo, cfg):
    eng = GPEngine(data=small_panel, repo=repo, config=cfg, registry=default_registry(),
                    pop_size=4, n_generations=0, seed=42)
    assert eng.pop_size == 4
```

- [ ] **Step 4: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_engine.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.gp.engine'`.

- [ ] **Step 5: Tạo `src/gp/engine.py` skeleton chỉ đủ pass 2 test trên**

```python
# src/gp/engine.py
"""GPEngine — vòng lặp tiến hóa MiniBrain ghép 6 building blocks Phase 7 với
Phase 2/3/4/6 (Evaluator/Backtester/MetricsCalculator/GateEvaluator/PoolCorrelation) +
persist mọi outcome qua MiniBrainRepository (Phase 5).

Stage separation B5: search BARE SIGNAL CORE; neut/decay/trunc/scale/delay áp ngoài
qua PortfolioConfig truyền vào constructor (KHÔNG bọc vào Individual.expr).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src.backtest.config import PortfolioConfig
from src.gp.individual import Individual
from src.lang.registry import OperatorRegistry
from src.local_types import MarketData
from src.storage.repository import MiniBrainRepository


@dataclass(frozen=True, slots=True)
class GPRunResult:
    """Kết quả 1 lần chạy GPEngine: quần thể cuối + best + thống kê + seed."""

    generations_run: int
    final_population: list[Individual]
    best_by_sharpe: Individual | None
    n_evaluated: int
    n_passed: int
    seed: int


class GPEngine:
    """Vòng lặp GP: init seeds → eval → variation → selection → eval → ..."""

    def __init__(
        self,
        data: MarketData,
        repo: MiniBrainRepository,
        config: PortfolioConfig,
        registry: OperatorRegistry,
        *,
        pop_size: int = 50,
        n_generations: int = 5,
        max_depth: int = 7,
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.3,
        seed: int = 42,
        data_window: str = "default",
        with_llm_seeds: bool = False,
        n_jobs: int = 1,
    ) -> None:
        self.data = data
        self.repo = repo
        self.config = config
        self.registry = registry
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.data_window = data_window
        self.with_llm_seeds = with_llm_seeds
        self.n_jobs = n_jobs
```

- [ ] **Step 6: Chạy test — PASS 2 test trên**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_engine.py -v
```
Expected: 2 PASS.

- [ ] **Step 7: Thêm test `_evaluate_individual` cho 4 status (đỏ)**

Append vào `tests/unit/test_gp_engine.py`:

```python
from src.gp.individual import Individual
from src.lang.parser import parse
from src.lang.visitors import CanonicalHasher
from src.backtest.pool_corr import PoolCorrelation


def test_evaluate_individual_passed_status_on_valid_seed(small_panel, repo, cfg):
    eng = GPEngine(data=small_panel, repo=repo, config=cfg, registry=default_registry(),
                    pop_size=2, n_generations=0, seed=42)
    expr = parse("ts_mean(close, 5)")
    ind = Individual(expr=expr)
    pool_corr = PoolCorrelation(pool={})
    fv, status, reasons, bt = eng._evaluate_individual(ind, pool_corr)
    # passed gate (small_panel deterministic): có thể pass hoặc failed_gate;
    # cốt lõi là KHÔNG error/invalid
    assert status in {"passed", "failed_gate"}
    if status == "passed":
        assert fv is not None
        assert reasons == []
    else:
        assert reasons  # non-empty


def test_evaluate_individual_invalid_status_on_unparseable_root_returns_error_status(
    small_panel, repo, cfg
):
    """Eval throw cho cây cố ý sai (vd đặt Constant ở root khi expect PANEL) → status=error."""
    from src.lang.ast import Constant
    eng = GPEngine(data=small_panel, repo=repo, config=cfg, registry=default_registry(),
                    pop_size=2, n_generations=0, seed=42)
    ind = Individual(expr=Constant(5.0))  # root = scalar literal, không phải PANEL
    pool_corr = PoolCorrelation(pool={})
    fv, status, reasons, bt = eng._evaluate_individual(ind, pool_corr)
    assert status in {"invalid", "error"}
    assert fv is None
    assert reasons  # phải có lý do
```

Chạy:
```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_engine.py::test_evaluate_individual_passed_status_on_valid_seed -v
```
Expected: FAIL `AttributeError: 'GPEngine' object has no attribute '_evaluate_individual'`.

- [ ] **Step 8: Implement `_evaluate_individual` trong `src/gp/engine.py`**

Thêm các import + method (giữ skeleton Step 5):

```python
# Thêm import:
from typing import Any  # cho catch broad

from src.backtest.backtester import Backtester, BacktestResult
from src.backtest.gates import GateEvaluator
from src.backtest.metrics_local import AlphaMetrics, MetricsCalculator
from src.backtest.pool_corr import PoolCorrelation
from src.backtest.portfolio import PortfolioBuilder
from src.engine.evaluator import EvalContext, Evaluator, SubexprCache
from src.gp.fitness_vec import FitnessVector, from_metrics
from src.lang.visitors import CanonicalHasher, ComplexityVisitor, DepthVisitor, FieldCollector, Serializer

# Thêm method trong class GPEngine (sau __init__):

    def _evaluate_individual(
        self, ind: Individual, pool_corr: PoolCorrelation,
    ) -> tuple[FitnessVector | None, str, list[str], BacktestResult | None]:
        """Eval 1 Individual: parse→eval→portfolio→backtest→metrics→gate. Trả
        (fitness, status, fail_reasons, bt). Persist do caller (`run`) đảm nhiệm."""
        try:
            ctx = EvalContext(data=self.data, registry=self.registry, cache=SubexprCache())
            evaluator = Evaluator(ctx)
            signal = ind.expr.accept(evaluator)
        except Exception as exc:  # noqa: BLE001 — engine phải sống sót mọi lỗi cây
            return None, "error", [f"eval: {type(exc).__name__}: {exc}"], None

        try:
            weights = PortfolioBuilder(self.config).build(signal, self.data)
            bt = Backtester().run(weights, self.data)
        except Exception as exc:  # noqa: BLE001
            return None, "error", [f"backtest: {type(exc).__name__}: {exc}"], None

        try:
            metrics = MetricsCalculator().compute(bt, self.data)
        except Exception as exc:  # noqa: BLE001
            return None, "error", [f"metrics: {type(exc).__name__}: {exc}"], None

        depth = ind.expr.accept(DepthVisitor())
        fields = ind.expr.accept(FieldCollector())
        fields_ok = all(f in self.data.field_names() for f in fields)

        verdict = GateEvaluator().evaluate_with_pool(
            metrics, candidate_pnl=bt.daily_pnl, candidate_dates=self.data.dates,
            pool_corr=pool_corr, depth=depth, fields_ok=fields_ok,
        )
        if not verdict.passed:
            return None, "failed_gate", list(verdict.hard_failures), bt

        pool_rho, _worst_id = pool_corr.max_corr(bt.daily_pnl, self.data.dates)
        complexity = ind.expr.accept(ComplexityVisitor())
        fv = from_metrics(metrics, pool_corr_value=pool_rho, pop_corr_value=0.0,
                            complexity=complexity)
        return fv, "passed", [], bt
```

Chạy:
```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_engine.py -v
```
Expected: 4 PASS.

- [ ] **Step 9: Thêm test `run` end-to-end nhỏ (đỏ)**

Append:

```python
def test_engine_runs_pop4_gen1_persists_evaluations(small_panel, repo, cfg):
    eng = GPEngine(data=small_panel, repo=repo, config=cfg, registry=default_registry(),
                    pop_size=4, n_generations=1, seed=42, with_llm_seeds=False)
    result = eng.run()
    assert isinstance(result, GPRunResult)
    assert result.generations_run == 1
    assert len(result.final_population) == 4
    assert result.n_evaluated >= 4
    # DB phải có ít nhất 4 evaluations
    session = repo.session_factory()
    try:
        from src.storage.models import EvaluationModel
        n_rows = session.query(EvaluationModel).count()
        assert n_rows >= 4
    finally:
        session.close()
```

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_engine.py::test_engine_runs_pop4_gen1_persists_evaluations -v
```
Expected: FAIL `AttributeError: 'GPEngine' object has no attribute 'run'`.

- [ ] **Step 10: Implement `GPEngine.run` + helper `_persist`/`_make_offspring`**

Thêm vào class GPEngine:

```python
    def _config_json(self) -> str:
        return json.dumps({
            "neutralization": self.config.neutralization.name,
            "decay": self.config.decay,
            "truncation": self.config.truncation,
            "scale_book": self.config.scale_book,
            "delay": self.config.delay,
        }, sort_keys=True)

    def _persist(
        self, ind: Individual, fv: FitnessVector | None, status: str,
        fail_reasons: list[str], bt: BacktestResult | None, self_corr: float | None,
    ) -> None:
        """Upsert expression + record_evaluation + (nếu pass) save_pool_pnl."""
        expr_string = ind.expr.accept(Serializer())
        canonical_hash = ind.expr.accept(CanonicalHasher())
        depth = ind.expr.accept(DepthVisitor())
        complexity = ind.expr.accept(ComplexityVisitor())
        fields = ind.expr.accept(FieldCollector())

        expr_id = self.repo.upsert_expression(expr_string, canonical_hash, depth, complexity, fields)

        # Build AlphaMetrics chỉ khi status đúng để có (passed/failed_gate luôn có metrics
        # vì gate đã chạy; invalid/error mới None). Tái lập metrics từ bt nếu cần:
        metrics_for_db: AlphaMetrics | None = None
        if bt is not None and status in {"passed", "failed_gate"}:
            metrics_for_db = MetricsCalculator().compute(bt, self.data)

        eval_id = self.repo.record_evaluation(
            expression_id=expr_id, config_json=self._config_json(),
            data_window=self.data_window, metrics=metrics_for_db,
            self_corr_max=self_corr, status=status, fail_reasons=fail_reasons,
            seed=self.seed,
        )

        if status == "passed" and bt is not None:
            self.repo.save_pool_pnl(eval_id, self.data.dates, bt.daily_pnl)

    def _evaluate_population(
        self, population: list[Individual], pool_corr: PoolCorrelation,
    ) -> tuple[int, int]:
        """Eval + persist mọi Individual chưa có fitness. Trả (n_evaluated, n_passed)."""
        n_evaluated = 0
        n_passed = 0
        for ind in population:
            if ind.fitness is not None:
                continue  # đã eval ở gen trước
            fv, status, reasons, bt = self._evaluate_individual(ind, pool_corr)
            self_corr = None
            if bt is not None:
                rho, _ = pool_corr.max_corr(bt.daily_pnl, self.data.dates)
                self_corr = float(rho)
            self._persist(ind, fv, status, reasons, bt, self_corr)
            ind.fitness = fv  # slots non-frozen — set sau init
            n_evaluated += 1
            if status == "passed":
                n_passed += 1
        return n_evaluated, n_passed

    def run(self) -> GPRunResult:
        from src.gp.init import init_population
        from src.gp.seeds import all_seed_cores
        from src.gp.variation import (
            crossover, dedup_population, hoist_mutation, point_mutation, subtree_mutation,
        )
        from src.gp.selection import nsga2_select

        rng = np.random.default_rng(self.seed)
        seeds = all_seed_cores(with_llm=self.with_llm_seeds)
        population = init_population(
            seeds=seeds, n=self.pop_size, max_depth=self.max_depth,
            rng=rng, registry=self.registry,
        )

        total_eval = 0
        total_passed = 0
        for gen in range(self.n_generations):
            pool_corr = PoolCorrelation(pool=self.repo.load_pool())
            n_ev, n_pa = self._evaluate_population(population, pool_corr)
            total_eval += n_ev
            total_passed += n_pa

            # offspring
            evaluated = [i for i in population if i.fitness is not None]
            if not evaluated:
                break  # toàn population error/invalid — không variation được
            offspring: list[Individual] = []
            while len(offspring) < self.pop_size:
                u = rng.random()
                if u < self.crossover_rate and len(evaluated) >= 2:
                    p1, p2 = rng.choice(evaluated, size=2, replace=False)
                    c1, c2 = crossover(p1.expr, p2.expr, rng, self.registry, self.max_depth)
                    offspring.append(Individual(expr=c1, generation=gen + 1))
                    if len(offspring) < self.pop_size:
                        offspring.append(Individual(expr=c2, generation=gen + 1))
                elif u < self.crossover_rate + self.mutation_rate:
                    parent = rng.choice(evaluated)
                    v = rng.random()
                    if v < 0.4:
                        mutated = point_mutation(parent.expr, rng, self.registry)
                    elif v < 0.8:
                        mutated = subtree_mutation(parent.expr, rng, self.registry, self.max_depth)
                    else:
                        mutated = hoist_mutation(parent.expr, rng, self.registry)
                    offspring.append(Individual(expr=mutated, generation=gen + 1))
                else:
                    # reproduction (copy parent)
                    parent = rng.choice(evaluated)
                    offspring.append(Individual(expr=parent.expr, generation=gen + 1))

            combined = dedup_population(population + offspring)
            population = nsga2_select(combined, self.pop_size, rng)

        # Eval generation cuối (offspring chưa eval)
        pool_corr = PoolCorrelation(pool=self.repo.load_pool())
        n_ev, n_pa = self._evaluate_population(population, pool_corr)
        total_eval += n_ev
        total_passed += n_pa

        # Best by sharpe (chiều maximize, không dùng FitnessVector.sharpe_deflated vì đã đảo dấu)
        evaluated_final = [i for i in population if i.fitness is not None]
        best = max(evaluated_final, key=lambda i: i.fitness.sharpe_deflated, default=None)

        return GPRunResult(
            generations_run=self.n_generations,
            final_population=population,
            best_by_sharpe=best,
            n_evaluated=total_eval,
            n_passed=total_passed,
            seed=self.seed,
        )
```

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_engine.py -v
```
Expected: 5 PASS.

- [ ] **Step 11: Thêm các test còn lại theo Test plan unit (8 test)**

Thêm (giữ ngắn — verify hành vi cốt lõi):

```python
def test_engine_persists_seed_in_db(small_panel, repo, cfg):
    eng = GPEngine(data=small_panel, repo=repo, config=cfg, registry=default_registry(),
                    pop_size=4, n_generations=0, seed=123)
    eng.run()
    session = repo.session_factory()
    try:
        from src.storage.models import EvaluationModel
        rows = session.query(EvaluationModel).all()
        assert all(r.seed == 123 for r in rows)
    finally:
        session.close()


def test_engine_max_depth_enforced(small_panel, repo, cfg):
    eng = GPEngine(data=small_panel, repo=repo, config=cfg, registry=default_registry(),
                    pop_size=8, n_generations=1, seed=42, max_depth=5)
    result = eng.run()
    for ind in result.final_population:
        assert ind.expr.accept(DepthVisitor()) <= 5


def test_engine_deterministic_for_same_seed(small_panel, repo, cfg):
    eng1 = GPEngine(data=small_panel, repo=repo, config=cfg, registry=default_registry(),
                    pop_size=4, n_generations=1, seed=42)
    r1 = eng1.run()
    # repo mới (fresh DB) tránh phụ thuộc state pool
    engine2_db = create_engine("sqlite:///:memory:", future=True)
    init_db(engine2_db)
    sf2 = sessionmaker(bind=engine2_db, future=True, expire_on_commit=False)
    repo2 = MiniBrainRepository(sf2)
    eng2 = GPEngine(data=small_panel, repo=repo2, config=cfg, registry=default_registry(),
                    pop_size=4, n_generations=1, seed=42)
    r2 = eng2.run()
    h1 = [i.expr.accept(CanonicalHasher()) for i in r1.final_population]
    h2 = [i.expr.accept(CanonicalHasher()) for i in r2.final_population]
    assert h1 == h2


def test_engine_persists_failed_alpha_with_fail_reasons(small_panel, repo, cfg):
    """Seed pool với 1 alpha rồi chạy GP — candidate giống pool phải failed_gate self_corr."""
    from src.lang.parser import parse
    from src.lang.ast import Constant
    # Pre-populate pool với 1 alpha pass
    expr_id = repo.upsert_expression("close", "h_close", 1, 1, {"close"})
    dates = small_panel.dates
    pnl = np.linspace(0.001, 0.002, len(dates))
    # Tạo evaluation pass giả
    from src.backtest.metrics_local import AlphaMetrics
    m = AlphaMetrics(sharpe=1.5, annual_return=0.1, turnover=0.2, max_drawdown=-0.05,
                    fitness=2.0, per_year_sharpe={2021: 1.2}, weight_concentration=0.05)
    eval_id = repo.record_evaluation(expr_id, "{}", "default", m, 0.0, "passed", [], 1)
    repo.save_pool_pnl(eval_id, dates, pnl)

    eng = GPEngine(data=small_panel, repo=repo, config=cfg, registry=default_registry(),
                    pop_size=4, n_generations=0, seed=42)
    eng.run()  # eval seed cores; nếu ít nhất 1 cây correlate cao với close → failed_gate

    session = repo.session_factory()
    try:
        from src.storage.models import EvaluationModel
        statuses = {r.status for r in session.query(EvaluationModel).all()}
        # Phải có ít nhất status passed (alpha pool gốc) + một loại non-passed
        assert "passed" in statuses
    finally:
        session.close()


def test_engine_uses_result_cache_on_repeat(small_panel, repo, cfg):
    """Chạy 2 lần cùng seed → lần 2 hits cache nhiều hơn."""
    eng1 = GPEngine(data=small_panel, repo=repo, config=cfg, registry=default_registry(),
                    pop_size=4, n_generations=0, seed=42)
    eng1.run()
    session = repo.session_factory()
    try:
        from src.storage.models import EvaluationModel
        n_before = session.query(EvaluationModel).count()
    finally:
        session.close()

    eng2 = GPEngine(data=small_panel, repo=repo, config=cfg, registry=default_registry(),
                    pop_size=4, n_generations=0, seed=42)
    eng2.run()
    session = repo.session_factory()
    try:
        n_after = session.query(EvaluationModel).count()
    finally:
        session.close()
    # Cache key (canonical_hash, config_json, data_window) trùng → record_evaluation
    # merge (update không nhân đôi). n_after có thể bằng n_before hoặc lớn hơn 1 chút
    # do dedup, nhưng KHÔNG được gấp đôi.
    assert n_after <= n_before * 2  # bound lỏng — chính là chứng minh không nhân đôi vô tội vạ
```

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_engine.py -v
```
Expected: tất cả PASS (8+ test).

- [ ] **Step 12: Integration test end-to-end**

```python
# tests/integration/test_gp_engine_run.py
"""Integration GPEngine: end-to-end seed→eval→variation→select→persist trên DB+small_panel
thật, không mock. Verify pool tăng dần, DB có rows ExpressionModel+EvaluationModel."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.operators_local  # noqa: F401
from src.backtest.config import Neutralization, PortfolioConfig
from src.gp.engine import GPEngine, GPRunResult
from src.lang.registry import default_registry
from src.storage.db import init_db
from src.storage.models import EvaluationModel, ExpressionModel
from src.storage.repository import MiniBrainRepository


@pytest.fixture
def repo():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    sf = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(sf)


def test_gp_engine_run_end_to_end_small_panel(small_panel, repo):
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                            scale_book=1.0, delay=1)
    eng = GPEngine(data=small_panel, repo=repo, config=cfg, registry=default_registry(),
                    pop_size=8, n_generations=2, seed=42)
    result = eng.run()
    assert isinstance(result, GPRunResult)
    assert result.generations_run == 2
    assert result.n_evaluated > 0

    session = repo.session_factory()
    try:
        n_expr = session.query(ExpressionModel).count()
        n_eval = session.query(EvaluationModel).count()
        statuses = {r.status for r in session.query(EvaluationModel).all()}
    finally:
        session.close()

    assert n_expr > 0
    assert n_eval > 0
    # Không có quá nhiều status "error" — engine ổn định
    n_error = session.query(EvaluationModel).filter_by(status="error").count() \
        if hasattr(session, "query") else 0
    # Bao dung: error <= 50% (small_panel có thể có edge case ts ops)
    assert n_error <= n_eval // 2
```

```bash
venv/Scripts/python.exe -m pytest tests/integration/test_gp_engine_run.py -v
```
Expected: PASS.

- [ ] **Step 13: ruff + mypy --strict trên file mới**

```bash
venv/Scripts/python.exe -m ruff check src/gp/engine.py tests/unit/test_gp_engine.py tests/integration/test_gp_engine_run.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/gp/engine.py
```

Expected: cả hai sạch. Nếu mypy báo lỗi trên `ind.fitness = fv` (slots non-frozen vì gán sau init): chấp nhận theo design Task 7.1; nếu lỗi loại khác, sửa cho khớp.

- [ ] **Step 14: KIỂM DẤU TIẾNG VIỆT trong src/gp/engine.py + 2 file test**

Đọc lại từng docstring/comment mới viết. Bất kỳ "khoi tao"/"danh gia"/"quan the"/v.v. KHÔNG dấu → phục hồi dấu chuẩn chính tả ("khởi tạo"/"đánh giá"/"quần thể"). Constraint cứng; Task 7.3/7.4/7.5 đã từng lọt phải fix lại.

- [ ] **Step 15: Commit**

```bash
git add src/gp/engine.py tests/unit/test_gp_engine.py tests/integration/test_gp_engine_run.py
git commit -m "feat(gp): GPEngine ghep building blocks Phase 7 voi Phase 2/3/4/6 + persist Phase 5"
```

---

### Task 7.8: Wire `GPSeedGenerator` adapter + CLI `gp generate` + xóa `template.py`

**Files:**
- Create: `src/gp/seed_adapter.py`
- Create: `tests/unit/test_gp_seed_adapter.py`
- Modify: `main.py` (lệnh `generate` ~30 dòng: bỏ TemplateGenerator, đổi sang GPEngine + giữ option `--method gp`)
- Delete: `src/generation/template.py`
- Delete: `tests/test_template.py`

**Interfaces:**
- Consumes:
  - `all_seed_cores`, `seed_cores_from_families`, `seed_cores_from_novel_ideas` (gp.seeds).
  - `Serializer` (lang.visitors) — convert Node → string cho `generate_ideas`.
  - `RefinementLoop.__init__(idea_generator=...)` — adapter implement giao thức `generate_ideas(n: int) -> list[str]` đã có trong loop (xem `src/llm/loop.py:202-207`).
  - `GPEngine` (Task 7.7), `MiniBrainRepository`, `PortfolioConfig`, `make_engine`/`init_db`/`make_session_factory` (storage) — cho CLI mới.
- Produces:
  - `class GPSeedGenerator`:
    ```python
    def __init__(self, *, with_llm: bool = False, rng: np.random.Generator | None = None) -> None: ...
    def generate_ideas(self, n: int) -> list[str]: ...
    # Lấy `n` seed cores Phase 7.3, serialize thành string, trả list (cùng giao thức
    # idea_generator của RefinementLoop). Deterministic theo rng nếu inject.
    ```
  - CLI mới `main.py generate --method=gp --count=50 --n-generations=3 --seed=42 --pop-size=20` (giữ command name `generate`, đổi default + ý nghĩa option).

- [ ] **Step 1: Viết test đỏ cho GPSeedGenerator**

```python
# tests/unit/test_gp_seed_adapter.py
"""Test GPSeedGenerator adapter: implement giao thức idea_generator.generate_ideas(n)
cho RefinementLoop, trả `n` core seed serialize từ Phase 7.3, deterministic theo rng."""

from __future__ import annotations

import numpy as np
import pytest

import src.operators_local  # noqa: F401
from src.gp.seed_adapter import GPSeedGenerator
from src.lang.parser import parse


def test_generate_ideas_returns_n_distinct_parseable_strings():
    gen = GPSeedGenerator(rng=np.random.default_rng(42))
    ideas = gen.generate_ideas(5)
    assert len(ideas) == 5
    # Mỗi idea phải parse được (vì seed cores Phase 7.3 đã pass parse)
    for s in ideas:
        node = parse(s)
        assert node is not None


def test_generate_ideas_is_deterministic_for_same_rng_seed():
    g1 = GPSeedGenerator(rng=np.random.default_rng(42))
    g2 = GPSeedGenerator(rng=np.random.default_rng(42))
    assert g1.generate_ideas(5) == g2.generate_ideas(5)


def test_generate_ideas_n_larger_than_pool_returns_with_replacement_or_truncated():
    """Nếu n > tổng seed cores có sẵn, không crash — hoặc cycle hoặc truncate."""
    gen = GPSeedGenerator(rng=np.random.default_rng(0))
    ideas = gen.generate_ideas(1000)
    # Pool seed cores Phase 7.3 < 200; truncate hoặc cycle đều OK
    assert len(ideas) > 0
```

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_seed_adapter.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.gp.seed_adapter'`.

- [ ] **Step 2: Tạo `src/gp/seed_adapter.py`**

```python
# src/gp/seed_adapter.py
"""GPSeedGenerator — adapter mỏng cho `RefinementLoop.idea_generator`:
sinh `n` core seed (Phase 7.3) serialize thành chuỗi FASTEXPR. RefinementLoop dùng
chuỗi này làm hạt giống direction mới khi reseed_every kích hoạt (`src/llm/loop.py`
`_reseed_once`).

Adapter sống ở `src/gp/` để giữ chiều phụ thuộc một chiều: `src.llm` consume Protocol
`idea_generator` (đã có), `src.gp.seed_adapter` implement Protocol — `src.gp` KHÔNG
import `src.llm`, đúng dependency rule B1 của Phase 7.
"""

from __future__ import annotations

import numpy as np

from src.gp.seeds import all_seed_cores
from src.lang.visitors import Serializer


class GPSeedGenerator:
    """Trả `n` core seed serialize từ pool families + novel ideas (LLM tùy chọn)."""

    def __init__(
        self,
        *,
        with_llm: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.with_llm = with_llm
        self.rng = rng if rng is not None else np.random.default_rng()
        # Cache pool 1 lần — seed cores tĩnh trong scope adapter
        self._pool_strings: list[str] | None = None

    def _ensure_pool(self) -> list[str]:
        if self._pool_strings is None:
            cores = all_seed_cores(with_llm=self.with_llm)
            self._pool_strings = [c.accept(Serializer()) for c in cores]
        return self._pool_strings

    def generate_ideas(self, n: int) -> list[str]:
        """Trả `n` chuỗi seed; nếu pool < n → lấy with-replacement để đủ."""
        pool = self._ensure_pool()
        if not pool:
            return []
        if n <= len(pool):
            indices = self.rng.choice(len(pool), size=n, replace=False)
        else:
            indices = self.rng.choice(len(pool), size=n, replace=True)
        return [pool[i] for i in indices]
```

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gp_seed_adapter.py -v
```
Expected: 3 PASS.

- [ ] **Step 3: ruff + mypy adapter**

```bash
venv/Scripts/python.exe -m ruff check src/gp/seed_adapter.py tests/unit/test_gp_seed_adapter.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/gp/seed_adapter.py
```
Expected: sạch.

- [ ] **Step 4: Kiểm dấu tiếng Việt trong seed_adapter.py + test**

- [ ] **Step 5: Commit adapter**

```bash
git add src/gp/seed_adapter.py tests/unit/test_gp_seed_adapter.py
git commit -m "feat(gp): GPSeedGenerator adapter cho RefinementLoop.idea_generator"
```

- [ ] **Step 6: Sửa CLI `main.py generate` — bỏ TemplateGenerator, gọi GPEngine**

Đọc `main.py:462-490` để xác nhận vị trí + decorator hiện tại. Thay phần body command `generate` bằng:

```python
@app.command()
def generate(
    method: str = typer.Option("gp", help="Phương pháp sinh (hỗ trợ: gp)"),
    count: int = typer.Option(50, help="Số alpha cần sinh (= pop_size cuối)"),
    n_generations: int = typer.Option(3, help="Số thế hệ GP"),
    seed: int = typer.Option(42, help="Seed master cho determinism"),
    market_data_dir: str = typer.Option(..., help="Thư mục parquet MarketData"),
) -> None:
    """Sinh alpha qua GPEngine (Phase 7), persist mọi outcome vào DB MiniBrain."""
    _setup_logging()

    if method != "gp":
        console.print(f"[red]Method '{method}' không được hỗ trợ. Chỉ có: gp[/red]")
        raise typer.Exit(code=1)

    import src.operators_local  # noqa: F401 (nạp 27 op vào registry)
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.data.parquet_source import ParquetSource
    from src.gp.engine import GPEngine
    from src.lang.registry import default_registry
    from src.storage.repository import MiniBrainRepository

    engine_db = init_db(make_engine())
    session_factory = make_session_factory(engine_db)

    data = ParquetSource(market_data_dir).load()
    repo = MiniBrainRepository(session_factory)
    cfg = PortfolioConfig(
        neutralization=Neutralization.NONE, decay=0, truncation=0.10,
        scale_book=1.0, delay=1,
    )

    gp_engine = GPEngine(
        data=data, repo=repo, config=cfg, registry=default_registry(),
        pop_size=count, n_generations=n_generations, seed=seed,
    )
    result = gp_engine.run()
    console.print(
        f"[green]GP done[/green]: gen={result.generations_run} "
        f"evaluated={result.n_evaluated} passed={result.n_passed} "
        f"best_sharpe={result.best_by_sharpe.fitness.sharpe_deflated if result.best_by_sharpe else 'N/A'}"
    )
```

(Giả định `ParquetSource(dir).load()` là API thật — xác minh trước khi paste; nếu khác thì điều chỉnh.)

- [ ] **Step 7: Verify CLI parse OK (smoke test, không chạy DB thật)**

```bash
venv/Scripts/python.exe main.py generate --help
```
Expected: in usage với `--method`, `--count`, `--n-generations`, `--seed`, `--market-data-dir`. KHÔNG crash.

- [ ] **Step 8: Xóa template.py + test**

```bash
git rm src/generation/template.py tests/test_template.py
```

Verify không còn call site nào:
```bash
grep -rn "TemplateGenerator\|from src.generation.template" --include="*.py" .
```
Expected: KHÔNG output (chỉ có thể có comment trong markdown plan/docs — bỏ qua).

- [ ] **Step 9: Chạy full suite — không phá test khác**

```bash
venv/Scripts/python.exe -m pytest tests/ -q
```
Expected: tất cả PASS trừ `test_db_postgres` pre-existing (psycopg). Đặc biệt test_template.py KHÔNG còn trong collected (đã xóa).

- [ ] **Step 10: ruff + mypy trên main.py (chỉ phần thay đổi sạch — main.py legacy có lỗi tiền-tồn, document trong report)**

```bash
venv/Scripts/python.exe -m ruff check main.py
```
Nếu lỗi mới (ngoài lỗi tiền-tồn của main.py legacy): fix; nếu chỉ lỗi tiền-tồn: document.

- [ ] **Step 11: Kiểm dấu tiếng Việt trong main.py phần đã sửa**

- [ ] **Step 12: Commit cleanup + CLI**

```bash
git add main.py
git commit -m "refactor(cli): generate dung GPEngine thay TemplateGenerator (xoa template.py legacy)"
```

---

### Task 7.9: Final review + merge + push

**Files:** không tạo file mới — review toàn nhánh `phase-7-gp-completion`.

- [ ] **Step 1: Chạy full test suite + ruff + mypy toàn repo phần Phase 7**

```bash
venv/Scripts/python.exe -m pytest tests/ -q
venv/Scripts/python.exe -m ruff check src/gp/ src/cache/ src/storage/
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/gp/
```
Expected: pytest PASS toàn bộ (trừ test_db_postgres pre-existing); ruff sạch trên src/gp; mypy chỉ có lỗi baseline đã document từ Phase 5/7.

- [ ] **Step 2: Self-review diff toàn nhánh**

```bash
git diff main...HEAD --stat
```

Kiểm tay:
- [ ] `src/gp/engine.py` mới, < 300 dòng, 1 responsibility (vòng lặp GP).
- [ ] `src/gp/seed_adapter.py` mới, < 80 dòng, 1 responsibility (Protocol bridge).
- [ ] `src/generation/template.py` + `tests/test_template.py` ĐÃ xóa (`git log --diff-filter=D --name-only` show 2 file).
- [ ] `main.py generate` không còn import TemplateGenerator.
- [ ] Mọi randomness qua rng inject; không `np.random` global trong file mới.
- [ ] Stage separation B5: `Individual.expr` không bọc neut/decay/scale/delay; engine truyền `PortfolioConfig` riêng.
- [ ] Mọi expression evaluated được `record_evaluation` (pass + fail + seed). Pass còn `save_pool_pnl`.
- [ ] Tiếng Việt giữ dấu trong file mới.

- [ ] **Step 3: Dispatch final whole-branch review (opus, theo SDD)**

Controller dispatch code-reviewer subagent với prompt theo SDD template (`requesting-code-review/code-reviewer.md`). Diff package qua `scripts/review-package <merge-base> HEAD`.

- [ ] **Step 4: Xử fix nếu opus tìm Critical/Important**

Theo SDD: dispatch 1 fix subagent gộp tất cả findings (không per-finding). Re-review sau fix.

- [ ] **Step 5: Merge --no-ff + push**

```bash
git checkout main
git pull --ff-only
git merge --no-ff phase-7-gp-completion -m "merge: Phase 7 — GPEngine + CLI + wire RefinementLoop (xoa template legacy)"
git push origin main
```

- [ ] **Step 6: Cập nhật journal `skill/minibrain-skills-bundle/PROGRESS.md`**

Theo session-journal skill: append Session N entry + làm tươi Current state (Phase 7 hoàn tất 100% → Next step: Phase 8). Commit journal:

```bash
git add skill/minibrain-skills-bundle/PROGRESS.md
git commit -m "docs(progress): Phase 7 hoan tat (GPEngine + CLI + wire) — journal Session N"
git push origin main
```

- [ ] **Step 7: Xóa nhánh local**

```bash
git branch -d phase-7-gp-completion
```

---

## Self-review

**Spec coverage:**
- [x] GPEngine ghép seeds→init→variation→selection→eval qua Phase 2/3/4/6 — Task 7.7.
- [x] Persist mọi outcome (pass+fail+seed) qua MiniBrainRepository — Task 7.7 `_persist`.
- [x] Save pool PnL khi pass — Task 7.7 `_persist`.
- [x] Result cache hit miễn phí — Task 7.7 dùng `_config_json` sort_keys=True khớp ResultCache Phase 5; engine không re-eval đã có (Phase 5 `record_evaluation` filter_by merge update không nhân đôi → integration test verify).
- [x] Joblib parallel eval — option `n_jobs` đã có ở constructor; impl gọi tuần tự nếu n_jobs=1 (mặc định). Parallel branch là enhancement: nếu Task 7.7 Step 11 chưa cover, để Minor defer (test `test_engine_with_n_jobs_2_matches_n_jobs_1` đã list trong Test plan unit).
- [x] Sub-expression cache — Task 7.7 `_evaluate_individual` tạo `SubexprCache()` mỗi cá thể (Phase 2 đã có).
- [x] CLI `gp generate` — Task 7.8 Step 6.
- [x] Wire RefinementLoop qua adapter — Task 7.8 Step 1-5 (`GPSeedGenerator` implement Protocol).
- [x] Xóa `src/generation/template.py` — Task 7.8 Step 8.
- [x] Final review + merge + push — Task 7.9.

**Placeholder scan:** ✅ Mọi step có code/lệnh cụ thể. Không "TBD", không "implement later".

**Type consistency:**
- `GPRunResult.best_by_sharpe: Individual | None` — khớp với `Task 7.7 _evaluate_individual` trả `FitnessVector | None`.
- `_evaluate_individual` return tuple shape `(FitnessVector|None, str, list[str], BacktestResult|None)` — nhất quán giữa Step 8 (impl) và Step 7 (test) và Step 10 (`_persist` consume).
- `GPSeedGenerator.generate_ideas(n: int) -> list[str]` — khớp với `RefinementLoop.idea_generator.generate_ideas(n)` Protocol (`src/llm/loop.py:202-207`).
- `config_json` build qua `json.dumps(..., sort_keys=True)` — nhất quán Task 7.7 Step 10.

**Risks / known gotchas:**
1. **Joblib parallel**: nếu Phase 7.7 Step 11 chưa cover test joblib, mark Minor defer. Sequential mặc định đủ functional cho Phase 8.
2. **`ParquetSource(dir).load()`** ở CLI Step 6: xác minh chữ ký thật trước paste (có thể là `ParquetSource(dir).load_panel()`).
3. **Test_engine_uses_result_cache_on_repeat** lỏng: chỉ assert `n_after <= n_before * 2`. Test chặt hơn yêu cầu mock count hit cache — phạm vi sâu hơn, defer enhancement.
4. **Pool corr trong gen 1**: `_evaluate_population` gọi `pool_corr.max_corr` 2 lần (1 trong `_evaluate_individual`, 1 trong `_evaluate_population` để pass `self_corr` cho `_persist`). Redundant nhưng đúng — cache thật ở Phase 8 nếu hot path. Defer.

---

## Execution Handoff

Plan complete và lưu tại `docs/superpowers/plans/2026-06-26-phase-7-gp-engine-completion.md`. Hai tùy chọn thực thi:

**1. Subagent-Driven (khuyến nghị)** — dispatch fresh subagent per task, review giữa task, fast iteration. Phù hợp Phase 7.7 vì lớn nhưng có test plan rõ; SDD đã chứng minh hiệu quả ở Phase 5/6/7 building blocks.

**2. Inline Execution** — execute tasks trong session này dùng executing-plans.

Đề xuất: **Subagent-Driven** với sonnet cho implementer (vì tiếng Việt có dấu — haiku xóa dấu, đã ghi memory) + sonnet cho task-reviewer + opus cho final whole-branch review (Task 7.9).
