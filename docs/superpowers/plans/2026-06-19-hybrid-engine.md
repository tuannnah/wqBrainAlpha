# Hybrid Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hợp nhất hai engine song song (AI research-loop + Genetic Algorithm) thành một engine hybrid duy nhất: LLM seed quần thể → GA tiến hóa → mỗi K thế hệ LLM refine top alpha và bơm biến thể vào → chạy vô hạn đến Ctrl+C.

**Architecture:** Mở rộng `GeneticOptimizer` bằng hai hook nhỏ (`inject`/`inject_every` + `generations=None` để chạy vô hạn, bắt `KeyboardInterrupt` trong `run()`). Thêm orchestrator mỏng `HybridEngine` (`src/optimization/hybrid.py`) nối `LLMAlphaGenerator` (seed) + `AlphaRefiner` (refine trong vòng) + `ReferenceZoo` (khử tương quan) + `GeneticOptimizer` (tìm kiếm). `_run_auto` trong `main.py` chuyển sang dựng `HybridEngine` trực tiếp; xóa toàn bộ bề mặt ai/ga cũ.

**Tech Stack:** Python 3.12, Typer (CLI), Rich (progress/table), pytest, loguru. Backend LLM qua `_make_deepseek`/`_make_router` (claude-cli/codex-cli, không phụ thuộc DeepSeek vì đã 402).

## Global Constraints

- Mã, comment, commit message, output CLI: **tiếng Việt có dấu đầy đủ**.
- TDD bắt buộc: viết test fail trước, rồi mới implement; mỗi task ≥ 1 commit.
- Giữ tương thích ngược cho `GeneticOptimizer`: mặc định `inject=None`, `inject_every=0`, `generations=10` → test GA cũ không đổi hành vi.
- Mặc định engine chạy **vô hạn** (`max_simulations=None`, `generations=None`); chỉ dừng do `_budget_exhausted()` (khi set trần) hoặc `KeyboardInterrupt`.
- KHÔNG đụng lệnh `research` độc lập, `_make_research_loop`, `RefinementLoop`.
- Chạy test bằng `python -m pytest` ở thư mục gốc dự án.

---

### Task 1: Mở rộng GeneticOptimizer (inject hook + chạy vô hạn)

**Files:**
- Modify: `src/optimization/evolution.py` (dataclass fields + hàm `run`, dòng 47-66 và 180-229)
- Test: `tests/test_evolution.py`

**Interfaces:**
- Consumes: `Node`, `to_expression`, `NEG_INF` (đã có trong module).
- Produces:
  - `GeneticOptimizer` thêm field `inject: object = None` (callable(`scored: list[tuple[Node, float]]`) -> `list[Node]`), `inject_every: int = 0`.
  - `generations: int | None = 10` (None = vô hạn).
  - `run(on_generation=None, on_simulation=None)` giữ nguyên chữ ký, hành vi mới: bơm Node mỗi `inject_every` thế hệ; chạy vô hạn khi `generations is None`; bắt `KeyboardInterrupt` → trả best-so-far.

- [ ] **Step 1: Viết test fail cho inject hook**

Thêm vào `tests/test_evolution.py`:

```python
def test_inject_them_node_moi_dung_nhip(monkeypatch):
    """inject được gọi mỗi inject_every thế hệ; Node trả về vào quần thể."""
    import random as _random
    from src.optimization.evolution import GeneticOptimizer
    from src.generation.ast_utils import parse_expression, to_expression

    rng = _random.Random(0)
    pf = PreFilter(known_operators=None, known_fields=None)
    sim = FakeSimulator()
    seed = parse_expression("rank(close)")

    inject_calls = []

    def inject(scored):
        inject_calls.append(len(scored))
        return [parse_expression("ts_mean(volume, 5)")]

    opt = GeneticOptimizer(
        simulator=sim, prefilter=pf, seed_factory=lambda: seed.copy(),
        fields=["close", "volume"], scorer=lambda r: float(str(r).count("volume")),
        population_size=4, generations=4, elite_size=1,
        inject=inject, inject_every=2, rng=rng,
    )
    opt.run()
    # 4 thế hệ, inject_every=2 -> gọi sau gen index 1 và 3 => 2 lần.
    assert len(inject_calls) == 2
    # Biểu thức được bơm xuất hiện trong các expression đã simulate.
    assert any("ts_mean(volume" in c for c in sim.calls)


def test_generations_none_dung_theo_budget():
    """generations=None: chạy đến khi chạm max_simulations rồi dừng."""
    import random as _random
    from src.optimization.evolution import GeneticOptimizer
    from src.generation.ast_utils import parse_expression

    pf = PreFilter(known_operators=None, known_fields=None)
    sim = FakeSimulator()
    seed = parse_expression("rank(close)")
    opt = GeneticOptimizer(
        simulator=sim, prefilter=pf, seed_factory=lambda: seed.copy(),
        fields=["close"], scorer=lambda r: 1.0,
        population_size=3, generations=None, max_simulations=5,
        rng=_random.Random(0),
    )
    best = opt.run()
    assert opt.simulations_used <= 6  # không vượt trần quá 1 quần thể
    assert best  # vẫn trả danh sách Node
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest tests/test_evolution.py::test_inject_them_node_moi_dung_nhip tests/test_evolution.py::test_generations_none_dung_theo_budget -v`
Expected: FAIL — `GeneticOptimizer` chưa nhận tham số `inject`/`inject_every`; `generations=None` gây `TypeError` ở `range(None)`.

- [ ] **Step 3: Thêm field vào dataclass**

Trong `src/optimization/evolution.py`, sau dòng `generations: int = 10` (dòng 57), sửa thành:

```python
    generations: int | None = 10  # None = chạy vô hạn (dừng do budget/Ctrl+C)
```

Và thêm ngay sau `tournament_size: int = 3` (dòng 61):

```python
    inject: object = None       # callable(scored) -> list[Node]; None = không bơm
    inject_every: int = 0       # >0: mỗi N thế hệ gọi inject() và bơm Node trả về
```

- [ ] **Step 4: Viết lại hàm `run` hỗ trợ vô hạn + inject + Ctrl+C**

Thay toàn bộ thân `run` (dòng 180-229) bằng:

```python
    def run(self, on_generation=None, on_simulation=None) -> list[Node]:
        """Chạy tiến hóa.

        on_generation(stats): gọi sau khi tổng kết mỗi thế hệ.
        on_simulation(n, expr, score): gọi mỗi lần simulate thật (qua evaluate).

        generations=None -> chạy vô hạn, dừng do _budget_exhausted() hoặc Ctrl+C.
        inject (mỗi inject_every thế hệ) -> bơm Node mới vào quần thể.
        """
        self._on_simulation = on_simulation
        population = self._seed_population()
        gen = 0
        try:
            while self.generations is None or gen < self.generations:
                scored = [(ind, self.evaluate(ind)) for ind in population]
                scored.sort(key=lambda x: x[1], reverse=True)

                valid = [s for _, s in scored if s != NEG_INF]
                best = scored[0][1]
                avg = sum(valid) / len(valid) if valid else NEG_INF
                stats = GenerationStats(gen, best, avg, to_expression(scored[0][0]))
                self.history.append(stats)
                logger.info(
                    "Gen {}: best={:.4f} avg={:.4f} expr={}",
                    gen, best, avg, stats.best_expression,
                )
                if on_generation is not None:
                    on_generation(stats)

                if self._budget_exhausted():
                    logger.info(
                        "Đã đạt giới hạn {} simulation — dừng tiến hóa.", self.max_simulations
                    )
                    break

                elites = [ind.copy() for ind, _ in scored[: self.elite_size]]
                new_pop = elites
                while len(new_pop) < self.population_size:
                    r = self.rng.random()
                    if r < self.crossover_rate and len(scored) >= 2:
                        child, _ = self.crossover(self._tournament(scored), self._tournament(scored))
                    elif r < self.crossover_rate + self.mutation_rate:
                        child = self.mutate(self._tournament(scored))
                    else:
                        child = self.seed_factory()
                    new_pop.append(child)

                if self.inject is not None and self.inject_every > 0 and (gen + 1) % self.inject_every == 0:
                    for node in self.inject(scored):
                        if len(new_pop) < self.population_size:
                            new_pop.append(node)
                        else:
                            new_pop[-1] = node  # thay slot không-elite cuối

                population = new_pop
                gen += 1
        except KeyboardInterrupt:
            logger.info("Ctrl+C — dừng tiến hóa, trả best hiện có.")

        # Sắp xếp theo điểm đã cache, KHÔNG simulate thêm (an toàn sau Ctrl+C/budget).
        final = [(ind, self._cache.get(to_expression(ind), NEG_INF)) for ind in population]
        final.sort(key=lambda x: x[1], reverse=True)
        return [ind for ind, _ in final]
```

- [ ] **Step 5: Chạy test mới + toàn bộ test GA để chắc tương thích ngược**

Run: `python -m pytest tests/test_evolution.py -v`
Expected: PASS toàn bộ (test cũ giữ nguyên hành vi, 2 test mới xanh).

- [ ] **Step 6: Commit**

```bash
git add src/optimization/evolution.py tests/test_evolution.py
git commit -m "feat(ga): GeneticOptimizer thêm hook inject + chạy vô hạn (generations=None) cho engine hybrid"
```

---

### Task 2: Module HybridEngine

**Files:**
- Create: `src/optimization/hybrid.py`
- Test: `tests/test_hybrid.py`

**Interfaces:**
- Consumes: `GeneticOptimizer` (Task 1, có `inject`/`inject_every`/`generations`), `score_vector`/`weakest_dimension` (`src/scoring/vector.py`), `normalize` (`src/scoring/metrics.py`), `AlphaCandidate` + `AlphaTranslator` (`src/llm/translator.py`), `Hypothesis` (`src/llm/hypothesis.py`), `to_expression` (`src/generation/ast_utils.py`), `default_score` (`src/scoring/scorer.py` qua `score`).
- Produces:
  - Class `HybridEngine` (dataclass) với `run(on_generation=None, on_simulation=None, on_inject=None) -> list[Node]`.
  - Thuộc tính cấu hình: `inject_every=3`, `refine_top=2`, `seed_ideas=5`, `per_idea=2`, `originality_min=0.4`, `population_size=30`, `max_simulations=None`, `generations=None`.
  - `run()` trả danh sách `Node` đã sắp xếp giảm dần theo điểm.

- [ ] **Step 1: Viết test fail cho HybridEngine**

Tạo `tests/test_hybrid.py`:

```python
"""Test engine hybrid: seed LLM -> GA tiến hóa -> LLM refine bơm vào vòng."""

from __future__ import annotations

import random

from src.optimization.hybrid import HybridEngine
from src.simulation.pre_filter import PreFilter


class FakeSim:
    def __init__(self):
        self.calls = []

    def simulate(self, expr, settings=None):
        self.calls.append(expr)
        # Metrics dict để score_vector/normalize đọc được; volume -> điểm cao hơn.
        return {"sharpe": 1.0 + expr.count("volume"), "fitness": 1.0,
                "turnover": 0.3, "drawdown": 0.05}


class FakeLLMGen:
    def __init__(self, ideas=None, exprs=None, raise_on=None):
        self._ideas = ideas or ["ý tưởng A"]
        self._exprs = exprs or ["rank(close)"]
        self._raise_on = raise_on or set()

    def generate_ideas(self, n):
        if "ideas" in self._raise_on:
            raise RuntimeError("Error code: 402 - hết token")
        return self._ideas[:n]

    def generate(self, idea, n=5):
        return list(self._exprs)


class FakeRefiner:
    """Trả biến thể cố định; có thể ném 402 để test tắt LLM-in-loop."""

    def __init__(self, out="ts_mean(volume, 5)", raise_402=False):
        self.out = out
        self.raise_402 = raise_402
        self.calls = 0

    def refine(self, candidate, metrics, weak_dimension):
        self.calls += 1
        if self.raise_402:
            raise RuntimeError("Error code: 402 - hết token")
        from src.llm.translator import AlphaCandidate
        return AlphaCandidate(
            hypothesis=candidate.hypothesis, description="cải thiện",
            expression=self.out,
        )


class FakeZoo:
    def __init__(self, originality=1.0):
        self._orig = originality
        self.added = []

    def originality(self, expr):
        return self._orig

    def add(self, expr):
        self.added.append(expr)
        return True


def _engine(**kw):
    pf = PreFilter(known_operators=None, known_fields=None)
    defaults = dict(
        simulator=FakeSim(), prefilter=pf, fields=["close", "volume"],
        llm_generator=FakeLLMGen(), refiner=FakeRefiner(), zoo=FakeZoo(),
        inject_every=2, refine_top=1, population_size=4, generations=4,
        rng=random.Random(0),
    )
    defaults.update(kw)
    return HybridEngine(**defaults)


def test_seed_tu_llm_va_inject_bom_bien_the():
    """Seed lấy từ LLM; refiner được gọi và biến thể vào quần thể + zoo."""
    sim = FakeSim()
    refiner = FakeRefiner(out="ts_mean(volume, 5)")
    zoo = FakeZoo(originality=1.0)
    eng = _engine(simulator=sim, refiner=refiner, zoo=zoo)
    eng.run()
    assert refiner.calls >= 1
    assert "ts_mean(volume, 5)" in zoo.added
    assert any("ts_mean(volume" in c for c in sim.calls)


def test_bien_the_trung_zoo_bi_loai():
    """originality < ngưỡng -> không bơm, không add zoo."""
    refiner = FakeRefiner(out="ts_mean(volume, 5)")
    zoo = FakeZoo(originality=0.1)  # < originality_min=0.4
    eng = _engine(refiner=refiner, zoo=zoo, originality_min=0.4)
    eng.run()
    assert zoo.added == []


def test_llm_402_o_refine_khong_dung_ga():
    """Refiner ném 402 -> tắt LLM-in-loop nhưng GA vẫn chạy hết, trả Node."""
    refiner = FakeRefiner(raise_402=True)
    eng = _engine(refiner=refiner)
    best = eng.run()
    assert best  # GA vẫn trả quần thể, không raise


def test_seed_fallback_template_khi_llm_rong():
    """LLM ném 402 ở seed -> fallback template_generator."""
    class FakeTemplate:
        def generate(self, count, max_attempts=None):
            return ["rank(volume)"]

    eng = _engine(
        llm_generator=FakeLLMGen(raise_on={"ideas"}),
        template_generator=FakeTemplate(),
    )
    best = eng.run()
    assert best


def test_max_simulations_dung_xac_dinh():
    """generations=None + max_simulations nhỏ -> kết thúc xác định."""
    eng = _engine(generations=None, max_simulations=5)
    best = eng.run()
    assert best
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest tests/test_hybrid.py -v`
Expected: FAIL — `ModuleNotFoundError: src.optimization.hybrid`.

- [ ] **Step 3: Viết `HybridEngine`**

Tạo `src/optimization/hybrid.py`:

```python
"""Engine hybrid: LLM seed quần thể -> GA tiến hóa -> mỗi K thế hệ LLM refine top
alpha rồi bơm biến thể vào vòng. Chạy vô hạn đến khi LLM hết token (chỉ tắt phần
LLM, GA vẫn chạy) hoặc Ctrl+C.

Tái dùng: LLMAlphaGenerator (seed), AlphaRefiner (refine theo chiều yếu),
ReferenceZoo (khử tương quan biến thể), GeneticOptimizer (tìm kiếm).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from loguru import logger

from src.generation.ast_utils import Node, to_expression
from src.llm.hypothesis import Hypothesis
from src.llm.translator import AlphaCandidate
from src.optimization.evolution import GeneticOptimizer
from src.scoring.metrics import normalize
from src.scoring.scorer import score as default_score
from src.scoring.vector import score_vector, weakest_dimension


@dataclass
class HybridEngine:
    simulator: object            # .simulate(expr, settings=None) -> result
    prefilter: object            # .check(expr) -> (ok, reason)
    fields: list[str]
    llm_generator: object        # .generate_ideas(n), .generate(idea, n)
    refiner: object              # .refine(candidate, metrics, weak_dim) -> AlphaCandidate | None
    zoo: object                  # .originality(expr) -> float, .add(expr)
    template_generator: object = None  # fallback seed: .generate(count)
    scorer: object = default_score
    inject_every: int = 3
    refine_top: int = 2
    seed_ideas: int = 5
    per_idea: int = 2
    originality_min: float = 0.4
    population_size: int = 30
    generations: int | None = None
    max_simulations: int | None = None
    simulation_settings: dict | None = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        self._results: dict[str, object] = {}   # expr -> raw simulate result
        self._llm_disabled = False

    # --------------------------------------------------------------- seed pool
    def _seed_pool(self) -> list[str]:
        pool: list[str] = []
        try:
            ideas = self.llm_generator.generate_ideas(self.seed_ideas)
            for idea in ideas:
                pool.extend(self.llm_generator.generate(idea, self.per_idea))
        except Exception as exc:  # 402 / lỗi LLM -> tắt LLM, dùng fallback
            logger.warning("LLM seed lỗi ({}) — tắt LLM-in-loop, dùng fallback.", exc)
            self._llm_disabled = True
        # Khử trùng giữ thứ tự.
        pool = list(dict.fromkeys(p for p in pool if p))
        if not pool and self.template_generator is not None:
            pool = list(self.template_generator.generate(self.population_size))
        if not pool:
            pool = [f"rank({self.fields[0]})"] if self.fields else ["rank(close)"]
        return pool

    # ------------------------------------------------------------ inject hook
    def _build_inject(self):
        def inject(scored: list[tuple[Node, float]]) -> list[Node]:
            if self._llm_disabled:
                return []
            out: list[Node] = []
            for node, _score in scored[: self.refine_top]:
                expr = to_expression(node)
                result = self._results.get(expr)
                if result is None:
                    continue
                metrics = normalize(result)
                weak = weakest_dimension(score_vector(result))
                candidate = AlphaCandidate(
                    hypothesis=Hypothesis(), description="", expression=expr
                )
                try:
                    refined = self.refiner.refine(candidate, metrics, weak)
                except Exception as exc:  # 402 / lỗi LLM -> tắt phần LLM, GA chạy tiếp
                    logger.warning("LLM refine lỗi ({}) — tắt LLM-in-loop.", exc)
                    self._llm_disabled = True
                    return out
                if refined is None or not refined.expression:
                    continue
                new_expr = refined.expression
                ok, _reason = self.prefilter.check(new_expr)
                if not ok:
                    continue
                if self.zoo.originality(new_expr) < self.originality_min:
                    continue
                self.zoo.add(new_expr)
                out.append(GeneticOptimizer.expr_to_node(new_expr))
                logger.info("Bơm biến thể LLM vào quần thể: {} (chiều yếu={})", new_expr, weak)
            return out

        return inject

    # --------------------------------------------------------------------- run
    def run(self, on_generation=None, on_simulation=None, on_inject=None) -> list[Node]:
        pool = self._seed_pool()

        # Bọc simulator để bắt raw result theo expression (phục vụ inject).
        original_simulate = self.simulator.simulate

        def simulate_capture(expr, **kwargs):
            res = original_simulate(expr, **kwargs)
            self._results[expr] = res
            return res

        self.simulator.simulate = simulate_capture

        def seed_factory():
            return GeneticOptimizer.expr_to_node(self.rng.choice(pool))

        opt = GeneticOptimizer(
            simulator=self.simulator, prefilter=self.prefilter, seed_factory=seed_factory,
            fields=self.fields, scorer=self.scorer,
            population_size=self.population_size, generations=self.generations,
            max_simulations=self.max_simulations,
            simulation_settings=self.simulation_settings,
            inject=self._build_inject(), inject_every=self.inject_every,
            rng=self.rng,
        )
        try:
            best = opt.run(on_generation=on_generation, on_simulation=on_simulation)
        finally:
            self.simulator.simulate = original_simulate
        self.simulations_used = opt.simulations_used
        self.history = opt.history
        return best
```

- [ ] **Step 4: Chạy test để chắc nó pass**

Run: `python -m pytest tests/test_hybrid.py -v`
Expected: PASS toàn bộ 5 test.

- [ ] **Step 5: Commit**

```bash
git add src/optimization/hybrid.py tests/test_hybrid.py
git commit -m "feat(hybrid): HybridEngine nối LLM seed + GA tiến hóa + LLM refine bơm trong vòng"
```

---

### Task 3: Tích hợp HybridEngine vào `_run_auto` + helper refiner/progress

**Files:**
- Modify: `main.py` (helper mới `_make_refiner`, `_run_hybrid_with_progress`; viết lại `_run_auto`, dòng 1118-1221)
- Test: `tests/test_auto_command.py` (cập nhật các test gọi `_run_auto`)

**Interfaces:**
- Consumes: `HybridEngine` (Task 2), `_make_llm_generator`, `_make_router`, `_cached_symbols`, `_make_invalid_field_recorder`, `Simulator`, `AlphaRepository`, `FieldRepository`, `OperatorRepository`, `AlphaTranslator`, `AlphaRefiner`, `ReferenceZoo`, `TemplateGenerator`, `SimConfig`, `PreFilter`.
- Produces:
  - `_make_refiner(session_factory, prefilter, region, universe, delay) -> AlphaRefiner`.
  - `_run_hybrid_with_progress(engine) -> list[Node]` (Rich progress, đếm sim).
  - `_run_auto(region, universe, delay, max_sims=0, generations=0, existing_client=None, swallow_errors=False, decay=0, truncation=0.08, neutralization="SUBINDUSTRY") -> list[Node] | None` — KHÔNG còn tham số `engine`.

- [ ] **Step 1: Viết test fail cho `_run_auto` hybrid**

Thay test `test_run_auto_truyen_scope_cu_the` đầu file (dòng 45-63) và thêm test mới. Mở `tests/test_auto_command.py`, cập nhật phần đầu để có một fake HybridEngine và test đường hybrid:

```python
def test_run_auto_chay_hybrid_va_luu_db(monkeypatch):
    """_run_auto dựng HybridEngine, chạy, lưu top alpha source='hybrid'."""
    import main

    captured = {}

    class _FakeHybrid:
        def __init__(self, **kw):
            captured["kw"] = kw

        def run(self, on_generation=None, on_simulation=None, on_inject=None):
            self.simulations_used = 3
            self.history = []
            from src.generation.ast_utils import parse_expression
            return [parse_expression("rank(close)")]

    saved = []

    class _FakeRepo:
        def __init__(self, sf):
            pass

        def save_alpha(self, expr, source=None):
            saved.append((expr, source))

        def zoo(self, n):
            return []

    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(main, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(main, "_cached_symbols",
                        lambda sf: (["close", "volume"], [], {}, set(), {}))
    monkeypatch.setattr(main, "_make_llm_generator", lambda sf, pf: _FakeGen())
    monkeypatch.setattr(main, "_make_refiner", lambda sf, pf, r, u, d: object())
    monkeypatch.setattr(main, "Simulator", lambda *a, **k: object())
    monkeypatch.setattr(main, "AlphaRepository", _FakeRepo)
    monkeypatch.setattr(main, "HybridEngine", _FakeHybrid)
    monkeypatch.setattr(main, "_run_hybrid_with_progress", lambda eng: eng.run())

    result = main._run_auto("USA", "TOP3000", 1, max_sims=5)
    assert result is not None
    assert ("rank(close)", "hybrid") in saved
    # max_sims=5 -> truyền max_simulations=5 vào HybridEngine.
    assert captured["kw"]["max_simulations"] == 5
```

Lưu ý: `_FakeClient` và `_FakeGen` đã có sẵn trong file; nếu `_FakeGen` thiếu `generate_ideas`/`generate` thì bổ sung (xem Step 4).

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest tests/test_auto_command.py::test_run_auto_chay_hybrid_va_luu_db -v`
Expected: FAIL — `_run_auto` vẫn nhận `engine` làm tham số đầu / `main.HybridEngine` chưa tồn tại.

- [ ] **Step 3: Thêm helper `_make_refiner` và `_run_hybrid_with_progress` vào `main.py`**

Thêm ngay trước `def _run_auto(` (dòng ~1118):

```python
def _make_refiner(session_factory, prefilter, region, universe, delay):
    """Dựng AlphaRefiner (DeepSeek/router + AlphaTranslator có scope) cho LLM-in-loop."""
    from src.llm.refiner import AlphaRefiner
    from src.llm.translator import AlphaTranslator

    deepseek = _make_router()
    field_repo = FieldRepository(None, session_factory)
    op_repo = OperatorRepository(None, session_factory)
    translator = AlphaTranslator(deepseek, field_repo, op_repo, prefilter)
    translator.set_scope(region=region, universe=universe, delay=delay)
    return AlphaRefiner(deepseek, translator)


def _run_hybrid_with_progress(engine):
    """Chạy HybridEngine kèm thanh tiến trình (đếm sim + thế hệ)."""
    from rich.progress import (
        BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TimeElapsedColumn(), console=console, transient=True,
    ) as progress:
        task = progress.add_task("Hybrid: seed + tiến hóa...", total=None)

        def on_simulation(n, expr, score):
            progress.update(task, description=f"Hybrid: {n} sim, best gần nhất {score:.3f}"[:60])

        def on_generation(stats):
            progress.update(
                task,
                description=f"Hybrid gen {stats.generation} best={stats.best_score:.3f}"[:60],
            )

        return engine.run(on_generation=on_generation, on_simulation=on_simulation)
```

- [ ] **Step 4: Viết lại `_run_auto` dùng HybridEngine**

Thay toàn bộ `_run_auto` (dòng 1118-1221) bằng:

```python
def _run_auto(region, universe, delay, max_sims=0, generations=0,
              existing_client=None, swallow_errors=False,
              decay=0, truncation=0.08, neutralization="SUBINDUSTRY"):
    """Toàn trình hybrid: login → cache → seed LLM → GA tiến hóa + LLM-in-loop → lưu DB.

    max_sims/generations = 0 nghĩa là VÔ HẠN (None). swallow_errors giữ để tương
    thích chữ ký gọi từ menu; HybridEngine tự nuốt lỗi LLM nên không cần dùng.
    Trả danh sách Node tốt nhất, hoặc None nếu thiếu điều kiện (chưa có fields).
    """
    import random as _random

    from src.generation.ast_utils import to_expression
    from src.generation.template import TemplateGenerator
    from src.decorrelation.zoo import ReferenceZoo
    from src.optimization.hybrid import HybridEngine
    from src.simulation.config import SimConfig
    from src.simulation.pre_filter import PreFilter

    engine_box = init_db(make_engine())
    session_factory = make_session_factory(engine_box)

    client = existing_client or _make_client()
    if not getattr(client, "authenticated", False):
        client.authenticate()

    fields, operators, field_types, matrix_only_ops, operator_arity = _cached_symbols(session_factory)
    if not fields:
        console.print("[red]Chưa có fields — tải fields (menu 2) trước.[/red]")
        return None

    sim_config = SimConfig(
        region=region, universe=universe, delay=delay,
        decay=decay, truncation=truncation, neutralization=neutralization,
    )
    pf = PreFilter(
        known_operators=operators or None, known_fields=set(fields),
        field_types=field_types, matrix_only_ops=matrix_only_ops,
        operator_arity=operator_arity,
    )
    sim = Simulator(
        client,
        on_invalid_field=_make_invalid_field_recorder(session_factory, region, universe),
    )
    repo = AlphaRepository(session_factory)
    zoo = ReferenceZoo.default(extra=[a.expression for a in repo.zoo(200)])
    tgen = TemplateGenerator(fields, pf, rng=_random.Random())

    engine = HybridEngine(
        simulator=sim, prefilter=pf, fields=fields,
        llm_generator=_make_llm_generator(session_factory, pf),
        refiner=_make_refiner(session_factory, pf, region, universe, delay),
        zoo=zoo, template_generator=tgen,
        max_simulations=max_sims or None, generations=generations or None,
        simulation_settings=sim_config.to_settings(),
    )

    best_nodes = _run_hybrid_with_progress(engine)
    best_exprs = [to_expression(n) for n in best_nodes[:10]]
    for expr in best_exprs:
        repo.save_alpha(expr, source="hybrid")

    table = Table(title=f"Top alpha hybrid ({len(best_exprs)}) — {engine.simulations_used} sim")
    table.add_column("Expression", overflow="fold")
    for expr in best_exprs:
        table.add_row(expr)
    console.print(table)
    console.print(
        "[dim]Đã lưu DB — xem bằng lệnh 'top'. CHƯA nộp; nộp bằng 'submit' khi muốn.[/dim]"
    )
    return best_nodes
```

Nếu `_FakeGen` trong test thiếu method, thêm vào class `_FakeGen` ở đầu `tests/test_auto_command.py`:

```python
    def generate_ideas(self, n):
        return ["ý tưởng test"][:n]

    def generate(self, idea, n=5):
        return ["rank(close)"]
```

- [ ] **Step 5: Chạy test hybrid `_run_auto`**

Run: `python -m pytest tests/test_auto_command.py::test_run_auto_chay_hybrid_va_luu_db -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_auto_command.py
git commit -m "feat(auto): _run_auto chuyển sang HybridEngine (bỏ AutoPipeline/engine ai-ga)"
```

---

### Task 4: Cập nhật lệnh `auto` + menu (bỏ lựa chọn engine)

**Files:**
- Modify: `main.py` (`auto` command dòng 1224-1243; `_menu_ask_engine` dòng 1300-1302; menu mục 4/5 dòng 1349-1380)
- Test: `tests/test_auto_command.py`

**Interfaces:**
- Consumes: `_run_auto` (Task 3, chữ ký mới không có `engine`).
- Produces: lệnh `auto` không còn `--engine`; thêm `--max-sims`/`--generations` (mặc định 0 = vô hạn). Menu mục 4/5 gọi thẳng hybrid, không hỏi engine.

- [ ] **Step 1: Viết test fail cho lệnh `auto` không còn --engine**

Thêm vào `tests/test_auto_command.py`:

```python
def test_lenh_auto_khong_con_engine_option(monkeypatch):
    """auto gọi _run_auto KHÔNG truyền engine; có --max-sims."""
    import main
    from typer.testing import CliRunner

    called = {}

    def fake_run_auto(region, universe, delay, max_sims=0, generations=0,
                      existing_client=None, swallow_errors=False,
                      decay=0, truncation=0.08, neutralization="SUBINDUSTRY"):
        called["max_sims"] = max_sims
        called["generations"] = generations
        return ["node"]

    monkeypatch.setattr(main, "_run_auto", fake_run_auto)
    monkeypatch.setattr(main, "_setup_logging", lambda: None)
    runner = CliRunner()
    result = runner.invoke(main.app, ["auto", "--max-sims", "7", "--generations", "3"])
    assert result.exit_code == 0, result.output
    assert called["max_sims"] == 7
    assert called["generations"] == 3
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest tests/test_auto_command.py::test_lenh_auto_khong_con_engine_option -v`
Expected: FAIL — `auto` chưa có `--max-sims`/`--generations`, vẫn truyền `engine`.

- [ ] **Step 3: Viết lại lệnh `auto`**

Thay `auto` (dòng 1224-1243) bằng:

```python
@app.command()
def auto(
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    max_sims: int = typer.Option(0, "--max-sims", help="Trần tổng simulation (0 = vô hạn)"),
    generations: int = typer.Option(0, "--generations", help="Số thế hệ GA (0 = vô hạn)"),
    decay: int = typer.Option(0, "--decay", help="Decay simulation config"),
    truncation: float = typer.Option(0.08, "--truncation", help="Truncation simulation config"),
    neutralization: str = typer.Option("SUBINDUSTRY", "--neutralization", help="Neutralization simulation config"),
) -> None:
    """Chạy engine hybrid: login → cache → seed LLM → GA tiến hóa + LLM-in-loop. KHÔNG nộp."""
    _setup_logging()
    if _run_auto(
        region, universe, delay, max_sims=max_sims, generations=generations,
        decay=decay, truncation=truncation, neutralization=neutralization,
    ) is None:
        raise typer.Exit(code=1)
```

- [ ] **Step 4: Bỏ `_menu_ask_engine` và sửa menu mục 4/5**

Xóa hàm `_menu_ask_engine` (dòng 1300-1302). Trong `start()`, thay nhánh mục 4 (dòng 1349-1370) bằng:

```python
            elif choice == "4":
                sim_settings = _menu_ask_sim_settings()
                # Hybrid chạy vô hạn, chỉ dừng khi LLM hết token / Ctrl+C.
                _run_auto(
                    state.region, state.universe, state.delay,
                    swallow_errors=True, existing_client=state.client,
                    **sim_settings,
                )
```

Và thay nhánh mục 5 (dòng 1371-1380) bằng:

```python
            elif choice == "5":
                sim_settings = _menu_ask_sim_settings()
                console.print("[cyan]Thử luồng: seed + tiến hóa ngắn (trần nhỏ)...[/cyan]")
                _run_auto(
                    state.region, state.universe, state.delay,
                    max_sims=5, generations=2,
                    existing_client=state.client, **sim_settings,
                )
```

- [ ] **Step 5: Chạy test lệnh auto + menu**

Run: `python -m pytest tests/test_auto_command.py::test_lenh_auto_khong_con_engine_option -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_auto_command.py
git commit -m "feat(cli): bỏ lựa chọn engine ai/ga ở lệnh auto + menu, chỉ còn hybrid"
```

---

### Task 5: Xóa bề mặt cũ (run-ga, callback ai/ga, helper chết) + dọn test

**Files:**
- Modify: `main.py` (xóa `run_ga`, `_auto_run_direction_ai`, `_auto_run_direction_ga`, `_run_ga_with_progress`, import `passed_from_ga` + `AutoPipeline` nếu hết dùng)
- Modify: `tests/test_auto_command.py` (xóa test gắn engine ai/ga, `_auto_run_direction_*`, `AutoPipeline`)
- Modify: `tests/test_auto_pipeline.py` (giữ test `passed_from_ga` của `src/pipeline/auto.py` — KHÔNG xóa, hàm vẫn ở đó)

**Interfaces:**
- Consumes: không (chỉ xóa code chết sau khi Task 3-4 đã thay đường đi).
- Produces: `main.py` không còn lệnh `run-ga`, không còn `_auto_run_direction_ai/_ga`, `_run_ga_with_progress`, không import `passed_from_ga`.

- [ ] **Step 1: Xác định code chết còn lại**

Run: `python -m pytest tests/ -q` để có baseline trước khi xóa (ghi nhận test nào còn đỏ do tham chiếu hàm cũ).
Run: `grep -n "_auto_run_direction_ai\|_auto_run_direction_ga\|_run_ga_with_progress\|passed_from_ga\|run-ga\|def run_ga" main.py`
Expected: liệt kê đúng các định nghĩa/usage cần xóa (sau Task 3-4 chỉ còn định nghĩa, không còn lời gọi từ `_run_auto`).

- [ ] **Step 2: Xóa các hàm chết trong `main.py`**

Xóa nguyên các khối:
- `@app.command("run-ga")` + hàm `run_ga` (dòng ~453-534).
- `_auto_run_direction_ai` (dòng ~1042-1066).
- `_auto_run_direction_ga` (dòng ~1069-1115).
- `_run_ga_with_progress` (dòng ~832-870).
- Trong khối import đầu file: bỏ `passed_from_ga` khỏi import từ `src.pipeline.auto` (dòng 36). Giữ `AutoPipeline`, `PassedAlpha`, `DirectionOutcome`, `PrepareInfo`, `AutoEvent` nếu còn nơi khác dùng; nếu `grep` cho thấy `AutoPipeline`/`PrepareInfo`/`DirectionOutcome`/`PassedAlpha`/`AutoEvent` không còn dùng trong `main.py`, bỏ luôn khỏi import.

Sau khi xóa, chạy:
Run: `grep -n "passed_from_ga\|_auto_run_direction\|_run_ga_with_progress\|AutoPipeline\|PrepareInfo\|DirectionOutcome\|PassedAlpha\|AutoEvent\|_make_research_loop\|_run_research_with_progress" main.py`
Giữ lại: `_make_research_loop`, `_run_research_with_progress` (lệnh `research` vẫn dùng). Mọi import không còn tham chiếu thì xóa khỏi dòng import.

- [ ] **Step 3: Dọn test gắn engine cũ trong `tests/test_auto_command.py`**

Xóa các test tham chiếu hành vi cũ (theo `grep -n "def test_"`): `test_run_auto_truyen_scope_cu_the` (nếu chưa thay ở Task 3), `test_run_auto_ai_mac_dinh_khong_gioi_han_huong`, `test_run_auto_builds_sim_config_for_ai_builder`, `test_run_auto_per_direction_sims_co_dinh`, và mọi test `monkeypatch.setattr(main, "AutoPipeline", ...)` hoặc `_auto_run_direction_ai`. Giữ các test không liên quan engine (vd `test_simulate_command_truyen_day_du_sim_config`).

- [ ] **Step 4: Chạy toàn bộ test**

Run: `python -m pytest tests/ -q`
Expected: PASS toàn bộ. Đặc biệt `tests/test_auto_pipeline.py::test_passed_from_ga_loc_alpha_dat_nguong` vẫn xanh (hàm còn ở `src/pipeline/auto.py`). Không còn `ImportError`/`AttributeError` do hàm đã xóa.

- [ ] **Step 5: Kiểm tra CLI khởi động sạch**

Run: `python main.py --help`
Expected: liệt kê lệnh, KHÔNG còn `run-ga`; lệnh `auto` có `--max-sims`/`--generations`, không có `--engine`.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_auto_command.py
git commit -m "refactor(cli): xóa bề mặt engine ga cũ (run-ga, callback ai/ga, helper chết)"
```

---

## Self-Review

**Spec coverage:**
- Kiến trúc seed→evolve→inject vô hạn → Task 1 (GA hook + vô hạn) + Task 2 (HybridEngine).
- LLM trong vòng lặp (refine top theo chiều yếu, khử trùng zoo) → Task 2 `_build_inject`.
- Dừng vô hạn / 402 không dừng GA / Ctrl+C → Task 1 (KeyboardInterrupt) + Task 2 (`_llm_disabled`).
- Trần test/CI `--max-sims`/`--generations` → Task 4.
- Xóa bề mặt cũ, giữ `research` → Task 5.
- Tích hợp menu 4/5 → Task 4.
- Kiểm thử test_hybrid/test_evolution/test_auto_command → Task 1,2,3,4,5.

**Placeholder scan:** Không có TBD/TODO; mọi step có code/lệnh cụ thể.

**Type consistency:** `inject(scored)`/`inject_every` đồng nhất Task 1↔2. `HybridEngine(...)` field names khớp giữa Task 2 (định nghĩa) và Task 3 (dựng). `_run_auto(region, universe, delay, max_sims, generations, ...)` khớp Task 3↔4. `save_alpha(expr, source="hybrid")` khớp `AlphaRepository` hiện có. `score_vector`/`weakest_dimension`/`normalize` đúng chữ ký `src/scoring/vector.py` + `metrics.py`.
