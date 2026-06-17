# Alpha Settings Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve alpha formula settings for `decay`, `truncation`, and `neutralization` without sweeping config during expression search.

**Architecture:** Keep expression search and config search separate per GĐ5. Add one validated `SimConfig` path for fixed-per-run simulation settings, make generated alpha reports carry per-alpha settings, and wire AI/GA simulation calls through that path. `sweep-config` remains the explicit post-pass config optimization tool.

**Tech Stack:** Python dataclasses, Typer CLI, pytest, existing `Simulator`, `RefinementLoop`, `GeneticOptimizer`, and `Candidate.overrides`.

---

### Task 1: Validate `SimConfig`

**Files:**
- Modify: `src/simulation/config.py`
- Test: `tests/test_sim_config.py`

- [ ] **Step 1: Write failing validation tests**

Add these tests to `tests/test_sim_config.py`:

```python
import pytest


def test_decay_phai_trong_khoang_hop_le():
    with pytest.raises(ValueError, match="decay"):
        SimConfig(decay=-1)
    with pytest.raises(ValueError, match="decay"):
        SimConfig(decay=513)


def test_truncation_phai_trong_khoang_hop_le():
    with pytest.raises(ValueError, match="truncation"):
        SimConfig(truncation=0)
    with pytest.raises(ValueError, match="truncation"):
        SimConfig(truncation=0.51)


def test_neutralization_phai_hop_le_va_chuan_hoa_in_hoa():
    assert SimConfig(neutralization="market").neutralization == "MARKET"
    with pytest.raises(ValueError, match="neutralization"):
        SimConfig(neutralization="BAD_GROUP")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sim_config.py -q`

Expected: FAIL because invalid settings are accepted and neutralization is not normalized.

- [ ] **Step 3: Implement validation**

In `src/simulation/config.py`, add constants and `__post_init__`:

```python
VALID_NEUTRALIZATIONS = {
    "NONE",
    "MARKET",
    "SECTOR",
    "INDUSTRY",
    "SUBINDUSTRY",
    "COUNTRY",
    "EXCHANGE",
}


def _normalize_neutralization(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in VALID_NEUTRALIZATIONS:
        raise ValueError(f"neutralization không hợp lệ: {value}")
    return normalized
```

Then validate in `SimConfig.__post_init__` using `object.__setattr__` because the dataclass is frozen:

```python
def __post_init__(self) -> None:
    if not isinstance(self.decay, int) or not 0 <= self.decay <= 512:
        raise ValueError(f"decay phải là int trong [0, 512], got {self.decay!r}")
    if not isinstance(self.truncation, (int, float)) or not 0.0 < float(self.truncation) <= 0.5:
        raise ValueError(f"truncation phải trong (0, 0.5], got {self.truncation!r}")
    object.__setattr__(self, "truncation", float(self.truncation))
    object.__setattr__(self, "neutralization", _normalize_neutralization(self.neutralization))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sim_config.py -q`

Expected: PASS.

---

### Task 2: Give Generated Alpha Formulas Complete Settings

**Files:**
- Modify: `src/generation/families.py`
- Test: `tests/test_families.py`

- [ ] **Step 1: Write failing generated-settings tests**

Add these tests to `tests/test_families.py`:

```python
def test_moi_ung_vien_co_truncation_va_neutralization_override():
    for c in generate_candidates():
        assert "truncation" in c.overrides, f"thiếu truncation: {c.family} / {c.expression}"
        assert "neutralization" in c.overrides, f"thiếu neutralization: {c.family} / {c.expression}"


def test_market_variant_dat_neutralization_market():
    market = [
        c for c in generate_candidates()
        if "group_neutralize" not in c.expression
    ]
    assert market
    assert all(c.overrides["neutralization"] == "MARKET" for c in market)


def test_truncation_trong_khoang_hop_le_cho_alpha_kinh_dien():
    for c in generate_candidates():
        t = c.overrides["truncation"]
        assert 0.0 < t <= 0.5, f"truncation sai: {c.family}={t}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_families.py -q`

Expected: FAIL because classic generated candidates currently set only `decay`.

- [ ] **Step 3: Implement per-family settings**

In `src/generation/families.py`, add:

```python
_FAMILY_TRUNCATION = {
    "reversal": 0.06,
    "momentum": 0.08,
    "volatility": 0.08,
    "volume": 0.05,
    "value": 0.10,
    "analyst": 0.06,
    "seasonality": 0.05,
}

_GROUP_TO_NEUTRALIZATION = {
    "market": "MARKET",
    "sector": "SECTOR",
    "industry": "INDUSTRY",
    "subindustry": "SUBINDUSTRY",
}
```

Add helpers:

```python
def _truncation_for(family: str) -> float:
    return _FAMILY_TRUNCATION.get(family, 0.08)


def _neutralization_for_expression(expression: str) -> str:
    marker = "group_neutralize("
    if marker not in expression:
        return "MARKET"
    group = expression.rsplit(",", 1)[-1].rstrip(") ").strip().lower()
    return _GROUP_TO_NEUTRALIZATION.get(group, "SUBINDUSTRY")
```

In `generate_candidates()`, after setting `decay`, also set:

```python
c.overrides.setdefault("truncation", _truncation_for(c.family))
c.overrides.setdefault("neutralization", _neutralization_for_expression(c.expression))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_families.py -q`

Expected: PASS.

---

### Task 3: Route AI Research Simulations Through `SimConfig`

**Files:**
- Modify: `src/llm/loop.py`
- Modify: `main.py`
- Test: `tests/test_loop.py`

- [ ] **Step 1: Write failing AI settings test**

Add this test to `tests/test_loop.py`:

```python
def test_loop_truyen_sim_config_vao_simulator():
    from src.simulation.config import SimConfig

    class _SettingsSim:
        def __init__(self):
            self.calls = []

        def simulate(self, expr, settings=None):
            self.calls.append((expr, settings))
            return _result(expr, 1.5)

    sim = _SettingsSim()
    repo = _repo()
    loop = _loop(
        _FakeTranslator("rank(close)"),
        _FakeRefiner([]),
        sim,
        repo,
        max_simulations=1,
        sim_config=SimConfig(region="EUR", universe="TOP1200", delay=0, decay=4, truncation=0.05, neutralization="INDUSTRY"),
    )

    loop.run("X")

    assert sim.calls[0][1] == {
        "region": "EUR",
        "universe": "TOP1200",
        "delay": 0,
        "neutralization": "INDUSTRY",
        "decay": 4,
        "truncation": 0.05,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loop.py::test_loop_truyen_sim_config_vao_simulator -q`

Expected: FAIL because `RefinementLoop.__init__` does not accept `sim_config`.

- [ ] **Step 3: Implement `sim_config` in `RefinementLoop`**

In `src/llm/loop.py`, import `SimConfig` and add a `sim_config: SimConfig | None = None` parameter to `RefinementLoop.__init__`. Store:

```python
self.sim_config = sim_config or SimConfig.default(region=region, universe=universe, delay=delay)
```

Replace the hardcoded simulate call:

```python
result = self.simulator.simulate(expr, settings=self.sim_config.to_settings())
```

In `main.py`, update `_make_research_loop(..., sim_config=None)` and pass `sim_config=sim_config` to `RefinementLoop`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_loop.py::test_loop_truyen_sim_config_vao_simulator -q`

Expected: PASS.

---

### Task 4: Route GA and Auto Through Fixed `SimConfig`

**Files:**
- Modify: `src/optimization/evolution.py`
- Modify: `main.py`
- Test: `tests/test_evolution.py`
- Test: `tests/test_auto_command.py`

- [ ] **Step 1: Write failing GA settings test**

Add this test to `tests/test_evolution.py`:

```python
def test_optimizer_truyen_simulation_settings_khi_duoc_cau_hinh():
    class _SettingsSim:
        def __init__(self):
            self.calls = []

        def simulate(self, expr, settings=None):
            self.calls.append((expr, settings))
            return expr

    rng = random.Random(23)
    sim = _SettingsSim()
    pf = PreFilter(known_operators=None, known_fields=None)
    opt = GeneticOptimizer(
        simulator=sim,
        prefilter=pf,
        seed_factory=lambda: GeneticOptimizer.expr_to_node("rank(close)"),
        fields=["close"],
        scorer=_expr_scorer,
        population_size=1,
        generations=1,
        max_simulations=1,
        rng=rng,
        simulation_settings={"region": "EUR", "universe": "TOP1200", "delay": 0, "decay": 8, "truncation": 0.05, "neutralization": "SECTOR"},
    )

    opt.run()

    assert sim.calls == [
        ("rank(close)", {"region": "EUR", "universe": "TOP1200", "delay": 0, "decay": 8, "truncation": 0.05, "neutralization": "SECTOR"})
    ]
```

- [ ] **Step 2: Write failing auto wiring test**

Add this test to `tests/test_auto_command.py`:

```python
def test_run_auto_truyen_sim_config_xuong_ai_builder(monkeypatch):
    captured = {}

    class _FakePipe:
        def __init__(self, **kwargs):
            self.prepare = kwargs["prepare"]
            self.run_direction = kwargs["run_direction"]

        def run(self):
            self.prepare()
            self.run_direction("h1")
            return AutoResult([], directions_run=1, total_sims=0, stop_reason="hết_hướng")

    def _fake_run_builder(client_box, sf, region, universe, delay, per_direction_box, sim_config):
        captured["settings"] = sim_config.to_settings()
        def run(direction: str):
            from src.pipeline.auto import DirectionOutcome
            return DirectionOutcome(passed=[], sims_used=0)
        return run

    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(main, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(main, "FieldRepository", _FakeFieldRepo)
    monkeypatch.setattr(main, "OperatorRepository", _FakeOpRepo)
    monkeypatch.setattr(main, "AutoPipeline", _FakePipe)
    monkeypatch.setattr(main, "_auto_run_direction_ai", _fake_run_builder)

    main._run_auto(
        "ai", "EUR", "TOP1200", 0,
        target_passes=3, max_sims=1,
        decay=6, truncation=0.04, neutralization="SECTOR",
    )

    assert captured["settings"] == {
        "region": "EUR",
        "universe": "TOP1200",
        "delay": 0,
        "neutralization": "SECTOR",
        "decay": 6,
        "truncation": 0.04,
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_evolution.py::test_optimizer_truyen_simulation_settings_khi_duoc_cau_hinh tests/test_auto_command.py::test_run_auto_truyen_sim_config_xuong_ai_builder -q
```

Expected: FAIL because neither `GeneticOptimizer` nor `_run_auto` has those parameters.

- [ ] **Step 4: Implement GA simulation settings**

In `src/optimization/evolution.py`, add a dataclass field:

```python
simulation_settings: dict | None = None
```

Then in `evaluate()`:

```python
if self.simulation_settings is None:
    result = self.simulator.simulate(expr)
else:
    result = self.simulator.simulate(expr, settings=self.simulation_settings)
```

- [ ] **Step 5: Implement auto fixed settings**

In `main.py`, import or locally create `SimConfig` in `_run_auto`:

```python
from src.simulation.config import SimConfig

sim_config = SimConfig(
    region=region,
    universe=universe,
    delay=delay,
    decay=decay,
    truncation=truncation,
    neutralization=neutralization,
)
```

Add parameters to `_run_auto` with current defaults:

```python
decay: int = 0,
truncation: float = 0.08,
neutralization: str = "SUBINDUSTRY",
```

Pass `sim_config` into both `_auto_run_direction_ai` and `_auto_run_direction_ga`.

For AI, pass `sim_config` to `_make_research_loop`.

For GA, pass `simulation_settings=sim_config.to_settings()` to `GeneticOptimizer`.

Add Typer options to `auto()` only:

```python
decay: int = typer.Option(0, "--decay", help="Fixed decay setting for expression search"),
truncation: float = typer.Option(0.08, "--truncation", help="Fixed truncation setting for expression search"),
neutralization: str = typer.Option("SUBINDUSTRY", "--neutralization", help="Fixed neutralization setting for expression search"),
```

Keep menu `4/ai` defaults unchanged by not passing those arguments.

- [ ] **Step 6: Run task tests**

Run:

```bash
pytest tests/test_evolution.py::test_optimizer_truyen_simulation_settings_khi_duoc_cau_hinh tests/test_auto_command.py::test_run_auto_truyen_sim_config_xuong_ai_builder -q
```

Expected: PASS.

---

### Task 5: Config-Aware Simulation Cache

**Files:**
- Modify: `src/storage/repository.py`
- Modify: `src/llm/loop.py`
- Modify: `src/simulation/config.py`
- Test: `tests/test_storage.py`
- Test: `tests/test_loop.py`
- Test: `tests/test_sim_config.py`

- [ ] **Step 1: Add cache regression tests**

Add tests proving `AlphaRepository.save_simulation(..., config_key=...)` and
`get_cached_simulation(..., config_key=...)` distinguish the same expression under
different `SimConfig.key()` values. Add a `RefinementLoop` test proving a cached
default-config expression does not satisfy a tuned-config simulation request.

- [ ] **Step 2: Make repository cache config-aware**

Add optional `config_key` parameters to `save_simulation()` and `get_cached_simulation()`,
and use `expr_hash(expression, config_key)`. Keep the default `config_key=None`
behavior for existing callers.

- [ ] **Step 3: Wire `RefinementLoop` cache through `SimConfig.key()`**

Use the resolved `self.sim_config.key()` for both cache lookup and save. Derive
`self.region`, `self.universe`, and `self.delay` from the resolved `SimConfig` so DB
metadata matches the simulation settings.

- [ ] **Step 4: Fail closed for invalid neutralization type**

Reject non-string `neutralization` values in `SimConfig`; do not coerce `None` to
`"NONE"`.

---

### Task 6: Verify Focused and Regression Suites

**Files:**
- No new files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_sim_config.py tests/test_families.py tests/test_loop.py tests/test_evolution.py tests/test_auto_command.py tests/test_auto_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 3: Inspect worktree**

Run:

```bash
git status --short
git diff -- src/simulation/config.py src/generation/families.py src/llm/loop.py src/optimization/evolution.py main.py tests/test_sim_config.py tests/test_families.py tests/test_loop.py tests/test_evolution.py tests/test_auto_command.py
```

Expected: Only planned files changed, plus pre-existing unrelated files remain untouched.
