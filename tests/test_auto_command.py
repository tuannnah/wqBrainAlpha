"""Test lệnh `auto`, `simulate`, `run_ga`, `research` và menu `start` trong main.py.

Bao gồm: _run_auto dựng HybridEngine đúng cách, lệnh `auto` không còn --engine
(chỉ chạy hybrid), menu mục 4/5 gọi _run_auto với scope cụ thể (không phải
OptionInfo của Typer), và các lệnh GA/research truyền đúng sim config.
"""

from __future__ import annotations

import main
from src.simulation.config import SimConfig


class _FakeClient:
    authenticated = True

    def authenticate(self, *a, **k):
        return None


class _FakeGen:
    def generate_ideas(self, n):
        return ["ý tưởng test"][:n]

    def generate(self, idea, n=5):
        return ["rank(close)"]


def test_run_auto_chay_hybrid_va_luu_db(monkeypatch):
    """_run_auto dựng HybridEngine, chạy, lưu top alpha source='hybrid'."""

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


def test_simulate_command_truyen_day_du_sim_config(monkeypatch):
    from src.simulation.simulator import SimulationResult

    captured = {}

    class _FakeSimulator:
        def __init__(self, client):
            pass

        def simulate(self, expr, settings=None):
            captured["expr"] = expr
            captured["settings"] = settings
            return SimulationResult(expression=expr, status="passed", sharpe=1.2, fitness=1.0)

    class _FakeRepo:
        def __init__(self, sf):
            pass

        def save_simulation(self, result, **kwargs):
            captured["saved"] = kwargs

    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(main, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(main, "Simulator", _FakeSimulator)
    monkeypatch.setattr(main, "AlphaRepository", _FakeRepo)

    main.simulate(
        expr="rank(close)",
        region="EUR",
        universe="TOP1200",
        delay=0,
        decay=6,
        truncation=0.12,
        neutralization="industry",
    )

    assert captured["expr"] == "rank(close)"
    assert captured["settings"] == {
        "region": "EUR",
        "universe": "TOP1200",
        "delay": 0,
        "neutralization": "INDUSTRY",
        "decay": 6,
        "truncation": 0.12,
    }
    assert captured["saved"] == {
        "region": "EUR",
        "universe": "TOP1200",
        "config_key": "EUR|TOP1200|delay=0|INDUSTRY|decay=6|truncation=0.12",
    }


def test_run_ga_with_progress_noi_callback_va_tra_ket_qua():
    """_run_ga_with_progress phải gọi opt.run với on_generation/on_simulation
    (để hiện tiến trình) và trả đúng kết quả của opt.run."""
    from src.optimization.evolution import GenerationStats

    captured = {}

    class _FakeOpt:
        def run(self, on_generation=None, on_simulation=None):
            captured["has_gen"] = on_generation is not None
            captured["has_sim"] = on_simulation is not None
            # Mô phỏng vài lần gọi callback để chắc không vỡ (truy cập đúng field).
            on_simulation(1, "rank(close)", 0.5)
            on_generation(GenerationStats(0, 1.23, 0.4, "rank(close)"))
            return ["node1", "node2"]

    result = main._run_ga_with_progress(_FakeOpt(), total=3)

    assert result == ["node1", "node2"]
    assert captured == {"has_gen": True, "has_sim": True}


def test_run_ga_truyen_fixed_sim_config_xuong_optimizer(monkeypatch):
    import src.generation.template as template_mod
    import src.optimization.evolution as evolution_mod

    captured = {}

    class _FakeTemplateGenerator:
        def __init__(self, *a, **k):
            pass

        def generate(self, count):
            return ["rank(close)"]

    class _FakeOptimizer:
        history = [type("_H", (), {"best_expression": "rank(close)"})()]
        simulations_used = 0

        def __init__(self, **kwargs):
            captured["simulation_settings"] = kwargs.get("simulation_settings")

        @staticmethod
        def expr_to_node(expr):
            return expr

        def run(self):
            return []

    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(main, "_cached_symbols", lambda sf: (["close"], {"rank"}, {"close": "MATRIX"}, {"rank"}, {"rank": 1}))
    monkeypatch.setattr(main, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(template_mod, "TemplateGenerator", _FakeTemplateGenerator)
    monkeypatch.setattr(evolution_mod, "GeneticOptimizer", _FakeOptimizer)

    main.run_ga(
        population=1,
        generations=1,
        region="EUR",
        universe="TOP1200",
        delay=0,
        decay=6,
        truncation=0.12,
        neutralization="industry",
        seed_llm=False,
        max_sims=0,
    )

    assert captured["simulation_settings"] == {
        "region": "EUR",
        "universe": "TOP1200",
        "delay": 0,
        "neutralization": "INDUSTRY",
        "decay": 6,
        "truncation": 0.12,
    }


def test_run_ga_truyen_operator_arity_vao_prefilter(monkeypatch):
    import src.generation.template as template_mod
    import src.optimization.evolution as evolution_mod

    captured = {}

    class _FakeTemplateGenerator:
        def __init__(self, *a, **k):
            pass

        def generate(self, count):
            return ["rank(close)"]

    class _FakeOptimizer:
        history = [type("_H", (), {"best_expression": "rank(close)"})()]
        simulations_used = 0

        def __init__(self, **kwargs):
            captured["operator_arity"] = kwargs["prefilter"].operator_arity

        @staticmethod
        def expr_to_node(expr):
            return expr

        def run(self):
            return []

    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(
        main,
        "_cached_symbols",
        lambda sf: (["close"], {"rank"}, {"close": "MATRIX"}, {"rank"}, {"rank": 1}),
    )
    monkeypatch.setattr(main, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(template_mod, "TemplateGenerator", _FakeTemplateGenerator)
    monkeypatch.setattr(evolution_mod, "GeneticOptimizer", _FakeOptimizer)

    main.run_ga(
        population=1,
        generations=1,
        region="USA",
        universe="TOP3000",
        delay=1,
        decay=0,
        truncation=0.08,
        neutralization="SUBINDUSTRY",
        seed_llm=False,
        max_sims=0,
    )

    assert captured["operator_arity"] == {"rank": 1}


def test_research_truyen_fixed_sim_config_xuong_loop_builder(monkeypatch):
    captured = {}

    def _fake_builder(session_factory, client, region, universe, delay, max_sims, patience,
                      align=True, regularize=False, penalty_lambda=0.3, sim_config=None):
        captured["scope"] = (region, universe, delay)
        captured["sim_config"] = sim_config
        return object(), object()

    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(main, "_cached_symbols", lambda sf: (["close"], {"rank"}, {"close": "MATRIX"}, {"rank"}, {"rank": 1}))
    monkeypatch.setattr(main, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(main, "_make_research_loop", _fake_builder)
    monkeypatch.setattr(main, "_run_research_with_progress", lambda *a, **k: object())
    monkeypatch.setattr(main, "_render_research_result", lambda *a, **k: None)

    main.research(
        direction="test",
        region="EUR",
        universe="TOP1200",
        delay=0,
        max_sims=1,
        decay=6,
        truncation=0.12,
        neutralization="industry",
    )

    assert captured["scope"] == ("EUR", "TOP1200", 0)
    assert captured["sim_config"] == SimConfig(
        region="EUR",
        universe="TOP1200",
        delay=0,
        decay=6,
        truncation=0.12,
        neutralization="INDUSTRY",
    )


def test_start_menu_truyen_sim_settings_xuong_run_auto(monkeypatch):
    captured = {}
    answers = iter(["5", "6", "0.12", "industry", "0"])

    class _FakeState:
        region = "EUR"
        universe = "TOP1200"
        delay = 0
        client = _FakeClient()

        @property
        def logged_in(self):
            return True

    def _fake_run_auto(region, universe, delay, **kwargs):
        captured.update(
            {
                "scope": (region, universe, delay),
                "decay": kwargs["decay"],
                "truncation": kwargs["truncation"],
                "neutralization": kwargs["neutralization"],
            }
        )
        from src.pipeline.auto import AutoResult
        return AutoResult([], directions_run=0, total_sims=0, stop_reason="test")

    monkeypatch.setattr(main, "_MenuState", _FakeState)
    monkeypatch.setattr(main, "_run_auto", _fake_run_auto)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    main.start()

    assert captured == {
        "scope": ("EUR", "TOP1200", 0),
        "decay": 6,
        "truncation": 0.12,
        "neutralization": "INDUSTRY",
    }


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
