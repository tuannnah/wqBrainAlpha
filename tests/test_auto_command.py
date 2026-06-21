"""Test lệnh `auto`, `simulate`, `research` và menu `start` trong main.py.

Bao gồm: _run_auto dựng HybridEngine đúng cách, lệnh `auto` không còn --engine
(chỉ chạy hybrid), menu mục 4/5 gọi _run_auto với scope cụ thể (không phải
OptionInfo của Typer), và lệnh `research` truyền đúng sim config.
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

        def run(self, on_generation=None, on_simulation=None):
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


def test_research_truyen_fixed_sim_config_xuong_loop_builder(monkeypatch):
    captured = {}

    def _fake_builder(session_factory, client, region, universe, delay, max_sims, patience,
                      align=True, regularize=False, penalty_lambda=0.3, sim_config=None,
                      oos_min_ratio=None):
        captured["scope"] = (region, universe, delay)
        captured["sim_config"] = sim_config
        captured["oos_min_ratio"] = oos_min_ratio
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
        oos_ratio=0.0,
    )

    assert captured["scope"] == ("EUR", "TOP1200", 0)
    assert captured["oos_min_ratio"] is None  # oos_ratio=0 -> tắt gate
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
                      decay=0, truncation=0.08, neutralization="SUBINDUSTRY",
                      no_llm_seed=False):
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


def test_menu_neutralization_chon_theo_so(monkeypatch):
    """Menu neutralization: chọn bằng số trả về đúng tên option."""
    # Option số 1 là mặc định SUBINDUSTRY; chọn số khác (vd MARKET) phải đúng.
    idx = main._NEUTRALIZATION_MENU.index("MARKET") + 1
    monkeypatch.setattr("builtins.input", lambda prompt="": str(idx))
    assert main._menu_ask_neutralization() == "MARKET"


def test_menu_neutralization_enter_dung_mac_dinh(monkeypatch):
    """Bỏ trống (Enter) → mặc định SUBINDUSTRY."""
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert main._menu_ask_neutralization() == "SUBINDUSTRY"


def test_menu_neutralization_van_cho_go_ten(monkeypatch):
    """Vẫn cho gõ tên trực tiếp (không phân biệt hoa thường)."""
    monkeypatch.setattr("builtins.input", lambda prompt="": "industry")
    assert main._menu_ask_neutralization() == "INDUSTRY"


def test_menu_neutralization_khong_hop_le_ve_mac_dinh(monkeypatch):
    """Nhập rác (số ngoài dải hoặc tên sai) → quay về SUBINDUSTRY."""
    monkeypatch.setattr("builtins.input", lambda prompt="": "999")
    assert main._menu_ask_neutralization() == "SUBINDUSTRY"
    monkeypatch.setattr("builtins.input", lambda prompt="": "xyz")
    assert main._menu_ask_neutralization() == "SUBINDUSTRY"
