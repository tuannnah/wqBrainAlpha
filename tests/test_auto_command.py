"""Test lệnh `simulate` và `research` trong main.py (sau khi hợp nhất về RefinementLoop).

Engine hybrid (lệnh `auto`/`start`) đã được gỡ — chỉ còn `research` là lệnh sinh
alpha chính. Các test ở đây kiểm: lệnh `simulate` truyền đúng sim config, và lệnh
`research` truyền đúng config + cờ (oos/deflate/regime/reseed) xuống loop builder.
"""

from __future__ import annotations

from datetime import date

from src.app.cli import research as cli_research
from src.app.cli import simulate as cli_simulate
from src.app.power_pool_config import resolve_theme_sim_config
from src.simulation.config import SimConfig


class _FakeClient:
    authenticated = True

    def authenticate(self, *a, **k):
        return None


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

    monkeypatch.setattr(cli_simulate, "init_db", lambda e: e)
    monkeypatch.setattr(cli_simulate, "make_engine", lambda: None)
    monkeypatch.setattr(cli_simulate, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(cli_simulate, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(cli_simulate, "Simulator", _FakeSimulator)
    monkeypatch.setattr(cli_simulate, "AlphaRepository", _FakeRepo)

    cli_simulate.simulate(
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
        "testPeriod": "P0Y0M",
        "maxTrade": "OFF",
        "maxPosition": "OFF",
    }
    assert captured["saved"] == {
        "region": "EUR",
        "universe": "TOP1200",
        "config_key": (
            "EUR|TOP1200|delay=0|INDUSTRY|decay=6|truncation=0.12|"
            "test_period=P0Y0M|max_trade=OFF|max_position=OFF"
        ),
    }


def test_research_truyen_fixed_sim_config_xuong_loop_builder(monkeypatch):
    captured = {}

    def _fake_builder(session_factory, client, region, universe, delay, max_sims, patience,
                      align=True, regularize=False, penalty_lambda=0.3, sim_config=None,
                      oos_min_ratio=None, deflate_haircut=0.0, regime_min=None, align_gate=True,
                      improve_margin=0.0, reseed_every=0):
        captured["scope"] = (region, universe, delay)
        captured["align_gate"] = align_gate
        captured["sim_config"] = sim_config
        captured["oos_min_ratio"] = oos_min_ratio
        captured["deflate_haircut"] = deflate_haircut
        captured["regime_min"] = regime_min
        captured["reseed_every"] = reseed_every
        return object(), object()

    monkeypatch.setattr(cli_research, "init_db", lambda e: e)
    monkeypatch.setattr(cli_research, "make_engine", lambda: None)
    monkeypatch.setattr(cli_research, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(cli_research, "_cached_symbols", lambda sf: (["close"], {"rank"}, {"close": "MATRIX"}, {"rank"}, {"rank": 1}))
    monkeypatch.setattr(cli_research, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(cli_research, "_make_research_loop", _fake_builder)
    monkeypatch.setattr(cli_research, "_run_research_with_progress", lambda *a, **k: object())
    monkeypatch.setattr(cli_research, "_render_research_result", lambda *a, **k: None)

    cli_research.research(
        direction="test",
        region="EUR",
        universe="TOP1200",
        delay=0,
        max_sims=1,
        decay=6,
        truncation=0.12,
        neutralization="industry",
        oos_ratio=0.0,
        deflate=0.0,
        min_annual_sharpe=0.0,
        align_soft=False,
        improve_margin=0.0,
        reseed_every=0,
    )

    assert captured["scope"] == ("EUR", "TOP1200", 0)
    assert captured["reseed_every"] == 0
    assert captured["oos_min_ratio"] is None  # oos_ratio=0 -> tắt gate
    assert captured["deflate_haircut"] == 0.0
    assert captured["regime_min"] is None
    assert captured["sim_config"] == SimConfig(
        region="EUR",
        universe="TOP1200",
        delay=0,
        decay=6,
        truncation=0.12,
        neutralization="INDUSTRY",
    )


def test_wiring_theme_ap_top1000_cho_hom_nay_trong_lich():
    base = SimConfig.default(region="USA", universe="TOP3000", delay=1)
    res = resolve_theme_sim_config(base, date(2026, 7, 9))
    # Đây là hợp đồng main.py dựa vào: có theme -> TOP1000 + tập risk-neut không rỗng.
    assert res.sim_config.universe == "TOP1000"
    assert res.allowed_neutralizations
