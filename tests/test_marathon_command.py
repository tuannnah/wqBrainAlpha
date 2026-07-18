"""Lệnh `marathon` trong main.py: config khởi đầu decay=4/truncation=0.01/MARKET,
bật referee+config_tuner (marathon=True), và truyền đúng tham số xuống run_marathon."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import main
from src.simulation.config import SimConfig


class _FakeClient:
    authenticated = True

    def authenticate(self, *a, **k):
        return None


def test_marathon_defaults_decay4_trunc001_market():
    """Khoá yêu cầu: mặc định lệnh phải là decay=4, truncation=0.01, neutralization=MARKET."""
    params = inspect.signature(main.marathon).parameters
    assert params["decay"].default.default == 4
    assert params["truncation"].default.default == 0.01
    assert params["neutralization"].default.default == "MARKET"


def test_marathon_wiring(monkeypatch):
    captured = {}

    def _fake_builder(session_factory, client, region, universe, delay, max_sims, patience,
                      *, marathon=False, sim_config=None, **kw):
        captured["max_sims"] = max_sims
        captured["patience"] = patience
        captured["marathon"] = marathon
        captured["sim_config"] = sim_config
        deepseek = SimpleNamespace(
            usage=SimpleNamespace(total_tokens=0, estimated_cost=lambda: 0.0)
        )
        return object(), deepseek

    def _fake_run_marathon(provider, run_direction, *, max_retries=2, on_event=None):
        captured["max_retries"] = max_retries
        captured["has_provider"] = callable(provider)
        captured["has_run_direction"] = callable(run_direction)
        return main.MarathonReport(directions_completed=2, total_sims=7, total_zoo_added=3,
                                   stop_reason="quota")

    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(main.cli_common, "_cached_symbols",
                        lambda sf: (["close"], {"rank"}, {"close": "MATRIX"}, {"rank"}, {"rank": 1}))
    monkeypatch.setattr(main.cli_common, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(main, "_make_research_loop", _fake_builder)
    monkeypatch.setattr(main, "run_marathon", _fake_run_marathon)

    main.marathon(
        region="USA", universe="TOP3000", delay=1,
        decay=4, truncation=0.01, neutralization="MARKET",
        per_direction_sims=30, max_patience=8, retry=3,
    )

    assert captured["marathon"] is True
    assert captured["max_sims"] == 30
    assert captured["patience"] == 8
    assert captured["max_retries"] == 3
    assert captured["has_provider"] and captured["has_run_direction"]
    assert captured["sim_config"] == SimConfig(
        region="USA", universe="TOP3000", delay=1,
        decay=4, truncation=0.01, neutralization="MARKET",
    )
