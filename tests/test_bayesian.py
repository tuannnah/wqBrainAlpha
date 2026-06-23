"""Tests for optional Bayesian template tuning."""

from __future__ import annotations

from src.optimization.bayesian import tune_template


class _SettingsSimulator:
    def __init__(self):
        self.calls = []

    def simulate(self, expr, settings=None):
        self.calls.append((expr, settings))
        return object()


def test_tune_template_truyen_simulation_settings_vao_simulator():
    sim = _SettingsSimulator()
    settings = {
        "region": "EUR",
        "universe": "TOP1200",
        "delay": 0,
        "neutralization": "INDUSTRY",
        "decay": 6,
        "truncation": 0.12,
    }

    tune_template(
        "rank(ts_delta(close, {d1}))",
        sim,
        n_trials=1,
        scorer=lambda result: 1.0,
        simulation_settings=settings,
    )

    assert sim.calls
    assert sim.calls[0][1] == settings
