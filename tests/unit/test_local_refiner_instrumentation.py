"""Pha 0 wiring: LocalTunerRefiner phải ĐIỀN trường chẩn đoán vào IdeaOutcome
(stage_reached/fail_check/family/expr_depth/dedup_key/local_sharpe/timing) — không còn vứt
_reasons của hard_filter (closed_loop_adapters.py cũ dòng 188)."""

from __future__ import annotations

from src.app.closed_loop_adapters import LocalTunerRefiner
from src.backtest.config import PortfolioConfig
from src.backtest.local_tuner import TuneResult
from src.simulation.config import SimConfig
from src.simulation.simulator import SimulationResult
from src.pipeline.shortlist import ShortlistCandidate


class _Repo:
    def save_alpha(self, *a, **k):
        return "a1"

    def save_simulation(self, *a, **k):
        return None


class _SimPass:
    def simulate(self, expr, settings=None):
        return SimulationResult(expression=expr, alpha_id="wq-1", status="passed",
                                sharpe=1.7, fitness=1.2, turnover=0.3, drawdown=0.05, raw={})


class _SimLowFitness:
    def simulate(self, expr, settings=None):
        # Sharpe qua nhưng fitness thấp -> hard_filter fail LOW_FITNESS (drawdown thấp để
        # không lẫn HIGH_DRAWDOWN).
        return SimulationResult(expression=expr, alpha_id="wq-2", status="passed",
                                sharpe=1.3, fitness=0.5, turnover=0.3, drawdown=0.05, raw={})


def _cand(expr):
    import numpy as np
    return ShortlistCandidate(expr=expr, metrics=None, pnl=np.zeros(0),
                              dates=np.zeros(0, dtype="datetime64[ns]"))


def test_local_floor_dien_stage_va_family():
    expr = "multiply(-1, ts_mean(subtract(close, vwap), 10))"

    def fake_tune(e, cfg, data, **kw):
        return TuneResult(best_expr=e, best_config=PortfolioConfig(decay=4, truncation=0.08),
                          local_sharpe=0.30)  # dưới sàn 0.5 -> local_floor

    r = LocalTunerRefiner(simulator=_SimPass(), repo=_Repo(), data=object(),
                          local_config=PortfolioConfig(decay=4, truncation=0.08),
                          sim_config=SimConfig.default(), tune_fn=fake_tune)
    o = r.refine_and_sim(_cand(expr))
    assert o.stop_reason == "local_floor"
    assert o.stage_reached == "local_floor"
    assert o.fail_check == "LOW_SHARPE"        # local sharpe dưới sàn
    assert o.family == "pv_reversal"
    assert o.expr_depth is not None and o.expr_depth > 0
    assert o.dedup_key
    assert o.local_sharpe == 0.30
    assert o.backtest_ms is not None           # đã đo thời gian tune


def test_simmed_failed_giu_fail_check_tu_hard_filter():
    expr = "ts_delta(close, 60)"

    def fake_tune(e, cfg, data, **kw):
        return TuneResult(best_expr=e, best_config=PortfolioConfig(decay=4, truncation=0.08),
                          local_sharpe=1.6)  # qua sàn -> đi tới sim

    r = LocalTunerRefiner(simulator=_SimLowFitness(), repo=_Repo(), data=object(),
                          local_config=PortfolioConfig(decay=4, truncation=0.08),
                          sim_config=SimConfig.default(), tune_fn=fake_tune)
    o = r.refine_and_sim(_cand(expr))
    assert o.sims_used == 1
    assert o.stage_reached == "simmed"
    assert o.fail_check == "LOW_FITNESS"       # _reasons hard_filter được giữ, không vứt
    assert o.family == "momentum"
    assert o.sim_ms is not None


def test_passed_stage_la_passed():
    expr = "ts_delta(close, 60)"

    def fake_tune(e, cfg, data, **kw):
        return TuneResult(best_expr=e, best_config=PortfolioConfig(decay=4, truncation=0.08),
                          local_sharpe=1.6)

    r = LocalTunerRefiner(simulator=_SimPass(), repo=_Repo(), data=object(),
                          local_config=PortfolioConfig(decay=4, truncation=0.08),
                          sim_config=SimConfig.default(), tune_fn=fake_tune)
    o = r.refine_and_sim(_cand(expr))
    assert o.passed is True
    assert o.stage_reached == "passed"
    assert o.fail_check == ""
