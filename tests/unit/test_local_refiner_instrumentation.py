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
    assert o.stop_reason.startswith("local_floor")  # nay kèm ngưỡng calibrated để audit
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


def test_depth_guard_chan_truoc_backtest():
    """Pha 1.3: biểu thức depth > MAX_DEPTH bị loại TRƯỚC backtest (0 sim, không gọi tune)."""
    from config.thresholds import MAX_DEPTH

    # Cây lồng sâu > MAX_DEPTH (7): ts_delta lồng nhiều tầng.
    deep = "close"
    for _ in range(MAX_DEPTH + 2):
        deep = f"ts_delta({deep}, 5)"

    tune_called = {"n": 0}

    def fake_tune(e, cfg, data, **kw):
        tune_called["n"] += 1
        from src.backtest.local_tuner import TuneResult
        return TuneResult(best_expr=e, best_config=PortfolioConfig(decay=4, truncation=0.08),
                          local_sharpe=1.6)

    r = LocalTunerRefiner(simulator=_SimPass(), repo=_Repo(), data=object(),
                          local_config=PortfolioConfig(decay=4, truncation=0.08),
                          sim_config=SimConfig.default(), tune_fn=fake_tune)
    o = r.refine_and_sim(_cand(deep))
    assert o.sims_used == 0
    assert o.stop_reason == "depth"
    assert o.stage_reached == "depth"
    assert o.fail_check == "DEPTH"
    assert tune_called["n"] == 0          # KHÔNG backtest


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
    assert o.is_brain_sim is True                # đã chạm Brain thật


def test_simmed_failed_is_brain_sim_true():
    """Test (b) yêu cầu: sim thật rớt sharpe/fitness thấp vẫn phải stage_reached=='simmed' VÀ
    is_brain_sim=True (khác nhánh pre-sim reject bên dưới)."""
    expr = "ts_delta(close, 60)"

    def fake_tune(e, cfg, data, **kw):
        return TuneResult(best_expr=e, best_config=PortfolioConfig(decay=4, truncation=0.08),
                          local_sharpe=1.6)

    r = LocalTunerRefiner(simulator=_SimLowFitness(), repo=_Repo(), data=object(),
                          local_config=PortfolioConfig(decay=4, truncation=0.08),
                          sim_config=SimConfig.default(), tune_fn=fake_tune)
    o = r.refine_and_sim(_cand(expr))
    assert o.stage_reached == "simmed"
    assert o.is_brain_sim is True
    assert o.sims_used == 1
    assert o.presim_reason is None


class _SimPresimOperatorInvalid:
    """Giả lập Simulator thật khi PreFilter loại vì operator không có trong catalog — ĐÃ trả
    presim_reason (Task 3) thay vì chỉ status='error' chung chung."""

    def simulate(self, expr, settings=None):
        return SimulationResult(
            expression=expr, status="error",
            raw={"error": "pre-sim reject: Operator không tồn tại: fake_op"},
            presim_reason="Operator không tồn tại: fake_op",
        )


def test_presim_reject_operator_invalid_khong_gia_vo_da_sim():
    """Test (a) yêu cầu: expr có op ngoài catalog giả -> outcome trung thực, KHÔNG còn giả vờ
    'simmed/LOW_SHARPE' như bug cũ (spec C2). Expr phải PARSE được thật (registry ngôn ngữ đầy
    đủ) — reject xảy ra ở catalog HẸP HƠN của pre_sim_validator (vd danh sách operator Brain
    trả về), không phải ở parser; presim_reason mô phỏng đúng kịch bản đó."""
    expr = "rank(close)"

    def fake_tune(e, cfg, data, **kw):
        # local_sharpe đủ cao để KHÔNG bị chặn ở local_floor -> đi tới sim, nơi bug nằm.
        return TuneResult(best_expr=e, best_config=PortfolioConfig(decay=4, truncation=0.08),
                          local_sharpe=1.6)

    r = LocalTunerRefiner(simulator=_SimPresimOperatorInvalid(), repo=_Repo(), data=object(),
                          local_config=PortfolioConfig(decay=4, truncation=0.08),
                          sim_config=SimConfig.default(), tune_fn=fake_tune)
    o = r.refine_and_sim(_cand(expr))
    assert o.stage_reached == "op_invalid"
    assert o.fail_check == "OPERATOR_INVALID"
    assert o.sims_used == 0
    assert o.is_brain_sim is False
    assert o.presim_reason == "Operator không tồn tại: fake_op"
    assert o.passed is False


class _SimPresimFieldInvalid:
    def simulate(self, expr, settings=None):
        return SimulationResult(
            expression=expr, status="error",
            raw={"error": "pre-sim reject: Field/hằng không tồn tại: fake_field"},
            presim_reason="Field/hằng không tồn tại: fake_field",
        )


def test_presim_reject_field_invalid():
    expr = "rank(fake_field)"

    def fake_tune(e, cfg, data, **kw):
        return TuneResult(best_expr=e, best_config=PortfolioConfig(decay=4, truncation=0.08),
                          local_sharpe=1.6)

    r = LocalTunerRefiner(simulator=_SimPresimFieldInvalid(), repo=_Repo(), data=object(),
                          local_config=PortfolioConfig(decay=4, truncation=0.08),
                          sim_config=SimConfig.default(), tune_fn=fake_tune)
    o = r.refine_and_sim(_cand(expr))
    assert o.stage_reached == "field_invalid"
    assert o.fail_check == "FIELD_INVALID"
    assert o.sims_used == 0
    assert o.is_brain_sim is False
    assert o.presim_reason == "Field/hằng không tồn tại: fake_field"
