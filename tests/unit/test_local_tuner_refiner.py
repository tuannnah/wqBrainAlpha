"""Test đỏ->xanh cho LocalTunerRefiner (Task 4): adapter refine_and_sim KHÔNG dùng LLM,
tune local (Task 3) trước rồi mới quyết định có đáng sim Brain hay không."""

from __future__ import annotations

import numpy as np

from src.app.closed_loop_adapters import LocalTunerRefiner
from src.backtest.config import PortfolioConfig
from src.pipeline.closed_loop import QuotaExhausted
from src.pipeline.shortlist import ShortlistCandidate
from src.simulation.config import SimConfig
from src.simulation.simulator import QuotaExceededError, SimulationResult


class _Repo:
    def save_alpha(self, *a, **k):
        return "alpha-1"

    def save_simulation(self, *a, **k):
        return None


def _cand(expr="rank(ts_delta(close, 5))"):
    return ShortlistCandidate(expr=expr, metrics=None, pnl=np.zeros(3), dates=np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]"))


def _refiner(simulate, *, tune_local_sharpe=1.5, pool_corr_fn=None):
    from src.backtest.local_tuner import TuneResult

    class _Sim:
        def __init__(self):
            self.calls = 0

        def simulate(self, expr, settings=None):
            self.calls += 1
            return simulate(expr, settings)

    def fake_tune(expr, cfg, data, **kw):
        return TuneResult(best_expr="rank(ts_delta(close, 20))",
                          best_config=PortfolioConfig(decay=3, truncation=0.02),
                          local_sharpe=tune_local_sharpe)

    r = LocalTunerRefiner(
        simulator=_Sim(), repo=_Repo(), data=object(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(), pool_corr_fn=pool_corr_fn,
        tune_fn=fake_tune,
    )
    return r


def _passed_result(expr, settings):
    # drawdown PHẢI khai rõ: normalize() mặc định drawdown=1.0 khi thiếu (an toàn/xấu nhất),
    # mà hard_filter mặc định đòi drawdown < 0.20 -> thiếu drawdown sẽ luôn fail hard filter.
    return SimulationResult(expression=expr, alpha_id="wq-1", status="passed",
                            sharpe=1.9, fitness=1.3, turnover=0.3, drawdown=0.1, raw={})


def test_refiner_sim_dung_1_lan_va_ap_config_tuned():
    r = _refiner(_passed_result)
    out = r.refine_and_sim(_cand())
    assert r.simulator.calls == 1
    assert out.expr == "rank(ts_delta(close, 20))"
    assert out.passed is True
    assert out.sharpe == 1.9
    assert out.sims_used == 1


def test_refiner_duoi_floor_thi_khong_sim():
    r = _refiner(_passed_result, tune_local_sharpe=0.1)  # < PRE_SIM_LOCAL_SHARPE_FLOOR=0.5
    out = r.refine_and_sim(_cand())
    assert r.simulator.calls == 0
    assert out.passed is False
    assert out.sims_used == 0
    assert out.stop_reason == "local_floor"


def test_refiner_quota_thi_nem_QuotaExhausted():
    def boom(expr, settings):
        raise QuotaExceededError("het quota ngay")
    r = _refiner(boom)
    try:
        r.refine_and_sim(_cand())
        assert False, "phải ném QuotaExhausted"
    except QuotaExhausted:
        pass


def test_refiner_luu_local_eval_cho_calibration(monkeypatch):
    """Khi có calib_repo, refiner lưu expression + evaluation local của expr ĐÃ TUNE (theo
    hash) -> join brain_local_sharpe_pairs khớp -> ρ local↔Brain thu được dữ liệu.

    Fixture (Task 4): local_metrics KHÔNG None ở đây (khác các test refiner khác) vì đúng là
    thứ đang được kiểm — nên KHÔNG dùng data=object() suông được nữa: refine_and_sim giờ chạy
    gate sub_universe_ok thật khi local_metrics is not None, mà object() không phải MarketData
    (không có .field/.universe) sẽ vỡ. Test này chỉ nhắm calib-save (đã có test_sub_universe.py
    riêng cho gate) nên monkeypatch sub_universe_ok về True — giữ data=object() gọn, tách bạch
    mối quan tâm, tránh phụ thuộc kết quả backtest thật trên panel giả (dễ chập chờn)."""
    monkeypatch.setattr("src.backtest.sub_universe.sub_universe_ok", lambda *a, **kw: True)
    from src.backtest.local_tuner import TuneResult
    from src.backtest.metrics_local import AlphaMetrics

    metrics = AlphaMetrics(
        sharpe=1.6, annual_return=0.2, turnover=0.3, max_drawdown=0.1,
        fitness=1.2, per_year_sharpe={2020: 1.5}, weight_concentration=0.05,
    )

    class _Sim:
        def simulate(self, expr, settings=None):
            return _passed_result(expr, settings)

    class _Calib:
        def __init__(self):
            self.upserts = []
            self.evals = []

        def upsert_expression(self, expr_string, canonical_hash, depth, complexity, fields):
            self.upserts.append(expr_string)
            return 7

        def record_evaluation(self, expression_id, config_json, data_window,
                              metrics, self_corr_max, status, fail_reasons, seed):
            self.evals.append((expression_id, metrics))
            return 1

    def fake_tune(expr, cfg, data, **kw):
        return TuneResult(
            best_expr="rank(ts_delta(close, 20))",
            best_config=PortfolioConfig(decay=3, truncation=0.02),
            local_sharpe=1.6, local_metrics=metrics,
        )

    calib = _Calib()
    r = LocalTunerRefiner(
        simulator=_Sim(), repo=_Repo(), data=object(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(), tune_fn=fake_tune, calib_repo=calib,
    )
    out = r.refine_and_sim(_cand())
    assert calib.upserts == ["rank(ts_delta(close, 20))"]   # lưu expr ĐÃ tune (không phải core gốc)
    assert calib.evals and calib.evals[0][1] is metrics      # evaluation mang AlphaMetrics local
    assert out.passed is True


def test_refiner_crowded_thi_khong_pass():
    r = _refiner(_passed_result, pool_corr_fn=lambda aid: 0.9)  # >= 0.70
    out = r.refine_and_sim(_cand())
    assert out.passed is False
    assert out.self_corr == 0.9
