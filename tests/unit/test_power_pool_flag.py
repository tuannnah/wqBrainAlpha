"""Test đỏ->xanh cho helper is_power_pool (Task 3): cờ Power Pool eligibility
(Sharpe>=1.0, <=8 operator, <=3 field trừ grouping, self_corr None hoặc <=0.5)."""

from __future__ import annotations

import numpy as np

import src.operators_local  # noqa: F401
from src.app.closed_loop_adapters import LocalTunerRefiner, is_power_pool
from src.backtest.config import PortfolioConfig
from src.lang.registry import default_registry
from src.pipeline.shortlist import ShortlistCandidate
from src.simulation.config import SimConfig
from src.simulation.simulator import SimulationResult


def test_power_pool_dat_khi_don_gian_va_sharpe_du():
    reg = default_registry()
    assert is_power_pool("rank(ts_delta(close, 5))", 1.2, 0.3, reg) is True


def test_power_pool_khong_dat_khi_sharpe_thap():
    reg = default_registry()
    assert is_power_pool("rank(ts_delta(close, 5))", 0.8, 0.3, reg) is False


def test_power_pool_khong_dat_khi_self_corr_cao():
    reg = default_registry()
    assert is_power_pool("rank(ts_delta(close, 5))", 1.5, 0.6, reg) is False


def test_power_pool_khong_dat_khi_qua_nhieu_field():
    reg = default_registry()
    # 4 field khác nhau > 3
    expr = "add(add(close, open), add(high, low))"
    assert is_power_pool(expr, 1.5, 0.1, reg) is False


class _RepoGia:
    """Fake repo tối thiểu cho LocalTunerRefiner.refine_and_sim (save_alpha/save_simulation)."""

    def save_alpha(self, *a, **k):
        return "alpha-pp"

    def save_simulation(self, *a, **k):
        return None


def _cand_gia(expr: str = "close") -> ShortlistCandidate:
    return ShortlistCandidate(
        expr=expr, metrics=None, pnl=np.zeros(3),
        dates=np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]"),
    )


def test_power_pool_eligible_doc_lap_voi_passed_regular(monkeypatch):
    """power_pool_eligible phải tính ĐỘC LẬP với `passed` Regular: alpha Brain Sharpe=1.1,
    fitness=0.9 (fail Regular vì cần Sharpe>1.25) nhưng vẫn <=8 operator/<=3 field, self_corr=0.3
    (<=0.5), turnover=0.3 (trong [0.01, 0.70]) -> ĐẠT Power Pool (bar thấp hơn: Sharpe>=1.0).
    Trước đây `power_pool = passed and is_power_pool(...)` khiến các ứng viên này bị bỏ sót."""
    # Bỏ qua gate proxy sub-universe (Task 4) — không phải điều test này nhắm tới.
    monkeypatch.setattr("src.backtest.sub_universe.sub_universe_ok", lambda *a, **kw: True)
    from src.backtest.local_tuner import TuneResult

    def fake_tune(expr, cfg, data, **kw):
        return TuneResult(
            best_expr="rank(ts_delta(close, 5))",
            best_config=PortfolioConfig(decay=3, truncation=0.02),
            local_sharpe=1.6,
        )

    class _SimGia:
        def simulate(self, expr, settings=None):
            return SimulationResult(
                expression=expr, alpha_id="wq-pp", status="passed",
                sharpe=1.1, fitness=0.9, turnover=0.3, drawdown=0.1, raw={},
            )

    r = LocalTunerRefiner(
        simulator=_SimGia(), repo=_RepoGia(), data=object(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(), tune_fn=fake_tune,
        pool_corr_fn=lambda aid: 0.3,
    )
    out = r.refine_and_sim(_cand_gia())
    assert out.passed is False               # fail Regular (Sharpe 1.1 < 1.25, fitness 0.9 <= 1.0)
    assert out.power_pool_eligible is True    # nhưng vẫn đạt Power Pool (Sharpe >= 1.0)
