"""Nhánh SIM-THẲNG của LocalTunerRefiner cho seed alt-data: khi expr dùng field NGOÀI panel
local (`local_usable == False`) thì BỎ tune/floor local (không chấm được) và sim Brain 1 lần
với neutralization theo category dataset. Đây là đường đưa alpha alt-data (option/social) tới
Brain — gỡ rào 'mọi alt-data chết ở local_floor' của refiner mặc định."""

from __future__ import annotations

import numpy as np

from src.app.closed_loop_adapters import LocalTunerRefiner
from src.backtest.config import PortfolioConfig
from src.pipeline.closed_loop import QuotaExhausted
from src.pipeline.shortlist import ShortlistCandidate
from src.simulation.config import SimConfig
from src.simulation.simulator import QuotaExceededError, SimulationResult


class _Repo:
    def __init__(self):
        self.saved = []

    def save_alpha(self, expr, **k):
        self.saved.append((expr, k.get("source")))
        return "alpha-1"

    def save_simulation(self, *a, **k):
        return None


class _PVData:
    """Panel local chỉ có price/volume — alt-data field không nằm trong đây."""

    def field_names(self):
        return {"close", "open", "vwap", "volume", "high", "low", "returns"}


def _cand(expr):
    return ShortlistCandidate(
        expr=expr, metrics=None, pnl=np.zeros(0),
        dates=np.zeros(0, dtype="datetime64[ns]"),
    )


def _alt_result(expr, settings):
    return SimulationResult(
        expression=expr, alpha_id="wq-1", status="passed",
        sharpe=1.4, fitness=1.05, turnover=0.3, drawdown=0.1, raw={},
    )


def _boom_tune(*a, **k):
    raise AssertionError("nhánh alt-data KHÔNG được gọi tune local")


def _refiner(simulate, *, repo=None, pool_corr_fn=None):
    class _Sim:
        def __init__(self):
            self.calls = 0
            self.last_settings = None

        def simulate(self, expr, settings=None):
            self.calls += 1
            self.last_settings = settings
            return simulate(expr, settings)

    return LocalTunerRefiner(
        simulator=_Sim(), repo=repo or _Repo(), data=_PVData(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(), pool_corr_fn=pool_corr_fn,
        tune_fn=_boom_tune,  # nếu bị gọi -> test đỏ (chứng minh đã BỎ tune)
    )


_ALT_EXPR = "multiply(-1, ts_mean(snt_social_value, 5))"


def test_altdata_sim_thang_khong_tune():
    repo = _Repo()
    r = _refiner(_alt_result, repo=repo)
    out = r.refine_and_sim(_cand(_ALT_EXPR))
    assert r.simulator.calls == 1              # có sim Brain
    assert out.sims_used == 1
    assert out.stop_reason == "alt_data_direct"
    assert out.passed is True
    assert out.sharpe == 1.4
    assert repo.saved and repo.saved[0][1] == "alt_data"  # nguồn ghi rõ là alt_data


def test_altdata_ap_neutralization_theo_category():
    r = _refiner(_alt_result)
    r.refine_and_sim(_cand(_ALT_EXPR))          # social → SUBINDUSTRY
    assert r.simulator.last_settings.get("neutralization") == "SUBINDUSTRY"


def test_altdata_option_dung_sector():
    expr = ("multiply(-1, subtract(ts_backfill(implied_volatility_put_30, 22), "
            "ts_backfill(implied_volatility_call_30, 22)))")
    r = _refiner(_alt_result)
    r.refine_and_sim(_cand(expr))
    assert r.simulator.last_settings.get("neutralization") == "SECTOR"


def test_altdata_quota_thi_nem_QuotaExhausted():
    def boom(expr, settings):
        raise QuotaExceededError("het quota ngay")
    r = _refiner(boom)
    try:
        r.refine_and_sim(_cand(_ALT_EXPR))
        assert False, "phải ném QuotaExhausted"
    except QuotaExhausted:
        pass


def test_altdata_presim_reject_khong_gia_vo_da_sim():
    """Task 3 (spec C2): nhánh _sim_direct cũng phải xử lý presim_reason trung thực — trước đây
    _finalize luôn gán sims_used=1/stage='simmed' dù pre-sim reject chưa chạm Brain."""
    def presim_reject(expr, settings):
        return SimulationResult(
            expression=expr, status="error",
            raw={"error": "pre-sim reject: Operator không tồn tại: fake_op"},
            presim_reason="Operator không tồn tại: fake_op",
        )

    r = _refiner(presim_reject)
    out = r.refine_and_sim(_cand(_ALT_EXPR))
    assert r.simulator.calls == 1
    assert out.sims_used == 0
    assert out.stage_reached == "op_invalid"
    assert out.fail_check == "OPERATOR_INVALID"
    assert out.is_brain_sim is False
    assert out.presim_reason == "Operator không tồn tại: fake_op"
    assert out.passed is False


def test_pv_expr_van_di_duong_tune(monkeypatch):
    """Expr price/volume (local_usable=True) vẫn đi đường tune cũ — KHÔNG lạc sang sim-thẳng.
    Chứng minh nhánh alt-data chỉ kích hoạt khi field ngoài panel."""
    from src.backtest.local_tuner import TuneResult

    called = {"tune": 0}

    def fake_tune(expr, cfg, data, **kw):
        called["tune"] += 1
        return TuneResult(
            best_expr="rank(ts_delta(close, 20))",
            best_config=PortfolioConfig(decay=3, truncation=0.02),
            local_sharpe=1.6, local_metrics=None,
        )

    r = LocalTunerRefiner(
        simulator=type("S", (), {"simulate": lambda self, e, settings=None: _alt_result(e, settings)})(),
        repo=_Repo(), data=_PVData(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(), tune_fn=fake_tune,
    )
    out = r.refine_and_sim(_cand("rank(ts_delta(close, 5))"))
    assert called["tune"] == 1
    assert out.stop_reason == "local_tuned"
