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
    assert out.stop_reason.startswith("local_floor")  # nay kèm ngưỡng calibrated để audit


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


def test_refiner_ap_neutralization_da_tune_vao_sim():
    """Neutralization mà tune() (Task 1) chọn (sweep MARKET/SECTOR) PHẢI được áp vào
    SimConfig gửi Brain — trước đây refiner chỉ forward decay/truncation, bỏ quên
    neutralization đã tune -> Brain sim luôn chạy default SUBINDUSTRY dù local đã tune ra
    SECTOR/MARKET tốt hơn."""
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.backtest.local_tuner import TuneResult

    seen = {}

    class _Sim:
        def simulate(self, expr, settings=None):
            seen["neut"] = settings.get("neutralization")
            return _passed_result(expr, settings)

    def fake_tune(expr, cfg, data, **kw):
        return TuneResult(
            best_expr="rank(ts_delta(close, 20))",
            best_config=PortfolioConfig(neutralization=Neutralization.SECTOR, decay=3, truncation=0.02),
            local_sharpe=1.6, local_metrics=None,
        )

    r = LocalTunerRefiner(
        simulator=_Sim(), repo=_Repo(), data=object(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(), tune_fn=fake_tune,
    )
    r.refine_and_sim(_cand())
    assert seen["neut"] == "SECTOR"   # neutralization tune -> áp vào Brain sim


class _FakeTracker:
    """Fake CalibrationTracker tối thiểu (chỉ 2 attribute mà refiner đọc)."""

    def __init__(self, last_rho, rho_bar=0.5):
        self.last_rho = last_rho
        self.rho_bar = rho_bar


def test_refiner_rho_thap_thi_bo_qua_floor_local():
    """Task 5: ρ hiện tại (0.3) < rho_bar (0.5) -> ranking local không tin -> refiner PHẢI
    bỏ qua min_local_sharpe floor, đi tiếp sim Brain dù local_sharpe (0.1) dưới floor mặc định
    (0.5) — floor hiệu lực = 0.0 khi ρ không tin."""
    r = _refiner(_passed_result, tune_local_sharpe=0.1)
    r.set_calibration_tracker(_FakeTracker(last_rho=0.3, rho_bar=0.5))
    out = r.refine_and_sim(_cand())
    assert r.simulator.calls == 1        # KHÔNG bị chặn ở local_floor -> tới simulate()
    assert out.stage_reached != "local_floor"
    assert out.is_brain_sim is True


def test_refiner_rho_cao_thi_van_giu_floor_local():
    """ρ hiện tại (0.8) >= rho_bar (0.5) -> ranking local vẫn tin -> floor local_sharpe áp
    dụng như cũ (không đổi hành vi khi ρ đủ tin)."""
    r = _refiner(_passed_result, tune_local_sharpe=0.1)
    r.set_calibration_tracker(_FakeTracker(last_rho=0.8, rho_bar=0.5))
    out = r.refine_and_sim(_cand())
    assert r.simulator.calls == 0
    assert out.sims_used == 0
    assert out.stage_reached == "local_floor"


def test_refiner_rho_none_thi_van_giu_floor_local():
    """Chưa đo được ρ (last_rho=None, VD chưa đủ cặp local/Brain) -> hành vi mặc định KHÔNG
    đổi: floor calibrated vẫn áp dụng như trước khi có Task 5 (backward-compat)."""
    r = _refiner(_passed_result, tune_local_sharpe=0.1)
    r.set_calibration_tracker(_FakeTracker(last_rho=None, rho_bar=0.5))
    out = r.refine_and_sim(_cand())
    assert r.simulator.calls == 0
    assert out.sims_used == 0
    assert out.stage_reached == "local_floor"


def test_refiner_khong_co_tracker_thi_hanh_vi_nhu_cu():
    """Không gắn tracker (mặc định None, drop-in cũ) -> floor calibrated vẫn áp dụng y hệt
    trước Task 5 — bảo đảm tương thích ngược cho mọi call site chưa nối tracker."""
    r = _refiner(_passed_result, tune_local_sharpe=0.1)
    out = r.refine_and_sim(_cand())
    assert r.simulator.calls == 0
    assert out.sims_used == 0
    assert out.stage_reached == "local_floor"


def test_refiner_sub_universe_fail_thi_khong_sim(monkeypatch):
    """Winner KHÔNG đạt proxy sub-universe -> gate chặn TRƯỚC sim: không gọi simulator,
    sims_used=0, stop_reason='sub_universe'. (Phủ trực tiếp nhánh reject của gate Task 4.)"""
    monkeypatch.setattr("src.backtest.sub_universe.sub_universe_ok", lambda *a, **kw: False)
    from src.backtest.local_tuner import TuneResult
    from src.backtest.metrics_local import AlphaMetrics

    metrics = AlphaMetrics(
        sharpe=1.6, annual_return=0.2, turnover=0.3, max_drawdown=0.1,
        fitness=1.2, per_year_sharpe={2020: 1.5}, weight_concentration=0.05,
    )

    class _Sim:
        def __init__(self):
            self.calls = 0

        def simulate(self, expr, settings=None):
            self.calls += 1
            return _passed_result(expr, settings)

    def fake_tune(expr, cfg, data, **kw):
        return TuneResult(
            best_expr="rank(ts_delta(close, 20))",
            best_config=PortfolioConfig(decay=3, truncation=0.02),
            local_sharpe=1.6, local_metrics=metrics,
        )

    sim = _Sim()
    r = LocalTunerRefiner(
        simulator=sim, repo=_Repo(), data=object(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(), tune_fn=fake_tune,
    )
    out = r.refine_and_sim(_cand())
    assert sim.calls == 0
    assert out.sims_used == 0
    assert out.stop_reason == "sub_universe"
    assert out.passed is False


def _refiner_local_metrics(simulate, *, tune_local_sharpe, sub_universe_result, monkeypatch):
    """Refiner với local_metrics THẬT (không None) -> cả 2 gate (floor + sub_universe) có mặt
    cùng lúc, đúng đường thật trong refine_and_sim (test cũ ở trên tách riêng từng gate bằng
    local_metrics=None nên KHÔNG phủ được tương tác giữa 2 gate — đây là gap Finding 1 vá)."""
    monkeypatch.setattr("src.backtest.sub_universe.sub_universe_ok",
                         lambda *a, **kw: sub_universe_result)
    from src.backtest.local_tuner import TuneResult
    from src.backtest.metrics_local import AlphaMetrics

    metrics = AlphaMetrics(
        sharpe=tune_local_sharpe, annual_return=0.2, turnover=0.3, max_drawdown=0.1,
        fitness=1.2, per_year_sharpe={2020: 1.5}, weight_concentration=0.05,
    )

    class _Sim:
        def __init__(self):
            self.calls = 0

        def simulate(self, expr, settings=None):
            self.calls += 1
            return simulate(expr, settings)

    def fake_tune(expr, cfg, data, **kw):
        return TuneResult(
            best_expr="rank(ts_delta(close, 20))",
            best_config=PortfolioConfig(decay=3, truncation=0.02),
            local_sharpe=tune_local_sharpe, local_metrics=metrics,
        )

    r = LocalTunerRefiner(
        simulator=_Sim(), repo=_Repo(), data=object(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(), tune_fn=fake_tune,
    )
    return r


def test_refiner_rho_thap_thi_bo_qua_ca_floor_va_sub_universe(monkeypatch):
    """Finding 1 (follow-up a404874): ρ thấp (untrusted) -> floor VÀ sub_universe_ok PHẢI
    cùng bị bỏ qua, không chỉ floor. Setup: local_metrics THẬT (không None) + local_sharpe
    (0.1) dưới floor mặc định (0.5) + sub_universe_ok stub trả False (nếu chạy sẽ chặn) +
    ρ untrusted (last_rho=0.3 < rho_bar=0.5). Nếu chỉ floor được tắt mà sub_universe_ok vẫn
    chạy thật, ứng viên vẫn bị giết oan ở gate thứ hai -> simulate() KHÔNG được gọi, hỏng.
    Assert PHẢI tới simulate() -> cả 2 gate cùng bị bypass đồng bộ."""
    r = _refiner_local_metrics(_passed_result, tune_local_sharpe=0.1,
                                sub_universe_result=False, monkeypatch=monkeypatch)
    r.set_calibration_tracker(_FakeTracker(last_rho=0.3, rho_bar=0.5))
    out = r.refine_and_sim(_cand())
    assert r.simulator.calls == 1          # tới simulate() -> CẢ 2 gate đều bị bypass
    assert out.stage_reached not in ("local_floor", "sub_universe")
    assert out.is_brain_sim is True


def test_refiner_rho_cao_thi_van_bi_sub_universe_chan(monkeypatch):
    """Đối chứng: ρ CAO (trusted, last_rho=0.8 >= rho_bar=0.5) -> local vẫn tin -> local_sharpe
    (1.6, TRÊN floor 0.5 nên không bị floor chặn) nhưng sub_universe_ok trả False -> ứng viên
    PHẢI bị chặn ở gate sub_universe như hành vi cũ (test_refiner_sub_universe_fail_thi_khong_sim),
    dù bây giờ có tracker gắn vào. simulate() KHÔNG được gọi."""
    r = _refiner_local_metrics(_passed_result, tune_local_sharpe=1.6,
                                sub_universe_result=False, monkeypatch=monkeypatch)
    r.set_calibration_tracker(_FakeTracker(last_rho=0.8, rho_bar=0.5))
    out = r.refine_and_sim(_cand())
    assert r.simulator.calls == 0          # sub_universe chặn TRƯỚC simulate()
    assert out.sims_used == 0
    assert out.stop_reason == "sub_universe"
    assert out.passed is False


# --- Task 4: gate backtest-cheap "degenerate position" -- sau self._tune, TRƯỚC floor/sim.
# turnover local < DEGENERATE_TURNOVER (0.005) VÀ |sharpe| local < DEGENERATE_SHARPE (0.05)
# ĐỒNG THỜI -> vị thế suy biến/gần hằng số (bằng chứng log thật 07-12: sim Brain thật ra đúng
# Sharpe 0.00/turnover 0.00 cho dạng biểu thức này) -> chặn TRƯỚC khi đốt sim Brain.


def _degenerate_metrics_refiner(*, sharpe: float, turnover: float, monkeypatch):
    """Refiner với local_metrics có sharpe/turnover TUỲ CHỈNH (khác `_refiner_local_metrics`
    vốn hardcode turnover=0.3) -- cần để dựng đúng ranh giới DEGENERATE_TURNOVER/SHARPE.
    sub_universe_ok mock True (không liên quan gate đang test, cô lập đúng gate degenerate)."""
    monkeypatch.setattr("src.backtest.sub_universe.sub_universe_ok", lambda *a, **kw: True)
    from src.backtest.local_tuner import TuneResult
    from src.backtest.metrics_local import AlphaMetrics

    metrics = AlphaMetrics(
        sharpe=sharpe, annual_return=0.0, turnover=turnover, max_drawdown=0.0,
        fitness=0.01, per_year_sharpe={2020: sharpe}, weight_concentration=0.05,
    )

    class _Sim:
        def __init__(self):
            self.calls = 0

        def simulate(self, expr, settings=None):
            self.calls += 1
            return _passed_result(expr, settings)

    def fake_tune(expr, cfg, data, **kw):
        return TuneResult(
            best_expr="power(sign(close), 2)",
            best_config=PortfolioConfig(decay=3, truncation=0.02),
            local_sharpe=sharpe, local_metrics=metrics,
        )

    sim = _Sim()
    r = LocalTunerRefiner(
        simulator=sim, repo=_Repo(), data=object(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(), tune_fn=fake_tune,
    )
    return r, sim


def test_refiner_degenerate_position_thi_khong_sim(monkeypatch):
    """turnover (0.001) < 0.005 VÀ |sharpe| (0.02) < 0.05 ĐỒNG THỜI -> chặn TRƯỚC sim, 0 sim,
    outcome trung thực (stage/fail_check RIÊNG, khác 'local_floor' chung chung -- phân biệt rõ
    'vị thế suy biến' khỏi 'sharpe thấp bình thường' cho mục đích chẩn đoán/audit)."""
    r, sim = _degenerate_metrics_refiner(sharpe=0.02, turnover=0.001, monkeypatch=monkeypatch)
    out = r.refine_and_sim(_cand())
    assert sim.calls == 0
    assert out.sims_used == 0
    assert out.passed is False
    assert out.is_brain_sim is False
    assert out.stage_reached == "degenerate"
    assert out.fail_check == "DEGENERATE_POSITION"


def test_refiner_turnover_thap_nhung_sharpe_du_thi_khong_bi_chan(monkeypatch):
    """turnover CỰC thấp (0.001 < 0.005) NHƯNG |sharpe| ĐỦ (1.6 >= 0.05) -- CHỈ 1 trong 2 điều
    kiện đúng (không phải AND) -> KHÔNG phải vị thế suy biến (tín hiệu chậm nhưng CÓ ý nghĩa
    thật) -- KHÔNG bị gate degenerate chặn, đi tiếp tới sim Brain bình thường."""
    r, sim = _degenerate_metrics_refiner(sharpe=1.6, turnover=0.001, monkeypatch=monkeypatch)
    out = r.refine_and_sim(_cand())
    assert sim.calls == 1
    assert out.is_brain_sim is True
    assert out.stage_reached != "degenerate"


def test_refiner_sharpe_thap_nhung_turnover_du_thi_khong_bi_chan_degenerate(monkeypatch):
    """|sharpe| thấp (0.02 < 0.05) NHƯNG turnover ĐỦ (0.3 >= 0.005) -- không phải AND -> KHÔNG
    bị gate degenerate chặn. Local sharpe (0.02) vẫn dưới floor calibrated mặc định (~0.5) nên
    bị chặn ở gate `local_floor` NHƯ CŨ (khác gate degenerate), không tới simulate()."""
    r, sim = _degenerate_metrics_refiner(sharpe=0.02, turnover=0.3, monkeypatch=monkeypatch)
    out = r.refine_and_sim(_cand())
    assert sim.calls == 0
    assert out.stage_reached == "local_floor"        # đúng gate cũ, KHÔNG phải degenerate mới
