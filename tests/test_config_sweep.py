"""Test ConfigSweeper: quét cấu hình cho alpha tốt, OOS làm trọng tài (GĐ5: T5.3, T5.6)."""

from __future__ import annotations

from src.simulation.config import SimConfig
from src.simulation.simulator import SimulationResult
from src.simulation.sweep import ConfigSweeper


class _ConfigSimulator:
    """Fake simulator: trả result theo (decay, truncation, neutralization) trong settings."""

    def __init__(self, fn):
        self._fn = fn
        self.calls = []

    def simulate(self, expression, settings=None):
        self.calls.append((expression, settings))
        return self._fn(expression, settings or {})


def _result(sharpe, os_sharpe, status="passed"):
    return SimulationResult(
        expression="e", status=status, sharpe=sharpe, fitness=1.2,
        turnover=0.3, drawdown=0.1, os_sharpe=os_sharpe,
    )


def test_sweep_quet_moi_to_hop_grid():
    sim = _ConfigSimulator(lambda e, s: _result(1.5, 1.0))
    sweeper = ConfigSweeper(sim)
    grid = {"decay": [0, 4], "truncation": [0.05, 0.1]}
    sweeper.sweep("rank(close)", SimConfig.default(), grid)
    assert len(sim.calls) == 4  # 2 decay x 2 truncation


def test_sweep_chon_cau_hinh_diem_cao_nhat_qua_oos():
    # decay=4 cho sharpe cao nhất VÀ qua OOS -> được chọn.
    def fn(e, s):
        if s.get("decay") == 4:
            return _result(2.0, 1.6)   # IS cao, OOS giữ tốt
        return _result(1.2, 0.9)
    sweeper = ConfigSweeper(_ConfigSimulator(fn))
    grid = {"decay": [0, 4]}
    res = sweeper.sweep("rank(close)", SimConfig.default(), grid, oos_min_ratio=0.5)
    assert res.best_config.decay == 4
    assert res.best_result.sharpe == 2.0


def test_sweep_loai_cau_hinh_chi_dep_o_is():
    # decay=4 IS cao nhất NHƯNG OOS sụt mạnh -> bị loại; decay=0 qua OOS -> được chọn.
    def fn(e, s):
        if s.get("decay") == 4:
            return _result(2.5, 0.2)   # overfit IS
        return _result(1.4, 1.0)       # ổn cả OOS
    sweeper = ConfigSweeper(_ConfigSimulator(fn))
    grid = {"decay": [0, 4]}
    res = sweeper.sweep("rank(close)", SimConfig.default(), grid, oos_min_ratio=0.5)
    assert res.best_config.decay == 0
    assert res.best_result.sharpe == 1.4


def test_sweep_khong_cau_hinh_nao_qua_oos_tra_none():
    sim = _ConfigSimulator(lambda e, s: _result(2.0, 0.1))  # tất cả overfit
    sweeper = ConfigSweeper(sim)
    res = sweeper.sweep("rank(close)", SimConfig.default(), {"decay": [0, 4]}, oos_min_ratio=0.5)
    assert res.best_config is None
    assert res.best_result is None


def test_sweep_bo_qua_ket_qua_error():
    def fn(e, s):
        if s.get("decay") == 4:
            return _result(0.0, 0.0, status="error")
        return _result(1.5, 1.0)
    sweeper = ConfigSweeper(_ConfigSimulator(fn))
    res = sweeper.sweep("rank(close)", SimConfig.default(), {"decay": [0, 4]}, oos_min_ratio=0.5)
    assert res.best_config.decay == 0


def test_sweep_luu_lich_su_cac_to_hop():
    sim = _ConfigSimulator(lambda e, s: _result(1.5, 1.0))
    sweeper = ConfigSweeper(sim)
    res = sweeper.sweep("rank(close)", SimConfig.default(), {"truncation": [0.05, 0.1, 0.2]})
    assert len(res.trials) == 3  # mỗi tổ hợp một dòng lịch sử
    assert all("config" in t and "oos_ok" in t for t in res.trials)
