"""Test parse OOS + kiểm chứng Out-of-Sample cho quét cấu hình (GĐ5: T5.6)."""

from __future__ import annotations

from src.simulation.oos import oos_passes
from src.simulation.simulator import Simulator
from tests.fakes import FakeClient, FakeResponse
from src.simulation.rate_limiter import RateLimiter


def _no_sleep_limiter() -> RateLimiter:
    return RateLimiter(min_delay=0, sleep_func=lambda *_: None, time_func=lambda: 0.0)


def _sim(client) -> Simulator:
    return Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )


def test_simulate_parse_block_os():
    """Simulator đọc cả block `os` (Out-of-Sample) ngoài `is`."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-1"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-1"}))
    client.queue_get(
        FakeResponse(
            200,
            json_data={
                "is": {"sharpe": 1.8, "fitness": 1.3, "checks": []},
                "os": {"sharpe": 1.4, "fitness": 1.0},
            },
        )
    )
    result = _sim(client).simulate("rank(close)")
    assert result.sharpe == 1.8        # IS giữ nguyên ở field cũ
    assert result.os_sharpe == 1.4     # OOS sharpe
    assert result.os_fitness == 1.0


def test_simulate_khong_co_os_thi_none():
    """Không có block `os` -> os_* là None (không vỡ)."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-2"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-2"}))
    client.queue_get(FakeResponse(200, json_data={"is": {"sharpe": 1.5, "checks": []}}))
    result = _sim(client).simulate("rank(close)")
    assert result.sharpe == 1.5
    assert result.os_sharpe is None


def test_oos_passes_giu_cau_hinh_tot_ca_is_va_oos():
    """OOS sharpe đủ cao so với IS -> qua (không overfit IS)."""
    from src.simulation.simulator import SimulationResult

    r = SimulationResult(expression="e", status="passed", sharpe=2.0, os_sharpe=1.6)
    assert oos_passes(r, min_ratio=0.5) is True


def test_oos_passes_loai_cau_hinh_chi_dep_o_is():
    """OOS sharpe sụt mạnh so với IS -> loại (overfit IS)."""
    from src.simulation.simulator import SimulationResult

    r = SimulationResult(expression="e", status="passed", sharpe=2.0, os_sharpe=0.3)
    assert oos_passes(r, min_ratio=0.5) is False


def test_oos_passes_thieu_os_thi_khong_qua():
    """Thiếu OOS -> coi như chưa kiểm chứng được -> không qua (an toàn)."""
    from src.simulation.simulator import SimulationResult

    r = SimulationResult(expression="e", status="passed", sharpe=2.0, os_sharpe=None)
    assert oos_passes(r, min_ratio=0.5) is False


def test_oos_passes_is_am_thi_khong_qua():
    """IS sharpe <= 0 -> tỉ lệ vô nghĩa -> không qua."""
    from src.simulation.simulator import SimulationResult

    r = SimulationResult(expression="e", status="passed", sharpe=0.0, os_sharpe=0.5)
    assert oos_passes(r, min_ratio=0.5) is False
