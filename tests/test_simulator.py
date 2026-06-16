"""Test Simulator end-to-end với FakeClient."""

from __future__ import annotations

from src.simulation.rate_limiter import RateLimiter
from src.simulation.simulator import Simulator
from tests.fakes import FakeClient, FakeResponse


def _no_sleep_limiter() -> RateLimiter:
    return RateLimiter(min_delay=0, sleep_func=lambda *_: None, time_func=lambda: 0.0)


def test_simulate_luong_day_du():
    client = FakeClient()
    # POST /simulations → 201 + Location
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-1"}))
    # Poll lần 1: chưa xong; lần 2: COMPLETE kèm alpha id
    client.queue_get(FakeResponse(200, json_data={"status": "RUNNING"}, headers={"Retry-After": "0"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-9"}))
    # GET /alphas/alpha-9 → metrics
    client.queue_get(
        FakeResponse(
            200,
            json_data={
                "is": {
                    "sharpe": 1.8,
                    "fitness": 1.3,
                    "turnover": 0.25,
                    "returns": 0.12,
                    "drawdown": 0.08,
                    "margin": 0.0015,
                    "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
                }
            },
        )
    )

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")

    assert result.status == "passed"
    assert result.alpha_id == "alpha-9"
    assert result.sharpe == 1.8
    assert result.fitness == 1.3
    assert result.metrics()["turnover"] == 0.25


def test_simulate_failed_khi_check_fail():
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-2"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-x"}))
    client.queue_get(
        FakeResponse(
            200,
            json_data={"is": {"sharpe": 0.2, "checks": [{"name": "LOW_SHARPE", "result": "FAIL"}]}},
        )
    )

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")
    assert result.status == "failed"


def test_simulate_error_giu_message_giai_thich():
    """status=ERROR kèm message → message phải được nêu trong raw['error'] để chẩn đoán."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-3"}))
    client.queue_get(
        FakeResponse(
            200,
            json_data={
                "status": "ERROR",
                "message": "Datafield 'foo_bar' is not supported in this region.",
            },
        )
    )

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(foo_bar)")

    assert result.status == "error"
    assert "foo_bar" in result.raw["error"]
    assert "ERROR" in result.raw["error"]


def test_simulate_thieu_location_tra_error():
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={}))
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")
    assert result.status == "error"
