"""Test RateLimiter: delay tối thiểu và phân loại status retryable."""

from __future__ import annotations

from src.simulation.rate_limiter import RateLimiter, is_retryable_response
from tests.fakes import FakeResponse


def test_wait_for_slot_enforces_min_delay():
    slept = []
    fake_time = {"t": 0.0}

    limiter = RateLimiter(
        min_delay=2.0,
        sleep_func=lambda s: slept.append(s),
        time_func=lambda: fake_time["t"],
    )

    limiter.wait_for_slot()  # lần đầu: last_post=0, elapsed=0 → ngủ 2.0
    assert slept == [2.0]


def test_is_retryable_response():
    assert is_retryable_response(FakeResponse(429)) is True
    assert is_retryable_response(FakeResponse(503)) is True
    assert is_retryable_response(FakeResponse(200)) is False
