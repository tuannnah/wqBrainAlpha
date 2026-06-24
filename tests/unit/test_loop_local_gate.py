"""Test RefinementLoop._evaluate: score_local_gate fail -> bỏ candidate, KHÔNG gọi
simulator.simulate (không đốt sim). Toàn bộ phụ thuộc là fake/monkeypatch — không gọi
mạng/Brain thật.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.backtest.config import PortfolioConfig
from src.backtest.gate import LocalGateVerdict
from src.llm.loop import RefinementLoop
from src.simulation.config import SimConfig


@dataclass
class _FakeCandidate:
    expression: str

    class hypothesis:
        @staticmethod
        def to_dict():
            return {}

    description: str = "fake"


class _FakePrefilter:
    def check(self, expr):
        return True, "ok"


class _FakeSimulator:
    def __init__(self):
        self.calls = 0

    def simulate(self, expr, settings):
        self.calls += 1
        raise AssertionError("simulate() KHÔNG được gọi khi local gate fail")


class _FakeRepo:
    def __init__(self):
        self.failures = []

    def record_failure(self, expr, kind, reason, source):
        self.failures.append((expr, kind, reason))

    def get_cached_simulation(self, expr, config_key=None):
        return None

    def save_alpha(self, *a, **kw):
        return "fake-alpha-id"

    def save_simulation(self, *a, **kw):
        return None

    def recent_failures(self, n):
        return self.failures[:n]


def _make_loop(local_gate_fn, market_data=object()):
    sim_config = SimConfig.default(region="USA", universe="TOP3000", delay=1)
    return RefinementLoop(
        hypothesis_gen=None, translator=None, refiner=None,
        simulator=_FakeSimulator(), prefilter=_FakePrefilter(), repo=_FakeRepo(),
        region="USA", universe="TOP3000", delay=1, sim_config=sim_config,
        max_simulations=10,
        local_gate_fn=local_gate_fn, market_data=market_data,
        local_gate_cfg=PortfolioConfig(),
    )


def test_local_gate_fail_blocks_simulate_and_records_failure():
    def fake_gate(expr, cfg, data):
        return LocalGateVerdict(False, "fake fail reason")

    loop = _make_loop(fake_gate)
    cand = _FakeCandidate(expression="rank(close)")
    ev = loop._evaluate(cand, parent_id=None)

    assert ev is None
    assert loop.simulator.calls == 0
    assert any(kind == "local_gate_fail" for _, kind, _ in loop.repo.failures)


def test_local_gate_pass_allows_simulate_to_be_called():
    def fake_gate(expr, cfg, data):
        return LocalGateVerdict(True, "ok")

    # Simulator fake ở đây PHẢI thực sự cho phép gọi (không assert chặn) để xác nhận
    # nhánh pass đi xuống simulate như cũ.
    class _AllowSimulator:
        def __init__(self):
            self.calls = 0

        def simulate(self, expr, settings):
            self.calls += 1
            raise RuntimeError("dừng sớm có chủ đích — chỉ cần xác nhận ĐÃ gọi simulate")

    loop = _make_loop(fake_gate)
    loop.simulator = _AllowSimulator()
    cand = _FakeCandidate(expression="rank(close)")
    with pytest.raises(RuntimeError, match="dừng sớm"):
        loop._evaluate(cand, parent_id=None)
    assert loop.simulator.calls == 1


def test_local_gate_skipped_when_market_data_is_none():
    """market_data=None (chưa wire data thật) -> gate bị bỏ qua, hành vi cũ giữ nguyên."""
    gate_called = []

    def fake_gate(expr, cfg, data):
        gate_called.append(expr)
        return LocalGateVerdict(False, "should not matter")

    class _AllowSimulator:
        def __init__(self):
            self.calls = 0

        def simulate(self, expr, settings):
            self.calls += 1
            raise RuntimeError("dừng sớm có chủ đích")

    loop = _make_loop(fake_gate, market_data=None)
    loop.simulator = _AllowSimulator()
    cand = _FakeCandidate(expression="rank(close)")
    with pytest.raises(RuntimeError, match="dừng sớm"):
        loop._evaluate(cand, parent_id=None)
    assert gate_called == []  # gate không được gọi khi market_data=None
    assert loop.simulator.calls == 1
