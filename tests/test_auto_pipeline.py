"""Test AutoPipeline bằng fake callback — không gọi mạng."""

from __future__ import annotations

from src.pipeline.auto import (
    AutoPipeline,
    DirectionOutcome,
    PassedAlpha,
    PrepareInfo,
)


def _pa(expr: str, direction: str = "") -> PassedAlpha:
    return PassedAlpha(expression=expr, sharpe=1.5, fitness=1.1, direction=direction)


def test_dung_khi_het_huong():
    calls = {"run": 0}

    def prepare() -> PrepareInfo:
        return PrepareInfo(fields=10, operators=5)

    def propose(n: int) -> list[str]:
        return ["h1", "h2"]

    def run_direction(direction: str) -> DirectionOutcome:
        calls["run"] += 1
        return DirectionOutcome(passed=[], sims_used=1)

    pipe = AutoPipeline(
        prepare=prepare,
        propose_directions=propose,
        run_direction=run_direction,
        target_passes=99,
        max_total_sims=999,
        max_directions=5,
    )
    result = pipe.run()

    assert calls["run"] == 2           # chạy đúng 2 hướng được đề xuất
    assert result.directions_run == 2
    assert result.total_sims == 2
    assert result.stop_reason == "hết_hướng"
    assert result.passed_alphas == []
