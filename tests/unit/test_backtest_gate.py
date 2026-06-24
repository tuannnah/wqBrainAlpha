"""Test score_local_gate: cổng local tối thiểu Phase 3 (parse + eval + pnl sinh được)."""

from __future__ import annotations

from src.backtest.config import PortfolioConfig
from src.backtest.gate import LocalGateVerdict, score_local_gate
from src.data.market_panel import MarketData


def test_valid_simple_expression_passes(small_panel: MarketData) -> None:
    verdict = score_local_gate("close", PortfolioConfig(delay=1), small_panel)
    assert isinstance(verdict, LocalGateVerdict)
    assert verdict.passed is True


def test_parse_error_fails_with_reason(small_panel: MarketData) -> None:
    verdict = score_local_gate("not_a_real_op(close,", PortfolioConfig(), small_panel)
    assert verdict.passed is False
    assert "parse" in verdict.reason.lower()


def test_unknown_field_fails(small_panel: MarketData) -> None:
    verdict = score_local_gate("totally_unknown_field_xyz", PortfolioConfig(), small_panel)
    assert verdict.passed is False
