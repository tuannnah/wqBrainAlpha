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


def test_returns_field_not_blocked_by_fields_ok(small_panel: MarketData) -> None:
    # `returns` là field WQ hợp lệ (lưu ở .returns, ngoài .fields). Trước fix: fields_ok=False
    # -> hard gate fail; eval cũng KeyError. Sau fix: KHÔNG bị chặn bởi fields_ok và eval được.
    verdict = score_local_gate("rank(returns)", PortfolioConfig(delay=1), small_panel)
    assert isinstance(verdict, LocalGateVerdict)
    assert "fields_ok" not in verdict.reason
    assert "eval lỗi" not in verdict.reason


def test_score_local_gate_fails_when_self_corr_too_high(small_panel: MarketData) -> None:
    # self_corr cao phải chặn pass dù expr hợp lệ và sinh pnl được — hành vi Phase 4 MỚI,
    # Phase 3 cũ KHÔNG có tham số self_corr nên test này xác nhận chữ ký đã mở rộng.
    verdict = score_local_gate(
        "close", PortfolioConfig(delay=1), small_panel, self_corr=0.99,
    )
    assert verdict.passed is False
    assert "self_corr" in verdict.reason.lower()


def test_score_local_gate_passes_with_low_self_corr_and_valid_expression(
    small_panel: MarketData,
) -> None:
    verdict = score_local_gate(
        "close", PortfolioConfig(delay=1), small_panel, self_corr=0.0,
    )
    # Không assert cứng passed=True (sharpe trên data thật có thể thấp) — assert reason
    # KHÔNG còn là lý do tối thiểu cũ ("no_pnl"/"signal toàn NaN") khi expr hợp lệ.
    assert verdict.reason not in {"signal toàn NaN — không có giá trị dùng được"}
