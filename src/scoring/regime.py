"""Đo độ ổn định theo regime (Sharpe theo năm) của một alpha từ daily PnL (review 3).

Metric tổng (Sharpe/Fitness IS) có thể đẹp nhờ một vài năm tốt che lấp một năm sập
(vd 2020). Tách Sharpe theo từng năm rồi lấy năm tệ nhất làm thước "mỏng manh theo
regime": alpha tốt phải tốt ở MỌI năm, không chỉ trung bình.
"""

from __future__ import annotations

import math

TRADING_DAYS = 252


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _year_of(date) -> int:
    """Lấy năm từ 'YYYY-...' (chuỗi) hoặc int năm."""
    if isinstance(date, str):
        return int(date[:4])
    return int(date)


def _sharpe(daily: list[float]) -> float:
    """Sharpe năm hoá từ chuỗi PnL ngày (đã là gia số ngày, không phải tích luỹ)."""
    n = len(daily)
    if n < 2:
        return 0.0
    mean = sum(daily) / n
    var = sum((v - mean) ** 2 for v in daily) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(TRADING_DAYS)


def yearly_sharpe(pnl_series) -> dict[int, float]:
    """{năm: Sharpe} từ iterable (date, daily_pnl). date là 'YYYY-...' hoặc int năm."""
    by_year: dict[int, list[float]] = {}
    for date, pnl in pnl_series:
        by_year.setdefault(_year_of(date), []).append(float(pnl))
    return {year: _sharpe(vals) for year, vals in by_year.items()}


def min_annual_sharpe(yearly: dict[int, float]) -> float:
    """Sharpe của năm tệ nhất. Rỗng -> 0.0."""
    return min(yearly.values()) if yearly else 0.0


def regime_fit(yearly: dict[int, float], target: float = 1.0) -> float:
    """[0,1]: Sharpe năm tệ nhất chuẩn hoá theo target. Năm lỗ -> 0. Không đo được
    (rỗng) -> 1.0 (không phạt chiều không quan sát được)."""
    if not yearly:
        return 1.0
    return _clamp01(min_annual_sharpe(yearly) / target)
