"""Tập alpha tham chiếu kiểu Alpha101, dịch sang FASTEXPR (T3.3).

Đây là tập biểu thức đại diện các bộ khung alpha phổ biến (momentum, reversal,
volume, volatility, correlation...). Mục đích: làm "zoo tham chiếu" để đo độ độc
đáo cấu trúc — KHÔNG nhằm tái tạo chính xác Alpha101 gốc của WorldQuant. Có thể bổ
sung dần các alpha bạn đã nộp.
"""

from __future__ import annotations

ALPHA101_FASTEXPR = [
    "rank(ts_delta(close, 1))",
    "-rank(ts_delta(close, 5))",
    "rank(ts_corr(close, volume, 10))",
    "-rank(ts_corr(open, volume, 10))",
    "rank(ts_mean(close, 5) - ts_mean(close, 20))",
    "ts_rank(volume, 20)",
    "-ts_rank(ts_delta(close, 7), 10)",
    "rank(ts_std_dev(returns, 20))",
    "rank(ts_zscore(volume, 20))",
    "-rank(ts_zscore(close, 10))",
    "group_neutralize(rank(returns), sector)",
    "group_neutralize(rank(ts_delta(vwap, 5)), industry)",
    "rank(close - vwap)",
    "rank(ts_max(high, 10) - close)",
    "rank(close - ts_min(low, 10))",
    "ts_corr(rank(close), rank(volume), 15)",
    "-ts_delta(rank(volume), 3)",
    "rank(ts_sum(returns, 5))",
    "rank(vwap - close) * rank(volume)",
    "ts_decay_linear(rank(ts_delta(close, 2)), 10)",
]
