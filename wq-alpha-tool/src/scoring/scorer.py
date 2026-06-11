"""Điểm tổng hợp một alpha từ metrics."""

from __future__ import annotations

from src.scoring.metrics import normalize

TARGET_TURNOVER = 0.3


def score(source) -> float:
    m = normalize(source)
    sharpe = m["sharpe"]
    fitness = m["fitness"]
    turnover = m["turnover"]
    drawdown = m["drawdown"]
    turnover_penalty = abs(turnover - TARGET_TURNOVER)
    return (
        0.40 * sharpe
        + 0.30 * fitness
        + 0.15 * (1 - drawdown)
        + 0.15 * (1 - turnover_penalty)
    )
