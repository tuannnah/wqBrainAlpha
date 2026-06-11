"""Lọc alpha đa tiêu chí theo ngưỡng."""

from __future__ import annotations

from dataclasses import dataclass

from src.scoring.metrics import normalize


@dataclass
class FilterThresholds:
    min_sharpe: float = 1.25
    min_fitness: float = 1.0
    turnover_low: float = 0.01
    turnover_high: float = 0.70
    max_drawdown: float = 0.20


def passes(source, thresholds: FilterThresholds | None = None) -> tuple[bool, list[str]]:
    """Trả (đạt, danh sách lý do fail)."""
    t = thresholds or FilterThresholds()
    m = normalize(source)
    reasons: list[str] = []

    if m["sharpe"] < t.min_sharpe:
        reasons.append(f"sharpe {m['sharpe']:.2f} < {t.min_sharpe}")
    if m["fitness"] <= t.min_fitness:
        reasons.append(f"fitness {m['fitness']:.2f} <= {t.min_fitness}")
    if not (t.turnover_low <= m["turnover"] <= t.turnover_high):
        reasons.append(f"turnover {m['turnover']:.2f} ngoài [{t.turnover_low}, {t.turnover_high}]")
    if m["drawdown"] >= t.max_drawdown:
        reasons.append(f"drawdown {m['drawdown']:.2f} >= {t.max_drawdown}")

    return (len(reasons) == 0, reasons)
