"""ScoreVector: điểm đa chiều chuẩn hoá [0,1] + chọn chiều yếu nhất (GĐ2).

Khác `scorer.score()` (điểm tổng thô cho GA): đây là vector điểm từng chiều đã
chuẩn hoá, phục vụ tinh chỉnh nhắm "chiều yếu nhất" (T2.11). Không thay scorer cũ.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.scoring.metrics import normalize

# Mốc chuẩn hoá: đạt mốc -> điểm 1.0.
SHARPE_TARGET = 2.0
FITNESS_TARGET = 1.5
TARGET_TURNOVER = 0.3
DRAWDOWN_LIMIT = 0.20

# Trọng số gộp về điểm tổng (khớp tinh thần scorer cũ: sharpe>fitness>=phụ).
WEIGHTS = {
    "sharpe": 0.40,
    "fitness": 0.30,
    "drawdown_fit": 0.15,
    "turnover_fit": 0.15,
}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


@dataclass
class ScoreVector:
    sharpe: float       # chuẩn hoá theo SHARPE_TARGET
    fitness: float      # chuẩn hoá theo FITNESS_TARGET
    turnover_fit: float  # 1 khi đúng target, giảm khi lệch
    drawdown_fit: float  # 1 khi drawdown=0, 0 khi >= DRAWDOWN_LIMIT
    total: float

    def dimensions(self) -> dict[str, float]:
        return {
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "turnover_fit": self.turnover_fit,
            "drawdown_fit": self.drawdown_fit,
        }


def score_vector(source) -> ScoreVector:
    m = normalize(source)
    sharpe = _clamp01(m["sharpe"] / SHARPE_TARGET)
    fitness = _clamp01(m["fitness"] / FITNESS_TARGET)
    turnover_fit = _clamp01(1 - abs(m["turnover"] - TARGET_TURNOVER) / TARGET_TURNOVER)
    drawdown_fit = _clamp01(1 - m["drawdown"] / DRAWDOWN_LIMIT)
    total = (
        WEIGHTS["sharpe"] * sharpe
        + WEIGHTS["fitness"] * fitness
        + WEIGHTS["drawdown_fit"] * drawdown_fit
        + WEIGHTS["turnover_fit"] * turnover_fit
    )
    return ScoreVector(sharpe, fitness, turnover_fit, drawdown_fit, total)


def weakest_dimension(
    vector: ScoreVector,
    priority: dict[str, float] | None = None,
    restrict: set[str] | None = None,
) -> str:
    """Tên chiều yếu nhất. `priority` (>1 = ưu tiên cải thiện) chia điểm để thiên
    về chiều đó. `restrict` (nếu không rỗng) -> chỉ xét các chiều này (vd chỉ các
    chiều đang chặn hard filter), để refine nhắm đúng biên cần vượt."""
    priority = priority or {}
    dims = vector.dimensions()
    if restrict:
        dims = {k: v for k, v in dims.items() if k in restrict} or vector.dimensions()
    return min(dims, key=lambda k: dims[k] / priority.get(k, 1.0))
