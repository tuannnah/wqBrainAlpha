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
# Trần self-correlation với pool: corr >= mốc này -> pool_fit = 0 (crowded, không nộp được).
CORR_LIMIT = 0.70

# Trọng số gộp về điểm tổng. pool_fit (độ trực giao với pool) là chiều HẠNG NHẤT:
# trọng số ngang fitness để best-selection kéo ra khỏi đỉnh đông, không chỉ là phạt phụ.
WEIGHTS = {
    "sharpe": 0.30,
    "fitness": 0.25,
    "pool_fit": 0.25,
    "drawdown_fit": 0.10,
    "turnover_fit": 0.10,
}


def _pool_fit(pool_corr: float | None) -> float:
    """Độ trực giao với pool: 1.0 khi không đo được/corr=0, giảm tuyến tính tới 0
    tại CORR_LIMIT. corr >= CORR_LIMIT -> 0 (crowded)."""
    if pool_corr is None:
        return 1.0
    return _clamp01(1.0 - abs(pool_corr) / CORR_LIMIT)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


@dataclass
class ScoreVector:
    sharpe: float       # chuẩn hoá theo SHARPE_TARGET
    fitness: float      # chuẩn hoá theo FITNESS_TARGET
    turnover_fit: float  # 1 khi đúng target, giảm khi lệch
    drawdown_fit: float  # 1 khi drawdown=0, 0 khi >= DRAWDOWN_LIMIT
    total: float
    pool_fit: float = 1.0  # độ trực giao với pool (1 = không trùng, 0 = crowded)

    def dimensions(self) -> dict[str, float]:
        return {
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "turnover_fit": self.turnover_fit,
            "drawdown_fit": self.drawdown_fit,
            "pool_fit": self.pool_fit,
        }


def _total(sharpe, fitness, turnover_fit, drawdown_fit, pool_fit) -> float:
    return (
        WEIGHTS["sharpe"] * sharpe
        + WEIGHTS["fitness"] * fitness
        + WEIGHTS["pool_fit"] * pool_fit
        + WEIGHTS["drawdown_fit"] * drawdown_fit
        + WEIGHTS["turnover_fit"] * turnover_fit
    )


def score_vector(source, pool_corr: float | None = None) -> ScoreVector:
    m = normalize(source)
    sharpe = _clamp01(m["sharpe"] / SHARPE_TARGET)
    fitness = _clamp01(m["fitness"] / FITNESS_TARGET)
    turnover_fit = _clamp01(1 - abs(m["turnover"] - TARGET_TURNOVER) / TARGET_TURNOVER)
    drawdown_fit = _clamp01(1 - m["drawdown"] / DRAWDOWN_LIMIT)
    pool_fit = _pool_fit(pool_corr)
    total = _total(sharpe, fitness, turnover_fit, drawdown_fit, pool_fit)
    return ScoreVector(sharpe, fitness, turnover_fit, drawdown_fit, total, pool_fit)


def with_pool_corr(vector: ScoreVector, pool_corr: float | None) -> ScoreVector:
    """Trả vector mới với pool_fit cập nhật theo `pool_corr` (đo sau sim), total tính
    lại. Dùng khi self-correlation chỉ lấy được SAU khi có WQ alpha_id."""
    pool_fit = _pool_fit(pool_corr)
    total = _total(
        vector.sharpe, vector.fitness, vector.turnover_fit, vector.drawdown_fit, pool_fit
    )
    return ScoreVector(
        vector.sharpe, vector.fitness, vector.turnover_fit, vector.drawdown_fit, total, pool_fit
    )


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
