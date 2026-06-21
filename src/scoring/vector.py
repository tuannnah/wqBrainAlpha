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

# Trọng số gộp về điểm tổng. pool_fit (trực giao với pool) và regime_fit (ổn định
# theo năm) là hai chiều "nộp được / bền" hạng nhất, không phải phạt phụ.
WEIGHTS = {
    "sharpe": 0.25,
    "fitness": 0.20,
    "pool_fit": 0.25,
    "regime_fit": 0.15,
    "drawdown_fit": 0.075,
    "turnover_fit": 0.075,
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
    pool_fit: float = 1.0    # độ trực giao với pool (1 = không trùng, 0 = crowded)
    regime_fit: float = 1.0  # độ ổn định theo năm (1 = bền mọi năm, 0 = có năm sập)

    def dimensions(self) -> dict[str, float]:
        return {
            "sharpe": self.sharpe,
            "fitness": self.fitness,
            "turnover_fit": self.turnover_fit,
            "drawdown_fit": self.drawdown_fit,
            "pool_fit": self.pool_fit,
            "regime_fit": self.regime_fit,
        }


def _total(sharpe, fitness, turnover_fit, drawdown_fit, pool_fit, regime_fit) -> float:
    return (
        WEIGHTS["sharpe"] * sharpe
        + WEIGHTS["fitness"] * fitness
        + WEIGHTS["pool_fit"] * pool_fit
        + WEIGHTS["regime_fit"] * regime_fit
        + WEIGHTS["drawdown_fit"] * drawdown_fit
        + WEIGHTS["turnover_fit"] * turnover_fit
    )


def score_vector(
    source, pool_corr: float | None = None, regime: float | None = None
) -> ScoreVector:
    m = normalize(source)
    sharpe = _clamp01(m["sharpe"] / SHARPE_TARGET)
    fitness = _clamp01(m["fitness"] / FITNESS_TARGET)
    turnover_fit = _clamp01(1 - abs(m["turnover"] - TARGET_TURNOVER) / TARGET_TURNOVER)
    drawdown_fit = _clamp01(1 - m["drawdown"] / DRAWDOWN_LIMIT)
    pool_fit = _pool_fit(pool_corr)
    regime_fit = 1.0 if regime is None else _clamp01(regime)
    total = _total(sharpe, fitness, turnover_fit, drawdown_fit, pool_fit, regime_fit)
    return ScoreVector(sharpe, fitness, turnover_fit, drawdown_fit, total, pool_fit, regime_fit)


def with_pool_corr(vector: ScoreVector, pool_corr: float | None) -> ScoreVector:
    """Trả vector mới với pool_fit cập nhật theo `pool_corr` (đo sau sim), total tính
    lại. Dùng khi self-correlation chỉ lấy được SAU khi có WQ alpha_id."""
    pool_fit = _pool_fit(pool_corr)
    total = _total(
        vector.sharpe, vector.fitness, vector.turnover_fit, vector.drawdown_fit,
        pool_fit, vector.regime_fit,
    )
    return ScoreVector(
        vector.sharpe, vector.fitness, vector.turnover_fit, vector.drawdown_fit,
        total, pool_fit, vector.regime_fit,
    )


def with_regime_fit(vector: ScoreVector, regime: float) -> ScoreVector:
    """Trả vector mới với regime_fit cập nhật (đo từ Sharpe theo năm sau sim), giữ
    nguyên pool_fit và các chiều khác, total tính lại."""
    regime_fit = _clamp01(regime)
    total = _total(
        vector.sharpe, vector.fitness, vector.turnover_fit, vector.drawdown_fit,
        vector.pool_fit, regime_fit,
    )
    return ScoreVector(
        vector.sharpe, vector.fitness, vector.turnover_fit, vector.drawdown_fit,
        total, vector.pool_fit, regime_fit,
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
