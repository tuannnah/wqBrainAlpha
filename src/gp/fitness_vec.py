"""FitnessVector — 6 chiều multi-objective, correlation- và regime-aware (B13 master
design, Gap #4 R4). Hướng tối ưu: sharpe_deflated/per_year_min_sharpe MAXIMIZE; turnover_
penalty/complexity_penalty/pool_corr_penalty/pop_corr_penalty MINIMIZE. Không tự gọi
PoolCorrelation/MetricsCalculator ở đây — caller (GPEngine, Task 7.7) tính trước rồi
truyền vào, giữ module này thuần tính toán, dễ test độc lập.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.thresholds import MAX_DEPTH, TURNOVER_BAND
from src.backtest.metrics_local import AlphaMetrics

_COMPLEXITY_NORM = 50.0  # chuẩn hóa node-count thô; không phải threshold submission


@dataclass(frozen=True, slots=True)
class FitnessVector:
    """6 chiều fitness GP. sharpe_deflated/per_year_min_sharpe: càng cao càng tốt.
    turnover_penalty/complexity_penalty/pool_corr_penalty/pop_corr_penalty: càng thấp
    càng tốt (0 = lý tưởng)."""

    sharpe_deflated: float
    per_year_min_sharpe: float
    turnover_penalty: float
    complexity_penalty: float
    pool_corr_penalty: float
    pop_corr_penalty: float


def deflated_sharpe(sharpe: float, n_trials: int) -> float:
    """Haircut đa kiểm định xấp xỉ (Bailey-López de Prado rút gọn): trừ độ lệch kỳ vọng
    của max trong n_trials phép thử ngẫu nhiên trên 252 quan sát/năm. n_trials<=1 -> không
    haircut (chưa có nhiều lần thử để overfit). Đây là XẤP XỈ TƯƠNG ĐỐI, không phải công
    thức Brain công bố — chỉ dùng để xếp hạng nội bộ GP, không báo cáo tuyệt đối."""
    if n_trials <= 1:
        return sharpe
    haircut = math.sqrt(2 * math.log(n_trials)) / math.sqrt(252)
    return sharpe - haircut


def _turnover_penalty(turnover: float) -> float:
    lo, hi = TURNOVER_BAND
    if turnover < lo:
        return lo - turnover
    if turnover > hi:
        return turnover - hi
    return 0.0


def from_metrics(
    m: AlphaMetrics,
    complexity: int,
    depth: int,
    pool_corr: float,
    pop_corr: float,
    n_trials: int,
) -> FitnessVector:
    """Dựng FitnessVector từ AlphaMetrics (Phase 4) + corr đã tính sẵn (Phase 6 / quần thể
    hiện tại) + số node (Phase 1 ComplexityVisitor) + độ sâu (Phase 1 DepthVisitor) + số
    lần thử (cho deflation).

    (T2.3) ``complexity_penalty = max(complexity/_COMPLEXITY_NORM, depth/MAX_DEPTH)`` —
    phương án GỘP (không tăng thêm chiều Pareto, giữ nguyên 6 chiều FitnessVector): trước
    Task 2, penalty này THUẦN đếm node nên một cây CHUỖI sâu (vd ts_delay lồng 7 tầng) nhưng
    chỉ ~7 node bị phạt gần như KHÔNG ĐÁNG KỂ dù rất dễ overfit/không thể combiner ghép
    (combiner cần depth nông, xem T1/T2.1/T2.2) — lấy max với depth/MAX_DEPTH buộc cây càng
    SÂU càng bị phạt nặng bất kể số node, bổ khuyết đúng lỗ hổng đó mà không cần thêm chiều
    tối ưu hóa mới (tránh đổi hành vi Pareto quá mạnh khi chưa A/B — xem task-2-brief.md)."""
    per_year_min = min(m.per_year_sharpe.values()) if m.per_year_sharpe else 0.0
    return FitnessVector(
        sharpe_deflated=deflated_sharpe(m.sharpe, n_trials),
        per_year_min_sharpe=per_year_min,
        turnover_penalty=_turnover_penalty(m.turnover),
        complexity_penalty=max(complexity / _COMPLEXITY_NORM, depth / MAX_DEPTH),
        pool_corr_penalty=pool_corr,
        pop_corr_penalty=pop_corr,
    )
