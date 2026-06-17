"""Gộp ba khoản phạt vào điểm hiệu quả thành điểm điều chuẩn (T4.4).

Công thức: điểm điều chuẩn = điểm hiệu quả − λ · (Σ trọng số · phạt từng chiều).
Ba chiều phạt: độ độc đáo, mức khớp giả thuyết, độ phức tạp. Tất cả là điểm
PHẠT ∈ [0,1] (cao = phạt nặng). Trọng số và λ cấu hình được, mặc định mức vừa.

Lưu ý hướng: originality (độ độc đáo) và alignment (khớp giả thuyết) là điểm
"tốt" (cao = tốt). `from_scores` đổi chúng thành phạt = 1 − điểm. complexity vốn
đã là phạt nên giữ nguyên.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


@dataclass
class PenaltyWeights:
    originality: float = 0.34
    alignment: float = 0.33
    complexity: float = 0.33


@dataclass
class Penalties:
    """Giá trị PHẠT từng chiều ∈ [0,1] (cao = phạt nặng)."""

    originality: float = 0.0
    alignment: float = 0.0
    complexity: float = 0.0

    @classmethod
    def from_scores(cls, originality: float, alignment: float, complexity: float) -> "Penalties":
        """Từ điểm 'tốt' -> phạt. originality/alignment cao = tốt -> phạt = 1−điểm;
        complexity vốn là phạt nên giữ nguyên."""
        return cls(
            originality=_clamp01(1.0 - originality),
            alignment=_clamp01(1.0 - alignment),
            complexity=_clamp01(complexity),
        )


def regularized_score(
    effective_score: float,
    penalties: Penalties,
    weights: PenaltyWeights | None = None,
    lambda_: float = 0.3,
) -> float:
    """điểm hiệu quả − λ·Σ(trọng số·phạt). Clamp tại 0 (không trả số âm)."""
    w = weights or PenaltyWeights()
    total_penalty = (
        w.originality * penalties.originality
        + w.alignment * penalties.alignment
        + w.complexity * penalties.complexity
    )
    return max(0.0, effective_score - lambda_ * total_penalty)
