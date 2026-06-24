"""PortfolioConfig — tách "cấu hình" khỏi "tín hiệu" (stage separation, Gap #8 master spec).

Expression (AST từ src/lang) chỉ là signal core; mọi WQ "settings" (neutralization, decay,
truncation, scale, delay) sống ở đây và được áp ở tầng portfolio (Task 3.2), KHÔNG trong
cây AST. Lý do: search GP (Phase 7) tìm core trong ngân sách độ sâu ≈7 — trộn config vào
cây AST lãng phí depth và làm nhiễu attribution (alpha tốt vì core hay vì config?).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Neutralization(Enum):
    """Cách trừ trung bình cross-sectional khi build weights (Task 3.2)."""

    NONE = auto()
    MARKET = auto()
    SECTOR = auto()
    INDUSTRY = auto()
    SUBINDUSTRY = auto()


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    """Toàn bộ tham số "config stage" của một alpha — tách khỏi expression.

    `decay`: window ts_decay_linear áp lên signal trước neutralize; 0 = tắt (signal nhanh,
    turnover là alpha, không nên decay). `truncation`: cap |w_i| theo tỉ lệ book (gate tập
    trung). `scale_book`: tổng |w| sau scale (1.0 = dollar-neutral chuẩn long-short).
    `delay`: weight tại t áp cho return tại t+delay (delay-1 mặc định, đúng convention WQ
    Delay-1)."""

    neutralization: Neutralization = Neutralization.SECTOR
    decay: int = 0
    truncation: float = 0.10
    scale_book: float = 1.0
    delay: int = 1
