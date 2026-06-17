"""Kiểm chứng Out-of-Sample cho quét cấu hình (T5.6).

Tinh chỉnh cấu hình chỉ theo In-Sample là một dạng overfitting. Chỉ giữ cấu hình
nào tốt cả IS lẫn OS: OOS sharpe phải đạt một tỉ lệ tối thiểu so với IS sharpe.
Thiếu OOS -> coi như chưa kiểm chứng được -> không qua (an toàn).
"""

from __future__ import annotations


def oos_passes(result, min_ratio: float = 0.5) -> bool:
    """OOS sharpe >= min_ratio * IS sharpe? Thiếu OS hoặc IS<=0 -> False."""
    is_sharpe = result.sharpe
    os_sharpe = result.os_sharpe
    if os_sharpe is None or is_sharpe is None or is_sharpe <= 0:
        return False
    return os_sharpe >= min_ratio * is_sharpe
