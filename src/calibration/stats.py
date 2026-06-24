"""Spearman rank-correlation tự viết bằng numpy — KHÔNG phụ thuộc scipy.

Quyết định: scipy chưa có trong requirements.txt của repo; Spearman = Pearson trên rank,
công thức đơn giản không cần thuật toán số trị scipy mới cung cấp được. Thêm scipy chỉ để
có 1 hàm là dependency nặng không tương xứng (~30-50MB). Tie dùng average rank (định nghĩa
Spearman chuẩn khi có giá trị trùng — vd nhiều alpha cùng Sharpe làm tròn)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def average_rank(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rank 1-based tăng dần; giá trị trùng nhận rank trung bình của nhóm."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks_sorted = np.empty(len(values), dtype=np.float64)

    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        # rank 1-based trung bình của nhóm [i, j] (inclusive)
        avg = (i + 1 + j + 1) / 2.0
        ranks_sorted[i : j + 1] = avg
        i = j + 1

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def spearman(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Spearman rho giữa x và y: Pearson rho trên average_rank(x)/average_rank(y).

    Cặp có NaN ở x HOẶC y bị loại trước khi rank (pairwise-complete). Trả NaN nếu còn
    < 2 cặp hợp lệ, hoặc x/y hằng số sau khi loại NaN (rho vô nghĩa khi std=0)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid], y[valid]
    if xv.size < 2:
        return float("nan")

    rx = average_rank(xv)
    ry = average_rank(yv)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return float("nan")

    return float(np.corrcoef(rx, ry)[0, 1])
