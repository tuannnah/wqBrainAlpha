"""IS-Ladder Sharpe robustness gate (docs consultant-submission-tests / finding-consultant-alphas).

Sharpe TOÀN KỲ có thể che giấu alpha đã suy thoái ở các năm gần đây. IS-Ladder xét Sharpe
trên cửa sổ trượt N=2..10 năm GẦN NHẤT (N=2 chặt nhất) và:
- Sharpe[N] < FAIL -> FAIL ngay (đúng dấu hiệu "suy thoái gần đây" mà Sharpe-tổng bỏ sót).
- Sharpe[N] >= PASS[N] -> PASS ngay.
- ở giữa -> tăng N. Turnover < cutoff -> nhân PASS ×mult (không nhân FAIL).

Đây là SOFT signal của tầng local (pre-filter tiết kiệm quota): Brain mới là trọng tài
cuối. `ladder_decision` là logic quyết định thuần (dễ test chính xác); `is_ladder_verdict`
dựng windows từ daily_pnl + data.years() rồi gọi nó.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config.thresholds import (
    IS_LADDER_FAIL,
    IS_LADDER_LOW_TURNOVER_CUTOFF,
    IS_LADDER_LOW_TURNOVER_MULT,
    IS_LADDER_PASS,
)

_PERIODS_PER_YEAR = 252
_MAX_LADDER_YEAR = max(IS_LADDER_PASS)


@dataclass(frozen=True, slots=True)
class LadderVerdict:
    passed: bool
    detail: str
    fail_window: int | None = None  # số năm cửa sổ gây FAIL (None nếu không FAIL)
    pass_window: int | None = None  # số năm cửa sổ đạt PASS tường minh (None nếu chỉ borderline)
    windows: dict[int, float] = field(default_factory=dict)  # {N: Sharpe trailing N năm}


def pass_threshold(n_years: int, *, low_turnover: bool) -> float:
    """Ngưỡng PASS cho cửa sổ n_years; cửa sổ > 10 năm dùng ngưỡng năm 10 (thấp nhất).
    Turnover thấp -> nhân ×mult (alpha low-turnover after-cost tốt hơn nên được nới)."""
    base = IS_LADDER_PASS.get(n_years, IS_LADDER_PASS[_MAX_LADDER_YEAR])
    return base * IS_LADDER_LOW_TURNOVER_MULT if low_turnover else base


def ladder_decision(windows: dict[int, float], *, low_turnover: bool) -> LadderVerdict:
    """Áp thang IS-Ladder trên các Sharpe cửa sổ trượt đã tính (`windows[N] = Sharpe`).
    Duyệt N tăng dần (từ 2): FAIL nếu tụt sàn; PASS nếu vượt ngưỡng; nếu ở giữa suốt thì
    borderline (passed=True nhưng pass_window=None — không suy thoái nhưng chưa đủ mạnh)."""
    if not windows:
        return LadderVerdict(True, "không có cửa sổ nào để xét — bỏ qua ladder", windows={})
    for n in sorted(windows):
        s = windows[n]
        if s < IS_LADDER_FAIL:
            return LadderVerdict(
                False, f"IS-Ladder FAIL: Sharpe {n}Y {s:.2f} < {IS_LADDER_FAIL}",
                fail_window=n, windows=dict(windows),
            )
        thr = pass_threshold(n, low_turnover=low_turnover)
        if s >= thr:
            return LadderVerdict(
                True, f"IS-Ladder PASS ở {n}Y: Sharpe {s:.2f} >= {thr:.2f}",
                pass_window=n, windows=dict(windows),
            )
    return LadderVerdict(
        True, "IS-Ladder borderline: không FAIL nhưng chưa đạt PASS tường minh",
        windows=dict(windows),
    )


def _sharpe(pnl: np.ndarray) -> float:
    """Sharpe năm hóa (khớp MetricsCalculator._sharpe): mean/std·√252, bỏ NaN, ddof=0."""
    valid = pnl[np.isfinite(pnl)]
    if valid.size < 2:
        return 0.0
    std = valid.std(ddof=0)
    if std == 0.0:
        return 0.0
    return float(valid.mean() / std * np.sqrt(_PERIODS_PER_YEAR))


def is_ladder_verdict(daily_pnl: np.ndarray, data, turnover: float) -> LadderVerdict:
    """Dựng Sharpe cửa sổ trượt N=2..min(10, số năm) từ `data.years()` (slice liên tục,
    tăng dần) rồi áp `ladder_decision`. <2 năm dữ liệu -> không chặn (passed=True)."""
    years = sorted(data.years().items())  # [(year, slice), ...] tăng dần
    if len(years) < 2:
        return LadderVerdict(True, "ít hơn 2 năm dữ liệu — bỏ qua ladder", windows={})
    low_turnover = turnover < IS_LADDER_LOW_TURNOVER_CUTOFF
    windows: dict[int, float] = {}
    for n in range(2, min(_MAX_LADDER_YEAR, len(years)) + 1):
        recent = years[-n:]  # n năm gần nhất (slice liên tục nên ghép = [start, stop))
        start, stop = recent[0][1].start, recent[-1][1].stop
        windows[n] = _sharpe(daily_pnl[start:stop])
    return ladder_decision(windows, low_turnover=low_turnover)
