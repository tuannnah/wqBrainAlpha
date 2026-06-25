"""PoolCorrelation — self-correlation cục bộ, tính năng đòn bẩy cao nhất của MiniBrain
(B9 master design): max |Pearson rho| của PnL candidate so với từng alpha đã PASS trong
pool, align trên ngày giao nhau. Đây là PROXY LOCAL, miễn phí quota; checker thật của Brain
là authoritative trước khi submit thật — không thay thế, chỉ lọc trước.

Pool được truyền vào dưới dạng dict đã vật chất hóa trong RAM (đọc từ DB ở tầng
storage/pipeline, KHÔNG ở đây) — pool_corr.py không import src.storage để giữ dependency
rule (lang/operators_local/engine/backtest không phụ thuộc storage/gp/llm).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.local_types import Dates

_MIN_OVERLAP_POINTS = 2  # Pearson cần >=2 điểm hữu hạn để có phương sai xác định


class PoolCorrelation:
    """Max |Pearson rho| của candidate PnL so với từng alpha trong pool."""

    def __init__(
        self, pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]]
    ) -> None:
        self._pool = pool

    def max_corr(
        self, candidate_pnl: npt.NDArray[np.float64], dates: Dates
    ) -> tuple[float, int | None]:
        if not self._pool:
            return 0.0, None

        best_abs_rho = 0.0
        best_id: int | None = None

        for pool_id, (pool_dates, pool_pnl) in self._pool.items():
            rho = self._pairwise_rho(candidate_pnl, dates, pool_pnl, pool_dates)
            if rho is None:
                continue
            abs_rho = abs(rho)
            if best_id is None or abs_rho > best_abs_rho:
                best_abs_rho = abs_rho
                best_id = pool_id

        if best_id is None:
            return 0.0, None
        return best_abs_rho, best_id

    def _pairwise_rho(
        self,
        candidate_pnl: npt.NDArray[np.float64],
        candidate_dates: Dates,
        pool_pnl: npt.NDArray[np.float64],
        pool_dates: Dates,
    ) -> float | None:
        common = np.intersect1d(candidate_dates, pool_dates)
        if common.size < _MIN_OVERLAP_POINTS:
            return None

        cand_idx = np.searchsorted(candidate_dates, common)
        pool_idx = np.searchsorted(pool_dates, common)
        cand_aligned = candidate_pnl[cand_idx]
        pool_aligned = pool_pnl[pool_idx]

        finite = np.isfinite(cand_aligned) & np.isfinite(pool_aligned)
        if finite.sum() < _MIN_OVERLAP_POINTS:
            return None
        cand_aligned = cand_aligned[finite]
        pool_aligned = pool_aligned[finite]

        if cand_aligned.std(ddof=0) == 0.0 or pool_aligned.std(ddof=0) == 0.0:
            return None

        rho = float(np.corrcoef(cand_aligned, pool_aligned)[0, 1])
        if not np.isfinite(rho):
            return None
        return rho
