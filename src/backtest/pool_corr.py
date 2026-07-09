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


def pairwise_abs_rho(
    pnl_a: npt.NDArray[np.float64], dates_a: Dates,
    pnl_b: npt.NDArray[np.float64], dates_b: Dates,
) -> float | None:
    """|Pearson rho| của hai chuỗi PnL trên NGÀY GIAO NHAU; None nếu thiếu điểm hoặc
    phương sai bằng 0 — KHÔNG bịa rho=0 giả.

    Helper cấp thấp dùng chung: `PoolCorrelation` (self-corr với pool) và `combiner`
    (chọn tín hiệu ít tương quan để ghép) đều gọi hàm này để logic tương quan chỉ tồn
    tại MỘT chỗ. Tự sort mỗi chuỗi theo dates trước khi align nên caller không cần
    sort trước (PnL từ DB và từ run hiện tại có thể lệch trục ngày)."""
    dates_a = np.asarray(dates_a)
    pnl_a = np.asarray(pnl_a, dtype=np.float64)
    dates_b = np.asarray(dates_b)
    pnl_b = np.asarray(pnl_b, dtype=np.float64)
    ord_a = np.argsort(dates_a)
    dates_a, pnl_a = dates_a[ord_a], pnl_a[ord_a]
    ord_b = np.argsort(dates_b)
    dates_b, pnl_b = dates_b[ord_b], pnl_b[ord_b]
    common = np.intersect1d(dates_a, dates_b)
    if common.size < _MIN_OVERLAP_POINTS:
        return None
    x = pnl_a[np.searchsorted(dates_a, common)]
    y = pnl_b[np.searchsorted(dates_b, common)]
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < _MIN_OVERLAP_POINTS:
        return None
    x, y = x[finite], y[finite]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    rho = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(rho):
        return None
    return abs(rho)


class PoolCorrelation:
    """Max |Pearson rho| của candidate PnL so với từng alpha trong pool."""

    def __init__(
        self, pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]]
    ) -> None:
        # PRE-SORT mỗi pool member MỘT LẦN theo dates (trước đây sort lại ở MỖI max_corr
        # call -> với pool lớn là ~48k argsort/batch = nút thắt production 54%). Sort ở đây
        # 1 lần/pool member; max_corr chỉ sort candidate 1 lần rồi align.
        self._pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] = {}
        for pid, (d, p) in pool.items():
            d_arr = np.asarray(d)
            p_arr = np.asarray(p, dtype=np.float64)
            order = np.argsort(d_arr)
            self._pool[pid] = (d_arr[order], p_arr[order])

    def max_corr(
        self, candidate_pnl: npt.NDArray[np.float64], dates: Dates
    ) -> tuple[float, int | None]:
        """Trả (max(|rho|), id) qua toàn bộ pool; pool rỗng -> (0.0, None).

        Caller KHÔNG cần sort dates trước: hàm tự sort candidate MỘT LẦN (pool đã pre-sort
        ở __init__) trước khi align, nên dates lệch thứ tự vẫn cho kết quả đúng.
        """
        if not self._pool:
            return 0.0, None

        # Sort candidate MỘT LẦN (thay vì lặp lại trong từng _pairwise_rho như trước).
        cand_dates = np.asarray(dates)
        cand_pnl = np.asarray(candidate_pnl, dtype=np.float64)
        cand_order = np.argsort(cand_dates)
        cand_dates = cand_dates[cand_order]
        cand_pnl = cand_pnl[cand_order]

        best_abs_rho = 0.0
        best_id: int | None = None

        for pool_id, (pool_dates, pool_pnl) in self._pool.items():
            rho = self._rho_sorted(cand_pnl, cand_dates, pool_pnl, pool_dates)
            if rho is None:
                continue
            abs_rho = abs(rho)
            if best_id is None or abs_rho > best_abs_rho:
                best_abs_rho = abs_rho
                best_id = pool_id

        if best_id is None:
            return 0.0, None
        return best_abs_rho, best_id

    @staticmethod
    def _rho_sorted(
        candidate_pnl: npt.NDArray[np.float64],
        candidate_dates: Dates,
        pool_pnl: npt.NDArray[np.float64],
        pool_dates: Dates,
    ) -> float | None:
        """|Pearson rho| trên ngày giao nhau; GIẢ ĐỊNH cả hai đã sort tăng dần theo dates.

        Fast-path CÙNG TRỤC NGÀY (mọi alpha lưu qua save_pool_pnl với cùng data.dates ->
        đa số cặp trùng ngày): bỏ intersect1d/searchsorted (đắt), align = identity. Kết
        quả TƯƠNG ĐƯƠNG nhánh tổng quát vì dates trùng chính xác."""
        if candidate_dates.shape == pool_dates.shape and np.array_equal(candidate_dates, pool_dates):
            cand_aligned = candidate_pnl
            pool_aligned = pool_pnl
        else:
            common = np.intersect1d(candidate_dates, pool_dates)
            if common.size < _MIN_OVERLAP_POINTS:
                return None
            cand_aligned = candidate_pnl[np.searchsorted(candidate_dates, common)]
            pool_aligned = pool_pnl[np.searchsorted(pool_dates, common)]

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
