"""PortfolioBuilder: signal (T,N) -> weights (T,N) qua 5 bước cấu hình (B7 master spec).

Thứ tự CỐ ĐỊNH: decay -> neutralize -> truncate -> scale -> delay. Mỗi bước chỉ tác động
cell in-universe; cell ngoài universe luôn NaN xuyên suốt (no-survivorship, B3 invariant).
"""

from __future__ import annotations

import numpy as np

from src.backtest.config import Neutralization, PortfolioConfig
from src.data.market_panel import MarketData
from src.local_types import Panel

_GROUP_KEY = {
    Neutralization.SECTOR: "sector",
    Neutralization.INDUSTRY: "industry",
    Neutralization.SUBINDUSTRY: "subindustry",
}


class PortfolioBuilder:
    """Áp `PortfolioConfig` lên một signal thô để ra weights tradable."""

    def build(self, signal: Panel, cfg: PortfolioConfig, data: MarketData) -> Panel:
        masked = np.where(data.universe, signal, np.nan)
        decayed = self._decay(masked, cfg.decay)
        neutralized = self._neutralize(decayed, cfg.neutralization, data)
        truncated = self._truncate(neutralized, cfg.truncation)
        scaled = self._scale(truncated, cfg.scale_book)
        return self._delay(scaled, cfg.delay)

    def _decay(self, signal: Panel, window: int) -> Panel:
        """Trung bình trọng số tuyến tính giảm dần trên trailing `window` ngày.
        window<=0 hoặc 1 -> không đổi (decay tắt)."""
        if window <= 1:
            return signal
        t = signal.shape[0]
        out = np.full_like(signal, np.nan)
        weights = np.arange(1, window + 1, dtype=np.float64)  # xa nhất=1 ... gần nhất=window
        for row in range(t):
            lo = max(0, row - window + 1)
            chunk = signal[lo : row + 1]
            w = weights[-(chunk.shape[0]) :]
            with np.errstate(invalid="ignore"):
                num = np.nansum(chunk * w[:, None], axis=0)
                valid_mask = ~np.isnan(chunk)
                denom = np.where(valid_mask, w[:, None], 0.0).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[row] = np.where(denom > 0, num / denom, np.nan)
        return out

    def _neutralize(self, signal: Panel, kind: Neutralization, data: MarketData) -> Panel:
        if kind is Neutralization.NONE:
            return signal
        if kind is Neutralization.MARKET:
            row_mean = np.nanmean(signal, axis=1, keepdims=True)
            demeaned: Panel = signal - row_mean
            return demeaned
        group_key = _GROUP_KEY[kind]
        groups = data.groups[group_key]  # raise KeyError nếu thiếu — đúng hợp đồng
        out = np.full_like(signal, np.nan)
        t = signal.shape[0]
        for row in range(t):
            row_signal = signal[row]
            row_groups = groups[row]
            for g in np.unique(row_groups):
                idx = row_groups == g
                vals = row_signal[idx]
                if np.all(np.isnan(vals)):
                    continue
                gmean = np.nanmean(vals)
                out[row, idx] = vals - gmean
        return out

    def _truncate(self, signal: Panel, cap: float) -> Panel:
        """Giới hạn tỉ lệ mỗi vị thế: `|w_i| <= cap * gross` (gross = tổng |w| trong ngày).

        Dùng water-filling LẶP: cap 1 lần làm gross giảm -> cap_abs giảm -> mã vừa cap có
        thể lại vượt; lặp tới khi không còn mã nào vượt (hội tụ). Đây là điểm khác tài
        liệu (bản plan cap 1-pass + renorm về gross cũ KHÔNG đảm bảo cap sau scale — đã
        xác nhận bằng phản ví dụ). KHÔNG renorm về gross cũ ở đây: bước `_scale` phía sau
        chia theo gross nên tỉ lệ (fraction) được bảo toàn — `_truncate` chỉ cần ghim
        fraction <= cap. Trường hợp suy biến `cap * n_valid <= 1` (quá ít mã để mọi vị thế
        đạt fraction <= cap) hội tụ về phân bổ đều tốt nhất khả thi."""
        if cap <= 0:
            return signal
        out = signal.copy()
        t = signal.shape[0]
        for row in range(t):
            r = out[row]
            valid = ~np.isnan(r)
            if not np.any(valid):
                continue
            for _ in range(1000):  # guard hội tụ; panel thật ít vòng, biên chậm vẫn đủ
                gross = float(np.nansum(np.abs(r)))
                if gross <= 0.0:
                    break
                cap_abs = cap * gross
                over = valid & (np.abs(r) > cap_abs + 1e-15)
                if not np.any(over):
                    break
                r[over] = np.sign(r[over]) * cap_abs
            out[row] = r
        return out

    def _scale(self, signal: Panel, scale_book: float) -> Panel:
        gross = np.nansum(np.abs(signal), axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized = np.where(gross > 0, signal / gross, np.nan)
        return normalized * scale_book

    def _delay(self, signal: Panel, delay: int) -> Panel:
        if delay <= 0:
            return signal
        out = np.full_like(signal, np.nan)
        out[delay:] = signal[:-delay]
        return out
