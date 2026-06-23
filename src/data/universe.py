"""Xây universe mask per-day (an toàn survivorship) và mã sector dạng int dày đặc."""

from __future__ import annotations

import numpy as np

from src.local_types import Mask, Panel


def build_universe_mask(tradable: Panel) -> Mask:
    """True khi cell hữu hạn và > 0 (vd volume/giá hợp lệ). Mỗi ngày một mask riêng."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(tradable) & (tradable > 0.0)


def sector_codes(raw_sector: np.ndarray) -> np.ndarray:
    """Map nhãn sector (chuỗi) -> int code dày đặc, giữ shape (T,N). NaN/None -> -1."""
    flat = raw_sector.ravel()
    labels = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
              for v in flat]
    uniq = {lab: i for i, lab in enumerate(sorted({x for x in labels if x is not None}))}
    codes = np.array([uniq[lab] if lab is not None else -1 for lab in labels], dtype=np.int64)
    return codes.reshape(raw_sector.shape)
