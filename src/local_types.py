"""Type alias dùng chung cho tầng local backtester (panel (T,N), mask, axes)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

type Panel = npt.NDArray[np.float64]  # (T, N); NaN = missing / ngoài universe
type Mask = npt.NDArray[np.bool_]  # (T, N); True = trong universe ngày đó
type Dates = npt.NDArray[np.datetime64]  # (T,)
type Assets = npt.NDArray[np.str_]  # (N,)
