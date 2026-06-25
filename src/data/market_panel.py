"""Panel thị trường bất biến: các mảng field/universe/returns cùng trục (T,N).

MarketData là nguồn dữ liệu duy nhất cho Evaluator. Out-of-universe = NaN (không phải 0);
universe là mask per-day (an toàn look-ahead/survivorship).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.local_types import Assets, Dates, Mask, Panel


@dataclass(frozen=True, slots=True)
class MarketData:
    dates: Dates  # (T,)
    assets: Assets  # (N,)
    fields: dict[str, Panel]  # name -> (T, N)
    universe: Mask  # (T, N)
    returns: Panel  # (T, N) close-to-close simple returns
    groups: dict[str, np.ndarray]  # "sector" -> (T, N) int codes

    def __post_init__(self) -> None:
        t, n = len(self.dates), len(self.assets)
        shape = (t, n)
        for name, arr in self.fields.items():
            if arr.shape != shape:
                raise ValueError(f"field {name!r} shape {arr.shape} != {shape}")
        if self.universe.shape != shape:
            raise ValueError(f"universe shape {self.universe.shape} != {shape}")
        if self.returns.shape != shape:
            raise ValueError(f"returns shape {self.returns.shape} != {shape}")
        for name, arr in self.groups.items():
            if arr.shape != shape:
                raise ValueError(f"group {name!r} shape {arr.shape} != {shape}")
        if len(self.dates) > 1 and not (
            np.diff(self.dates.astype("datetime64[ns]")).astype("int64") > 0
        ).all():
            raise ValueError("dates phải tăng nghiêm ngặt (sorted, không trùng)")
        if self.universe.dtype != np.bool_:
            raise ValueError("universe phải là mảng bool")
        if self.dates.dtype.kind != "M":
            raise ValueError("dates phải là datetime64")

    def field(self, name: str) -> Panel:
        """Mảng (T,N) của field; KeyError nếu không có.

        `returns` là field WQ hợp lệ nhưng được lưu riêng ở thuộc tính `.returns` (backtester
        dùng cho PnL), KHÔNG trong `.fields`. Resolve nó như field phái sinh để mọi biểu thức
        tham chiếu `returns` eval được ở MỌI nơi (Evaluator/score_local_gate/calibration), không
        chỉ riêng calibration. `fields["returns"]` tường minh (nếu caller đưa vào) được ưu tiên.
        """
        if name in self.fields:
            return self.fields[name]
        if name == "returns":
            return self.returns
        raise KeyError(name)

    def field_names(self) -> set[str]:
        """Tên MỌI field truy vấn được qua `field()`: `fields` + `returns` phái sinh.

        Dùng cho validate `fields_ok` (gate) — nhất quán với `field()`, tránh chặn nhầm alpha
        dùng `returns` (field WQ hợp lệ nhưng lưu riêng ngoài `.fields`)."""
        return set(self.fields) | {"returns"}

    def years(self) -> dict[int, slice]:
        """Slice hàng theo từng năm dương lịch (cho per-year Sharpe)."""
        yrs = self.dates.astype("datetime64[Y]").astype(int) + 1970
        out: dict[int, slice] = {}
        for y in np.unique(yrs):
            idx = np.nonzero(yrs == y)[0]
            out[int(y)] = slice(int(idx[0]), int(idx[-1]) + 1)
        return out
