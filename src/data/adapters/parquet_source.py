"""Adapter Parquet cho MarketDataSource: đọc/ghi panel (T,N) ra đĩa để tái dùng.

Layout: root/axes.parquet (dates,assets), root/fields/<name>.parquet,
root/universe.parquet, root/returns.parquet, root/groups/<g>.parquet.
Mỗi bảng field: index=dates, columns=assets, value=float64 (NaN giữ nguyên).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.market_panel import MarketData


def _to_frame(arr: np.ndarray, dates: np.ndarray, assets: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(arr, index=pd.Index(dates, name="date"), columns=list(assets))


def save(md: MarketData, root: str) -> None:
    """Ghi MarketData ra parquet partitioned dưới `root`."""
    base = Path(root)
    (base / "fields").mkdir(parents=True, exist_ok=True)
    (base / "groups").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": md.dates, "date_idx": range(len(md.dates))}).to_parquet(
        base / "axes_dates.parquet")
    pd.DataFrame({"asset": md.assets}).to_parquet(base / "axes_assets.parquet")
    for name, arr in md.fields.items():
        _to_frame(arr, md.dates, md.assets).to_parquet(base / "fields" / f"{name}.parquet")
    _to_frame(md.universe.astype(np.int8), md.dates, md.assets).to_parquet(base / "universe.parquet")
    _to_frame(md.returns, md.dates, md.assets).to_parquet(base / "returns.parquet")
    for g, arr in md.groups.items():
        _to_frame(arr, md.dates, md.assets).to_parquet(base / "groups" / f"{g}.parquet")


class ParquetSource:
    """Đọc panel đã lưu. Thỏa Protocol MarketDataSource."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def available_fields(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "fields").glob("*.parquet"))

    def load(self, start: str, end: str, universe: str = "TOP3000") -> MarketData:
        assets = pd.read_parquet(self.root / "axes_assets.parquet")["asset"].to_numpy().astype(np.str_)
        fields: dict[str, np.ndarray] = {}
        dates_ref: np.ndarray | None = None
        for name in self.available_fields():
            df = pd.read_parquet(self.root / "fields" / f"{name}.parquet")
            mask = (df.index >= np.datetime64(start)) & (df.index <= np.datetime64(end))
            df = df.loc[mask]
            dates_ref = df.index.to_numpy().astype("datetime64[D]")
            fields[name] = df.to_numpy(dtype=np.float64)
        assert dates_ref is not None, "không có field nào để suy ra trục dates"

        def _load(name: str) -> np.ndarray:
            df = pd.read_parquet(self.root / name)
            m = (df.index >= np.datetime64(start)) & (df.index <= np.datetime64(end))
            return df.loc[m].to_numpy()

        universe_arr = _load("universe.parquet").astype(bool)
        returns_arr = _load("returns.parquet").astype(np.float64)
        groups: dict[str, np.ndarray] = {}
        gdir = self.root / "groups"
        if gdir.exists():
            for p in gdir.glob("*.parquet"):
                df = pd.read_parquet(p)
                m = (df.index >= np.datetime64(start)) & (df.index <= np.datetime64(end))
                groups[p.stem] = df.loc[m].to_numpy()
        return MarketData(dates=dates_ref, assets=assets, fields=fields,
                          universe=universe_arr, returns=returns_arr, groups=groups)
