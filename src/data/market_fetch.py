"""Kéo OHLCV + universe + sector từ WQ Brain → MarketData → parquet.

Tách logic ghép (thuần, test được) khỏi I/O mạng. Phần mạng (`fetch_to_parquet`) gọi
WQBrainClient thật; vì WQ không cấp bulk OHLCV sạch (Gap #3), endpoint thực tế phải probe khi
chạy tay — nếu chưa xác định, raise NotImplementedError có chỉ dẫn thay vì giả vờ thành công.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.data.adapters.parquet_source import save  # noqa: F401 (dùng khi fetch_to_parquet hết spike)
from src.data.market_panel import MarketData
from src.data.universe import build_universe_mask, sector_codes
from src.local_types import Panel

if TYPE_CHECKING:
    from src.data.client import WQBrainClient

RawField = tuple[np.ndarray, np.ndarray, np.ndarray]  # (dates, assets, values(T,N))


def _simple_returns(close: Panel) -> Panel:
    """Close-to-close simple returns; hàng đầu = NaN (không look-ahead)."""
    prev = np.empty_like(close)
    prev[0] = np.nan
    prev[1:] = close[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        # astype trả ndarray có kiểu rõ (float64), tránh mypy "Returning Any"
        return ((close - prev) / prev).astype(np.float64)


def _assemble_panel(
    raw_by_field: dict[str, RawField],
    sector_raw: np.ndarray,
    tradable_field: str = "volume",
) -> MarketData:
    """Ghép raw theo field thành MarketData căn trục. Giả định mọi field cùng (dates,assets)."""
    if not raw_by_field:
        raise ValueError("raw_by_field rỗng")
    dates, assets, _ = next(iter(raw_by_field.values()))
    fields = {name: vals for name, (_, _, vals) in raw_by_field.items()}
    tradable = fields[tradable_field]
    universe = build_universe_mask(tradable)
    returns = _simple_returns(fields["close"])
    groups = {"sector": sector_codes(sector_raw)}
    return MarketData(dates=dates.astype("datetime64[D]"),
                      assets=assets.astype(np.str_), fields=fields,
                      universe=universe, returns=returns, groups=groups)


def fetch_to_parquet(
    client: WQBrainClient,
    fields: list[str],
    start: str,
    end: str,
    universe: str = "TOP3000",
    out_dir: str = "data/market",
) -> str:
    """Kéo `fields` cho cửa sổ [start,end] → ghi parquet, trả root path.

    CHƯA chốt endpoint bulk: probe khi chạy tay (xem docstring module). Tới khi xác định,
    nâng NotImplementedError có chỉ dẫn để không tạo data sai âm thầm.
    """
    raise NotImplementedError(
        "Endpoint bulk OHLCV của WQ Brain chưa được xác định (Gap #3). "
        "Probe API khi có phiên rồi điền logic; tạm thời nạp panel qua ParquetSource.save(). "
        "Ghi cách lấy data thực tế vào PROGRESS.md."
    )
