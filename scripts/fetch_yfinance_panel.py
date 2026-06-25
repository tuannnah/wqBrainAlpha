"""Lấy OHLCV+sector cho universe S&P 500 (subset thanh khoản của TOP3000 USA) qua yfinance
-> MarketData -> parquet, làm nguồn `--market-data-dir` cho `calibrate` (Gap#3 fallback).

XẤP XỈ có chủ đích (KHÔNG khớp Brain chính xác): universe = S&P 500 (~500 mã, không phải
TOP3000 đầy đủ); vwap ≈ typical price (H+L+C)/3 (không có VWAP intraday miễn phí); sector từ
GICS Wikipedia; PIT theo ngày niêm yết yfinance. Đủ để đo ρ THẬT đầu tiên — nếu ρ thấp thì
biết do data fidelity, đúng mục đích calibration (B10).

Dùng: venv/Scripts/python.exe scripts/fetch_yfinance_panel.py [out_dir] [start] [end]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.market_panel import MarketData  # noqa: E402
from src.data.adapters.parquet_source import save  # noqa: E402

SP500_CSV = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/"
    "data/constituents.csv"
)


def _sp500() -> tuple[list[str], dict[str, str]]:
    """Trả (tickers, {ticker: gics_sector}) từ CSV constituents S&P 500 (datahub/GitHub).

    Dùng GitHub raw thay Wikipedia vì Wikipedia chặn request tự động (403)."""
    import io

    import httpx

    text = httpx.get(SP500_CSV, timeout=30, follow_redirects=True).text
    df = pd.read_csv(io.StringIO(text))
    syms = [str(t).replace(".", "-") for t in df["Symbol"]]  # BRK.B -> BRK-B (yfinance)
    sectors = dict(zip(syms, df["GICS Sector"].astype(str)))
    return syms, sectors


def _field_frame(data: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """Trích 1 field thành DataFrame (dates x tickers) từ download group_by='ticker'."""
    cols = {}
    for t in tickers:
        if (t, field) in data.columns:
            cols[t] = data[(t, field)]
    return pd.DataFrame(cols)


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/market_yf"
    start = sys.argv[2] if len(sys.argv) > 2 else "2015-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2025-06-01"

    tickers, sectors = _sp500()
    print(f"[1] S&P500: {len(tickers)} mã, {len(set(sectors.values()))} sector GICS")

    print(f"[2] Tải OHLCV yfinance {start}..{end} (có thể vài phút)...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True,
                       group_by="ticker", threads=True, progress=False)
    if data.empty:
        raise SystemExit("yfinance trả rỗng — kiểm tra mạng/tickers.")

    close = _field_frame(data, "Close", tickers)
    open_ = _field_frame(data, "Open", tickers)
    high = _field_frame(data, "High", tickers)
    low = _field_frame(data, "Low", tickers)
    volume = _field_frame(data, "Volume", tickers)

    # Giữ mã có >=60% ngày có giá (loại mã mới niêm yết/thiếu data nặng).
    coverage = close.notna().mean()
    keep = coverage[coverage >= 0.60].index.tolist()
    close, open_, high, low, volume = (df[keep] for df in (close, open_, high, low, volume))
    assets = list(close.columns)
    dates = close.index.to_numpy().astype("datetime64[D]")
    print(f"[3] Sau lọc coverage>=60%: {len(assets)} mã, {len(dates)} ngày")

    def arr(df: pd.DataFrame) -> np.ndarray:
        return df.to_numpy(dtype=np.float64)

    close_a, open_a, high_a, low_a, vol_a = arr(close), arr(open_), arr(high), arr(low), arr(volume)
    vwap_a = (high_a + low_a + close_a) / 3.0  # proxy typical price
    prev = np.empty_like(close_a)
    prev[0] = np.nan
    prev[1:] = close_a[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        returns_a = ((close_a - prev) / prev).astype(np.float64)

    # universe per-day: có giá đóng cửa + khối lượng dương (tradable ngày đó).
    universe = np.isfinite(close_a) & np.isfinite(vol_a) & (vol_a > 0)

    # sector codes: GICS sector -> int, hằng theo thời gian, tile (T,N).
    sec_labels = pd.Series([sectors.get(a, "Unknown") for a in assets])
    sec_codes = sec_labels.astype("category").cat.codes.to_numpy(dtype=np.int64)
    sector = np.tile(sec_codes, (len(dates), 1))

    md = MarketData(
        dates=dates, assets=np.array(assets, dtype=np.str_),
        fields={"close": close_a, "open": open_a, "high": high_a, "low": low_a,
                "volume": vol_a, "vwap": vwap_a},
        universe=universe, returns=returns_a, groups={"sector": sector},
    )
    save(md, out_dir)
    print(f"[4] Đã lưu parquet -> {out_dir}  (panel {close_a.shape[0]}x{close_a.shape[1]})")
    print(f"    universe in-day trung bình: {universe.mean():.2%}")


if __name__ == "__main__":
    main()
