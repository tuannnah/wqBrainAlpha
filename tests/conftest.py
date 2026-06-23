"""Cấu hình chung cho test suite.

Mục tiêu chính: KHÔNG để test ghi đè vào log production
(`logs/wq_alpha_<date>.log`). Loguru dùng handler toàn cục, nên nếu một test
gọi `main._setup_logging()` (qua lệnh Typer chẳng hạn) thì file sink dính cho
cả phiên → mọi logger.error của các test fixture (foo_bar, rank(a,b,c)...) đổ
vào log thật, gây nhiễu khó soi lỗi. Ta đặt biến môi trường WQ_NO_FILE_LOG
ngay từ đầu phiên để `_setup_logging` bỏ qua file sink, và chủ động gỡ mọi
file sink có thể đã được thêm trước đó.
"""

from __future__ import annotations

import os

import pytest
from loguru import logger


@pytest.fixture(scope="session", autouse=True)
def _no_production_log_during_tests():
    os.environ["WQ_NO_FILE_LOG"] = "1"
    # Gỡ sạch handler hiện có (kể cả file sink lỡ dính) rồi chỉ giữ stderr.
    import sys

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    yield


# --- MiniBrain: panel nhỏ thật-hình-dạng cho test backtester (Phase 0+) ---
@pytest.fixture
def small_panel():
    """Panel (T=120, N=30) reproducible: close = random walk; volume dương;
    universe per-day (vài mã rời giữa kỳ); sector groups; returns close-to-close."""
    import numpy as np

    from src.data.market_panel import MarketData

    rng = np.random.default_rng(42)
    t, n = 120, 30
    dates = (np.datetime64("2020-01-01") + np.arange(t)).astype("datetime64[D]")
    assets = np.array([f"A{i:02d}" for i in range(n)], dtype=np.str_)
    steps = rng.normal(0.0, 0.02, size=(t, n))
    close = 100.0 * np.exp(np.cumsum(steps, axis=0))
    volume = rng.uniform(1e5, 1e6, size=(t, n))
    universe = np.ones((t, n), dtype=bool)
    universe[: t // 2, -3:] = False  # 3 mã cuối chỉ vào universe nửa sau
    prev = np.empty_like(close)
    prev[0] = np.nan
    prev[1:] = close[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = (close - prev) / prev
    sector = np.tile(np.arange(n) % 5, (t, 1)).astype(np.int64)
    return MarketData(dates=dates, assets=assets,
                      fields={"close": close, "volume": volume},
                      universe=universe, returns=returns, groups={"sector": sector})
