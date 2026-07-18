"""Test C1: worker song song `src.gp.parallel_eval` — `khoi_tao_worker`/`eval_thuan` chạy
IN-PROCESS (gọi thẳng, logic thuần không cần pool thật) + một test integration pool thật
2 worker (Windows spawn) để xác nhận initializer/pickling hoạt động đúng qua ranh giới
process — không chỉ đúng khi gọi trực tiếp trong cùng process.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

import src.operators_local  # noqa: F401  (side-effect: nạp 27 operator vào registry)
from src.backtest.backtester import BacktestResult
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.metrics_local import AlphaMetrics
from src.data.market_panel import MarketData
from src.gp.parallel_eval import eval_thuan, khoi_tao_worker
from src.lang.registry import default_registry


def _panel_gia() -> MarketData:
    """Panel nhỏ reproducible (T=60, N=10) — đủ cho eval_thuan chạy thật, không cần
    small_panel (fixture conftest) vì module-level: test integration pool thật cần hàm
    top-level picklable để build initargs, không thể dùng closure/fixture trực tiếp."""
    rng = np.random.default_rng(7)
    t, n = 60, 10
    dates = (np.datetime64("2021-01-01") + np.arange(t)).astype("datetime64[D]")
    assets = np.array([f"A{i:02d}" for i in range(n)], dtype=np.str_)
    steps = rng.normal(0.0, 0.02, size=(t, n))
    close = 100.0 * np.exp(np.cumsum(steps, axis=0))
    open_ = close * (1.0 + rng.normal(0.0, 0.005, size=(t, n)))
    universe = np.ones((t, n), dtype=bool)
    prev = np.empty_like(close)
    prev[0] = np.nan
    prev[1:] = close[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = (close - prev) / prev
    sector = np.tile(np.arange(n) % 3, (t, 1)).astype(np.int64)
    return MarketData(
        dates=dates, assets=assets, fields={"close": close, "open": open_},
        universe=universe, returns=returns, groups={"sector": sector},
    )


def _config_gia() -> PortfolioConfig:
    return PortfolioConfig(
        neutralization=Neutralization.NONE, decay=0, truncation=0.10,
        scale_book=1.0, delay=1,
    )


def test_eval_thuan_tra_ok_va_loi() -> None:
    """`eval_thuan` gọi thẳng (không qua pool) sau `khoi_tao_worker`: cây hợp lệ -> "ok"
    kèm BacktestResult + AlphaMetrics thật; operator không tồn tại -> "error" kèm lý do."""
    khoi_tao_worker(_panel_gia(), _config_gia(), default_registry())
    tag, bt, metrics = eval_thuan("ts_mean(subtract(close, open), 5)")
    assert tag == "ok"
    assert isinstance(bt, BacktestResult)
    assert isinstance(metrics, AlphaMetrics)
    assert bt.daily_pnl.shape == (60,)

    tag2, reasons = eval_thuan("op_khong_ton_tai(close)")
    assert tag2 == "error"
    assert reasons  # non-empty, có lý do cụ thể


def test_eval_thuan_dung_registry_cua_ctx_khong_phai_default_registry_ngam_dinh() -> None:
    """Registry rỗng (không nạp operators_local) qua `khoi_tao_worker` -> `eval_thuan` PHẢI
    lỗi parse ngay cả khi `default_registry()` toàn cục (đã bị test khác import
    operators_local) có đủ operator -- xác nhận `eval_thuan` dùng `_CTX["registry"]` tường
    minh, không âm thầm rơi về `default_registry()` mặc định của `parse()` (registry của
    process con là singleton MỚI khi spawn thật, không được lẫn với registry test hiện tại)."""
    from src.lang.registry import OperatorRegistry

    khoi_tao_worker(_panel_gia(), _config_gia(), OperatorRegistry())
    tag, reasons = eval_thuan("ts_mean(close, 5)")
    assert tag == "error"
    assert reasons


def test_eval_thuan_qua_pool_that_2_worker() -> None:
    """Integration: `ProcessPoolExecutor` thật (Windows spawn) với `khoi_tao_worker` làm
    initializer -- xác nhận data/config/registry pickle qua ranh giới process đúng và kết
    quả worker khớp kết quả gọi thẳng in-process (parity). Không đánh dấu `slow` -- suite
    hiện chưa có convention marker này (grep pytest.ini/pyproject rỗng), thêm marker chưa
    đăng ký sẽ chỉ tạo warning thừa."""
    data = _panel_gia()
    cfg = _config_gia()
    registry = default_registry()
    expr = "ts_mean(subtract(close, open), 5)"

    expected_tag, expected_bt, expected_metrics = _eval_in_process(data, cfg, registry, expr)

    with ProcessPoolExecutor(
        max_workers=2, initializer=khoi_tao_worker, initargs=(data, cfg, registry),
    ) as ex:
        fut1 = ex.submit(eval_thuan, expr)
        fut2 = ex.submit(eval_thuan, "op_khong_ton_tai(close)")
        tag1, bt1, metrics1 = fut1.result(timeout=60)
        tag2, reasons2 = fut2.result(timeout=60)

    assert tag1 == "ok" == expected_tag
    assert isinstance(bt1, BacktestResult)
    np.testing.assert_allclose(bt1.daily_pnl, expected_bt.daily_pnl)
    assert metrics1.sharpe == pytest.approx(expected_metrics.sharpe)
    assert tag2 == "error"
    assert reasons2


def _eval_in_process(data, config, registry, expr):  # noqa: ANN001
    khoi_tao_worker(data, config, registry)
    return eval_thuan(expr)
