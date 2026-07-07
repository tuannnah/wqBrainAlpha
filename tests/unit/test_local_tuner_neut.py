"""Test cho `tune` quét thêm chiều neutralization ∈ {MARKET, SECTOR} ở Giai đoạn 2.

Docs WQ Brain khuyến nghị alpha price/volume neutralize bằng MARKET hoặc SECTOR (Industry/
Subindustry làm giảm hiệu năng) — panel local chỉ có group "sector" nên chỉ quét 2 giá trị
này (không quét INDUSTRY/SUBINDUSTRY, không có group tương ứng để eval local).
"""

from __future__ import annotations

import src.operators_local  # noqa: F401
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.local_tuner import tune


def _cfg():
    return PortfolioConfig(neutralization=Neutralization.MARKET, decay=4, truncation=0.08)


def test_tune_quet_neutralization_chon_sector():
    # eval_fn cho điểm cao khi neutralization=SECTOR -> phải được chọn.
    def eval_fn(node, config):
        return 2.0 if config.neutralization is Neutralization.SECTOR else 0.5

    res = tune("rank(close)", _cfg(), data=None, budget=60, eval_fn=eval_fn)
    assert res.best_config.neutralization is Neutralization.SECTOR
    assert res.local_sharpe == 2.0
