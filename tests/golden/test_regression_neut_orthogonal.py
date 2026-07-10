"""Golden Pha 3.1: regression_neut(y, x) cho residual TRỰC GIAO với x mỗi hàng (corr chéo ≈ 0).

Đây là cơ sở toán để tune dùng regression_neut hạ self-corr: nếu alpha bị neutralize theo một
risk factor crowded thì thành phần chung với factor đó bị trừ đi -> tương quan pool giảm."""

from __future__ import annotations

import numpy as np

import src.operators_local  # noqa: F401  # đăng ký regression_neut
from src.data.market_panel import MarketData
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import parse
from src.lang.registry import REGISTRY


def _eval(expr, data):
    return Evaluator(EvalContext(data=data, registry=REGISTRY)).evaluate(parse(expr, registry=REGISTRY))


def test_residual_truc_giao_voi_risk_factor():
    rng = np.random.default_rng(1)
    n_days, n_assets = 30, 40
    close = 100 + rng.standard_normal((n_days, n_assets)).cumsum(axis=0)
    volume = np.abs(rng.standard_normal((n_days, n_assets))) * 1e6 + 1e5
    dates = (np.datetime64("2021-01-01") + np.arange(n_days)).astype("datetime64[D]")
    assets = np.array([f"A{i:02d}" for i in range(n_assets)], dtype=np.str_)
    data = MarketData(dates=dates, assets=assets,
                      fields={"close": close, "volume": volume},
                      universe=np.ones((n_days, n_assets), dtype=bool),
                      returns=np.zeros((n_days, n_assets)),
                      groups={"sector": np.zeros((n_days, n_assets), dtype=np.int64)})

    resid = _eval("regression_neut(ts_delta(close, 5), rank(volume))", data)
    factor = _eval("rank(volume)", data)

    # Với mỗi hàng đủ dữ liệu: corr chéo giữa residual và factor ≈ 0 (đã trừ OLS).
    checked = 0
    for t in range(n_days):
        r, f = resid[t], factor[t]
        m = np.isfinite(r) & np.isfinite(f)
        if m.sum() < 5 or np.std(r[m]) < 1e-9 or np.std(f[m]) < 1e-9:
            continue
        c = np.corrcoef(r[m], f[m])[0, 1]
        assert abs(c) < 1e-6, f"hàng {t}: corr={c}"
        checked += 1
    assert checked > 0
