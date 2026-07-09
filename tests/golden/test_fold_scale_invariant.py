"""Golden Pha 1.1: chứng minh fold scale DƯƠNG ở gốc KHÔNG đổi positions thực tế.

CanonicalHasher coi multiply(k>0, X) ≡ X. Test này khóa bất biến đó ở tầng portfolio THẬT
(PortfolioBuilder.build): positions của `multiply(2, X)` phải trùng `X` từng ô. Nếu ai đó
đổi _scale/_truncate thành phi tuyến với scale, test này gãy -> cảnh báo dedup gộp nhầm.

Cơ sở toán: _scale chia signal cho L1-norm (sum|signal|) nên (k·s)/sum|k·s| = s/sum|s| với
k>0; decay tuyến tính + neutralize (trừ mean/OLS) đều scale-preserving."""

from __future__ import annotations

import numpy as np
import pytest

import src.operators_local.arithmetic  # noqa: F401  # đăng ký impl thật vào REGISTRY
import src.operators_local.timeseries  # noqa: F401  # ts_delta/ts_mean...
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.portfolio import PortfolioBuilder
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import parse
from src.lang.registry import REGISTRY


def _positions(expr: str, data, cfg):
    node = parse(expr, registry=REGISTRY)
    ctx = EvalContext(data=data, registry=REGISTRY)
    signal = Evaluator(ctx).evaluate(node)
    return PortfolioBuilder().build(signal, cfg, data)


@pytest.mark.parametrize("cfg", [
    PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.0),
    PortfolioConfig(neutralization=Neutralization.MARKET, decay=4, truncation=0.08),
    PortfolioConfig(neutralization=Neutralization.SECTOR, decay=2, truncation=0.05),
])
def test_scale_duong_bat_bien_positions(small_panel, cfg):
    base = "ts_delta(close, 5)"
    scaled = "multiply(2, ts_delta(close, 5))"
    p_base = _positions(base, small_panel, cfg)
    p_scaled = _positions(scaled, small_panel, cfg)
    np.testing.assert_allclose(
        np.nan_to_num(p_base), np.nan_to_num(p_scaled), rtol=1e-9, atol=1e-12,
    )


def test_scale_am_KHONG_bat_bien(small_panel):
    """Chốt ngược: multiply(-1, X) ĐẢO positions -> KHÔNG được coi trùng."""
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.0)
    p = _positions("ts_delta(close, 5)", small_panel, cfg)
    p_neg = _positions("multiply(-1, ts_delta(close, 5))", small_panel, cfg)
    finite = np.isfinite(p) & np.isfinite(p_neg) & (np.abs(p) > 1e-9)
    assert finite.any()
    assert np.all(np.sign(p[finite]) == -np.sign(p_neg[finite]))
