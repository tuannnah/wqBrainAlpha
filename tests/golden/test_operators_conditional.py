"""Golden test trade_when/hump: logic carry-forward theo điều kiện, no-look-ahead."""

from __future__ import annotations

import numpy as np

import src.operators_local.arithmetic  # noqa: F401  # cần "sign" cho test trigger
import src.operators_local.conditional  # noqa: F401  # đăng ký impl thật vào REGISTRY
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Constant, Field
from src.lang.registry import default_registry


def _ctx_for(panel):
    return EvalContext(data=panel, registry=default_registry())


def test_trade_when_trigger_duong_lay_alpha(small_panel) -> None:
    ctx = _ctx_for(small_panel)
    node = Call("trade_when", (
        Call("sign", (Field("close"),)),  # luôn 1 (close>0) -> trigger luôn kích hoạt
        Field("close"),
        Constant(-1.0),  # exit luôn âm -> không carry-forward (n/a vì trigger luôn ưu tiên)
    ))
    out = Evaluator(ctx).evaluate(node)
    in_uni = small_panel.universe
    np.testing.assert_allclose(out[in_uni], small_panel.field("close")[in_uni], equal_nan=True)


def test_trade_when_khong_trigger_khong_exit_la_nan(small_panel) -> None:
    ctx = _ctx_for(small_panel)
    node = Call("trade_when", (
        Constant(-1.0),  # trigger luôn <=0 -> không bao giờ lấy alpha mới
        Field("close"),
        Constant(-1.0),  # exit luôn <=0 -> không carry-forward
    ))
    out = Evaluator(ctx).evaluate(node)
    assert np.all(np.isnan(out))


def test_hump_chan_thay_doi_nho(small_panel) -> None:
    ctx = _ctx_for(small_panel)
    out = Evaluator(ctx).evaluate(Call("hump", (Field("close"), Constant(1e9))))
    # threshold siêu lớn -> mọi thay đổi đều bị chặn -> chuỗi const = giá trị đầu tiên hợp lệ
    col = 0
    series = out[:, col]
    valid = ~np.isnan(series)
    vals = series[valid]
    assert np.allclose(vals, vals[0])


def test_hump_no_look_ahead(small_panel) -> None:
    from src.data.market_panel import MarketData

    row_t = 60
    mutated_close = small_panel.field("close").copy()
    mutated_close[row_t + 1 :] += 999.0
    mutated = MarketData(
        dates=small_panel.dates, assets=small_panel.assets,
        fields={**small_panel.fields, "close": mutated_close},
        universe=small_panel.universe, returns=small_panel.returns,
        groups=small_panel.groups,
    )
    node = Call("hump", (Field("close"), Constant(0.01)))
    out_orig = Evaluator(_ctx_for(small_panel)).evaluate(node)
    out_mut = Evaluator(_ctx_for(mutated)).evaluate(node)
    np.testing.assert_allclose(out_orig[row_t], out_mut[row_t], equal_nan=True)
