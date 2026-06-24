"""Golden test arithmetic ops trên small_panel: giá trị đúng + NaN-propagate."""

from __future__ import annotations

import numpy as np

import src.operators_local.arithmetic  # noqa: F401  # đăng ký impl thật vào REGISTRY
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Constant, Field
from src.lang.registry import default_registry


def test_add_dung_gia_tri(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("add", (Field("close"), Field("volume"))))
    expected = small_panel.field("close") + small_panel.field("volume")
    expected[~small_panel.universe] = np.nan
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_divide_nan_khi_mau_la_nan(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("divide", (Field("close"), Field("close"))))
    # mọi cell trong-universe: close/close == 1.0; ngoài universe -> NaN (mask sau impl)
    in_uni = small_panel.universe
    assert np.allclose(out[in_uni], 1.0)
    assert np.all(np.isnan(out[~in_uni]))


def test_log_abs_sign(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    close = small_panel.field("close")
    out_log = Evaluator(ctx).evaluate(Call("log", (Field("close"),)))
    expected_log = np.log(close)
    expected_log[~small_panel.universe] = np.nan
    np.testing.assert_allclose(out_log, expected_log, equal_nan=True)
    out_abs = Evaluator(ctx).evaluate(Call("abs", (Field("close"),)))
    expected_abs = np.abs(close)
    expected_abs[~small_panel.universe] = np.nan
    np.testing.assert_allclose(out_abs, expected_abs, equal_nan=True)
    out_sign = Evaluator(ctx).evaluate(Call("sign", (Field("close"),)))
    expected_sign = np.sign(close)
    expected_sign[~small_panel.universe] = np.nan
    np.testing.assert_allclose(out_sign, expected_sign, equal_nan=True)


def test_power_max_min(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    close = small_panel.field("close")
    out_pow = Evaluator(ctx).evaluate(Call("power", (Field("close"), Constant(2.0))))
    expected_pow = close**2.0
    expected_pow[~small_panel.universe] = np.nan
    np.testing.assert_allclose(out_pow, expected_pow, equal_nan=True)
    out_max = Evaluator(ctx).evaluate(Call("max", (Field("close"), Field("volume"))))
    expected_max = np.maximum(close, small_panel.field("volume"))
    expected_max[~small_panel.universe] = np.nan
    np.testing.assert_allclose(out_max, expected_max, equal_nan=True)


def test_nan_propagation_qua_chuoi_phep_toan(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(
        Call("add", (Call("log", (Field("close"),)), Constant(1.0)))
    )
    in_uni = small_panel.universe
    assert not np.any(np.isnan(out[in_uni]))  # close luôn >0 trong fixture
    assert np.all(np.isnan(out[~in_uni]))
