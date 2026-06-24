"""Test khung Evaluator: constant broadcast (T,N), field đọc từ MarketData, cache theo
canonical hash. KHÔNG test operator cụ thể ở đây (đó là golden test Task 2.3-2.8) — chỉ
test cơ chế visit_constant/visit_field/dispatch + cache, dùng operator giả lập đơn giản."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext, Evaluator
from src.engine.subexpr_cache import SubexprCache
from src.lang.ast import Call, Constant, Field
from src.lang.registry import ArgKind, OpCategory, OperatorRegistry, OperatorSpec


def _registry_voi_double() -> OperatorRegistry:
    """Registry test cục bộ (không đụng REGISTRY toàn cục) với 1 op giả `double(x) = x*2`."""
    reg = OperatorRegistry()
    reg.register(OperatorSpec(
        name="double", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL,), impl=lambda ctx, x: x * 2.0, bounded=False,
    ))
    return reg


def test_visit_constant_broadcast_shape_t_n(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=_registry_voi_double())
    out = Evaluator(ctx).evaluate(Constant(3.0))
    assert out.shape == small_panel.universe.shape
    assert np.all(out == 3.0)


def test_visit_field_doc_dung_du_lieu(small_panel) -> None:
    """visit_field áp universe mask (B6: out-of-universe = NaN) — so sánh phải tính
    luôn mask đó vào "expected", vì small_panel có 180 cell ngoài universe mà field
    `close` gốc KHÔNG có NaN sẵn (universe loại 3 mã cuối nửa đầu kỳ)."""
    ctx = EvalContext(data=small_panel, registry=_registry_voi_double())
    out = Evaluator(ctx).evaluate(Field("close"))
    expected = small_panel.field("close").copy()
    expected[~small_panel.universe] = np.nan
    np.testing.assert_array_equal(np.nan_to_num(out, nan=-1.0),
                                   np.nan_to_num(expected, nan=-1.0))


def test_visit_call_dispatch_dung_impl(small_panel) -> None:
    """visit_call áp universe mask sau impl — "expected" cũng phải mask cùng cách."""
    ctx = EvalContext(data=small_panel, registry=_registry_voi_double())
    out = Evaluator(ctx).evaluate(Call("double", (Field("close"),)))
    expected = small_panel.field("close") * 2.0
    expected[~small_panel.universe] = np.nan
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_cache_hit_khong_goi_lai_impl(small_panel) -> None:
    calls = {"n": 0}

    def _counting_impl(ctx, x):
        calls["n"] += 1
        return x * 2.0

    reg = OperatorRegistry()
    reg.register(OperatorSpec(
        name="double", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL,), impl=_counting_impl, bounded=False,
    ))
    ctx = EvalContext(data=small_panel, registry=reg, cache=SubexprCache(maxsize=8))
    node = Call("double", (Field("close"),))
    ev = Evaluator(ctx)
    ev.evaluate(node)
    ev.evaluate(node)  # node giống hệt -> cùng canonical hash -> cache hit
    assert calls["n"] == 1


def test_khong_co_cache_van_chay_dung(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=_registry_voi_double(), cache=None)
    out = Evaluator(ctx).evaluate(Call("double", (Field("close"),)))
    assert out.shape == small_panel.universe.shape
