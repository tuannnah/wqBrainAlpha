"""Integration: parse(expr_str) -> Evaluator.evaluate(node) -> (T,N) Panel đúng
NaN-propagation trên small_panel, dùng registry đầy đủ Phase 2 (không placeholder)."""

from __future__ import annotations

import numpy as np

import src.operators_local  # noqa: F401  side-effect: đăng ký toàn bộ operator
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import parse
from src.lang.registry import default_registry


def test_khong_con_placeholder_not_implemented(small_panel) -> None:
    reg = default_registry()
    ctx = EvalContext(data=small_panel, registry=reg)
    node = parse("rank(close)")
    out = Evaluator(ctx).evaluate(node)  # raise NotImplementedError nếu còn placeholder
    assert out.shape == small_panel.universe.shape


def test_pipeline_bieu_thuc_long(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    node = parse("rank(ts_delta(close, 10))")
    out = Evaluator(ctx).evaluate(node)
    in_uni = small_panel.universe
    assert np.all(np.isnan(out[~in_uni]))
    # hàng có đủ lịch sử (t>=9) và universe đầy đủ phải có rank hợp lệ
    row = 50
    assert not np.any(np.isnan(out[row][in_uni[row]]))


def test_tat_ca_operator_co_impl_khong_placeholder(small_panel) -> None:
    reg = default_registry()
    for name in ["add", "subtract", "multiply", "divide", "log", "abs", "sign", "power",
                 "max", "min", "rank", "winsorize", "scale", "zscore", "ts_mean", "ts_std",
                 "ts_delta", "ts_delay", "ts_rank", "ts_zscore", "ts_corr",
                 "ts_decay_linear", "ts_backfill", "group_neutralize", "regression_neut",
                 "vector_neut", "trade_when", "hump"]:
        spec = reg.get(name)
        assert spec.impl.__name__ != "_not_implemented", f"{name} vẫn là placeholder"


def test_gp_function_set_loai_wrapper_config() -> None:
    reg = default_registry()
    gp_names = {s.name for s in reg.gp_function_set()}
    assert "group_neutralize" not in gp_names
    assert "scale" not in gp_names
    # B5 stage separation: neutralization (regression_neut/vector_neut) là wrapper-stage
    # của PortfolioConfig Phase 3, KHÔNG phải signal core -> loại khỏi function set GP.
    assert "regression_neut" not in gp_names
    assert "vector_neut" not in gp_names
