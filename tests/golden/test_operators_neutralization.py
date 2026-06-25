"""Golden test regression_neut/vector_neut: residual có corr ~0 với biến neutralize,
per-row in-universe."""

from __future__ import annotations

import numpy as np

import src.operators_local.neutralization  # noqa: F401  # đăng ký impl thật vào REGISTRY
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Field
from src.lang.registry import default_registry


def test_regression_neut_residual_khong_tuong_quan_voi_x(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(
        Call("regression_neut", (Field("close"), Field("volume")))
    )
    row = 100
    in_uni = small_panel.universe[row]
    resid = out[row][in_uni]
    x = small_panel.field("volume")[row][in_uni]
    corr = np.corrcoef(resid, x)[0, 1]
    assert abs(corr) < 1e-6


def test_vector_neut_truc_giao_voi_y(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("vector_neut", (Field("close"), Field("volume"))))
    row = 100
    in_uni = small_panel.universe[row]
    resid = out[row][in_uni]
    y = small_panel.field("volume")[row][in_uni]
    dot = float(np.dot(resid, y))
    assert abs(dot) < 1e-6


def test_categoria_e_gp_usable_false(small_panel) -> None:
    """B5 stage separation: regression_neut/vector_neut là wrapper-stage (neutralization là
    PortfolioConfig của Phase 3, KHÔNG phải signal core của GP) -> gp_usable=False."""
    spec_r = default_registry().get("regression_neut")
    spec_v = default_registry().get("vector_neut")
    assert spec_r.gp_usable is False
    assert spec_v.gp_usable is False
