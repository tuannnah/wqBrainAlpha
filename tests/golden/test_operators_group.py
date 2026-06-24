"""Golden test group_neutralize: trừ mean theo group mỗi ngày, per-row in-universe."""

from __future__ import annotations

import numpy as np

import src.operators_local.group  # noqa: F401  # đăng ký impl thật vào REGISTRY
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Field
from src.lang.registry import default_registry


def test_group_neutralize_mean_0_moi_group(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("group_neutralize", (Field("close"), Field("sector"))))
    row = 100  # universe đầy đủ ở nửa sau
    sector_row = small_panel.groups["sector"][row]
    in_uni = small_panel.universe[row]
    for g in np.unique(sector_row[in_uni]):
        mask = in_uni & (sector_row == g)
        if mask.sum() < 1:
            continue
        assert abs(float(np.mean(out[row][mask]))) < 1e-8


def test_group_neutralize_gp_usable_false() -> None:
    spec = default_registry().get("group_neutralize")
    assert spec.gp_usable is False
