"""Pha 1.4: GP sinh scalar từ tập RỜI RẠC, không float ngẫu nhiên 15 chữ số
(IMPROVEMENT_SPEC §3 Pha 1.3). Tránh winsorize(open, -1.9423623924877862) vô nghĩa."""

from __future__ import annotations

import numpy as np

from src.gp.init import DISCRETE_SCALARS, _random_leaf, random_tree
from src.lang.ast import Call, Constant
from src.lang.registry import ArgKind, default_registry
from src.lang.visitors import iter_leaves


def test_discrete_scalars_la_tap_dinh_nghia():
    assert set(DISCRETE_SCALARS) == {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}


def test_random_leaf_scalar_thuoc_tap_roi_rac():
    rng = np.random.default_rng(0)
    for _ in range(50):
        leaf = _random_leaf(rng, ("close", "open"), kind=ArgKind.SCALAR)
        assert isinstance(leaf, Constant)
        assert leaf.value in DISCRETE_SCALARS


def test_random_tree_moi_scalar_thuoc_tap_roi_rac():
    """Mọi Constant ở slot SCALAR trong cây sinh ra phải thuộc tập rời rạc (không float lẻ)."""
    reg = default_registry()
    rng = np.random.default_rng(3)
    fields = ("close", "open", "volume")
    for _ in range(30):
        tree = random_tree(reg, rng, depth=4, fields=fields, full=False)
        for leaf in iter_leaves(tree):
            if isinstance(leaf, Constant):
                v = leaf.value
                # window (int>=2) hoặc scalar rời rạc — KHÔNG float lẻ nhiều chữ số
                assert v == int(v) or v in DISCRETE_SCALARS, f"scalar lẻ: {v}"
