"""Stress test bất biến kiểu (typed GP): mỗi đối số ở slot PANEL của một Call PHẢI là một
tín hiệu thật (``Call`` hoặc ``Field``) — KHÔNG bao giờ là ``Constant`` (literal số). Đây là
khoá regression cho lỗi xuyên-task Phase 7: ``random_tree`` cũ chèn Constant vào leaf PANEL,
và ``subtree_mutation``/``hoist_mutation`` cũ tráo subtree vào/từ slot WINDOW/SCALAR khiến
~50-75% cây sinh ra type-invalid.
"""

from __future__ import annotations

import numpy as np

import src.operators_local  # noqa: F401  (side-effect: nạp toàn bộ operator thật vào REGISTRY)
from src.gp.init import random_tree
from src.lang.ast import Call, Constant, Node
from src.lang.registry import ArgKind, OperatorRegistry, default_registry

_FIELDS = ("close", "volume", "returns")


def _check_panel_invariant(node: Node, registry: OperatorRegistry) -> None:
    """Đi qua cây: với mỗi Call, đối số nào nằm ở slot ``ArgKind.PANEL`` theo signature của
    operator thì PHẢI là ``Call`` hoặc ``Field`` (tín hiệu), không được là ``Constant``."""
    if not isinstance(node, Call):
        return
    spec = registry.get(node.op)
    for child, kind in zip(node.args, spec.signature):
        if kind is ArgKind.PANEL:
            assert not isinstance(child, Constant), (
                f"vi phạm bất biến PANEL: operator {node.op!r} có Constant({child.value!r}) "
                f"ở slot PANEL — phải là Call hoặc Field"
            )
        _check_panel_invariant(child, registry)


def test_random_tree_no_constant_at_panel_position() -> None:
    """100 cây sinh qua ``random_tree(kind=PANEL)`` — gốc không bao giờ là Constant và mọi
    slot PANEL bên trong cũng vậy."""
    registry = default_registry()
    for seed in range(100):
        rng = np.random.default_rng(seed)
        depth = int(rng.integers(2, 6))
        full = bool(rng.integers(0, 2))
        tree = random_tree(
            registry, rng, depth=depth, fields=_FIELDS, full=full, kind=ArgKind.PANEL,
        )
        assert not isinstance(tree, Constant), (
            f"seed={seed}: gốc cây PANEL không được là Constant"
        )
        _check_panel_invariant(tree, registry)
