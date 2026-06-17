"""Phạt độ phức tạp của một alpha (T4.3).

Cây càng sâu, càng nhiều hằng số tự do và feature thì càng dễ overfit/decay.
Phạt là điểm mềm [0,1] (0 = đơn giản, 1 = rất phức tạp / parse lỗi), cộng vào số
hạng điều chuẩn ở T4.4. Parse lỗi -> phạt tối đa (an toàn).
"""

from __future__ import annotations

from src.generation.ast_utils import Leaf, iter_leaves, parse_expression, tree_depth

# Mốc chuẩn hoá: đạt mốc -> đóng góp tối đa cho chiều đó.
MAX_DEPTH = 8
MAX_CONSTANTS = 8
MAX_FIELDS = 8

WEIGHTS = {"depth": 0.5, "constants": 0.25, "fields": 0.25}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def complexity_features(expr: str) -> dict[str, int]:
    """Đếm độ sâu cây, số hằng số tự do, số field phân biệt."""
    tree = parse_expression(expr)
    n_constants = 0
    fields: set[str] = set()
    for leaf in iter_leaves(tree):
        if isinstance(leaf.value, (int, float)) and not isinstance(leaf.value, bool):
            n_constants += 1
        else:
            fields.add(str(leaf.value))
    return {
        "depth": tree_depth(tree),
        "n_constants": n_constants,
        "n_fields": len(fields),
    }


def complexity_penalty(expr: str) -> float:
    """Phạt độ phức tạp ∈ [0,1]. Parse lỗi -> 1.0 (phạt tối đa)."""
    try:
        f = complexity_features(expr)
    except ValueError:
        return 1.0
    return _clamp01(
        WEIGHTS["depth"] * f["depth"] / MAX_DEPTH
        + WEIGHTS["constants"] * f["n_constants"] / MAX_CONSTANTS
        + WEIGHTS["fields"] * f["n_fields"] / MAX_FIELDS
    )
