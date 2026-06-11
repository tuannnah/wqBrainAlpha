"""Lọc syntax TRƯỚC khi simulate để khỏi phí quota."""

from __future__ import annotations

from src.generation.ast_utils import (
    BINARY_OPS,
    Leaf,
    Node,
    node_count,
    parse_expression,
    tree_depth,
)

DEFAULT_GROUPS = {"market", "sector", "industry", "subindustry", "country", "exchange"}


class PreFilter:
    def __init__(
        self,
        known_operators: set[str] | None = None,
        known_fields: set[str] | None = None,
        known_groups: set[str] | None = None,
        max_depth: int = 6,
        max_nodes: int = 30,
    ):
        self.known_operators = known_operators
        self.known_fields = known_fields
        self.known_groups = known_groups or set(DEFAULT_GROUPS)
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def check(self, expr: str) -> tuple[bool, str]:
        if expr.count("(") != expr.count(")"):
            return False, "Dấu ngoặc không cân bằng"

        try:
            tree = parse_expression(expr)
        except ValueError as exc:
            return False, f"Parse lỗi: {exc}"

        if tree_depth(tree) > self.max_depth:
            return False, f"Độ sâu > {self.max_depth}"
        if node_count(tree) > self.max_nodes:
            return False, f"Số node > {self.max_nodes}"

        ok, reason = self._check_symbols(tree)
        if not ok:
            return False, reason

        return True, "ok"

    def _check_symbols(self, node) -> tuple[bool, str]:
        if isinstance(node, Leaf):
            if isinstance(node.value, (int, float)):
                return True, "ok"
            name = str(node.value)
            if self.known_fields is not None and name not in self.known_fields:
                if name not in self.known_groups:
                    return False, f"Field/hằng không tồn tại: {name}"
            return True, "ok"

        if node.op not in BINARY_OPS and node.op != "neg":
            if self.known_operators is not None and node.op not in self.known_operators:
                return False, f"Operator không tồn tại: {node.op}"

        for child in node.children:
            ok, reason = self._check_symbols(child)
            if not ok:
                return False, reason
        return True, "ok"
