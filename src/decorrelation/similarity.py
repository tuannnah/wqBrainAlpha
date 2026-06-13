"""Đo tương đồng cấu trúc giữa hai cây AST (T3.1, T3.2).

Canon hoá: field -> "F", số -> "N", nên đổi field hay đổi tham số cửa sổ đều cho
cùng canon — chủ ý coi các biến thể tầm thường là "trùng cấu trúc". Chỉ tính các
nhánh con là Node (bỏ qua lá đơn) để tương đồng phản ánh bộ khung operator.

LƯU Ý: đây KHÁC return-correlation thật của WQ. AST-similarity chỉ là bộ lọc rẻ,
chạy local, để loại trùng hiển nhiên trước khi tốn quota simulate.
"""

from __future__ import annotations

from collections import Counter

from src.generation.ast_utils import (
    Leaf,
    Node,
    all_subtrees,
    node_count,
    parse_expression,
)


def _as_node(tree):
    return parse_expression(tree) if isinstance(tree, str) else tree


def subtree_canon(node) -> str:
    """Chuỗi chuẩn hoá của một subtree (field->F, số->N, giữ thứ tự con)."""
    if isinstance(node, Leaf):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return "N"
        return "F"
    return node.op + "(" + ",".join(subtree_canon(c) for c in node.children) + ")"


def _node_canon_sizes(tree) -> dict[str, int]:
    """canon -> kích thước (node_count) cho mọi nhánh con là Node."""
    sizes: dict[str, int] = {}
    for sub in all_subtrees(tree):
        if isinstance(sub, Node):
            c = subtree_canon(sub)
            sizes[c] = max(sizes.get(c, 0), node_count(sub))
    return sizes


def largest_common_subtree(a, b) -> int:
    """Số node của nhánh con (Node) đẳng cấu lớn nhất chung giữa a và b."""
    a, b = _as_node(a), _as_node(b)
    sa, sb = _node_canon_sizes(a), _node_canon_sizes(b)
    common = set(sa) & set(sb)
    return max((min(sa[c], sb[c]) for c in common), default=0)


def similarity_ratio(a, b) -> float:
    """Tỉ lệ tương đồng ∈ [0,1] = nhánh chung lớn nhất / min(số node hai cây)."""
    a, b = _as_node(a), _as_node(b)
    denom = min(node_count(a), node_count(b))
    if denom <= 0:
        return 0.0
    return largest_common_subtree(a, b) / denom


def common_subtrees(
    expressions, min_count: int = 2, top_n: int | None = None
) -> list[tuple[str, int]]:
    """Thống kê canon subtree (operator) xuất hiện ở NHIỀU alpha (T3.6).

    Mỗi alpha đóng góp tối đa 1 cho mỗi canon (đếm theo số alpha chứa, không
    theo số lần lặp trong một alpha). Chỉ giữ canon đạt `min_count`, sort giảm
    theo số lần, cắt còn `top_n` nếu có. Biểu thức parse lỗi bị bỏ qua.
    """
    counter: Counter[str] = Counter()
    for expr in expressions:
        try:
            tree = parse_expression(expr) if isinstance(expr, str) else expr
        except ValueError:
            continue
        canons = {
            subtree_canon(sub)
            for sub in all_subtrees(tree)
            if isinstance(sub, Node)
        }
        counter.update(canons)
    items = [(c, n) for c, n in counter.items() if n >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_n] if top_n is not None else items
