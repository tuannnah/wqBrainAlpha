"""Đo tương đồng cấu trúc giữa hai cây AST (T3.1, T3.2).

Canon hoá có 2 chế độ qua cờ `field_aware`:
- `field_aware=False` (MẶC ĐỊNH): field -> "F", số -> "N" — đổi field hay window
  đều cùng canon. Dùng cho khử-trùng-CẤU-TRÚC (common_subtrees/avoid, đa dạng hoá
  local/nộp): mục tiêu là phát hiện lặp BỘ KHUNG operator bất kể field.
- `field_aware=True`: field GIỮ tên (chỉ số -> "N"). Dùng cho ĐỘ ĐỘC ĐÁO vs zoo:
  dùng dataset/field thay thế phải được tính là độc đáo, chỉ đổi window mới là
  biến thể tầm thường (xem ReferenceZoo).
Chỉ tính các nhánh con là Node (bỏ qua lá đơn).

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


def subtree_canon(node, field_aware: bool = False) -> str:
    """Chuỗi chuẩn hoá của một subtree (số->N; field->F hoặc giữ tên nếu field_aware)."""
    if isinstance(node, Leaf):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return "N"
        return str(node.value) if field_aware else "F"
    return node.op + "(" + ",".join(subtree_canon(c, field_aware) for c in node.children) + ")"


def _node_canon_sizes(tree, field_aware: bool = False) -> dict[str, int]:
    """canon -> kích thước (node_count) cho mọi nhánh con là Node."""
    sizes: dict[str, int] = {}
    for sub in all_subtrees(tree):
        if isinstance(sub, Node):
            c = subtree_canon(sub, field_aware)
            sizes[c] = max(sizes.get(c, 0), node_count(sub))
    return sizes


def largest_common_subtree(a, b, field_aware: bool = False) -> int:
    """Số node của nhánh con (Node) đẳng cấu lớn nhất chung giữa a và b."""
    a, b = _as_node(a), _as_node(b)
    sa, sb = _node_canon_sizes(a, field_aware), _node_canon_sizes(b, field_aware)
    common = set(sa) & set(sb)
    return max((min(sa[c], sb[c]) for c in common), default=0)


def similarity_ratio(a, b, field_aware: bool = False) -> float:
    """Tỉ lệ tương đồng ∈ [0,1] = nhánh chung lớn nhất / min(số node hai cây)."""
    a, b = _as_node(a), _as_node(b)
    denom = min(node_count(a), node_count(b))
    if denom <= 0:
        return 0.0
    return largest_common_subtree(a, b, field_aware) / denom


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


def avoid_subtree_canons(
    passed_exprs,
    failed_exprs=None,
    passed_min: int = 3,
    passed_top: int = 8,
    failed_min: int = 2,
    failed_top: int = 12,
) -> set[str]:
    """Tập canon subtree LLM nên tránh để giữ đa dạng (T3.6 mở rộng).

    Gộp 2 nguồn: bộ khung phổ biến trong alpha ĐÃ PASS (tránh lặp cái ai cũng có)
    và bộ khung LẶP LẠI trong các thất bại (failed_exprs, vd duplicate/low_score —
    tránh đi lại vết xe đổ). `failed_min>=2` để chỉ tránh cấu trúc thực sự bị lặp,
    không phải mọi expr hỏng đơn lẻ."""
    avoid = {c for c, _ in common_subtrees(passed_exprs, min_count=passed_min, top_n=passed_top)}
    if failed_exprs:
        avoid |= {c for c, _ in common_subtrees(failed_exprs, min_count=failed_min, top_n=failed_top)}
    return avoid
