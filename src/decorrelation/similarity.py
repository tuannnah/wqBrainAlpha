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

from src.lang.ast import Call, Constant, Field, Node
from src.lang.parser import ParseError, parse_expression
from src.lang.visitors import ComplexityVisitor, all_subtrees


def _as_node(tree: Node | str) -> Node:
    return parse_expression(tree) if isinstance(tree, str) else tree


def subtree_canon(node: Node, field_aware: bool = False) -> str:
    """Chuỗi chuẩn hoá của một subtree (số->N; field->F hoặc giữ tên nếu field_aware)."""
    if isinstance(node, Constant):
        return "N"
    if isinstance(node, Field):
        return node.name if field_aware else "F"
    assert isinstance(node, Call)
    return node.op + "(" + ",".join(subtree_canon(c, field_aware) for c in node.args) + ")"


def _node_canon_sizes(tree: Node, field_aware: bool = False) -> dict[str, int]:
    """canon -> kích thước (node_count) cho mọi nhánh con là Call (tức Node có con)."""
    sizes: dict[str, int] = {}
    cv = ComplexityVisitor()
    for sub in all_subtrees(tree):
        if isinstance(sub, Call):
            c = subtree_canon(sub, field_aware)
            sizes[c] = max(sizes.get(c, 0), cv.visit(sub))
    return sizes


def largest_common_subtree(
    a: Node | str, b: Node | str, field_aware: bool = False
) -> int:
    """Số node của nhánh con (Call) đẳng cấu lớn nhất chung giữa a và b."""
    a, b = _as_node(a), _as_node(b)
    sa, sb = _node_canon_sizes(a, field_aware), _node_canon_sizes(b, field_aware)
    common = set(sa) & set(sb)
    return max((min(sa[c], sb[c]) for c in common), default=0)


def similarity_ratio(
    a: Node | str, b: Node | str, field_aware: bool = False
) -> float:
    """Tỉ lệ tương đồng ∈ [0,1] = nhánh chung lớn nhất / min(số node hai cây)."""
    a, b = _as_node(a), _as_node(b)
    cv = ComplexityVisitor()
    denom = min(cv.visit(a), cv.visit(b))
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
        except (ValueError, ParseError):
            continue
        canons = {
            subtree_canon(sub)
            for sub in all_subtrees(tree)
            if isinstance(sub, Call)
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
