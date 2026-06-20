"""Lõi dùng chung của hai bộ sinh biểu thức LLM (generator + translator).

Gom phần trùng lặp: dựng ngữ cảnh prompt (symbol + field type), vòng
prefilter-repair, và auto-wrap field VECTOR. Hai lớp công khai
(LLMAlphaGenerator, AlphaTranslator) chỉ uỷ thác phần lõi cho module này.
"""

from __future__ import annotations

from loguru import logger

from src.generation.ast_utils import Leaf, Node, parse_expression, to_expression

MAX_REPAIR_ATTEMPTS = 3
MAX_FIELDS_IN_PROMPT = 40

# Ví dụ minh hoạ CÚ PHÁP, đa dạng cấu trúc, tránh khung kinh điển trùng Alpha101.
FEWSHOT_EXAMPLES = [
    "ts_decay_linear(rank(ts_std_dev(returns, 20)), 5)",
    "group_neutralize(ts_zscore(vwap, 60), industry)",
    "rank(divide(ts_mean(volume, 10), ts_mean(volume, 60)))",
    "ts_rank(ts_corr(close, volume, 20), 120)",
]


def autowrap_vector_fields(expr: str, field_types, matrix_only_ops) -> str:
    """Bọc vec_avg() quanh leaf field VECTOR bị đưa thẳng vào matrix-only op.

    Khớp ĐÚNG luật pre_filter._check_symbols: với Node có op ∈ matrix_only_ops,
    con TRỰC TIẾP là Leaf field có field_types[name]=='VECTOR' -> thay bằng
    vec_avg(leaf). Thiếu dữ liệu kiểu -> trả nguyên. Không parse được -> trả
    nguyên để prefilter báo lỗi (không nuốt lỗi).
    """
    if not field_types or not matrix_only_ops:
        return expr
    try:
        tree = parse_expression(expr)
    except ValueError:
        return expr

    def _walk(node):
        if isinstance(node, Leaf):
            return node
        wrap_here = node.op in matrix_only_ops
        new_children = []
        for child in node.children:
            child = _walk(child)
            if (
                wrap_here
                and isinstance(child, Leaf)
                and not isinstance(child.value, (int, float))
                and field_types.get(str(child.value)) == "VECTOR"
            ):
                child = Node("vec_avg", [child])
            new_children.append(child)
        node.children = new_children
        return node

    return to_expression(_walk(tree))
