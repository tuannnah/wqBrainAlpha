"""LocalTuner: quét tham số (window/hệ số) + config quanh MỘT biểu thức bằng đường backtest
MiniBrain (không mạng, không LLM). Deterministic — coordinate descent dưới ngân sách, luôn
giữ biểu thức gốc làm cận dưới (không bao giờ trả kết quả tệ hơn gốc)."""

from __future__ import annotations

from src.lang.ast import Call, Constant, Node
from src.lang.registry import ArgKind, OperatorRegistry


def iter_constants(
    node: Node, registry: OperatorRegistry, _path: tuple[int, ...] = ()
) -> list[tuple[tuple[int, ...], float, bool]]:
    """Liệt kê mọi hằng số trong cây kèm đường tới nó và cờ 'là window'."""
    if isinstance(node, Constant):
        return [(_path, node.value, False)]  # gốc là hằng đơn — hiếm; không đánh dấu window
    if not isinstance(node, Call):
        return []
    try:
        kinds = registry.get(node.op).signature
    except KeyError:
        kinds = ()
    out: list[tuple[tuple[int, ...], float, bool]] = []
    for i, arg in enumerate(node.args):
        kind = kinds[i] if i < len(kinds) else None
        if isinstance(arg, Constant):
            out.append((_path + (i,), arg.value, kind is ArgKind.WINDOW))
        else:
            out.extend(iter_constants(arg, registry, _path + (i,)))
    return out


def set_constant(node: Node, path: tuple[int, ...], new_value: float) -> Node:
    """Trả node mới với hằng tại `path` = new_value; phần còn lại giữ nguyên (bất biến)."""
    if not path:
        return Constant(float(new_value))
    if isinstance(node, Call):
        i = path[0]
        new_args = list(node.args)
        new_args[i] = set_constant(node.args[i], path[1:], new_value)
        return Call(node.op, tuple(new_args))
    return node
