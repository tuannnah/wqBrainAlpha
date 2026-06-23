"""Visitor cụ thể trên AST: DepthVisitor, FieldCollector, Serializer, CanonicalHasher,
ComplexityVisitor. Mỗi visitor một trách nhiệm (B4 design) — không tangle với evaluator.
"""

from __future__ import annotations

import hashlib

from src.lang.ast import Call, Constant, Field, Node, NodeVisitor
from src.lang.registry import OperatorRegistry, default_registry


class DepthVisitor(NodeVisitor[int]):
    """Độ sâu tối đa của cây, ĐẾM CẢ wrapper Call (vd rank(...) tính 1 tầng độc lập với
    số args). Leaf có depth 1; Call có depth 1 + max(depth con, mặc định 0 nếu rỗng)."""

    def visit(self, node: Node) -> int:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> int:
        return 1

    def visit_field(self, node: Field) -> int:
        return 1

    def visit_call(self, node: Call) -> int:
        child_depths = [c.accept(self) for c in node.children()]
        return 1 + (max(child_depths) if child_depths else 0)


class FieldCollector(NodeVisitor["set[str]"]):
    """Tập tên field được tham chiếu trong cây — phục vụ validate field tồn tại và
    dead-field blacklist (Phase 0.7/Phase 5)."""

    def visit(self, node: Node) -> set[str]:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> set[str]:
        return set()

    def visit_field(self, node: Field) -> set[str]:
        return {node.name}

    def visit_call(self, node: Call) -> set[str]:
        result: set[str] = set()
        for c in node.children():
            result |= c.accept(self)
        return result


class Serializer(NodeVisitor[str]):
    """AST -> chuỗi FASTEXPR canonical. Round-trip với parser:
    parse(Serializer().visit(node)) == node. Toán tử nhị phân luôn render dạng hàm
    (vd `add(a, b)`), không dạng infix — đơn giản hóa round-trip (grammar Task 1.4 chấp
    nhận cả hai dạng nhưng AST không phân biệt nguồn gốc cú pháp)."""

    def visit(self, node: Node) -> str:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> str:
        if node.value.is_integer():
            return str(int(node.value))
        return repr(node.value)

    def visit_field(self, node: Field) -> str:
        return node.name

    def visit_call(self, node: Call) -> str:
        args = ", ".join(c.accept(self) for c in node.children())
        return f"{node.op}({args})"


class CanonicalHasher(NodeVisitor[str]):
    """Hash sha256-hex ổn định sau canonicalize: literal normalize qua repr(float),
    args của operator commutative (theo registry) được sort trước khi ghép — đảm bảo
    add(a,b) và add(b,a) cho cùng hash. Dùng cho sub-expression cache, result cache,
    dedup quần thể GP (B12)."""

    def __init__(self, registry: OperatorRegistry | None = None) -> None:
        self._registry = registry if registry is not None else default_registry()

    def visit(self, node: Node) -> str:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> str:
        return self._digest(f"const:{repr(float(node.value))}")

    def visit_field(self, node: Field) -> str:
        return self._digest(f"field:{node.name}")

    def visit_call(self, node: Call) -> str:
        child_hashes = [c.accept(self) for c in node.children()]
        if self._is_commutative(node.op):
            child_hashes = sorted(child_hashes)
        return self._digest(f"call:{node.op}({','.join(child_hashes)})")

    def _is_commutative(self, op: str) -> bool:
        try:
            return self._registry.get(op).commutative
        except KeyError:
            return False

    @staticmethod
    def _digest(payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
