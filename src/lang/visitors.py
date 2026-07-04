"""Visitor cụ thể trên AST: DepthVisitor, FieldCollector, Serializer, CanonicalHasher,
ComplexityVisitor. Mỗi visitor một trách nhiệm (B4 design) — không tangle với evaluator.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

from src.lang.ast import Call, Constant, Field, Node, NodeVisitor
from src.lang.registry import ArgKind, OperatorRegistry, default_registry


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
    dead-field blacklist (Phase 0.7/Phase 5). Chỉ thu thập field ở đúng vị trí ArgKind.PANEL
    của mỗi operator (tín hiệu thật) -- bỏ qua WINDOW/SCALAR/GROUP (literal, không phải field
    tham chiếu, vd tên group "sector" trong group_neutralize(x, sector)). Khớp semantics
    Evaluator.visit_call (src/engine/evaluator.py) -- registry bắt buộc để tra signature.
    Nếu operator không có trong registry local (vd phân tích alpha THẬT đã nộp trên WQ Brain,
    dùng operator chưa implement local -- xem power_pool.py/dataset_usage.py/genius_report.py),
    fallback về hành vi cũ: duyệt hết mọi con, không phân biệt ArgKind (các module gọi ở ngữ
    cảnh này đã tự lọc riêng qua _GROUPING_FIELDS/_EXEMPT_OPERATORS cho đúng mục đích nghiệp vụ
    của họ, nên fallback bảo thủ này không sai lệch kết quả nghiệp vụ)."""

    def __init__(self, registry: OperatorRegistry) -> None:
        self.registry = registry

    def visit(self, node: Node) -> set[str]:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> set[str]:
        return set()

    def visit_field(self, node: Field) -> set[str]:
        return {node.name}

    def visit_call(self, node: Call) -> set[str]:
        try:
            spec = self.registry.get(node.op)
        except KeyError:
            # Operator không có trong registry local (vd phân tích alpha thật dùng operator
            # WQ Brain chưa implement local, xem power_pool.py/dataset_usage.py) -- không biết
            # signature nên fallback: duyệt hết con như hành vi cũ (các module gọi ở ngữ cảnh
            # này đã tự lọc riêng qua _GROUPING_FIELDS/_EXEMPT_OPERATORS cho đúng mục đích
            # nghiệp vụ của họ).
            result: set[str] = set()
            for c in node.children():
                result |= c.accept(self)
            return result
        result: set[str] = set()
        for arg, kind in zip(node.args, spec.signature, strict=True):
            if kind is ArgKind.PANEL:
                result |= arg.accept(self)
        return result


class OperatorCollector(NodeVisitor["set[str]"]):
    """Tập tên operator (Call.op) dùng trong cây — phục vụ đếm operator unique cho
    Power Pool eligibility (sub-project A) và phát hiện single-dataset alpha
    (sub-project D, operator inst_pnl/convert tính là dùng dataset pv1)."""

    def visit(self, node: Node) -> set[str]:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> set[str]:
        return set()

    def visit_field(self, node: Field) -> set[str]:
        return set()

    def visit_call(self, node: Call) -> set[str]:
        result: set[str] = {node.op}
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


class ComplexityVisitor(NodeVisitor[int]):
    """Số node toàn cây (leaf + Call) — proxy độ phức tạp cho GP anti-bloat penalty
    (Phase 7, FitnessVector.complexity_penalty)."""

    def visit(self, node: Node) -> int:
        return node.accept(self)

    def visit_constant(self, node: Constant) -> int:
        return 1

    def visit_field(self, node: Field) -> int:
        return 1

    def visit_call(self, node: Call) -> int:
        return 1 + sum(c.accept(self) for c in node.children())


def all_subtrees(node: Node) -> list[Node]:
    """Mọi sub-node của cây (gồm cả leaf và chính ``node``) — duyệt pre-order.

    Dùng cho điểm crossover/mutation của GP (Phase 7); tương đương
    ``ast_utils.all_subtrees`` cũ nhưng trên AST mới (Constant/Field/Call).
    """
    result: list[Node] = [node]
    for child in node.children():
        result.extend(all_subtrees(child))
    return result


def iter_leaves(node: Node) -> Iterator[Constant | Field]:
    """Duyệt mọi leaf (``Constant`` hoặc ``Field``) của cây, theo thứ tự trái-phải."""
    if isinstance(node, (Constant, Field)):
        yield node
    else:
        for child in node.children():
            yield from iter_leaves(child)
