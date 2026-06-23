"""Lọc syntax TRƯỚC khi simulate để khỏi phí quota."""

from __future__ import annotations

from src.lang.ast import Call, Constant, Field, Node
from src.lang.parser import ParseError, parse_expression
from src.lang.visitors import ComplexityVisitor, DepthVisitor

# 4 phép số học nhị phân: + - * / được parser map sang add/subtract/multiply/divide.
# Đây là nhóm operator hạ tầng, miễn kiểm operator-tồn-tại + arity (luôn hợp lệ vì
# do grammar tạo, không phải tên hàm do user gõ trong call(...)).
BINARY_OPS = {"add", "subtract", "multiply", "divide"}

DEFAULT_GROUPS = {"market", "sector", "industry", "subindustry", "country", "exchange"}

# Operator nhận số input bất kỳ (variadic) -> miễn kiểm "thừa input". Chữ ký chỉ
# ghi 2 tham số nhưng thực tế WQ cho truyền nhiều hơn (add(a,b,c,...)).
DEFAULT_VARIADIC_OPS = {"add", "multiply", "max", "min", "or", "and"}


class PreFilter:
    def __init__(
        self,
        known_operators: set[str] | None = None,
        known_fields: set[str] | None = None,
        known_groups: set[str] | None = None,
        max_depth: int = 7,
        max_nodes: int = 30,
        field_types: dict[str, str] | None = None,
        matrix_only_ops: set[str] | None = None,
        operator_arity: dict[str, int] | None = None,
        variadic_ops: set[str] | None = None,
    ):
        self.known_operators = known_operators
        self.known_fields = known_fields
        self.known_groups = known_groups or set(DEFAULT_GROUPS)
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        # Kiểm arity (tolerant): chỉ chặn khi số input THỪA so với chữ ký operator.
        # Chữ ký gồm cả tham số tùy chọn nên là arity TỐI ĐA -> vượt quá là chắc
        # sai (tái hiện lỗi WQ "Invalid number of inputs"). Thiếu thì bỏ qua (có
        # thể là tham số tùy chọn). Thiếu dữ liệu arity -> không kiểm.
        self.operator_arity = operator_arity
        self.variadic_ops = variadic_ops if variadic_ops is not None else set(DEFAULT_VARIADIC_OPS)
        # Kiểm tương thích kiểu: operator Time Series/Cross Sectional đòi input
        # MATRIX, không nhận field VECTOR trực tiếp (WQ trả status=ERROR). Cần
        # vec_*/group_* để rút VECTOR về MATRIX trước. Thiếu dữ liệu -> bỏ qua.
        self.field_types = field_types
        self.matrix_only_ops = matrix_only_ops or set()

    def check(self, expr: str) -> tuple[bool, str]:
        if expr.count("(") != expr.count(")"):
            return False, "Dấu ngoặc không cân bằng"

        try:
            tree = parse_expression(expr)
        except (ValueError, ParseError) as exc:
            return False, f"Parse lỗi: {exc}"

        if DepthVisitor().visit(tree) > self.max_depth:
            return False, f"Độ sâu > {self.max_depth}"
        if ComplexityVisitor().visit(tree) > self.max_nodes:
            return False, f"Số node > {self.max_nodes}"

        ok, reason = self._check_symbols(tree)
        if not ok:
            return False, reason

        return True, "ok"

    def _check_symbols(self, node: Node) -> tuple[bool, str]:
        if isinstance(node, Constant):
            return True, "ok"
        if isinstance(node, Field):
            name = node.name
            if self.known_fields is not None and name not in self.known_fields:
                if name not in self.known_groups:
                    return False, f"Field/hằng không tồn tại: {name}"
            return True, "ok"

        assert isinstance(node, Call)

        if node.op not in BINARY_OPS:
            if self.known_operators is not None and node.op not in self.known_operators:
                return False, f"Operator không tồn tại: {node.op}"

            # Tolerant arity: chặn khi THỪA input so với chữ ký, trừ operator variadic.
            if self.operator_arity and node.op not in self.variadic_ops:
                expected = self.operator_arity.get(node.op)
                if expected and len(node.args) > expected:
                    return False, (
                        f"Operator {node.op} nhận tối đa {expected} input, "
                        f"có {len(node.args)}"
                    )

        # Operator đòi MATRIX không được nhận field VECTOR làm đối số TRỰC TIẾP.
        if self.field_types and node.op in self.matrix_only_ops:
            for child in node.args:
                if isinstance(child, Field):
                    ftype = self.field_types.get(child.name)
                    if ftype == "VECTOR":
                        return False, (
                            f"Operator {node.op} đòi input MATRIX, không nhận field "
                            f"VECTOR trực tiếp: {child.name} (cần vec_avg/vec_sum trước)"
                        )

        for child in node.args:
            ok, reason = self._check_symbols(child)
            if not ok:
                return False, reason
        return True, "ok"
