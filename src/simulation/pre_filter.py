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

# Operator mà Brain THẬT chấp nhận tham số TÙY CHỌN ở cuối (optional trailing arg) vượt
# quá độ dài `OperatorSpec.signature` cục bộ (registry cục bộ chỉ khai báo tham số bắt
# buộc, không mô hình hóa tham số tùy chọn có default trên Brain). Nếu KHÔNG loại các op
# này khỏi fallback local-arity, khi catalog Brain vắng mặt (operator_arity={}) — DB
# fresh/chưa cache — PreFilter sẽ dùng ĐÚNG local_arity làm cap và SIẾT OAN các lời gọi
# hợp lệ trên Brain (vd `rank(close, 2)` dùng optional `rate`, `ts_backfill(close, 22)`
# dùng optional `k`), khiến ý tưởng hợp lệ không bao giờ tới được Brain để simulate.
#
# Với các op trong tập này: BỎ QUA fallback local_arity hoàn toàn — cap chỉ tính từ
# `operator_arity` (catalog Brain). Catalog vắng mặt -> cap=0 (falsy) -> không kiểm arity
# cho op đó (an toàn hơn: bỏ sót lỗi arity rẻ hơn siết oan tín hiệu hợp lệ). Đây đúng là
# hành vi TRƯỚC task này (không regression) cho các op có mặt trong catalog.
#
# Danh sách tra cứu theo skill `worldquant-brain`
# (references/fastexpr-operators.md — rank/winsorize được liệt kê rõ có tham số optional;
# các op ts_* khác suy từ kiến thức operator FASTEXPR thật của Brain). `ts_delta` KHÔNG
# đưa vào đây: không có tham số tùy chọn nào được biết trên Brain, chữ ký cục bộ (x, d) đã
# khớp đúng arity thật -> fallback local vẫn an toàn cho op này.
OPTIONAL_TRAILING_ARG_OPS = {
    "rank",  # rank(x, rate=2) — rate tùy chọn
    "winsorize",  # winsorize(x, std=4)
    "ts_backfill",  # ts_backfill(x, d, k=1)
    "ts_rank",  # ts_rank(x, d, constant=0)
    "ts_zscore",  # ts_zscore(x, d, ...) — chưa chắc có tham số tùy chọn, thà miễn kiểm
    "quantile",  # quantile(x, driver="gaussian", sigma=1.0)
    "normalize",  # normalize(x, useStd=false, limit=0.0)
    "scale",  # scale(x, scale=1, longscale=1, shortscale=1)
}


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
        local_arity: dict[str, int] | None = None,
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
        # Nguồn arity BỔ SUNG lấy từ chữ ký OperatorRegistry cục bộ (local_arity =
        # {tên op -> len(signature)}). Lấp lỗ hổng: op vắng mặt trong catalog Brain (hoặc
        # definition không parse được -> count_max_arity trả 0) trước đây bị BỎ QUA kiểm
        # arity hoàn toàn vì `operator_arity.get(op)` falsy. Cap hiệu lực = max(catalog,
        # local) — catalog đã tính cả tham số tùy chọn (rank/winsorize/ts_backfill...) nên
        # lấy max không siết oan các op đó khi catalog có mặt và lớn hơn.
        self.local_arity = local_arity
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
            # Cap hiệu lực = max(catalog_arity, local_arity) -> op vắng mặt/arity=0 ở MỘT
            # nguồn vẫn được nguồn kia lấp (đóng lỗ hổng "Invalid number of inputs" khi
            # catalog thiếu entry cho op). NGOẠI LỆ: op thuộc OPTIONAL_TRAILING_ARG_OPS
            # (Brain chấp nhận tham số tùy chọn cuối mà chữ ký cục bộ không mô hình hóa)
            # -> KHÔNG áp fallback local_arity, chỉ tin catalog (như hành vi trước khi có
            # local_arity) để tránh siết oan (vd rank(close, 2), ts_backfill(close, 22)).
            if node.op not in self.variadic_ops:
                catalog_cap = (self.operator_arity or {}).get(node.op, 0)
                if node.op in OPTIONAL_TRAILING_ARG_OPS:
                    cap = catalog_cap
                else:
                    local_cap = (self.local_arity or {}).get(node.op, 0)
                    cap = max(catalog_cap, local_cap)
                # TODO(min-arity): chỉ kiểm THỪA (len > cap), chưa kiểm THIẾU input so với
                # arity tối thiểu của operator (vd gọi thiếu tham số bắt buộc) — số tham số
                # bắt buộc tối thiểu không có sẵn từ catalog/local hiện tại nên bỏ qua kiểm.
                if cap and len(node.args) > cap:
                    return False, (
                        f"Operator {node.op} nhận tối đa {cap} input, "
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
