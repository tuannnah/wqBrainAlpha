"""Bộ lọc CẤU TRÚC (structural): từ chối biểu thức AST vô nghĩa kinh tế/degenerate TRƯỚC khi
chấm điểm hay đốt sim Brain.

Khác với gate local (Sharpe/turnover/fitness...) vốn dựa trên ĐIỂM SỐ backtest, bộ lọc này
bắt các mẫu CẤU TRÚC luôn luôn vô nghĩa BẤT KỂ điểm số cục bộ — vì hệ số tương quan
local↔Brain (ρ) thấp nên rác vẫn có thể qua gate local do may rủi rồi tốn sim Brain thật.
Ba nhóm quy tắc (xem `check_meaningful`):

1. No-op/degenerate nhị phân: `min(x,x)`/`max(x,x)`/`subtract(x,x)`/`divide(x,x)` và
   `add(x, -x)` — hai nhánh GIỐNG HỆT CẤU TRÚC (so bằng `Serializer`, KHÔNG bóc scale) nên
   kết quả suy biến (hằng số/0/1). Cố ý KHÔNG dùng `CanonicalHasher` ở đây: hasher đó bóc
   scale dương Ở GỐC để dedup toàn-alpha (multiply(2,X)≡X), nhưng áp cho từng NHÁNH con của
   min/max/subtract sẽ coi `close` và `multiply(2, close)` là "giống nhau" -> chặn oan
   `subtract(close, multiply(2, close))` (= -close, tín hiệu hợp lệ). Xem `_check_noop`.
2. Toán học domain-invalid: `log`/`sqrt` áp lên input CÓ THỂ ÂM (field `returns`, hoặc kết
   quả của các operator hay ra giá trị âm/tương quan) -> NaN-heavy hoặc vô nghĩa kinh tế;
   `power(x, mũ ÂM)` -> nổ giá trị khi cơ số gần 0.
3. Tự lồng dư thừa: cùng một ts-operator lồng trực tiếp vào chính nó quá `MAX_SAME_TS_NEST`
   tầng (vd `ts_std_dev(ts_std_dev(ts_std_dev(x, d), d), d)`).
4. (Task 4) `power(sign(x), k)` với `k` số mũ CHẴN: `sign(x)` ∈ {-1,0,1} nên nâng lũy thừa
   chẵn luôn suy biến thành hằng số ∈ {0,1} — bằng chứng thật (log 07-12): sim Brain ra
   Sharpe 0.00/turnover 0.00 (đốt 13 phút sim cho một vị thế hằng số). Xem `_check_constant_power`.
5. (Task 4) Biểu thức mà TOÀN BỘ tập field tham chiếu ⊆ nhóm "chỉ khối lượng" (không field
   hướng giá) — vô nghĩa kinh tế vì không mang thông tin hướng giá nào (chỉ đếm/đo khối
   lượng giao dịch). Bằng chứng thật: sim Brain ra Sharpe -0.13. Áp trên TOÀN CÂY (không phải
   từng subtree — một biểu thức con dùng riêng `volume` vẫn ổn nếu biểu thức TỔNG kết hợp với
   field giá khác). Xem `_check_no_price_direction`.

Nguyên tắc CONSERVATIVE: từ chối oan một biểu thức có thể tốt còn tốn kém hơn để lọt một
biểu thức rác (biểu thức rác dù sao cũng bị gate/backtest chặn ở bước sau nếu KHÔNG khớp quy
tắc rõ ràng ở đây) — nên chỉ liệt kê các mẫu THẬT SỰ rõ ràng là suy biến/vô nghĩa.
"""

from __future__ import annotations

from src.lang.ast import Call, Constant, Field, Node
from src.lang.registry import OpCategory, OperatorRegistry, default_registry
from src.lang.visitors import FieldCollector, Serializer, all_subtrees

# Ngưỡng tự lồng cùng một ts-operator: cho phép tối đa 2 tầng (vd
# ts_std_dev(ts_std_dev(x, d), d) vẫn hợp lệ); từ 3 tầng trở lên (>MAX_SAME_TS_NEST) coi là
# dư thừa/vô nghĩa (làm mượt một tín hiệu đã được làm mượt nhiều lần không thêm thông tin gì).
MAX_SAME_TS_NEST = 2

# log/sqrt trên input CÓ THỂ ÂM -> NaN (log) hoặc NaN (sqrt của số âm) -- vô nghĩa kinh tế.
_LOG_SQRT_OPS = frozenset({"log", "sqrt"})

# Field nào tự thân có thể âm (returns hàng ngày âm khi giá giảm).
_MAYBE_NEGATIVE_FIELDS = frozenset({"returns"})

# Operator nào cho output thường xuyên ÂM (hệ số tương quan, hiệu số, phần dư neutralize).
_MAYBE_NEGATIVE_CALL_OPS = frozenset(
    {"ts_corr", "correlation", "subtract", "vector_neut", "regression_neut"}
)

# Toán tử nhị phân coi là no-op khi 2 tham số GIỐNG HỆT CẤU TRÚC (so bằng Serializer).
_NOOP_SAME_ARG_OPS = frozenset({"min", "max", "subtract", "divide"})

# Field CHỈ đo khối lượng giao dịch, KHÔNG mang thông tin hướng giá (Task 4). Danh sách rút
# từ các seed/family hiện có trong repo (src/generation/families.py, alt_data_seeds.py) —
# chỉ liệt kê field THẬT SỰ vô hướng, tránh nhận nhầm field khác (vd "cap"/market-cap dù
# không phải giá OHLC vẫn phản ánh gián tiếp mức định giá, KHÔNG đưa vào đây để bảo thủ).
_VOLUME_ONLY_FIELDS = frozenset({"volume", "adv20", "sharesout"})

# Field mang thông tin hướng giá (kể cả returns, phái sinh từ close). Biểu thức CHỈ cần dùng
# 1 trong các field này (kết hợp với volume hay không) là đã CÓ hướng giá -> không bị chặn.
_PRICE_DIRECTION_FIELDS = frozenset({"close", "open", "high", "low", "vwap", "returns"})


def check_meaningful(
    node: Node, registry: OperatorRegistry | None = None,
) -> tuple[bool, str]:
    """Duyệt TOÀN BỘ cây (preorder, cha trước con -- xem `all_subtrees`), trả
    `(False, lý_do)` ngay tại node vi phạm ĐẦU TIÊN gặp (nếu một chuỗi tự lồng vi phạm, node
    cao nhất trong chuỗi được báo cáo, không cần dò tiếp xuống con). Trả `(True, "")` nếu
    không node nào trong cây vi phạm quy tắc nào.

    `_check_no_price_direction` chạy MỘT LẦN trên TOÀN CÂY (không phải per-subtree như các
    rule khác) -- rule này xét tập field của CẢ biểu thức, không phải từng node con (xem
    docstring module, nhóm quy tắc 5)."""
    reg = registry if registry is not None else default_registry()
    no_direction_reason = _check_no_price_direction(node, reg)
    if no_direction_reason is not None:
        return False, no_direction_reason
    serializer = Serializer()
    for sub in all_subtrees(node):
        if not isinstance(sub, Call):
            continue
        ok, reason = _check_call(sub, serializer, reg)
        if not ok:
            return False, reason
    return True, ""


def _check_call(
    node: Call, serializer: Serializer, registry: OperatorRegistry,
) -> tuple[bool, str]:
    """Áp lần lượt các nhóm quy tắc PER-NODE lên một node `Call`; dừng ở vi phạm đầu tiên tìm
    được. (`_check_no_price_direction` KHÔNG ở đây -- rule đó chạy một lần trên toàn cây,
    xem `check_meaningful`.)"""
    reason = (
        _check_noop(node, serializer)
        or _check_domain_invalid(node)
        or _check_self_nest(node, registry)
        or _check_constant_power(node)
    )
    if reason is not None:
        return False, reason
    return True, ""


def _check_noop(node: Call, serializer: Serializer) -> str | None:
    """Nhị phân suy biến: 2 nhánh GIỐNG HỆT VỀ CẤU TRÚC (min/max/subtract/divide(x,x)), hoặc
    add(x, -x)/add(-x, x) (một nhánh giống hệt `multiply(-1, nhánh_kia)`).

    QUAN TRỌNG: so sánh bằng `Serializer` (chuỗi FASTEXPR, KHÔNG bóc scale) chứ KHÔNG dùng
    `CanonicalHasher` -- hasher đó bóc scale DƯƠNG Ở GỐC (`_fold_positive_scale_at_root`)
    cho mục đích dedup toàn-alpha (multiply(2,X)≡X vì rank không đổi thứ hạng). Áp fold đó
    cho TỪNG NHÁNH con ở đây là SAI: nó khiến `close` và `multiply(2, close)` bị coi là
    "giống nhau", nên `subtract(close, multiply(2, close))` (= -close, tín hiệu HỢP LỆ) hay
    `min(close, multiply(2, close))` bị chặn oan -- đúng bug mà spec yêu cầu sửa. Serializer
    không fold gì cả nên chỉ 2 nhánh THẬT SỰ giống hệt cấu trúc (vd `close` với `close`) mới
    bị coi là no-op; `divide(x, multiply(k,x))` tuy về mặt toán là hằng số nhưng bị BỎ QUA
    (chấp nhận under-reject, ưu tiên không từ chối oan tín hiệu hợp lệ)."""
    if node.op in _NOOP_SAME_ARG_OPS and len(node.args) == 2:
        a, b = node.args
        if serializer.visit(a) == serializer.visit(b):
            return f"no-op: {node.op}(x, x) — hai nhánh giống hệt nhau"
    if node.op == "add" and len(node.args) == 2:
        a, b = node.args
        neg_b = Call("multiply", (Constant(-1.0), b))
        if serializer.visit(a) == serializer.visit(neg_b):
            return "no-op: add(x, -x) xấp xỉ hằng số 0"
        neg_a = Call("multiply", (Constant(-1.0), a))
        if serializer.visit(b) == serializer.visit(neg_a):
            return "no-op: add(-x, x) xấp xỉ hằng số 0"
    return None


def _check_domain_invalid(node: Call) -> str | None:
    """log/sqrt trên input có thể âm, hoặc power với số mũ âm (nổ giá trị gần 0)."""
    if node.op in _LOG_SQRT_OPS and len(node.args) == 1:
        x = node.args[0]
        if isinstance(x, Field) and x.name in _MAYBE_NEGATIVE_FIELDS:
            return f"{node.op}({x.name}) — input có thể ÂM -> NaN/vô nghĩa"
        if isinstance(x, Call) and x.op in _MAYBE_NEGATIVE_CALL_OPS:
            return f"{node.op}({x.op}(...)) — input có thể ÂM -> NaN/vô nghĩa"
    if node.op == "power" and len(node.args) == 2:
        exp = node.args[1]
        if isinstance(exp, Constant) and exp.value < 0:
            return f"power(x, {exp.value}) — số mũ ÂM có thể nổ giá trị khi cơ số gần 0"
    return None


def _check_self_nest(node: Call, registry: OperatorRegistry) -> str | None:
    """Cùng một ts-operator tự lồng vào chính nó quá `MAX_SAME_TS_NEST` tầng. Chỉ áp cho
    operator category TIME_SERIES (registry) -- op không tồn tại trong registry (vd hằng
    số/test dựng tay) hoặc không phải TIME_SERIES thì bỏ qua quy tắc này."""
    try:
        spec = registry.get(node.op)
    except KeyError:
        return None
    if spec.category is not OpCategory.TIME_SERIES:
        return None
    depth = _self_nest_depth(node, node.op)
    if depth > MAX_SAME_TS_NEST:
        return f"{node.op} tự lồng {depth} tầng (> {MAX_SAME_TS_NEST}) — dư thừa/vô nghĩa"
    return None


def _self_nest_depth(node: Node, op: str) -> int:
    """Độ sâu chuỗi tự lồng cùng `op` bắt đầu TẠI `node` (0 nếu `node` không phải Call cùng
    `op`; ngược lại 1 + độ sâu lớn nhất trong các con)."""
    if not isinstance(node, Call) or node.op != op:
        return 0
    child_depths = [_self_nest_depth(c, op) for c in node.children()]
    return 1 + (max(child_depths) if child_depths else 0)


def _check_constant_power(node: Call) -> str | None:
    """`power(sign(x), k)` với `k` nguyên CHẴN -- `sign(x)` chỉ nhận {-1,0,1} nên lũy thừa
    chẵn của nó luôn suy biến thành hằng số ∈ {0,1} (mất hết thông tin hướng). Số mũ LẺ được
    BỎ QUA (sign(x)**lẻ == sign(x), không suy biến); base KHÔNG phải `sign(...)` cũng bỏ qua
    (power(close, 2) hợp lệ -- xem `test_accepts_power_with_positive_exponent`)."""
    if node.op != "power" or len(node.args) != 2:
        return None
    base, exp = node.args
    if not (isinstance(base, Call) and base.op == "sign"):
        return None
    if not isinstance(exp, Constant):
        return None
    k = exp.value
    if k != int(k) or int(k) % 2 != 0:
        return None
    return f"power(sign(x), {exp.value}) — số mũ CHẴN của sign(x) luôn suy biến thành hằng số"


def _check_no_price_direction(node: Node, registry: OperatorRegistry) -> str | None:
    """Tập field tham chiếu trong TOÀN BỘ biểu thức ⊆ nhóm chỉ-khối-lượng
    (`_VOLUME_ONLY_FIELDS`) -- không field nào mang hướng giá -> vô nghĩa kinh tế (chỉ đếm
    khối lượng giao dịch, không biết giá tăng hay giảm). Biểu thức không tham chiếu field nào
    (toàn hằng số) không khớp rule này (tập rỗng luôn là subset -- nhưng khi đó không có gì để
    "chặn oan", cố tình loại trừ để tránh false-positive vô nghĩa trên biểu thức hằng đơn thuần)."""
    fields = FieldCollector(registry).visit(node)
    if fields and fields.issubset(_VOLUME_ONLY_FIELDS):
        return f"chỉ dùng field khối lượng {sorted(fields)} — không có hướng giá"
    return None
