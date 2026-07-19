"""Stage combiner: ghép nhiều tín hiệu con ÍT TƯƠNG QUAN thành một alpha mạnh hơn.

Cơ sở lý thuyết (Grinold–Kahn, Luật cơ bản của quản lý danh mục): ghép N tín hiệu kỹ
năng ngang nhau và ÍT TƯƠNG QUAN thì Information Ratio (≈ Sharpe) của tổ hợp tăng ~√N.
Điều kiện sống còn là các tín hiệu con ít trùng nhau — nếu tương quan cao, lợi ích √N sụp
về 1. Vì vậy đòn bẩy thật nằm ở KHÂU CHỌN (greedy khử tương quan theo PnL local), không
phải ở cách cộng.

Hai trách nhiệm (logic thuần, tuân dependency-rule: KHÔNG import storage — nhận candidate
đã vật chất hóa qua tham số, giống pool_corr.py):

- `select_decorrelated_combos`: từ pool ứng viên, greedy chọn các tổ hợp 2..N_max tín hiệu
  mà |rho(PnL)| đôi một < tau.
- `build_combined_expression`: dựng FASTEXPR `add(rank(s1), rank(s2), ...)` (chuẩn hóa +
  trọng số đều), canh trần độ sâu WQ. Hướng A nướng group_neutralize sẵn trong tín hiệu
  con (setting combo = NONE); vượt trần -> hướng B tước group_neutralize, trung hòa ở tầng
  setting; vẫn quá sâu -> giảm N; không lọt -> trả None (KHÔNG nộp biểu thức sai).

Xem spec docs/superpowers/specs/2026-07-09-alpha-combiner-design.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.thresholds import COMBINER_MAX_COMPONENT_DEPTH, MAX_DEPTH
from src.backtest.pool_corr import pairwise_abs_rho
from src.lang.ast import Call
from src.lang.parser import ParseError, parse
from src.lang.registry import OperatorRegistry, default_registry
from src.lang.visitors import DepthVisitor, Serializer
from src.local_types import Dates

DEFAULT_TAU = 0.3       # ngưỡng |rho| PnL: cặp >= tau coi là trùng, không ghép chung
DEFAULT_N_MIN = 2       # combo phải >= 2 tín hiệu mới có nghĩa
DEFAULT_N_MAX = 4       # trần số tín hiệu / combo (giữ biểu thức không quá sâu)
DEFAULT_MAX_COMBOS = 5  # số combo tối đa mỗi run

# (T1.3) Operator gốc coi như ĐÃ chuẩn hóa cross-sectional/bounded [0,1] — _standardize bỏ
# qua bọc rank() cho các operator này (tiết kiệm đúng 1 tầng độ sâu khi build).
_ALREADY_STANDARDIZED_OPS = frozenset({"rank", "zscore", "ts_rank"})


@dataclass(frozen=True, slots=True)
class SubSignal:
    """Một tín hiệu con ứng viên để ghép: biểu thức + PnL local (đo tương quan) + điểm.

    `source` phân biệt tín hiệu từ run hiện tại ("run") hay kho alpha tốt trong DB ("db")
    — chỉ để chẩn đoán/log, không ảnh hưởng thuật toán."""

    expr: str
    pnl: npt.NDArray[np.float64]
    dates: Dates
    score: float
    source: str = "run"


@dataclass(frozen=True, slots=True)
class CombinedAlpha:
    """Kết quả dựng: biểu thức ghép RAW + các tín hiệu con thực sự dùng.

    Biểu thức đã TƯỚC group_neutralize khỏi mọi tín hiệu con (rank+add thô) để downstream
    `local_tuner` tự quét CHỌN MỘT neutralization tốt nhất cho cả combo — tránh trung hòa
    kép (Brain chỉ áp một neutralization/alpha). `sub_exprs` có thể ít hơn đầu vào nếu phải
    giảm N cho lọt trần độ sâu."""

    expr: str
    sub_exprs: tuple[str, ...]


# ------------------------- khâu chọn -------------------------

def select_decorrelated_combos(
    signals: list[SubSignal],
    *,
    tau: float = DEFAULT_TAU,
    n_min: int = DEFAULT_N_MIN,
    n_max: int = DEFAULT_N_MAX,
    max_combos: int = DEFAULT_MAX_COMBOS,
    max_component_depth: int = COMBINER_MAX_COMPONENT_DEPTH,
    registry: OperatorRegistry | None = None,
) -> list[list[SubSignal]]:
    """Greedy khử tương quan: mỗi combo bắt đầu từ tín hiệu ĐỨNG ĐẦU danh sách đã xếp còn
    lại (seed), lần lượt thêm tín hiệu tiếp theo CHỈ KHI |rho(PnL)| với MỌI thành viên đã
    chọn < tau, dừng ở n_max. Combo >= n_min mới xuất. Bỏ các tín hiệu đã dùng rồi lặp để
    tạo thêm combo (tối đa max_combos). Tương quan đo trên PnL local (không phải văn bản
    biểu thức) để chống 'đa dạng giả'. Không sửa đổi `signals` đầu vào.

    (T1.1) Xếp theo COMBINABILITY, không phải fitness thô: khóa lexicographic
    `(depth_bucket_asc, standardized_bucket_asc, score_desc)` — tín hiệu depth <=
    `max_component_depth` (bucket độ sâu 0, "đủ nông, còn cơ hội lọt trần sau khi
    `build_combined_expression` bọc rank+add") luôn đứng trước tín hiệu sâu hơn (bucket 1,
    "dễ chết trần"); CHỈ trong CÙNG bucket độ sâu mới xét tiếp. Trước đây
    `sorted(..., key=score, reverse=True)` chọn đúng các biểu thức GP sâu nhất làm seed
    (điểm cao nhưng chết trần MAX_DEPTH khi bọc rank+add) — nguyên nhân chính combiner ra
    ~0 combo (xem Bối cảnh task-1-brief.md).

    (T1.3) Trong CÙNG bucket độ sâu, tín hiệu gốc ĐÃ chuẩn hóa (rank/zscore/ts_rank —
    `_is_already_standardized`) được ưu tiên trước tín hiệu thô: `_standardize` bỏ qua bọc
    rank() cho các operator này nên chọn chúng làm thành viên combo tiết kiệm đúng 1 tầng
    độ sâu khi build, tăng tỉ lệ lọt trần. CHỈ khi cùng bucket độ sâu VÀ cùng trạng thái
    chuẩn hóa mới so fitness giảm dần."""
    reg = registry or default_registry()
    ranked = sorted(
        signals,
        key=lambda s: (
            0 if _depth_of(s.expr, reg) <= max_component_depth else 1,
            0 if _is_already_standardized(s.expr, reg) else 1,
            -s.score,
        ),
    )
    used: set[int] = set()  # id() các SubSignal đã dùng
    combos: list[list[SubSignal]] = []

    for seed in ranked:
        if len(combos) >= max_combos:
            break
        if id(seed) in used:
            continue
        combo = [seed]
        for cand in ranked:
            if len(combo) >= n_max:
                break
            if id(cand) in used or cand is seed:
                continue
            if _too_correlated(cand, combo, tau):
                continue
            combo.append(cand)
        if len(combo) >= n_min:
            combos.append(combo)
            used.update(id(s) for s in combo)
    return combos


def _too_correlated(cand: SubSignal, combo: list[SubSignal], tau: float) -> bool:
    """True nếu |rho(PnL)| của cand với BẤT KỲ thành viên combo >= tau. rho=None (thiếu
    overlap/phương sai 0) coi như KHÔNG trùng — thà giữ còn hơn loại oan."""
    for member in combo:
        rho = pairwise_abs_rho(cand.pnl, cand.dates, member.pnl, member.dates)
        if rho is not None and rho >= tau:
            return True
    return False


# ------------------------- khâu dựng biểu thức -------------------------

def build_combined_expression(
    exprs: list[str],
    *,
    max_depth: int = MAX_DEPTH,
    registry: OperatorRegistry | None = None,
) -> CombinedAlpha | None:
    """Dựng biểu thức ghép trọng số đều từ `exprs` (đã xếp theo điểm giảm dần).

    Tước group_neutralize ngoài cùng khỏi mỗi tín hiệu con (để tuner tự chọn 1 neutralization,
    tránh trung hòa kép), chuẩn hóa bằng rank rồi fold add cân bằng. Nếu vượt trần độ sâu ->
    bỏ tín hiệu điểm thấp nhất và thử lại, không xuống dưới 2. Không lọt trần -> None (KHÔNG
    nộp biểu thức sai)."""
    reg = registry or default_registry()
    ordered = list(exprs)
    while len(ordered) >= 2:
        stripped = [_strip_group_neutralize(e, reg) for e in ordered]
        combined = _fold_balanced_add([_standardize(e, reg) for e in stripped])
        if combined is not None and _depth_of(combined, reg) <= max_depth:
            return CombinedAlpha(combined, tuple(ordered))
        if len(ordered) == 2:
            return None
        ordered = ordered[:-1]  # bỏ tín hiệu điểm thấp nhất (cuối danh sách đã xếp)
    return None


def _is_already_standardized(expr: str, reg: OperatorRegistry) -> bool:
    """(T1.3) True nếu node GỐC của expr đã là rank/zscore/ts_rank (`_ALREADY_STANDARDIZED_OPS`)
    — coi như ĐÃ chuẩn hóa cross-sectional/bounded [0,1], không cần bọc thêm rank() ở
    `_standardize`. Dùng chung cho cả khâu dựng (tiết kiệm 1 tầng độ sâu) lẫn khóa sort
    combinability trong `select_decorrelated_combos` (ưu tiên chọn thành viên combo tiết
    kiệm được tầng này). Parse lỗi -> False (an toàn: vẫn bọc rank() như tín hiệu thô)."""
    try:
        node = parse(expr, registry=reg)
    except ParseError:
        return False
    return isinstance(node, Call) and node.op in _ALREADY_STANDARDIZED_OPS


def _standardize(expr: str, reg: OperatorRegistry) -> str:
    """Bọc rank() để đưa tín hiệu về cùng thang cross-sectional [0,1] (trọng số đều công
    bằng). Bỏ qua nếu gốc đã là rank/zscore/ts_rank (đã chuẩn hóa/bounded, T1.3 mở rộng
    thêm ts_rank so với bản gốc chỉ rank/zscore) — tiết kiệm ngân sách độ sâu."""
    if _is_already_standardized(expr, reg):
        return expr
    return f"rank({expr})"


def _fold_balanced_add(terms: list[str]) -> str | None:
    """Gộp các số hạng bằng cây add NHỊ PHÂN CÂN BẰNG (tối thiểu độ sâu). add trong registry
    local là nhị phân nên add(a,b,c) sẽ fail arity -> phải fold. Cân bằng: ghép cặp kề nhau
    theo tầng, N=4 -> add(add(a,b), add(c,d)) (2 tầng) thay vì trái-lệch (3 tầng)."""
    if not terms:
        return None
    level = list(terms)
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(f"add({level[i]}, {level[i + 1]})")
            else:
                nxt.append(level[i])  # số hạng lẻ dồn lên tầng sau
        level = nxt
    return level[0]


def _strip_group_neutralize(expr: str, reg: OperatorRegistry) -> str:
    """Nếu gốc biểu thức là group_neutralize(inner, <group>), trả inner (đã serialize) để
    tuner tự trung hòa ở tầng setting; ngược lại trả nguyên expr. Families luôn bọc
    group_neutralize ở NGOÀI CÙNG nên chỉ cần xét node gốc."""
    try:
        node = parse(expr, registry=reg)
    except ParseError:
        return expr
    if isinstance(node, Call) and node.op == "group_neutralize" and len(node.args) == 2:
        return Serializer().visit(node.args[0])
    return expr


def _depth_of(expr: str, reg: OperatorRegistry) -> int:
    return DepthVisitor().visit(parse(expr, registry=reg))


def component_depth_cap(n_max: int, *, max_depth: int = MAX_DEPTH) -> int:
    """(T1.2) Trần độ sâu MỘT tín hiệu con được phép có để combo N=n_max thành viên còn cơ
    hội lọt trần `max_depth` sau khi `build_combined_expression` bọc: 1 tầng `rank()` chuẩn
    hóa + cây `add` cân bằng tốn `ceil(log2(n_max))` tầng cho N lá. Thay hằng số cố định
    `COMBINER_MAX_COMPONENT_DEPTH` (luôn giả định N=4) bằng trần suy theo N THỰC TẾ combo
    dùng — N nhỏ hơn thì cây add nông hơn, trần được PHÉP nới ra tương ứng.

    N=4 -> ceil(log2(4))=2 tầng add + 1 rank = 3 -> cap = max_depth-3 (khớp hằng số cũ
    COMBINER_MAX_COMPONENT_DEPTH=4 khi max_depth=7). N=2 -> ceil(log2(2))=1 tầng add + 1
    rank = 2 -> cap nới ra max_depth-2=5. Dùng bởi `combine_stage` khi thử lại greedy với
    n_max nhỏ hơn (4 -> 3 -> 2) lúc N lớn không lọt trần (xem `_n_max_retry_sequence`)."""
    if n_max < 1:
        raise ValueError(f"n_max phải >= 1, nhận {n_max}")
    add_levels = math.ceil(math.log2(n_max)) if n_max > 1 else 0
    return max_depth - 1 - add_levels
