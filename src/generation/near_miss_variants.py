"""Sinh biến thể CÓ KIỂM SOÁT quanh near-miss (sim thật Sharpe dương nhưng chưa đạt).

Bằng chứng (log 2026-07-16): sau khi mọi core alt-data vào avoid-list (389 lần "bão hoà"
trong 2 vòng menu-5), vòng kín rơi về GP nhiễu (best Sharpe 0.68 suốt ~6h) trong khi các
near-miss 0.8-0.9 (broker_dealer_vol_imbalance...) chưa từng được thử biến thể window/wrapper
— đúng loại đòn bẩy đã đưa KP92dQAx (0.89 -> 1.71 nhờ đổi cấu trúc) qua ngưỡng Power Pool.

Nguyên tắc (skill worldquant-brain):
- Biến thể là EXPRESSION SEARCH quanh core đã có tín hiệu — config (neutralization/decay)
  giữ nguyên ở refiner, không trộn 2 tầng.
- Chỉ phép biến đổi bảo toàn ngữ nghĩa kinh tế: bọc rank (robust cross-section), nâng window
  một bậc (giảm nhiễu/turnover), bọc ts_rank bounded (ổn định regime).
- Giữ cấu trúc Power Pool (≤8 operator, ≤3 field) và depth ≤ 7 — biến thể vi phạm bị loại.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from src.backtest.local_tuner import iter_constants, set_constant
from src.lang.ast import Call, Node
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import FieldCollector, Serializer
from src.pipeline.shortlist import ShortlistCandidate

# Bậc thang window (khớp triết lý _WINDOW_LADDER của LocalTuner) — chỉ nâng LÊN một bậc:
# window dài hơn = mượt hơn = turnover thấp hơn (hướng có lợi cho fitness/Power Pool).
_WINDOW_LADDER = (3, 5, 10, 22, 66, 120)
_MAX_DEPTH = 7
_MAX_PP_OPERATORS = 8


def _depth(node: Node) -> int:
    if isinstance(node, Call):
        return 1 + max((_depth(c) for c in node.args), default=0)
    return 1


def _next_window_up(value: float) -> float | None:
    cur = int(round(value))
    for step in _WINDOW_LADDER:
        if step > cur:
            return float(step)
    return None


def generate_variants(expr: str, max_variants: int = 4) -> list[str]:
    """Biến thể có kiểm soát của `expr` (không gồm chính nó, không trùng lặp, tối đa
    `max_variants`): (1) bọc rank nếu gốc chưa rank; (2) nâng từng WINDOW lên một bậc thang;
    (3) bọc ts_rank(x, 66) bounded. Biểu thức parse lỗi -> [] (không ném — nguồn ý tưởng
    không được làm sập vòng kín). Biến thể vượt depth 7 hoặc >8 operator (Power Pool) bị loại."""
    registry = default_registry()
    try:
        node = parse(expr, registry)  # strict: operator lạ/arity sai -> loại ngay
    except Exception:  # noqa: BLE001 - biểu thức DB cũ có thể chứa operator đã gỡ
        return []

    ser = Serializer()
    root_op = node.op if isinstance(node, Call) else None
    candidates: list[Node] = []

    if root_op not in ("rank", "ts_rank"):
        candidates.append(Call("rank", (node,)))

    for path, value, is_window in iter_constants(node, registry):
        if not is_window:
            continue
        nxt = _next_window_up(value)
        if nxt is not None:
            candidates.append(set_constant(node, path, nxt))

    if root_op != "ts_rank":
        from src.lang.ast import Constant

        candidates.append(Call("ts_rank", (node, Constant(66.0))))

    out: list[str] = []
    seen: set[str] = {expr}
    for cand in candidates:
        if len(out) >= max_variants:
            break
        if _depth(cand) > _MAX_DEPTH:
            continue
        try:
            from src.scoring.power_pool import count_operators_fields

            text = ser.visit(cand)
            n_ops, _n_fields = count_operators_fields(text)
            if n_ops > _MAX_PP_OPERATORS:
                continue
        except Exception:  # noqa: BLE001 - đếm lỗi -> bỏ biến thể, không chặn cả batch
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


_MAX_PP_FIELDS = 3


def combine_same_dataset(
    rows: "list[tuple[str, float]]", dataset_of_fields_fn, max_combos: int = 3,
) -> list[str]:
    """Ghép CẶP biểu thức near-miss mà TOÀN BỘ field của cả hai thuộc CÙNG MỘT dataset —
    template thắng thật KP9Aw3lj 2026-07-16 (0.89 -> 1.03): `rank(add(a, b))` (mỗi vế đã
    đúng dấu vì near-miss có Sharpe dương; rank bọc ngoài cho robust cross-section — cùng
    thứ hạng weights với multiply(-1, rank(add(-a, -b))) sau demean, không cần bóc dấu).

    Chỉ ghép khi: (1) dataset_of_fields_fn phủ ĐỦ mọi field cả hai vế và trả về đúng 1
    dataset chung (field thiếu map -> không dám đoán, bỏ); (2) union field mang thông tin
    MỚI (lớn hơn từng vế — không blend 2 công thức trên cùng tập field); (3) giữ chuẩn
    Power Pool: union <=3 field, <=8 operator, depth <=7. Cặp duyệt theo thứ tự đầu vào
    (repo đã sort Sharpe giảm dần -> cặp mạnh nhất trước), cắt `max_combos`."""
    registry = default_registry()
    ser = Serializer()
    parsed: list[tuple[Node, frozenset[str], str]] = []  # (node, fields, dataset)
    for expr, _sharpe in rows:
        try:
            node = parse(expr, registry)
            fields = frozenset(FieldCollector(registry).visit(node))
        except Exception:  # noqa: BLE001 - biểu thức DB cũ có thể chứa operator đã gỡ
            continue
        if not fields:
            continue
        ds_map = dataset_of_fields_fn(set(fields)) or {}
        datasets = {ds_map.get(f) for f in fields}
        if None in datasets or len(datasets) != 1:
            continue  # field thiếu map / lẫn nhiều dataset -> không dám ghép
        parsed.append((node, fields, next(iter(datasets))))

    out: list[str] = []
    seen: set[str] = set()
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            if len(out) >= max_combos:
                return out
            node_a, fields_a, ds_a = parsed[i]
            node_b, fields_b, ds_b = parsed[j]
            if ds_a != ds_b:
                continue
            union = fields_a | fields_b
            if len(union) > _MAX_PP_FIELDS:
                continue
            if union == fields_a or union == fields_b:
                continue  # không thêm field mới -> chỉ là blend công thức, bỏ
            combo = Call("rank", (Call("add", (node_a, node_b)),))
            if _depth(combo) > _MAX_DEPTH:
                continue
            text = ser.visit(combo)
            try:
                from src.scoring.power_pool import count_operators_fields

                n_ops, _ = count_operators_fields(text)
                if n_ops > _MAX_PP_OPERATORS:
                    continue
            except Exception:  # noqa: BLE001
                continue
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


class NearMissVariantSource:
    """Nguồn ý tưởng cho ClosedLoop: biến thể quanh near-miss sim thật (Sharpe trong
    [min_sharpe, max_sharpe)) từ repo. Phục vụ MỘT lần (như AltDataIdeaSource) rồi ủy quyền
    fallback. origin='alt_data' để đi đường _sim_direct (field ngoài panel local — sim thẳng
    Brain); nguồn gốc thật ghi trong docstring/log, không tách nhãn riêng để khỏi đổi routing."""

    def __init__(
        self, *, repo, fallback, dedup_key_fn=None, avoided_hashes=None,
        min_sharpe: float = 0.6, max_sharpe: float = 1.0,
        top_k: int = 5, max_variants_per_expr: int = 3,
        dataset_of_fields_fn=None,
    ) -> None:
        self._repo = repo
        self._fallback = fallback
        self._dedup_key_fn = dedup_key_fn
        self._avoided_hashes = avoided_hashes
        self._min_sharpe = min_sharpe
        self._max_sharpe = max_sharpe
        self._top_k = top_k
        self._max_variants = max_variants_per_expr
        # Map {field -> dataset_id} cho tổ hợp cùng-dataset (bài học KP9Aw3lj: combo 2 field
        # cùng dataset thắng 1.03 vs biến thể đơn lẻ <=0.9). None -> tắt combo (tương thích cũ).
        self._dataset_of_fields_fn = dataset_of_fields_fn
        self._served = False

    def set_saturated_families(self, fams) -> None:
        if hasattr(self._fallback, "set_saturated_families"):
            self._fallback.set_saturated_families(fams)

    def next_batch(self):
        if self._served:
            return self._fallback.next_batch()
        self._served = True

        near_miss_fn = getattr(self._repo, "near_miss_exprs", None)
        rows = near_miss_fn(self._min_sharpe, self._max_sharpe, self._top_k) if callable(near_miss_fn) else []
        empty = np.zeros(0, dtype=np.float64)
        dates = np.zeros(0, dtype="datetime64[ns]")

        def _blocked(expr: str) -> bool:
            return (
                self._dedup_key_fn is not None
                and self._avoided_hashes is not None
                and self._dedup_key_fn(expr) in self._avoided_hashes
            )

        # Combo cùng-dataset đứng TRƯỚC biến thể đơn lẻ (bằng chứng KP9Aw3lj 2026-07-16:
        # combo thắng 1.03 vs biến thể đơn lẻ <=0.9).
        exprs: list[str] = []
        if self._dataset_of_fields_fn is not None and rows:
            combos = combine_same_dataset(rows, self._dataset_of_fields_fn)
            if combos:
                logger.info("NearMissVariant: {} combo cùng-dataset từ {} near-miss.", len(combos), len(rows))
            exprs.extend(combos)
        for expr, sharpe in rows:
            variants = generate_variants(expr, max_variants=self._max_variants)
            if variants:
                logger.info(
                    "NearMissVariant: sinh {} biến thể quanh near-miss Sharpe={:.2f}: {!r}",
                    len(variants), sharpe, expr if len(expr) <= 60 else expr[:57] + "...",
                )
            exprs.extend(variants)

        seen: set[str] = set()
        out: list[ShortlistCandidate] = []
        for expr in exprs:
            if expr in seen or _blocked(expr):
                continue
            seen.add(expr)
            out.append(
                ShortlistCandidate(
                    expr=expr, metrics=None, pnl=empty, dates=dates, origin="alt_data",
                )
            )
        if out:
            return out
        return self._fallback.next_batch()
