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
from src.lang.visitors import Serializer
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


class NearMissVariantSource:
    """Nguồn ý tưởng cho ClosedLoop: biến thể quanh near-miss sim thật (Sharpe trong
    [min_sharpe, max_sharpe)) từ repo. Phục vụ MỘT lần (như AltDataIdeaSource) rồi ủy quyền
    fallback. origin='alt_data' để đi đường _sim_direct (field ngoài panel local — sim thẳng
    Brain); nguồn gốc thật ghi trong docstring/log, không tách nhãn riêng để khỏi đổi routing."""

    def __init__(
        self, *, repo, fallback, dedup_key_fn=None, avoided_hashes=None,
        min_sharpe: float = 0.6, max_sharpe: float = 1.0,
        top_k: int = 5, max_variants_per_expr: int = 3,
    ) -> None:
        self._repo = repo
        self._fallback = fallback
        self._dedup_key_fn = dedup_key_fn
        self._avoided_hashes = avoided_hashes
        self._min_sharpe = min_sharpe
        self._max_sharpe = max_sharpe
        self._top_k = top_k
        self._max_variants = max_variants_per_expr
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
        out: list[ShortlistCandidate] = []
        for expr, sharpe in rows:
            for variant in generate_variants(expr, max_variants=self._max_variants):
                if (
                    self._dedup_key_fn is not None
                    and self._avoided_hashes is not None
                    and self._dedup_key_fn(variant) in self._avoided_hashes
                ):
                    continue
                out.append(
                    ShortlistCandidate(
                        expr=variant, metrics=None, pnl=empty, dates=dates, origin="alt_data",
                    )
                )
            if out:
                logger.info(
                    "NearMissVariant: sinh {} biến thể quanh near-miss Sharpe={:.2f}: {!r}",
                    len(out), sharpe, expr if len(expr) <= 60 else expr[:57] + "...",
                )
        if out:
            return out
        return self._fallback.next_batch()
