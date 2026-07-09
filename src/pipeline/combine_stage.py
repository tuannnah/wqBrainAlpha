"""combine_stage — điều phối tầng pipeline cho combiner: chọn tổ hợp ít tương quan, dựng
biểu thức ghép, chấm local, chỉ giữ combo QUA GATE và VƯỢT tín hiệu con tốt nhất.

Tách khỏi `src/generation/combiner.py` (logic thuần chọn+dựng) để giữ combiner không phụ
thuộc backtest/scoring: combine_stage nhận `score_fn` injected (thường bọc
`runner._score_one_full` với base config + data), test được bằng scorer giả.

'Vượt tín hiệu con tốt nhất' đo CÔNG BẰNG: chấm cả combo lẫn từng tín hiệu con bằng CÙNG
`score_fn` (cùng base config) — tránh so combo (base cfg) với điểm tín hiệu con đã tune
riêng (khác cfg). Combo chỉ có nghĩa nếu tổ hợp mạnh hơn thành phần mạnh nhất."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from src.generation.combiner import (
    DEFAULT_MAX_COMBOS,
    DEFAULT_N_MAX,
    DEFAULT_N_MIN,
    DEFAULT_TAU,
    SubSignal,
    build_combined_expression,
    select_decorrelated_combos,
)
from src.lang.registry import OperatorRegistry
from src.pipeline.shortlist import ShortlistCandidate


class _Scored(Protocol):
    """Kết quả chấm local tối thiểu combine_stage cần (khớp runner._ScoreResult)."""

    @property
    def metrics(self) -> object: ...  # có .fitness
    @property
    def verdict(self) -> object: ...  # có .passed
    @property
    def pnl(self) -> object: ...
    @property
    def dates(self) -> object: ...


def combine_stage(
    signals: list[SubSignal],
    score_fn: Callable[[str], _Scored],
    *,
    tau: float = DEFAULT_TAU,
    n_min: int = DEFAULT_N_MIN,
    n_max: int = DEFAULT_N_MAX,
    max_combos: int = DEFAULT_MAX_COMBOS,
    max_depth: int | None = None,
    registry: OperatorRegistry | None = None,
) -> list[ShortlistCandidate]:
    """Chọn combo khử tương quan từ `signals`, dựng biểu thức ghép, chấm bằng `score_fn`,
    trả về ShortlistCandidate cho các combo QUA GATE và fitness > tín hiệu con tốt nhất
    (chấm cùng score_fn). Combo không dựng được (quá trần độ sâu) bị bỏ qua."""
    combos = select_decorrelated_combos(
        signals, tau=tau, n_min=n_min, n_max=n_max, max_combos=max_combos
    )
    depth_kw = {} if max_depth is None else {"max_depth": max_depth}
    out: list[ShortlistCandidate] = []
    for combo in combos:
        built = build_combined_expression(
            [s.expr for s in combo], registry=registry, **depth_kw
        )
        if built is None:
            continue
        scored = score_fn(built.expr)
        if not scored.verdict.passed:  # type: ignore[attr-defined]
            continue
        best_component = max(
            (score_fn(e).metrics.fitness for e in built.sub_exprs),  # type: ignore[attr-defined]
            default=float("-inf"),
        )
        if scored.metrics.fitness <= best_component:  # type: ignore[attr-defined]
            continue
        out.append(
            ShortlistCandidate(
                expr=built.expr,
                metrics=scored.metrics,  # type: ignore[arg-type]
                pnl=scored.pnl,  # type: ignore[arg-type]
                dates=scored.dates,  # type: ignore[arg-type]
            )
        )
    return out
