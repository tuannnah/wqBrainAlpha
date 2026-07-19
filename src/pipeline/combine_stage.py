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

from config.thresholds import SUBMIT_FITNESS_REF, SUBMIT_SHARPE_REF
from src.generation.combiner import (
    DEFAULT_MAX_COMBOS,
    DEFAULT_N_MAX,
    DEFAULT_N_MIN,
    DEFAULT_TAU,
    SubSignal,
    _depth_of,  # tái dùng đúng phép đo depth combiner dùng nội bộ (DepthVisitor)
    build_combined_expression,
    component_depth_cap,  # T1.2: trần component ĐỘNG theo N thay hằng số cố định
    select_decorrelated_combos,
)
from src.lang.registry import OperatorRegistry, default_registry
from src.pipeline.shortlist import ShortlistCandidate


class _Scored(Protocol):
    """Kết quả chấm local tối thiểu combine_stage cần (khớp runner._ScoreResult)."""

    @property
    def metrics(self) -> object: ...  # có .fitness và .sharpe (Fix 4: điểm-nộp)
    @property
    def verdict(self) -> object: ...  # có .passed
    @property
    def pnl(self) -> object: ...
    @property
    def dates(self) -> object: ...


def _submit_score(metrics: object) -> float:
    """Điểm-nộp (Fix 4, Task 2): min(sharpe/SUBMIT_SHARPE_REF, fitness/SUBMIT_FITNESS_REF) —
    đo combo tiến GẦN NGƯỠNG NỘP thật (Sharpe~1.58, fitness~1) tới đâu, thay vì so fitness thô
    (`fitness <= best_component` cũ): fitness thô có thể tăng dù sharpe tệ đi (vd combo tăng
    turnover/giảm tập trung mà không tăng risk-adjusted return) -- điểm-nộp buộc combo phải
    tiến bộ trên CẢ HAI trục mới được coi là 'vượt trội' đáng giữ."""
    return min(metrics.sharpe / SUBMIT_SHARPE_REF, metrics.fitness / SUBMIT_FITNESS_REF)  # type: ignore[attr-defined]


def _bump(drop_stats: dict[str, int] | None, key: str) -> None:
    if drop_stats is not None:
        drop_stats[key] = drop_stats.get(key, 0) + 1


def _n_max_retry_sequence(n_max: int, n_min: int) -> list[int]:
    """(T1.2) Dãy N thử giảm dần khi combo N=n_max không lọt trần: n_max, n_max-1, ..., n_min
    — thử lại TOÀN BỘ greedy (`select_decorrelated_combos`) với N nhỏ hơn NGAY TỪ ĐẦU, chứ
    không chỉ bỏ tín hiệu điểm thấp nhất như vòng lặp nội bộ cũ của `build_combined_expression`
    (vòng đó vẫn giữ nguyên làm phòng vệ thứ hai, không xung đột). N=4 (mặc định) -> [4, 3, 2]
    đúng thứ tự nêu trong task-1-brief.md."""
    if n_max <= n_min:
        return [n_max]
    return list(range(n_max, n_min - 1, -1))


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
    score_fn_factory: Callable[[list[SubSignal]], Callable[[str], _Scored]] | None = None,
    drop_stats: dict[str, int] | None = None,
) -> list[ShortlistCandidate]:
    """Chọn combo khử tương quan từ `signals`, dựng biểu thức ghép, chấm, trả về
    ShortlistCandidate cho các combo QUA GATE và ĐIỂM-NỘP vượt tín hiệu con tốt nhất (chấm
    cùng scorer). Combo không dựng được (quá trần độ sâu) bị bỏ qua.

    `score_fn_factory` (Fix 2, Task 2 — thay pool `repo.load_pool()` 1321+ eval LOCAL bão
    hòa đã giết oan combo self-corr 0.70-0.86 trong khi Brain thật đo 0.40-0.46, xem
    `logs/diag_combiner_20260712.md`): nếu có, ƯU TIÊN dùng — gọi lại cho MỖI combo với
    danh sách `signals` NGOÀI combo đó (loại chính thành phần của combo cũng khử luôn
    tự-so) để dựng scorer chấm gate bằng pool = PnL local của CHÍNH các tín hiệu Brain-
    proven, không phải toàn bộ pool tích luỹ. Không có factory -> giữ `score_fn` cũ (tương
    thích ngược cho test/call site hiện hữu).
    Chốt ngữ nghĩa pool (deviation plan Fix 2/Task 2, review final): "ngoài combo" nghĩa là
    MỌI signal trong `signals` không thuộc combo hiện tại (khớp test spec {A,B,C}→combo(A,B)
    thì pool={C}), KHÔNG PHẢI thu hẹp về "chỉ signal Brain-proven ngoài combo" — mọi phần tử
    của `signals` được coi là ứng viên pool hợp lệ, bất kể `source` ("run" hay "db").
    Loại "ngoài combo" tính theo CHUỖI expr (Finding #3 review, không phải id() Python): 2
    SubSignal khác object nhưng CÙNG expr (vd 1 bản "run" + 1 bản "db" trùng nhau) đều bị loại
    khỏi pool nếu expr đó có mặt trong combo — id() để lọt bản sao vào pool sẽ tạo tự-so
    (|rho|≈1 với chính combo) khiến gate giết oan combo.

    `drop_stats` (Fix 4, Task 2 — instrumentation): mutate in-place, đếm tại 4 điểm rớt —
    "greedy_empty" (0 combo thô sau greedy, thường vì < n_min tín hiệu đủ shallow/ít tương
    quan), "depth" (build_combined_expression không lọt trần), "gate" (verdict fail — vd
    self_corr pool), "not_better" (điểm-nộp combo không vượt thành phần mạnh nhất). Để mặc
    định None nếu caller không cần chẩn đoán (không tốn gì thêm).

    (T1.2) Trần component KHÔNG còn là hằng số cố định `COMBINER_MAX_COMPONENT_DEPTH`: mỗi
    lần thử được suy ĐỘNG qua `component_depth_cap(attempt_n_max)` — N nhỏ hơn thì cây add
    nông hơn, trần được nới ra tương ứng. Nếu greedy rỗng HOẶC mọi combo đều chết trần ở
    n_max yêu cầu, `combine_stage` tự thử lại TOÀN BỘ (lọc pool + greedy) với n_max nhỏ hơn
    (4 -> 3 -> 2, xem `_n_max_retry_sequence`) thay vì chỉ bỏ tín hiệu điểm thấp nhất như
    vòng lặp nội bộ cũ của `build_combined_expression` (vẫn giữ nguyên làm phòng vệ thứ hai).
    Dừng thử ngay khi N nào đó dựng được >= 1 combo (không cần N càng nhỏ càng tốt, chỉ cần
    thoát khỏi bức tường độ sâu)."""
    reg = registry or default_registry()
    depth_kw = {} if max_depth is None else {"max_depth": max_depth}
    out: list[ShortlistCandidate] = []
    for attempt_n_max in _n_max_retry_sequence(n_max, n_min):
        # (T1.2) Loại tín hiệu quá sâu NGAY TRƯỚC greedy — component depth > cap của N đang
        # thử chắc chắn vượt trần sau khi build_combined_expression bọc rank+add, đo được
        # 3/5 rồi 2/5 combo chết ở tầng dựng biểu thức (diag 20260712/20260713) vì greedy
        # chọn nhầm seed quá sâu trước khi biết nó sẽ hỏng. Không đụng `signals` gốc
        # (score_fn_factory Fix 2 vẫn cần TOÀN BỘ signals để dựng pool "ngoài combo").
        cap = component_depth_cap(attempt_n_max)
        shallow_signals = [s for s in signals if _depth_of(s.expr, reg) <= cap]
        combos = select_decorrelated_combos(
            shallow_signals, tau=tau, n_min=n_min, n_max=attempt_n_max, max_combos=max_combos,
            max_component_depth=cap, registry=reg,
        )
        if not combos:
            _bump(drop_stats, "greedy_empty")
            continue
        produced = False
        for combo in combos:
            built = build_combined_expression(
                [s.expr for s in combo], registry=reg, **depth_kw
            )
            if built is None:
                _bump(drop_stats, "depth")
                continue
            produced = True
            if score_fn_factory is not None:
                # Finding #3 (review): loại theo CHUỖI expr, không phải id() — `signals` có thể
                # chứa CÙNG một expr 2 bản (vd 1 bản "run" phiên hiện tại + 1 bản "db" nạp lại từ
                # kho, khác object/id). Lọc bằng id() để lọt bản sao vào pool "others" dù expr đã
                # nằm trong combo -> pool tự-so với chính combo (|rho|≈1) -> gate giết oan combo
                # (đúng loại tự-so Fix 2 phải khử).
                combo_exprs = {s.expr for s in combo}
                others = [s for s in signals if s.expr not in combo_exprs]
                local_score_fn = score_fn_factory(others)
            else:
                local_score_fn = score_fn
            scored = local_score_fn(built.expr)
            if not scored.verdict.passed:  # type: ignore[attr-defined]
                _bump(drop_stats, "gate")
                continue
            best_component = max(
                (_submit_score(local_score_fn(e).metrics) for e in built.sub_exprs),  # type: ignore[attr-defined]
                default=float("-inf"),
            )
            if _submit_score(scored.metrics) <= best_component:  # type: ignore[attr-defined]
                _bump(drop_stats, "not_better")
                continue
            out.append(
                ShortlistCandidate(
                    expr=built.expr,
                    metrics=scored.metrics,  # type: ignore[arg-type]
                    pnl=scored.pnl,  # type: ignore[arg-type]
                    dates=scored.dates,  # type: ignore[arg-type]
                    origin="combiner",
                )
            )
        if produced:
            # (T1.2) Đã vượt bức tường độ sâu ở N này (>= 1 combo build được, bất kể sau đó
            # có qua gate/điểm-nộp hay không) -> KHÔNG cần thử N nhỏ hơn nữa. Retry chỉ để
            # giải quyết vấn đề DEPTH, không phải để cứu combo rớt gate/not_better.
            break
    return out
