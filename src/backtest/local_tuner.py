"""LocalTuner: quét tham số (window/hệ số) + config quanh MỘT biểu thức bằng đường backtest
MiniBrain (không mạng, không LLM). Deterministic — coordinate descent dưới ngân sách, luôn
giữ biểu thức gốc làm cận dưới (không bao giờ trả kết quả tệ hơn gốc)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.backtest.config import PortfolioConfig
from src.lang.ast import Call, Constant, Node
from src.lang.parser import parse
from src.lang.registry import ArgKind, OperatorRegistry, default_registry
from src.lang.visitors import Serializer


def iter_constants(
    node: Node, registry: OperatorRegistry, _path: tuple[int, ...] = ()
) -> list[tuple[tuple[int, ...], float, bool]]:
    """Liệt kê mọi hằng số trong cây kèm đường tới nó và cờ 'là window'."""
    if isinstance(node, Constant):
        return [(_path, node.value, False)]  # gốc là hằng đơn — hiếm; không đánh dấu window
    if not isinstance(node, Call):
        return []
    try:
        kinds = registry.get(node.op).signature
    except KeyError:
        kinds = ()
    out: list[tuple[tuple[int, ...], float, bool]] = []
    for i, arg in enumerate(node.args):
        kind = kinds[i] if i < len(kinds) else None
        if isinstance(arg, Constant):
            out.append((_path + (i,), arg.value, kind is ArgKind.WINDOW))
        else:
            out.extend(iter_constants(arg, registry, _path + (i,)))
    return out


def set_constant(node: Node, path: tuple[int, ...], new_value: float) -> Node:
    """Trả node mới với hằng tại `path` = new_value; phần còn lại giữ nguyên (bất biến)."""
    if not path:
        return Constant(float(new_value))
    if isinstance(node, Call):
        i = path[0]
        new_args = list(node.args)
        new_args[i] = set_constant(node.args[i], path[1:], new_value)
        return Call(node.op, tuple(new_args))
    return node


# --- Task 3: tune() — coordinate descent quanh MỘT biểu thức + config sweep ------------

_WINDOW_LADDER = (3, 5, 10, 20, 40, 60, 120)
_DECAYS = (2, 3, 4, 6)
_TRUNCS = (0.02, 0.05, 0.08)


@dataclass(frozen=True, slots=True)
class TuneResult:
    """Kết quả tune: biểu thức tốt nhất (đã serialize), config tốt nhất, sharpe local đạt được."""

    best_expr: str
    best_config: PortfolioConfig
    local_sharpe: float


def _window_candidates(value: float) -> list[float]:
    """Các bậc thang lân cận (gần giá trị hiện tại nhất trước) — bỏ chính nó."""
    cur = int(round(value))
    ladder = sorted(_WINDOW_LADDER, key=lambda w: abs(w - cur))
    return [float(w) for w in ladder if w != cur]


def _coef_candidates(value: float) -> list[float]:
    """Hệ số lân cận: nửa và gấp đôi giá trị hiện tại (bỏ qua nếu giá trị bằng 0)."""
    if value == 0:
        return []
    return [value * 0.5, value * 2.0]


def local_sharpe(node: Node, config: PortfolioConfig, data, registry: OperatorRegistry) -> float:
    """Sharpe local qua đúng đường backtest của gate (Evaluator -> PortfolioBuilder ->
    Backtester -> MetricsCalculator); trả −inf nếu lỗi/NaN/không có pnl hữu hạn."""
    from src.backtest.backtester import Backtester
    from src.backtest.metrics_local import MetricsCalculator
    from src.backtest.portfolio import PortfolioBuilder
    from src.engine.evaluator import EvalContext, Evaluator

    try:
        signal = Evaluator(EvalContext(data=data, registry=registry, cache=None)).evaluate(node)
        if np.all(np.isnan(signal)):
            return float("-inf")
        weights = PortfolioBuilder().build(signal, config, data)
        result = Backtester().run(weights, data)
        if not np.isfinite(result.daily_pnl).any():
            return float("-inf")
        s = MetricsCalculator().compute(result, data).sharpe
    except (KeyError, ValueError, ZeroDivisionError):
        return float("-inf")
    return float(s) if s is not None and np.isfinite(s) else float("-inf")


def tune(
    expr: str,
    base_config: PortfolioConfig,
    data,
    *,
    registry: OperatorRegistry | None = None,
    budget: int = 40,
    eval_fn=None,
) -> TuneResult:
    """Coordinate descent quanh `expr`: quét window (thang bậc) + hệ số (×0.5/×2) của từng
    hằng số trong cây, RỒI quét config (decay × truncation). Luôn giữ biểu thức/config gốc
    làm cận dưới (bất biến đơn điệu) — không bao giờ trả kết quả tệ hơn gốc.

    `eval_fn(node, config) -> float` inject được cho test deterministic không cần backtest
    thật; None thì dùng `local_sharpe` thật trên `data`/`registry`. Biến thể làm `eval_fn`
    raise lỗi bị coi là −inf (bỏ qua), KHÔNG làm sập `tune`.
    """
    registry = registry or default_registry()
    scorer = eval_fn or (lambda n, c: local_sharpe(n, c, data, registry))

    def score(node: Node, config: PortfolioConfig) -> float:
        try:
            return scorer(node, config)
        except (KeyError, ValueError, ZeroDivisionError):
            return float("-inf")

    base_node = parse(expr)
    best_node = base_node
    best_config = base_config
    best = score(base_node, best_config)
    evals = 1

    # Giai đoạn 1: quét window/hệ số của từng hằng số trong biểu thức.
    for path, value, is_window in iter_constants(base_node, registry):
        if evals >= budget:
            break
        cands = _window_candidates(value) if is_window else _coef_candidates(value)
        for cand in cands:
            if evals >= budget:
                break
            trial = set_constant(best_node, path, cand)
            s = score(trial, best_config)
            evals += 1
            if s > best:
                best, best_node = s, trial

    # Giai đoạn 2: quét config (decay x truncation) quanh biểu thức tốt nhất tìm được.
    for d in _DECAYS:
        for t in _TRUNCS:
            if evals >= budget:
                break
            cfg = replace(base_config, decay=d, truncation=t)
            if cfg == best_config:
                continue
            s = score(best_node, cfg)
            evals += 1
            if s > best:
                best, best_config = s, cfg

    return TuneResult(
        best_expr=Serializer().visit(best_node), best_config=best_config, local_sharpe=best
    )
