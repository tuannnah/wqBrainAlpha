"""LocalTuner: quét tham số (window/hệ số) + config quanh MỘT biểu thức bằng đường backtest
MiniBrain (không mạng, không LLM). Deterministic — coordinate descent dưới ngân sách, luôn
giữ biểu thức gốc làm cận dưới (không bao giờ trả kết quả tệ hơn gốc)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from src.backtest.config import Neutralization, PortfolioConfig
from src.lang.ast import Call, Constant, Node
from src.lang.parser import parse
from src.lang.registry import ArgKind, OperatorRegistry, default_registry
from src.lang.visitors import Serializer

if TYPE_CHECKING:
    from src.backtest.metrics_local import AlphaMetrics


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
# Chỉ MARKET/SECTOR: docs khuyến nghị cho price/volume + eval local được (panel có group sector).
_NEUTS = (Neutralization.MARKET, Neutralization.SECTOR)
_MAX_TURNOVER = 0.70  # Brain đòi 1%-70%; config vượt trần là rác chắc chắn fail.
# Ngưỡng nộp Delay-1 (docs): Sharpe > 1.25, Fitness > 1.0. Tuner tối ưu ĐIỂM NỘP = biên
# chuẩn hoá NHỎ NHẤT giữa hai cổng (min(Sharpe/1.25, Fitness/1.0)) thay vì Sharpe trần — bằng
# chứng live: core đạt Sharpe 1.45 nhưng FAIL vì fitness 0.80, do tuner chỉ đuổi Sharpe.
_SUB_SHARPE_MIN = 1.25
_SUB_FITNESS_MIN = 1.0


def _submission_score(m: "AlphaMetrics") -> float:
    """Điểm hướng nộp: min(Sharpe/1.25, Fitness/1.0). ≥1 ⇔ qua CẢ hai cổng. Đẩy tuner tìm
    config cân bằng Sharpe–Fitness (giảm turnover nâng fitness) thay vì Sharpe cao mà fail fitness."""
    s = m.sharpe
    f = m.fitness
    if s is None or f is None or not (np.isfinite(s) and np.isfinite(f)):
        return float("-inf")
    return min(float(s) / _SUB_SHARPE_MIN, float(f) / _SUB_FITNESS_MIN)


@dataclass(frozen=True, slots=True)
class TuneResult:
    """Kết quả tune: biểu thức tốt nhất (đã serialize), config tốt nhất, sharpe local đạt được."""

    best_expr: str
    best_config: PortfolioConfig
    local_sharpe: float
    # AlphaMetrics local của best (đường backtest thật) — để refiner lưu vào kho calibration
    # (join local↔Brain theo hash). None khi tune qua eval_fn inject (test, không backtest thật).
    local_metrics: "AlphaMetrics | None" = None


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


def local_metrics(
    node: Node, config: PortfolioConfig, data, registry: OperatorRegistry
) -> "AlphaMetrics | None":
    """AlphaMetrics local qua đúng đường backtest của gate (Evaluator -> PortfolioBuilder ->
    Backtester -> MetricsCalculator); trả None nếu lỗi/NaN/không có pnl hữu hạn."""
    from src.backtest.backtester import Backtester
    from src.backtest.metrics_local import MetricsCalculator
    from src.backtest.portfolio import PortfolioBuilder
    from src.engine.evaluator import EvalContext, Evaluator

    try:
        signal = Evaluator(EvalContext(data=data, registry=registry, cache=None)).evaluate(node)
        if np.all(np.isnan(signal)):
            return None
        weights = PortfolioBuilder().build(signal, config, data)
        result = Backtester().run(weights, data)
        if not np.isfinite(result.daily_pnl).any():
            return None
        return MetricsCalculator().compute(result, data)
    except (KeyError, ValueError, ZeroDivisionError):
        return None


def local_sharpe(node: Node, config: PortfolioConfig, data, registry: OperatorRegistry) -> float:
    """Sharpe local (bọc `local_metrics`); trả −inf nếu lỗi/NaN/không có pnl hữu hạn."""
    m = local_metrics(node, config, data, registry)
    if m is None:
        return float("-inf")
    s = m.sharpe
    return float(s) if s is not None and np.isfinite(s) else float("-inf")


def tune(
    expr: str,
    base_config: PortfolioConfig,
    data,
    *,
    registry: OperatorRegistry | None = None,
    budget: int = 48,  # lưới config gấp đôi (decay×trunc×neut=24); chừa ~24 eval cho window/hệ số
    max_turnover: float = _MAX_TURNOVER,
    eval_fn=None,
) -> TuneResult:
    """Coordinate descent quanh `expr`: quét window (thang bậc) + hệ số (×0.5/×2) của từng
    hằng số trong cây, RỒI quét config (decay × truncation). Luôn giữ biểu thức/config gốc
    làm cận dưới (bất biến đơn điệu) — không bao giờ trả kết quả tệ hơn gốc.

    `eval_fn(node, config) -> float` inject được cho test deterministic không cần backtest
    thật; None thì dùng `local_sharpe` thật trên `data`/`registry`. Biến thể làm `eval_fn`
    raise lỗi bị coi là −inf (bỏ qua), KHÔNG làm sập `tune`.

    `max_turnover` chỉ áp ở đường thật (khi `eval_fn is None`): Brain đòi turnover 1%-70%,
    config vượt trần bị loại (điểm −inf) dù Sharpe local cao hơn, để winner luôn nộp được.
    """
    registry = registry or default_registry()

    def score(node: Node, config: PortfolioConfig) -> "tuple[float, AlphaMetrics | None]":
        """Trả (điểm, metrics|None). eval_fn inject (test) chỉ cho điểm, không có metrics;
        đường thật trả AlphaMetrics để refiner lưu vào kho calibration. Biến thể lỗi -> −inf."""
        try:
            if eval_fn is not None:
                return float(eval_fn(node, config)), None
            m = local_metrics(node, config, data, registry)
            if m is None:
                return float("-inf"), None
            if m.turnover is not None and m.turnover > max_turnover:
                return float("-inf"), m   # vượt trần turnover -> loại (giữ metrics để báo cáo)
            # Xếp hạng theo ĐIỂM NỘP (min biên Sharpe/Fitness) — không phải Sharpe trần — để
            # winner qua CẢ hai cổng submission, không chỉ Sharpe.
            return _submission_score(m), m
        except (KeyError, ValueError, ZeroDivisionError):
            return float("-inf"), None

    base_node = parse(expr)
    best_node = base_node
    best_config = base_config
    best, best_metrics = score(base_node, best_config)
    evals = 1

    # Chừa ngân sách cho Giai đoạn 2 (config): biểu thức nhiều hằng có thể nuốt hết budget ở
    # Giai đoạn 1, khiến decay/truncation — thứ quan trọng nhất — không bao giờ được quét.
    # Giới hạn Giai đoạn 1 ở `budget - kích thước lưới config` để Giai đoạn 2 luôn có chỗ.
    phase1_cap = max(1, budget - len(_DECAYS) * len(_TRUNCS) * len(_NEUTS))

    # Giai đoạn 1: quét window/hệ số của từng hằng số trong biểu thức.
    for path, value, is_window in iter_constants(base_node, registry):
        if evals >= phase1_cap:
            break
        cands = _window_candidates(value) if is_window else _coef_candidates(value)
        for cand in cands:
            if evals >= phase1_cap:
                break
            trial = set_constant(best_node, path, cand)
            s, m = score(trial, best_config)
            evals += 1
            if s > best:
                best, best_node, best_metrics = s, trial, m

    # Giai đoạn 2: quét config (decay x truncation x neutralization) quanh biểu thức tốt nhất.
    for neut in _NEUTS:
        for d in _DECAYS:
            for t in _TRUNCS:
                if evals >= budget:
                    break
                cfg = replace(base_config, decay=d, truncation=t, neutralization=neut)
                if cfg == best_config:
                    continue
                s, m = score(best_node, cfg)
                evals += 1
                if s > best:
                    best, best_config, best_metrics = s, cfg, m

    # local_sharpe báo cáo Sharpe THẬT của best (cho pre-sim floor 0.5), KHÔNG phải điểm nộp
    # dùng để xếp hạng. Đường eval_fn (test) không có metrics -> giữ `best` (giá trị eval_fn).
    reported_sharpe = best if best_metrics is None else best_metrics.sharpe
    return TuneResult(
        best_expr=Serializer().visit(best_node), best_config=best_config,
        local_sharpe=reported_sharpe, local_metrics=best_metrics,
    )
