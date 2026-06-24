"""Lọc alpha đa tiêu chí theo ngưỡng."""

from __future__ import annotations

from dataclasses import dataclass

from src.backtest.gates import GateEvaluator, GateVerdict
from src.backtest.metrics_local import AlphaMetrics
from src.scoring.metrics import normalize


@dataclass
class FilterThresholds:
    min_sharpe: float = 1.25
    min_fitness: float = 1.0
    turnover_low: float = 0.01
    turnover_high: float = 0.70
    max_drawdown: float = 0.20


def passes(source, thresholds: FilterThresholds | None = None) -> tuple[bool, list[str]]:
    """Trả (đạt, danh sách lý do fail)."""
    t = thresholds or FilterThresholds()
    m = normalize(source)
    reasons: list[str] = []

    if m["sharpe"] < t.min_sharpe:
        reasons.append(f"sharpe {m['sharpe']:.2f} < {t.min_sharpe}")
    if m["fitness"] <= t.min_fitness:
        reasons.append(f"fitness {m['fitness']:.2f} <= {t.min_fitness}")
    if not (t.turnover_low <= m["turnover"] <= t.turnover_high):
        reasons.append(f"turnover {m['turnover']:.2f} ngoài [{t.turnover_low}, {t.turnover_high}]")
    if m["drawdown"] >= t.max_drawdown:
        reasons.append(f"drawdown {m['drawdown']:.2f} >= {t.max_drawdown}")

    return (len(reasons) == 0, reasons)


def blocking_dimensions(
    source,
    thresholds: FilterThresholds | None = None,
    pool_corr: float | None = None,
    max_pool_corr: float = 0.70,
) -> set[str]:
    """Tập tên chiều ScoreVector đang KHÔNG đạt ngưỡng hard filter.

    Dùng để refiner nhắm đúng chiều chặn việc pass (đặc biệt fitness), thay vì
    chiều có điểm chuẩn-hoá thấp nhất tuyệt đối. Rỗng nghĩa là alpha đã đạt.

    `pool_corr` (self-correlation với pool, đo sau sim): vượt `max_pool_corr` ->
    thêm 'pool_fit' để refiner nhắm khử trùng (residual neutralize). None -> bỏ qua."""
    t = thresholds or FilterThresholds()
    m = normalize(source)
    dims: set[str] = set()
    if m["sharpe"] < t.min_sharpe:
        dims.add("sharpe")
    if m["fitness"] <= t.min_fitness:
        dims.add("fitness")
    if not (t.turnover_low <= m["turnover"] <= t.turnover_high):
        dims.add("turnover_fit")
    if m["drawdown"] >= t.max_drawdown:
        dims.add("drawdown_fit")
    if pool_corr is not None and abs(pool_corr) >= max_pool_corr:
        dims.add("pool_fit")
    return dims


def evaluate_local(
    metrics: AlphaMetrics, self_corr: float, depth: int, fields_ok: bool
) -> GateVerdict:
    """Cổng local đầy đủ (Phase 4, B8 master spec) — wrap GateEvaluator cho loop/CLI dùng.

    Khác `passes`/`blocking_dimensions` ở trên: hai hàm đó chấm điểm KẾT QUẢ SIM BRAIN THẬT
    (qua `ScoreVector`/`normalize`); `evaluate_local` chấm điểm `AlphaMetrics` tính LOCAL
    (Phase 3/4, không tốn quota sim) — dùng trước khi quyết định có đáng đốt sim hay không.
    """
    return GateEvaluator().evaluate(metrics, self_corr=self_corr, depth=depth, fields_ok=fields_ok)
