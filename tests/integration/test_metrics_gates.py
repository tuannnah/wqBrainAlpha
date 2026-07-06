"""Integration Phase 4: parse -> eval -> portfolio -> backtest -> metrics -> gate,
end-to-end thật trên fixture small_panel (Phase 0), không mock bất cứ thành phần nào.

Lưu ý field/API (khác bản brief nguyên văn — đã verify thật trước khi viết):
- `small_panel` (tests/conftest.py) có fields={"close", "volume"} — "close" hợp lệ, nên
  biểu thức `rank(ts_mean(close, 5))` giữ nguyên như brief, không cần đổi.
- `EvalContext` BẮT BUỘC truyền `registry` thật (không phải None) — dùng
  `default_registry()` sau khi import `src.operators_local` để đăng ký operator (side-effect),
  đúng pattern `src/backtest/gate.py`.
- `DepthVisitor`/`FieldCollector` có method `.visit(node)` thật (xem src/lang/visitors.py) —
  gọi trực tiếp, không cần fallback `hasattr(...).compute(...)` như brief.
"""

from __future__ import annotations

import numpy as np

import src.operators_local  # noqa: F401  # side-effect: đăng ký 28 operator thật vào registry
from src.backtest.backtester import Backtester
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.gates import GateVerdict
from src.backtest.metrics_local import AlphaMetrics, MetricsCalculator
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor, FieldCollector
from src.scoring.filter import evaluate_local


def test_handwritten_alpha_end_to_end_metrics_and_gate(small_panel: MarketData) -> None:
    # Biểu thức dùng field có sẵn trong small_panel (fields={"close", "volume"}, xem
    # tests/conftest.py) — "close" hợp lệ nên giữ nguyên biểu thức brief, không cần đổi.
    expr = "rank(ts_mean(close, 5))"
    node = parse(expr)

    depth = DepthVisitor().visit(node)
    fields = FieldCollector(default_registry()).visit(node)
    fields_ok = fields.issubset(set(small_panel.fields.keys()))

    ctx = EvalContext(data=small_panel, registry=default_registry(), cache=None)
    signal = Evaluator(ctx).evaluate(node)
    assert signal.shape == (len(small_panel.dates), len(small_panel.assets))

    cfg = PortfolioConfig(
        neutralization=Neutralization.SECTOR, decay=0, truncation=0.10,
        scale_book=1.0, delay=1,
    )
    weights = PortfolioBuilder().build(signal, cfg, small_panel)
    bt = Backtester().run(weights, small_panel)

    metrics = MetricsCalculator().compute(bt, small_panel)
    assert isinstance(metrics, AlphaMetrics)
    assert np.isfinite(metrics.sharpe)
    assert metrics.per_year_sharpe  # small_panel multi-day -> ít nhất 1 năm

    verdict = evaluate_local(metrics, self_corr=0.0, depth=depth, fields_ok=fields_ok)
    assert isinstance(verdict, GateVerdict)
    # Không assert verdict.passed is True cứng — alpha viết tay trên data nhỏ có thể không
    # đạt sharpe/turnover thật; assert ĐÚNG HÀNH VI: verdict luôn có cả hard_failures (list)
    # và soft_scores (dict) đầy đủ 5 khoá (thêm is_ladder robustness), bất kể pass/fail.
    assert isinstance(verdict.hard_failures, list)
    assert set(verdict.soft_scores) == {
        "sharpe", "fitness", "turnover_band", "per_year_min", "is_ladder"
    }
    print(
        f"[Phase4 demo] sharpe={metrics.sharpe:.3f} fitness={metrics.fitness:.3f} "
        f"turnover={metrics.turnover:.3f} concentration={metrics.weight_concentration:.3f} "
        f"gate_passed={verdict.passed} hard_failures={verdict.hard_failures}"
    )
