"""Báo cáo READ-ONLY cho tiêu chí tie-break BRAIN Genius tính được LOCAL từ alpha đã nộp —
KHÔNG phải gate nộp. Nguồn: docs/worldquantbrain/docs/consultant-information/brain-genius.md
(mục "What happens if more consultants qualify..."). Phạm vi KHÔNG làm (pyramid, streak,
community leader, Combined Alpha Performance) ghi ở
docs/superpowers/plans/2026-07-02-genius-tracking-report.md."""

from __future__ import annotations

from src.lang.parser import parse_expression
from src.lang.visitors import FieldCollector, OperatorCollector
from src.storage.models import AlphaModel, SimulationModel, SubmissionModel

# ts_backfill/group_backfill KHÔNG tính vào Avg/Total distinct Operators (tài liệu Genius).
_EXEMPT_OPERATORS = {"ts_backfill", "group_backfill"}


def _submitted_expressions(session_factory) -> list[str]:
    """Biểu thức của mọi alpha đã `status == "submitted"` (không trùng lặp theo wq_alpha_id)."""
    session = session_factory()
    try:
        wq_ids = {
            row[0]
            for row in session.query(SubmissionModel.alpha_id)
            .filter(SubmissionModel.status == "submitted")
            .all()
        }
        if not wq_ids:
            return []
        rows = (
            session.query(AlphaModel.expression)
            .join(SimulationModel, SimulationModel.alpha_id == AlphaModel.id)
            .filter(SimulationModel.wq_alpha_id.in_(wq_ids))
            .distinct()
            .all()
        )
        return [r[0] for r in rows]
    finally:
        session.close()


def average_distinct_operators_per_alpha(session_factory) -> float | None:
    exprs = _submitted_expressions(session_factory)
    if not exprs:
        return None
    counts = [
        len(OperatorCollector().visit(parse_expression(e)) - _EXEMPT_OPERATORS) for e in exprs
    ]
    return sum(counts) / len(counts)


def average_distinct_fields_per_alpha(session_factory) -> float | None:
    exprs = _submitted_expressions(session_factory)
    if not exprs:
        return None
    counts = [len(FieldCollector().visit(parse_expression(e))) for e in exprs]
    return sum(counts) / len(counts)


def total_distinct_operators(session_factory) -> int:
    exprs = _submitted_expressions(session_factory)
    all_ops: set[str] = set()
    for e in exprs:
        all_ops |= OperatorCollector().visit(parse_expression(e)) - _EXEMPT_OPERATORS
    return len(all_ops)


def total_distinct_fields(session_factory) -> int:
    exprs = _submitted_expressions(session_factory)
    all_fields: set[str] = set()
    for e in exprs:
        all_fields |= FieldCollector().visit(parse_expression(e))
    return len(all_fields)
