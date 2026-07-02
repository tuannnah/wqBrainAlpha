"""Điều kiện Power Pool Alphas — CHỈ phần tính được LOCAL (Sharpe, operator/field unique).
KHÔNG gồm Power Pool Correlation/Theme matching (endpoint/danh sách thật chưa xác nhận, xem
docs/superpowers/plans/2026-07-02-power-pool-alphas.md mục "Phạm vi KHÔNG làm").

Nguồn tiêu chí: docs/worldquantbrain/docs/consultant-information/power-pool-alphas.md."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.lang.parser import parse_expression
from src.lang.visitors import FieldCollector, OperatorCollector

MIN_SHARPE = 1.0
MAX_UNIQUE_OPERATORS = 8
MAX_UNIQUE_FIELDS = 3
MIN_DESCRIPTION_LEN = 100

# Operator KHÔNG tính vào giới hạn 8 (theo tài liệu Power Pool).
_EXEMPT_OPERATORS = {"ts_backfill", "group_backfill"}
# 7 grouping field KHÔNG tính vào giới hạn 3 field — CÓ 'currency' (khác Single Dataset Alphas,
# sub-project D, không có 'currency' — 2 danh sách khác nhau theo đúng tài liệu gốc).
_GROUPING_FIELDS = {
    "country", "industry", "subindustry", "currency", "market", "sector", "exchange",
}


def count_operators_fields(expr: str) -> tuple[int, int]:
    """Trả (số operator unique trừ ngoại lệ, số field unique trừ grouping field) — 2 con số
    quyết định giới hạn Power Pool (<=8 operator, <=3 field)."""
    node = parse_expression(expr)
    operators = OperatorCollector().visit(node) - _EXEMPT_OPERATORS
    fields = FieldCollector().visit(node) - _GROUPING_FIELDS
    return len(operators), len(fields)
