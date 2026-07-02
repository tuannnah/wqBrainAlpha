"""Phát hiện 'Single Dataset Alpha' (WQ Brain: mọi field trừ 6 grouping field đến từ ĐÚNG 1
dataset) — dùng để gắn nhãn/tham khảo khi sinh alpha, KHÔNG dùng để gate nộp (WQ Brain tự áp
ngưỡng nới lỏng cho loại này qua `is.checks`, xem
docs/worldquantbrain/docs/consultant-information/single-dataset-alphas.md và đính chính
sub-project B trong docs/superpowers/specs/2026-07-02-submission-compliance-roadmap-design.md)."""

from __future__ import annotations

from src.lang.parser import parse_expression
from src.lang.visitors import FieldCollector, OperatorCollector

# 6 grouping field được miễn trừ khi tính dataset (theo single-dataset-alphas.md — khác danh
# sách Power Pool có thêm 'currency', xem lưu ý trong roadmap spec).
_GROUPING_FIELDS = {"country", "exchange", "market", "sector", "industry", "subindustry"}
# 2 operator luôn tính là dùng dataset 'pv1' (theo tài liệu).
_PV1_OPERATORS = {"inst_pnl", "convert"}


def dataset_of_alpha(expr: str, field_dataset: dict[str, str]) -> str | None:
    """Trả dataset_id DUY NHẤT nếu `expr` là single-dataset; `None` nếu dùng >1 dataset hoặc
    có field không xác định được dataset trong `field_dataset`."""
    node = parse_expression(expr)
    fields = FieldCollector().visit(node)
    operators = OperatorCollector().visit(node)

    datasets: set[str] = set()
    for field_id in fields:
        if field_id in _GROUPING_FIELDS:
            continue
        dataset_id = field_dataset.get(field_id)
        if dataset_id is None:
            return None
        datasets.add(dataset_id)
    if operators & _PV1_OPERATORS:
        datasets.add("pv1")
    if len(datasets) == 1:
        return next(iter(datasets))
    return None


def is_single_dataset_alpha(expr: str, field_dataset: dict[str, str]) -> bool:
    return dataset_of_alpha(expr, field_dataset) is not None


def datasets_used(expr: str, field_dataset: dict[str, str]) -> set[str]:
    """Tập TẤT CẢ dataset_id dùng trong `expr` (bỏ qua grouping field và field không rõ
    dataset) — khác `dataset_of_alpha` (chỉ trả kết quả khi DUY NHẤT 1 dataset). Dùng để kiểm
    khớp Power Pool Theme (loại trừ dataset cụ thể, không yêu cầu single-dataset)."""
    node = parse_expression(expr)
    fields = FieldCollector().visit(node)
    operators = OperatorCollector().visit(node)

    datasets: set[str] = set()
    for field_id in fields:
        if field_id in _GROUPING_FIELDS:
            continue
        dataset_id = field_dataset.get(field_id)
        if dataset_id is not None:
            datasets.add(dataset_id)
    if operators & _PV1_OPERATORS:
        datasets.add("pv1")
    return datasets
