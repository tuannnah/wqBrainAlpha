"""Xác minh field LIVE trước khi seed (WS3 T3.3, .superpowers/sdd/20260719/task-3-brief.md —
cardinal rule #1 của dự án: đừng bịa field).

Bối cảnh: seed đường sim-thẳng (`_sim_direct`, frontier/alt-data/fundamental/hypothesis — xem
`src/app/closed_loop_adapters.py::build_closed_loop`) tham chiếu field theo tên suy đoán từ
docs Brain; field có thể KHÔNG tồn tại/khác tên trên platform thật. `tools/verify_frontier_
fields.py` và `tools/verify_datasets.py` verify LIVE (gọi API đọc) và ghi bằng chứng ra
`logs/verified_*.json`, nhưng trước Task 3 KHÔNG được nối vào đường closed-loop — seed dùng
field chưa verify vẫn lọt xuống Brain sim bình thường.

Module này CHỈ đọc file JSON đã ghi sẵn (KHÔNG gọi API live) + lọc core theo field đã verify —
an toàn unit-test (không mạng); nguồn sự thật (`verified_fields`) là THAM SỐ inject được, hàm
lọc không tự đọc file bên trong."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from src.lang.parser import parse
from src.lang.visitors import FieldCollector

# Field nhóm (sector/industry/...) không phải field DỮ LIỆU thật -> luôn coi là "đã verify".
# Lặp lại danh sách này (thay vì import từ closed_loop_adapters) để module này không phụ thuộc
# ngược vào tầng adapter — 6 field group ổn định, hiếm khi đổi.
_GROUP_FIELDS: frozenset[str] = frozenset(
    {"country", "exchange", "market", "sector", "industry", "subindustry", "currency"}
)


def extract_verified_fields(data: dict[str, Any]) -> set[str]:
    """Rút tập field ID đã verify LIVE từ 1 JSON đã parse — hỗ trợ CẢ HAI định dạng hiện có:
    - `tools/verify_frontier_fields.py`: `{"co": {field_id: {...}}, "thieu": [...]}`.
    - `tools/verify_datasets.py`: `{dataset_id: [{"id":.., "coverage":..}, ...], ...}`.
    Định dạng lạ (không khớp cả hai) -> trả rỗng, KHÔNG raise (tầng gọi tự quyết định fail-open
    hay không dựa trên rỗng đó)."""
    co = data.get("co")
    if isinstance(co, dict):
        return set(co.keys())
    fields: set[str] = set()
    for value in data.values():
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict) and "id" in entry:
                    fields.add(str(entry["id"]))
    return fields


def load_latest_verified_fields(logs_dir: Path) -> frozenset[str] | None:
    """Tìm file `verified_*.json` MỚI NHẤT (mtime) trong `logs_dir`, trả tập field đã verify
    LIVE. `None` nếu KHÔNG có file nào (thư mục vắng/không tồn tại) hoặc file mới nhất lỗi
    parse — CẢ HAI coi là "KHÔNG có bằng chứng" (quyết định T3.3: thiếu infra verify không nên
    chặn oan seed thật) -> tầng gọi (`filter_seeds_by_verified_fields`) FAIL-OPEN khi nhận None."""
    if not logs_dir.is_dir():
        return None
    candidates = sorted(
        logs_dir.glob("verified_*.json"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        return None
    try:
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "Field-verify guard: không đọc được {} ({}) — coi như KHÔNG có bằng chứng "
            "(fail-open).", candidates[0], exc,
        )
        return None
    return frozenset(extract_verified_fields(data))


def filter_seeds_by_verified_fields(
    cores: "tuple[str, ...] | list[str]", verified_fields: frozenset[str] | None, registry: Any,
) -> tuple[str, ...]:
    """Loại core dùng field KHÔNG nằm trong `verified_fields` (cardinal rule #1) TRƯỚC khi core
    được yield đi sim. `verified_fields=None` (không có bằng chứng — file/bảng vắng) -> FAIL-
    OPEN: không lọc gì (quyết định T3.3 — thiếu infra verify không nên chặn oan seed thật, chỉ
    chặn khi CÓ bằng chứng mà field không nằm trong đó). Core parse lỗi -> loại (không để lọt
    biểu thức hỏng xuống sim), log WARNING 1 dòng mỗi core bị loại."""
    if verified_fields is None:
        return tuple(cores)
    kept: list[str] = []
    for expr in cores:
        try:
            fields = FieldCollector(registry).visit(parse(expr)) - _GROUP_FIELDS
        except Exception as exc:  # noqa: BLE001 - core hỏng: loại + log, không để lọt xuống sim
            logger.warning("Field-verify guard: bỏ qua core (parse lỗi) {!r}: {}", expr, exc)
            continue
        missing = fields - verified_fields
        if missing:
            logger.warning(
                "Field-verify guard: loại seed {!r} — field CHƯA verify LIVE: {}",
                expr, sorted(missing),
            )
            continue
        kept.append(expr)
    return tuple(kept)
