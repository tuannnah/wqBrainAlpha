"""T3.3 (WS3, .superpowers/sdd/20260719/task-3-brief.md) — xác minh field LIVE trước khi seed
(cardinal rule #1: đừng bịa field). Test THUẦN (không mạng, không gọi API live): đọc file JSON
đã ghi sẵn (2 định dạng của tools/verify_frontier_fields.py và tools/verify_datasets.py) +
lọc core theo field đã verify, nguồn sự thật luôn INJECT ĐƯỢC qua tham số."""

from __future__ import annotations

from pathlib import Path

from src.generation.field_verification import (
    extract_verified_fields,
    filter_seeds_by_verified_fields,
    load_latest_verified_fields,
)
from src.lang.registry import default_registry


def test_extract_verified_fields_dinh_dang_verify_frontier_fields() -> None:
    """Định dạng tools/verify_frontier_fields.py: {"co": {field_id: {...}}, "thieu": [...]}."""
    data = {
        "ngay": "2026-07-14", "n_cores": 40, "n_fields": 2,
        "thieu": ["field_thieu"],
        "co": {"directional_indicator_score": {"type": "MATRIX", "dataset": "ds1"}, "vwap": {}},
    }
    assert extract_verified_fields(data) == {"directional_indicator_score", "vwap"}


def test_extract_verified_fields_dinh_dang_verify_datasets() -> None:
    """Định dạng tools/verify_datasets.py: {dataset_id: [{"id":.., "coverage":..}, ...], ...}."""
    data = {
        "ds_a": [{"id": "field_x", "coverage": 0.9}, {"id": "field_y", "coverage": None}],
        "ds_b": [],
    }
    assert extract_verified_fields(data) == {"field_x", "field_y"}


def test_extract_verified_fields_dinh_dang_la_tra_rong() -> None:
    assert extract_verified_fields({}) == set()
    assert extract_verified_fields({"khac": "chuoi_khong_phai_list"}) == set()


def test_load_latest_verified_fields_khong_co_file_tra_none(tmp_path: Path) -> None:
    """Thư mục vắng (không file verified_*.json nào) -> None (KHÔNG có bằng chứng)."""
    assert load_latest_verified_fields(tmp_path) is None


def test_load_latest_verified_fields_thu_muc_khong_ton_tai_tra_none(tmp_path: Path) -> None:
    assert load_latest_verified_fields(tmp_path / "khong_ton_tai") is None


def test_load_latest_verified_fields_chon_file_moi_nhat(tmp_path: Path) -> None:
    """2 file cùng thư mục -> chọn file MỚI NHẤT theo mtime."""
    import os
    import time

    old = tmp_path / "verified_fields_20260701.json"
    new = tmp_path / "verified_frontier_fields_20260714.json"
    old.write_text('{"co": {"field_cu": {}}}', encoding="utf-8")
    time.sleep(0.01)
    new.write_text('{"co": {"field_moi": {}}}', encoding="utf-8")
    # Đảm bảo mtime tách biệt rõ ràng dù filesystem có độ phân giải thô.
    now = time.time()
    os.utime(old, (now - 100, now - 100))
    os.utime(new, (now, now))

    fields = load_latest_verified_fields(tmp_path)
    assert fields == frozenset({"field_moi"})


def test_load_latest_verified_fields_file_hong_tra_none(tmp_path: Path) -> None:
    """File JSON hỏng (parse lỗi) -> coi như KHÔNG có bằng chứng (fail-open), không raise."""
    bad = tmp_path / "verified_fields_broken.json"
    bad.write_text("{khong phai json hop le", encoding="utf-8")
    assert load_latest_verified_fields(tmp_path) is None


def test_filter_seeds_fail_open_khi_khong_co_bang_chung() -> None:
    """verified_fields=None (không có bằng chứng verify) -> KHÔNG lọc gì (fail-open)."""
    cores = ("ts_backfill(field_bia, 66)", "rank(close)")
    out = filter_seeds_by_verified_fields(cores, None, default_registry())
    assert out == cores


def test_filter_seeds_loai_core_dung_field_chua_verify() -> None:
    """Core dùng field NGOÀI verified_fields bị loại; core toàn field đã verify được giữ."""
    cores = ("rank(close)", "ts_backfill(field_chua_verify, 66)")
    out = filter_seeds_by_verified_fields(cores, frozenset({"close"}), default_registry())
    assert out == ("rank(close)",)


def test_group_fields_luon_duoc_coi_la_da_verify() -> None:
    """Field nhóm (sector/industry/...) không phải field dữ liệu thật -> luôn miễn trừ khỏi
    kiểm tra verify (khớp `_POWER_POOL_GROUPS` ở closed_loop_adapters.py). Registry hiện tại
    KHÔNG có operator group_neutralize (wrapper cấu hình, ngoài vocab GP — xem
    src/lang/registry.py) nên không dựng được expr thật để test qua filter_seeds_by_verified_
    fields; kiểm trực tiếp hằng số miễn trừ."""
    from src.generation.field_verification import _GROUP_FIELDS

    assert {"sector", "industry", "subindustry", "market", "country", "exchange"} <= _GROUP_FIELDS


def test_filter_seeds_log_warning_1_dong_khi_loai(caplog) -> None:  # noqa: ANN001
    from loguru import logger

    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        filter_seeds_by_verified_fields(
            ("ts_backfill(field_chua_verify, 66)",), frozenset({"close"}), default_registry(),
        )
    finally:
        logger.remove(sink_id)
    assert any("field_chua_verify" in m for m in msgs)
    assert any("WARNING" in m or "Field-verify" in m for m in msgs)
