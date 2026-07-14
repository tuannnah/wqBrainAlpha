"""Test lịch Power Pool Theme — dữ liệu thủ công từ bài 'Current month Power Pool Themes'
(support.worldquantbrain.com), xác nhận 2026-07-02. Xem docstring
src/scoring/power_pool_theme.py để biết cách cập nhật khi sang tháng/theme mới."""

from __future__ import annotations

from datetime import date

from src.scoring.power_pool_theme import (
    JUNE_JULY_2026_CALENDAR,
    check_theme_compliance,
    matches_theme,
    parse_allowed_neutralizations,
    parse_theme_filter,
    theme_for_date,
)


def test_parse_theme_filter_tach_dung_region_delay_universe_datasets():
    raw = (
        "region=USA & delay=1 & universe=TOP1000 and neutralization in "
        "(slow, fast, slow and fast, ram, statistical, crowding) and datasets not in ['pv1']"
    )
    result = parse_theme_filter(raw)
    assert result["region"] == "USA"
    assert result["delay"] == 1
    assert result["universe"] == "TOP1000"
    assert result["datasets_excluded"] == ("pv1",)
    assert result["unparsed_constraints"] is not None
    assert "neutralization" in result["unparsed_constraints"]


def test_theme_for_date_tuan_co_filter_day_du():
    week = theme_for_date(date(2026, 7, 2))  # hôm nay, thuộc tuần 29/6-5/7
    assert week is not None
    assert week.region == "USA"
    assert week.delay == 1
    assert week.universe == "TOP1000"
    assert week.datasets_excluded == ("pv1",)


def test_theme_for_date_tuan_chi_co_ten():
    week = theme_for_date(date(2026, 6, 3))
    assert week is not None
    assert week.name == "USA D1 Fast Datasets"
    assert week.region is None  # không có filter chi tiết trong dữ liệu đã nhận


def test_theme_for_date_ngoai_lich_tra_none():
    assert theme_for_date(date(2026, 8, 1)) is None


def test_matches_theme_dat_het_dieu_kien():
    week = theme_for_date(date(2026, 7, 2))
    ok, reasons = matches_theme(
        week, region="USA", delay=1, universe="TOP1000", datasets_used={"fundamental6"},
    )
    assert ok is True
    assert reasons == []


def test_matches_theme_dung_dataset_bi_loai_tru():
    week = theme_for_date(date(2026, 7, 2))
    ok, reasons = matches_theme(
        week, region="USA", delay=1, universe="TOP1000", datasets_used={"pv1"},
    )
    assert ok is False
    assert any("pv1" in r for r in reasons)


def test_matches_theme_sai_region():
    week = theme_for_date(date(2026, 7, 2))
    ok, reasons = matches_theme(
        week, region="EUR", delay=1, universe="TOP1000", datasets_used=set(),
    )
    assert ok is False
    assert any("region" in r for r in reasons)


def test_matches_theme_tuan_khong_co_filter_chi_tiet_khong_chan_gi():
    week = theme_for_date(date(2026, 6, 3))  # chỉ có tên, region/delay/universe đều None
    ok, reasons = matches_theme(
        week, region="ASI", delay=0, universe="TOP500", datasets_used={"pv1"},
    )
    assert ok is True  # không có field nào để so -> không chặn
    assert reasons == []


def test_parse_allowed_neutralizations_du_6_token():
    raw = "neutralization in (slow, fast, slow and fast, ram, statistical, crowding)"
    assert parse_allowed_neutralizations(raw) == frozenset(
        {"SLOW", "FAST", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM", "STATISTICAL", "CROWDING"}
    )


def test_parse_allowed_neutralizations_bo_token_la():
    raw = "neutralization in (statistical, khong_biet, crowding)"
    assert parse_allowed_neutralizations(raw) == frozenset({"STATISTICAL", "CROWDING"})


def test_parse_allowed_neutralizations_none_va_rong():
    assert parse_allowed_neutralizations(None) == frozenset()
    assert parse_allowed_neutralizations("region=USA & delay=1") == frozenset()


def test_theme_tuan_hien_tai_2026_07_09():
    week = theme_for_date(date(2026, 7, 9))
    assert week is not None
    assert week.region == "USA"
    assert week.delay == 1
    assert week.universe == "TOP1000"
    assert week.datasets_excluded == ("pv1",)
    assert week.allowed_neutralizations == frozenset(
        {"SLOW", "FAST", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM", "STATISTICAL", "CROWDING"}
    )


def test_matches_theme_chan_khi_neut_ngoai_tap():
    week = theme_for_date(date(2026, 7, 9))
    ok, reasons = matches_theme(
        week, region="USA", delay=1, universe="TOP1000",
        datasets_used={"option8"}, neutralization="SUBINDUSTRY",
    )
    assert ok is False
    assert any("neutralization" in r.lower() for r in reasons)


def test_matches_theme_cho_qua_khi_neut_trong_tap():
    week = theme_for_date(date(2026, 7, 9))
    ok, reasons = matches_theme(
        week, region="USA", delay=1, universe="TOP1000",
        datasets_used={"option8"}, neutralization="STATISTICAL",
    )
    assert ok is True
    assert reasons == []


def test_matches_theme_khong_truyen_neut_thi_khong_chan_neut():
    week = theme_for_date(date(2026, 7, 9))
    ok, reasons = matches_theme(
        week, region="USA", delay=1, universe="TOP1000", datasets_used={"option8"},
    )
    assert ok is True  # neutralization=None -> giữ tương thích ngược, không xét neut


def test_check_theme_compliance_khop():
    ok, reasons = check_theme_compliance(
        region="USA", delay=1, universe="TOP1000", neutralization="STATISTICAL",
        datasets_used={"option8"}, on_date=date(2026, 7, 9),
    )
    assert ok is True and reasons == []


def test_check_theme_compliance_lech_neut_va_universe():
    ok, reasons = check_theme_compliance(
        region="USA", delay=1, universe="TOP3000", neutralization="SUBINDUSTRY",
        datasets_used={"pv1"}, on_date=date(2026, 7, 9),
    )
    assert ok is False
    assert len(reasons) >= 2  # universe + neutralization (+ pv1)


def test_theme_tuan_hien_tai_2026_07_14_july26_2():
    """Theme 'USA/D1 Power Pool July`26 2' — nguyên văn announcement 2026-07-12 (đọc qua
    GET /users/self/messages bằng session consultant): 12 Jul–26 Jul'26, USA/D1/TOP1000,
    PV1 bị loại (trừ support fields), KHÔNG ràng buộc neutralization; thay vào đó phải pass
    'High Turnover returns ratio test' (chưa parse được -> nằm trong unparsed_constraints)."""
    week = theme_for_date(date(2026, 7, 14))
    assert week is not None
    assert week.name == "USA/D1 Power Pool July`26 2"
    assert week.end_date == date(2026, 7, 26)
    assert week.region == "USA"
    assert week.delay == 1
    assert week.universe == "TOP1000"
    assert week.datasets_excluded == ("pv1",)
    assert week.allowed_neutralizations == frozenset()  # theme này KHÔNG giới hạn neut
    assert "High Turnover" in (week.unparsed_constraints or "")


def test_theme_july26_2_khong_chan_neut_market():
    """Khác theme trước: July`26 2 không có 'neutralization in (...)' -> MARKET cũng qua."""
    week = theme_for_date(date(2026, 7, 20))
    ok, reasons = matches_theme(
        week, region="USA", delay=1, universe="TOP1000",
        datasets_used={"option8"}, neutralization="MARKET",
    )
    assert ok is True
    assert reasons == []


def test_theme_ngay_12_07_van_thuoc_theme_cu():
    """12/7 là ngày CHỒNG LẤN (theme cũ 29/6–12/7, theme mới 12/7–26/7): first-match trong
    CALENDAR giữ theme cũ cho 12/7 — chốt hành vi để khỏi lệch khi ai đó đổi thứ tự list."""
    week = theme_for_date(date(2026, 7, 12))
    assert week is not None
    assert week.allowed_neutralizations  # theme cũ CÓ ràng buộc neut -> nhận diện được


def test_check_theme_compliance_khong_co_theme_khong_chan():
    ok, reasons = check_theme_compliance(
        region="USA", delay=1, universe="TOP3000", neutralization="SUBINDUSTRY",
        datasets_used={"pv1"}, on_date=date(2026, 8, 15),
    )
    assert ok is True and reasons == []
