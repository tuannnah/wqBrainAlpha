"""Test lịch Power Pool Theme — dữ liệu thủ công từ bài 'Current month Power Pool Themes'
(support.worldquantbrain.com), xác nhận 2026-07-02. Xem docstring
src/scoring/power_pool_theme.py để biết cách cập nhật khi sang tháng/theme mới."""

from __future__ import annotations

from datetime import date

from src.scoring.power_pool_theme import (
    JUNE_JULY_2026_CALENDAR,
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
