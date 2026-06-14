"""Test định dạng log chi tiết từng alpha ra text (để user mang lên WQ Brain).

Mỗi alpha in: mã, họ, giả thuyết, lý giải, biểu thức FASTEXPR, setting đầy đủ,
điểm local + lý do. Test kiểm tra các phần bắt buộc đều xuất hiện và setting
khớp scope USA/TOP3000/delay=1.
"""

from __future__ import annotations

from src.generation.alpha_logger import DEFAULT_SETTINGS, format_alpha, format_report
from src.generation.local_select import Candidate


def _scored(expr: str, family: str = "reversal") -> Candidate:
    c = Candidate(family=family, expression=expr, hypothesis="giả thuyết X", rationale="lý giải Y")
    c.score = 0.73
    c.originality = 0.80
    c.complexity = 0.10
    c.reasons = ["originality=0.80", "complexity=0.10", "cú pháp/field/operator hợp lệ"]
    return c


def test_format_alpha_co_du_phan_bat_buoc():
    text = format_alpha(_scored("-rank(ts_delta(close, 5))"), index=1)
    # biểu thức
    assert "-rank(ts_delta(close, 5))" in text
    # họ + giả thuyết + lý giải
    assert "reversal" in text
    assert "giả thuyết X" in text
    assert "lý giải Y" in text
    # điểm
    assert "0.73" in text


def test_format_alpha_in_setting_day_du():
    text = format_alpha(_scored("rank(close)"), index=1)
    for key in ("region", "universe", "delay", "decay", "neutralization", "truncation"):
        assert key in text
    assert "USA" in text
    assert "TOP3000" in text


def test_setting_mac_dinh_khop_scope():
    assert DEFAULT_SETTINGS["region"] == "USA"
    assert DEFAULT_SETTINGS["universe"] == "TOP3000"
    assert DEFAULT_SETTINGS["delay"] == 1
    assert DEFAULT_SETTINGS["language"] == "FASTEXPR"


def test_neutralization_suy_ra_tu_bieu_thuc():
    """Biểu thức bọc group_neutralize(..., industry) -> setting neutralization=INDUSTRY."""
    text = format_alpha(_scored("group_neutralize(rank(close), industry)"), index=1)
    assert "INDUSTRY" in text


def test_format_report_dem_so_luong_va_danh_so():
    cands = [_scored("rank(close)"), _scored("-rank(volume)", "volume")]
    report = format_report(cands)
    # tiêu đề có tổng số
    assert "2" in report
    # đánh số từng alpha
    assert "#1" in report
    assert "#2" in report
    # cả hai biểu thức xuất hiện
    assert "rank(close)" in report
    assert "-rank(volume)" in report


def test_format_report_rong_van_co_tieu_de():
    report = format_report([])
    assert "0" in report
