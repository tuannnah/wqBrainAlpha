"""Test sinh ứng viên alpha theo các họ kinh điển (Claude research -> FASTEXPR).

Mỗi họ định nghĩa các khung công thức, expand qua field x cửa sổ x neutralization
để ra nhiều biến thể. Test kiểm tra: số lượng đủ lớn, trường đầy đủ, biểu thức
parse được, và các họ mong đợi đều có mặt.
"""

from __future__ import annotations

from src.generation.ast_utils import parse_expression
from src.generation.families import EXPECTED_FAMILIES, generate_candidates


def test_sinh_so_luong_lon():
    """Sinh số lượng lớn ứng viên thô để lọc (yêu cầu user: sinh lớn)."""
    cands = generate_candidates()
    assert len(cands) >= 150


def test_moi_ung_vien_du_truong():
    """Mỗi ứng viên có family/expression/hypothesis/rationale không rỗng."""
    for c in generate_candidates():
        assert c.family
        assert c.expression
        assert c.hypothesis
        assert c.rationale


def test_moi_bieu_thuc_parse_duoc():
    """Biểu thức sinh ra phải parse được (cú pháp FASTEXPR hợp lệ)."""
    for c in generate_candidates():
        parse_expression(c.expression)  # ném ValueError nếu sai cú pháp


def test_du_cac_ho_kinh_dien():
    """Tất cả họ kinh điển mong đợi đều xuất hiện."""
    families = {c.family for c in generate_candidates()}
    for fam in EXPECTED_FAMILIES:
        assert fam in families, f"thiếu họ {fam}"


def test_khong_trung_lap_y_het():
    """Không có hai ứng viên trùng biểu thức y hệt."""
    exprs = [c.expression for c in generate_candidates()]
    assert len(exprs) == len(set(exprs))
