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


def test_moi_ung_vien_co_decay_override():
    """Mỗi ứng viên phải đặt decay riêng (núm điều khiển turnover) — không để mặc định 0 cứng."""
    for c in generate_candidates():
        assert "decay" in c.overrides, f"thiếu decay: {c.family} / {c.expression}"


def test_decay_trong_khoang_hop_le():
    """decay là số nguyên trong [0,512] ngày theo WQ Brain."""
    for c in generate_candidates():
        d = c.overrides["decay"]
        assert isinstance(d, int) and 0 <= d <= 512, f"decay sai: {c.family}={d}"


def test_bieu_thuc_co_decay_linear_thi_setting_decay_0():
    """Biểu thức đã có ts_decay_linear (mượt nội tại) thì setting decay = 0 để khỏi mượt kép."""
    for c in generate_candidates():
        if "ts_decay_linear" in c.expression:
            assert c.overrides["decay"] == 0, (
                f"mượt kép: {c.family} đã có ts_decay_linear nhưng setting decay={c.overrides['decay']}"
            )


def test_tin_hieu_nhieu_decay_cao_hon_tin_hieu_cham():
    """Tín hiệu nhiễu/spiky (volume) cần decay cao hơn tín hiệu cơ bản chậm (value)."""
    by_family: dict[str, int] = {}
    for c in generate_candidates():
        # lấy decay đại diện của họ từ biểu thức KHÔNG có mượt nội tại
        if "ts_decay_linear" not in c.expression:
            by_family.setdefault(c.family, c.overrides["decay"])
    assert by_family["volume"] > by_family["value"]
    assert by_family["reversal"] > 0  # tín hiệu đảo chiều thô phải có decay > 0
