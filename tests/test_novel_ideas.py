"""Test 10 alpha "mới mẻ" dựa trên dataset ít người khai thác (T-novel).

User phản hồi: các công thức PV/fundamental kinh điển trùng nhiều -> correlation
cao -> bị reject. 10 ý tưởng này chủ ý dùng option-implied vol, news novelty,
social buzz, analyst revision (sentiment1), supply-chain graph — nguồn cho
correlation thấp hơn. Test kiểm: đúng 10, parse được, field đã xác minh tồn tại,
không trùng, đủ giả thuyết + lý giải.
"""

from __future__ import annotations

from src.generation.ast_utils import Leaf, iter_leaves, parse_expression
from src.generation.novel_ideas import (
    VERIFIED_FIELDS,
    NOVEL_ALPHAS,
    fields_in,
)

# group hợp lệ (không phải field) — bỏ qua khi kiểm field tồn tại.
_GROUPS = {"market", "sector", "industry", "subindustry", "country", "exchange"}


def test_dung_10_alpha():
    assert len(NOVEL_ALPHAS) == 10


def test_khong_trung_bieu_thuc():
    exprs = [c.expression for c in NOVEL_ALPHAS]
    assert len(exprs) == len(set(exprs))


def test_moi_bieu_thuc_parse_duoc():
    for c in NOVEL_ALPHAS:
        parse_expression(c.expression)  # ném ValueError nếu sai cú pháp


def test_du_gia_thuyet_va_ly_giai():
    for c in NOVEL_ALPHAS:
        assert c.family
        assert len(c.hypothesis) > 10
        assert len(c.rationale) > 10


def test_field_da_xac_minh_ton_tai():
    """Mọi field dùng trong 10 alpha phải nằm trong tập đã xác minh tồn tại DB."""
    for c in NOVEL_ALPHAS:
        for f in fields_in(c.expression):
            if f in _GROUPS:
                continue
            assert f in VERIFIED_FIELDS, f"field chưa xác minh: {f} trong {c.expression}"


def test_khong_dung_pv_fundamental_co_ban():
    """Đảm bảo "mới mẻ": không alpha nào CHỈ dựa trên PV/fundamental kinh điển."""
    pv_basic = {"close", "open", "high", "low", "volume", "vwap", "returns",
                "cap", "adv20", "ebit", "ebitda", "revenue", "equity", "eps"}
    for c in NOVEL_ALPHAS:
        used = {f for f in fields_in(c.expression) if f not in _GROUPS}
        alt = used - pv_basic
        assert alt, f"alpha chỉ dùng PV/fundamental cơ bản: {c.expression}"


def test_fields_in_loai_so_va_lay_field():
    """Helper fields_in: chỉ trả field (bỏ số)."""
    fs = fields_in("rank(implied_volatility_mean_30 - historical_volatility_30)")
    assert "implied_volatility_mean_30" in fs
    assert "historical_volatility_30" in fs
    assert "30" not in fs
