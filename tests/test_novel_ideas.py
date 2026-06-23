"""Test 10 alpha "mới mẻ" dựa trên dataset ít người khai thác (T-novel).

User phản hồi: các công thức PV/fundamental kinh điển trùng nhiều -> correlation
cao -> bị reject. 10 ý tưởng này chủ ý dùng option-implied vol, news novelty,
social buzz, analyst revision (sentiment1), supply-chain graph — nguồn cho
correlation thấp hơn. Test kiểm: đúng 10, parse được, field đã xác minh tồn tại,
không trùng, đủ giả thuyết + lý giải.
"""

from __future__ import annotations

from src.generation.novel_ideas import (
    VERIFIED_FIELDS,
    NOVEL_ALPHAS,
    fields_in,
)
from src.lang.parser import parse_expression

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


def test_moi_alpha_co_overrides_day_du_decay_truncation_neutralization():
    """Mỗi alpha phải đặt đủ decay/truncation/neutralization riêng."""
    for c in NOVEL_ALPHAS:
        assert c.overrides, f"thiếu overrides: {c.family}"
        assert "decay" in c.overrides, f"thiếu decay: {c.family}"
        assert "truncation" in c.overrides, f"thiếu truncation: {c.family}"
        assert "neutralization" in c.overrides, f"thiếu neutralization: {c.family}"


def test_neutralization_override_khop_group_trong_bieu_thuc():
    expected_by_group = {
        "market": "MARKET",
        "sector": "SECTOR",
        "industry": "INDUSTRY",
        "subindustry": "SUBINDUSTRY",
    }
    for c in NOVEL_ALPHAS:
        group = c.expression.rsplit(",", 1)[-1].rstrip(") ").strip().lower()
        assert c.overrides["neutralization"] == expected_by_group[group]


def test_decay_truncation_trong_khoang_hop_le():
    """decay ∈ [0,512] (số ngày), truncation ∈ (0, 0.5] theo WQ Brain."""
    for c in NOVEL_ALPHAS:
        d = c.overrides["decay"]
        t = c.overrides["truncation"]
        assert isinstance(d, int) and 0 <= d <= 512, f"decay sai: {c.family}={d}"
        assert 0.0 < t <= 0.5, f"truncation sai: {c.family}={t}"


def test_tin_hieu_su_kien_decay_cao_hon():
    """Tín hiệu sự kiện/news/social nhiễu phải có decay cao hơn tín hiệu vol bền."""
    by_family = {c.family: c.overrides["decay"] for c in NOVEL_ALPHAS}
    # news/social nhiễu mạnh -> cần dàn tín hiệu nhiều ngày hơn VRP (vol bền vững)
    assert by_family["news-novelty"] > by_family["vol-risk-premium"]
    assert by_family["social-attention"] > by_family["vol-risk-premium"]


def test_du_lieu_thua_truncation_chat_hon():
    """Dữ liệu thưa (graph/news) truncation chặt hơn dữ liệu option dày."""
    by_family = {c.family: c.overrides["truncation"] for c in NOVEL_ALPHAS}
    # graph pv13 thưa -> truncation nhỏ hơn (chặt hơn) option IV dày
    assert by_family["supply-chain-graph"] < by_family["vol-risk-premium"]
