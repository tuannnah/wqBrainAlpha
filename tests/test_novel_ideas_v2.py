"""Novel-ideas v2 — nâng cấp CẤU TRÚC theo docs (understanding-data / examples).

Bài học docs: "alpha tốt sống trong GAP/GATE/RESIDUAL, không phải LEVEL". v2 áp 3 khuôn
mẫu lên field ĐÃ VERIFY (không bịa field mới): residual bằng `vector_neut` (khử self-corr
trực tiếp — lever DUY NHẤT theo skill), gap zscore chuẩn hóa, và gate bằng `trade_when`.
Giữ nguyên hợp đồng 10 alpha gốc (`NOVEL_ALPHAS`); v2 là danh sách RIÊNG.
"""

from __future__ import annotations

from src.generation.novel_ideas import (
    NOVEL_ALPHAS,
    NOVEL_ALPHAS_V2,
    VERIFIED_FIELDS,
    all_novel_alphas,
    fields_in,
)
from src.lang.parser import parse_expression

_GROUPS = {"market", "sector", "industry", "subindustry", "country", "exchange"}


def test_v2_khong_rong_va_tach_biet_v1():
    assert len(NOVEL_ALPHAS_V2) >= 6
    v1 = {c.expression for c in NOVEL_ALPHAS}
    for c in NOVEL_ALPHAS_V2:
        assert c.expression not in v1  # v2 KHÔNG trùng v1


def test_all_novel_gop_ca_hai():
    assert len(all_novel_alphas()) == len(NOVEL_ALPHAS) + len(NOVEL_ALPHAS_V2)


def test_v2_parse_duoc():
    for c in NOVEL_ALPHAS_V2:
        parse_expression(c.expression)  # ValueError nếu sai cú pháp


def test_v2_field_da_xac_minh():
    for c in NOVEL_ALPHAS_V2:
        for f in fields_in(c.expression):
            if f in _GROUPS:
                continue
            assert f in VERIFIED_FIELDS, f"field chưa verify: {f} trong {c.expression}"


def test_v2_dung_cau_truc_bac_cao():
    """Mỗi v2 phải dùng ÍT NHẤT một khuôn mẫu bậc cao: vector_neut / trade_when / ts_zscore
    (gap chuẩn hóa) / ts_corr (bậc hai) — không phải rank(level) trần."""
    markers = ("vector_neut", "trade_when", "ts_zscore", "ts_corr")
    for c in NOVEL_ALPHAS_V2:
        assert any(m in c.expression for m in markers), f"v2 thiếu cấu trúc bậc cao: {c.expression}"


def test_v2_co_ban_vector_neut_de_khu_self_corr():
    """Ít nhất một nửa v2 dùng vector_neut — lever trực tiếp hạ self-correlation (gate #1)."""
    n = sum(1 for c in NOVEL_ALPHAS_V2 if "vector_neut" in c.expression)
    assert n >= len(NOVEL_ALPHAS_V2) // 2


def test_v2_du_gia_thuyet_va_ly_giai():
    for c in NOVEL_ALPHAS_V2:
        assert c.family
        assert len(c.hypothesis) > 10
        assert len(c.rationale) > 10


def test_v2_khong_dung_pv_fundamental_co_ban():
    pv_basic = {"close", "open", "high", "low", "volume", "vwap", "returns",
                "cap", "adv20", "ebit", "ebitda", "revenue", "equity", "eps"}
    for c in NOVEL_ALPHAS_V2:
        used = {f for f in fields_in(c.expression) if f not in _GROUPS}
        assert used - pv_basic, f"v2 chỉ dùng PV/fundamental cơ bản: {c.expression}"


def test_v2_overrides_day_du_va_hop_le():
    for c in NOVEL_ALPHAS_V2:
        assert "decay" in c.overrides and "truncation" in c.overrides
        assert "neutralization" in c.overrides
        d, t = c.overrides["decay"], c.overrides["truncation"]
        assert isinstance(d, int) and 0 <= d <= 512
        assert 0.0 < t <= 0.5


def test_v2_neutralization_khop_group():
    expected = {"market": "MARKET", "sector": "SECTOR", "industry": "INDUSTRY",
                "subindustry": "SUBINDUSTRY"}
    for c in NOVEL_ALPHAS_V2:
        group = c.expression.rsplit(",", 1)[-1].rstrip(") ").strip().lower()
        assert c.overrides["neutralization"] == expected[group]
