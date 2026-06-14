"""Test chọn-lọc alpha bằng bộ lọc local (không có metric backtest).

Quy trình: cửa cứng (PreFilter) -> điểm originality (so zoo) + complexity ->
khử trùng nội bộ -> quota đa dạng theo họ -> xếp hạng.
"""

from __future__ import annotations

from src.generation.local_select import (
    Candidate,
    local_score,
    originality_score,
    select_alphas,
)


def _cand(expr: str, family: str = "reversal") -> Candidate:
    return Candidate(family=family, expression=expr, hypothesis="h", rationale="r")


# ----------------------------------------------------------- originality
def test_originality_cao_khi_khac_zoo():
    """Biểu thức cấu trúc lạ -> originality gần 1."""
    zoo = ["rank(ts_delta(close, 1))"]
    la = originality_score("ts_regression(ebit, revenue, 60)", zoo)
    quen = originality_score("rank(ts_delta(close, 5))", zoo)
    assert la > quen


def test_originality_trong_khoang_0_1():
    zoo = ["rank(close)"]
    s = originality_score("rank(close)", zoo)
    assert 0.0 <= s <= 1.0


# ----------------------------------------------------------- local_score
def test_local_score_thuong_originality_phat_complexity():
    """Cùng originality, cây đơn giản hơn -> điểm cao hơn."""
    zoo = ["rank(ts_delta(close, 1))"]
    don_gian = local_score("rank(ebit)", zoo)
    phuc_tap = local_score("rank(ts_corr(ts_mean(ebit, 5), ts_delta(revenue, 10), 20))", zoo)
    # cây đơn giản, lạ -> điểm không thấp hơn cây phức tạp
    assert don_gian >= phuc_tap


def test_local_score_trong_khoang_0_1():
    s = local_score("rank(close)", ["rank(close)"])
    assert 0.0 <= s <= 1.0


# ----------------------------------------------------------- select_alphas
def test_select_loai_bo_cu_phap_sai():
    """Ứng viên parse lỗi / operator-field lạ bị cửa cứng loại."""
    cands = [
        _cand("rank(close)"),
        _cand("bad ))("),                      # parse lỗi
        _cand("kgvkhongton(close)"),           # operator lạ
    ]
    selected = select_alphas(
        cands,
        zoo=["rank(ts_delta(open, 1))"],
        known_operators={"rank"},
        known_fields={"close"},
    )
    exprs = [c.expression for c in selected]
    assert "rank(close)" in exprs
    assert "bad ))(" not in exprs
    assert "kgvkhongton(close)" not in exprs


def test_select_khu_trung_noi_bo():
    """Hai ứng viên trùng cấu trúc gần như y hệt -> chỉ giữ 1."""
    cands = [
        _cand("rank(ts_delta(close, 5))"),
        _cand("rank(ts_delta(volume, 10))"),   # cùng canon rank(ts_delta(F,N))
    ]
    selected = select_alphas(
        cands,
        zoo=[],
        known_operators={"rank", "ts_delta"},
        known_fields={"close", "volume"},
        dedup_threshold=0.85,
    )
    assert len(selected) == 1


def test_select_quota_da_dang_theo_ho():
    """Giới hạn mỗi họ -> không để 1 họ chiếm hết output."""
    cands = (
        [_cand(f"rank(ts_mean(close, {n}))", "momentum") for n in (5, 10, 20, 30, 40)]
        + [_cand("rank(ebit)", "value")]
    )
    selected = select_alphas(
        cands,
        zoo=[],
        known_operators={"rank", "ts_mean"},
        known_fields={"close", "ebit"},
        per_family_quota=2,
        dedup_threshold=0.99,   # không khử trùng để test quota
    )
    momentum = [c for c in selected if c.family == "momentum"]
    assert len(momentum) <= 2


def test_select_sap_xep_giam_theo_diem():
    """Kết quả trả về sắp xếp giảm dần theo điểm local."""
    cands = [
        _cand("rank(close)"),
        _cand("ts_regression(ebit, revenue, 60)", "value"),
    ]
    selected = select_alphas(
        cands,
        zoo=["rank(close)"],
        known_operators={"rank", "ts_regression"},
        known_fields={"close", "ebit", "revenue"},
        dedup_threshold=0.99,
        per_family_quota=10,
    )
    scores = [c.score for c in selected]
    assert scores == sorted(scores, reverse=True)


def test_select_gan_diem_va_ly_do_vao_candidate():
    """Mỗi ứng viên được chọn có .score và .reasons để log."""
    cands = [_cand("rank(close)")]
    selected = select_alphas(
        cands,
        zoo=["rank(ts_delta(open, 1))"],
        known_operators={"rank"},
        known_fields={"close"},
    )
    assert len(selected) == 1
    c = selected[0]
    assert isinstance(c.score, float)
    assert c.originality is not None
    assert c.complexity is not None
