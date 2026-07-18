"""FRONTIER_HYPOTHESES — hypothesis 4 phần CẤU TRÚC cho từng category frontier seed.

Bối cảnh (log 2026-07-18): mọi alpha sinh từ đường sim-thẳng/near-miss có
`alphas.hypothesis = '{}'` (rỗng) → `select_power_pool_candidates` không dựng được
mô tả Idea/Rationale ≥100 ký tự → ứng viên khớp theme (LLdLVX0a Sharpe 1.08) vẫn bị
skip "thiếu mô tả" và khối ⭐ cuối phiên luôn 0 ready. Hypothesis 4 phần vốn ĐÃ có ở
comment từng nhóm trong frontier_seeds.py — chuyển thành dữ liệu cấu trúc để selector
tự dựng mô tả (mô tả gửi Brain phải tiếng Anh)."""

from __future__ import annotations

from src.generation.frontier_seeds import (
    FRONTIER_CATEGORY_BY_FIELD,
    FRONTIER_HYPOTHESES,
    frontier_hypothesis,
)
from src.scoring.power_pool import (
    build_power_pool_description,
    is_valid_power_pool_description,
)


def test_frontier_hypotheses_phu_kin_moi_category():
    """Mỗi category frontier phải có hypothesis cấu trúc — thiếu 1 category là lớp alpha
    của category đó vĩnh viễn không PP-ready tự động."""
    assert set(FRONTIER_HYPOTHESES) == set(FRONTIER_CATEGORY_BY_FIELD.values())


def test_frontier_hypotheses_dung_duoc_mo_ta_hop_le():
    """build_power_pool_description từ mỗi hypothesis phải đạt chuẩn docs (>=100 ký tự)."""
    for cat, hyp in FRONTIER_HYPOTHESES.items():
        desc = build_power_pool_description(hyp)
        assert is_valid_power_pool_description(desc), f"category {cat} mô tả quá ngắn"
        # Mô tả gửi Brain là 3 phần Idea/Rationale chuẩn.
        assert desc.startswith("Idea: ")
        assert "Rationale for data used:" in desc


def test_frontier_hypothesis_theo_bieu_thuc():
    """frontier_hypothesis(expr) trả hypothesis của category chứa field trong expr;
    None với biểu thức không dùng field frontier."""
    hyp = frontier_hypothesis("ts_rank(multiply(-1, ts_mean(firm_vol_imbalance, 5)), 66)")
    assert hyp is FRONTIER_HYPOTHESES["option_flow"]
    assert frontier_hypothesis("rank(ts_delta(close, 5))") is None
