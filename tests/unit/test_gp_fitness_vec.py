"""Test FitnessVector: deflated_sharpe haircut theo n_trials, from_metrics map đúng
AlphaMetrics + corr penalty + turnover band, hướng tối ưu nhất quán (max sharpe/min năm
tệ nhất, min mọi penalty). T2.3: complexity_penalty = max(complexity/NORM, depth/MAX_DEPTH)
-- cây SÂU vẫn bị phạt nặng dù ÍT NODE (trước đây complexity_penalty chỉ đếm node, không
xét độ sâu -- một cây chuỗi sâu 7 tầng nhưng chỉ 7 node bị phạt NHẸ HƠN một cây rộng 45 node
nhưng chỉ sâu 2 tầng, sai lệch bản chất "cây sâu = dễ overfit/khó ghép combiner")."""

from __future__ import annotations

import math

import pytest

from config.thresholds import MAX_DEPTH, TURNOVER_BAND
from src.backtest.metrics_local import AlphaMetrics
from src.gp.fitness_vec import FitnessVector, deflated_sharpe, from_metrics


def _metrics(**overrides) -> AlphaMetrics:
    base = dict(
        sharpe=1.5, annual_return=0.20, turnover=0.30, max_drawdown=0.10,
        fitness=2.0, per_year_sharpe={2021: 1.2, 2022: 0.5}, weight_concentration=0.05,
    )
    base.update(overrides)
    return AlphaMetrics(**base)


def test_deflated_sharpe_no_haircut_for_single_trial():
    assert deflated_sharpe(1.5, n_trials=1) == pytest.approx(1.5)
    assert deflated_sharpe(1.5, n_trials=0) == pytest.approx(1.5)


def test_deflated_sharpe_haircut_grows_with_trials():
    d10 = deflated_sharpe(1.5, n_trials=10)
    d1000 = deflated_sharpe(1.5, n_trials=1000)
    assert d10 < 1.5
    assert d1000 < d10  # nhiều lần thử hơn -> haircut nặng hơn


def test_deflated_sharpe_matches_formula():
    sharpe, n = 2.0, 100
    expected = sharpe - math.sqrt(2 * math.log(n)) / math.sqrt(252)
    assert deflated_sharpe(sharpe, n) == pytest.approx(expected)


# --- Các test dưới đây KHÔNG quan tâm depth (chỉ test hành vi khác của from_metrics) ---
# dùng depth=1 (đóng góp 1/MAX_DEPTH ~ 0.143, luôn nhỏ hơn complexity/NORM=10/50=0.2 với
# complexity=10 dùng xuyên suốt) để giữ nguyên kỳ vọng cũ, KHÔNG đổi ý nghĩa test.


def test_from_metrics_per_year_min_sharpe_is_worst_year():
    fv = from_metrics(_metrics(), complexity=10, depth=1, pool_corr=0.1, pop_corr=0.05, n_trials=1)
    assert fv.per_year_min_sharpe == pytest.approx(0.5)


def test_from_metrics_empty_per_year_gives_zero():
    fv = from_metrics(
        _metrics(per_year_sharpe={}), complexity=10, depth=1, pool_corr=0.0, pop_corr=0.0,
        n_trials=1,
    )
    assert fv.per_year_min_sharpe == pytest.approx(0.0)


def test_from_metrics_turnover_inside_band_has_zero_penalty():
    mid = (TURNOVER_BAND[0] + TURNOVER_BAND[1]) / 2
    fv = from_metrics(
        _metrics(turnover=mid), complexity=10, depth=1, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.turnover_penalty == pytest.approx(0.0)


def test_from_metrics_turnover_below_band_penalized_by_distance():
    too_low = TURNOVER_BAND[0] - 0.05
    fv = from_metrics(
        _metrics(turnover=too_low), complexity=10, depth=1, pool_corr=0.0, pop_corr=0.0,
        n_trials=1,
    )
    assert fv.turnover_penalty == pytest.approx(0.05, abs=1e-9)


def test_from_metrics_turnover_above_band_penalized_by_distance():
    too_high = TURNOVER_BAND[1] + 0.10
    fv = from_metrics(
        _metrics(turnover=too_high), complexity=10, depth=1, pool_corr=0.0, pop_corr=0.0,
        n_trials=1,
    )
    assert fv.turnover_penalty == pytest.approx(0.10, abs=1e-9)


def test_from_metrics_passes_through_corr_penalties_unchanged():
    fv = from_metrics(_metrics(), complexity=10, depth=1, pool_corr=0.42, pop_corr=0.31, n_trials=1)
    assert fv.pool_corr_penalty == pytest.approx(0.42)
    assert fv.pop_corr_penalty == pytest.approx(0.31)


def test_from_metrics_complexity_penalty_scales_with_node_count():
    """Depth cố định NHỎ (1) ở cả 2 vế -> depth/MAX_DEPTH không đổi và luôn nhỏ hơn
    complexity/NORM ở cả 2 mức complexity -> so sánh vẫn thuần túy phản ánh node-count."""
    fv_small = from_metrics(
        _metrics(), complexity=10, depth=1, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    fv_large = from_metrics(
        _metrics(), complexity=100, depth=1, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv_large.complexity_penalty > fv_small.complexity_penalty


def test_fitness_vector_is_frozen_and_hashable():
    fv = from_metrics(_metrics(), complexity=10, depth=1, pool_corr=0.0, pop_corr=0.0, n_trials=1)
    assert isinstance(fv, FitnessVector)
    with pytest.raises(AttributeError):
        fv.sharpe_deflated = 99.0  # type: ignore[misc]


# --- T2.3: complexity_penalty = max(complexity/_COMPLEXITY_NORM, depth/MAX_DEPTH) ---


def test_from_metrics_complexity_penalty_dung_depth_khi_depth_ap_dao():
    """Cây RẤT ÍT node (complexity nhỏ) nhưng RẤT SÂU (depth=MAX_DEPTH): trước Task 2,
    complexity_penalty thuần đếm node sẽ cho điểm phạt gần 0 (sai lệch — cây sâu dễ overfit/
    khó ghép combiner PHẢI bị phạt nặng dù ít node). Công thức mới lấy max -> chiều depth
    ÁP ĐẢO, đúng bằng depth/MAX_DEPTH = 1.0."""
    fv = from_metrics(
        _metrics(), complexity=5, depth=MAX_DEPTH, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.complexity_penalty == pytest.approx(1.0)


def test_from_metrics_complexity_penalty_dung_complexity_khi_complexity_ap_dao():
    """Ngược lại: cây NÔNG (depth=1) nhưng RẤT RỘNG (complexity=40) -> chiều complexity áp
    đảo, đúng bằng complexity/_COMPLEXITY_NORM = 40/50 = 0.8 (depth/MAX_DEPTH=1/7~0.143 nhỏ
    hơn nhiều)."""
    fv = from_metrics(
        _metrics(), complexity=40, depth=1, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.complexity_penalty == pytest.approx(40 / 50)


def test_from_metrics_cay_sau_it_node_bi_phat_nang_hon_cay_nong_nhieu_node():
    """Ví dụ dominance đúng nghĩa bối cảnh brief: cây NÔNG NHIỀU NODE (depth=2, complexity=45)
    so với cây SÂU ÍT NODE (depth=MAX_DEPTH, complexity=5) -- trước Task 2 (complexity_penalty
    thuần node-count), cây sâu-ít-node sẽ có penalty THẤP HƠN (5/50=0.1 < 45/50=0.9), tức
    được GP ưu tiên giữ lại hơn dù sâu/khó ghép hơn -- ngược hẳn ý muốn. Công thức mới phải
    đảo lại: cây sâu-ít-node bị phạt NẶNG HƠN cây nông-nhiều-node."""
    cay_nong_nhieu_node = from_metrics(
        _metrics(), complexity=45, depth=2, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    cay_sau_it_node = from_metrics(
        _metrics(), complexity=5, depth=MAX_DEPTH, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert cay_sau_it_node.complexity_penalty > cay_nong_nhieu_node.complexity_penalty


def test_from_metrics_complexity_penalty_giu_nguyen_khi_ca_hai_chieu_thap():
    """Cây vừa ít node vừa nông (complexity=5, depth=1) -> penalty thấp ở cả 2 vế, max ra
    max(5/50, 1/7) = 1/7 (depth vẫn áp đảo dù cả 2 đều thấp vì _COMPLEXITY_NORM=50 lớn hơn
    nhiều MAX_DEPTH=7 theo tỉ lệ) -- chỉ khóa giá trị đúng công thức, không giả định thêm."""
    fv = from_metrics(
        _metrics(), complexity=5, depth=1, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.complexity_penalty == pytest.approx(max(5 / 50, 1 / MAX_DEPTH))
