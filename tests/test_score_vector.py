"""Test ScoreVector đa chiều + chọn chiều yếu nhất (GĐ2: T2.8, T2.11)."""

from __future__ import annotations

from src.scoring.vector import ScoreVector, score_vector, weakest_dimension


def test_score_vector_chuan_hoa_trong_khoang_0_1():
    m = {"sharpe": 2.0, "fitness": 1.5, "turnover": 0.3, "drawdown": 0.0}
    v = score_vector(m)
    assert isinstance(v, ScoreVector)
    for name, val in v.dimensions().items():
        assert 0.0 <= val <= 1.0, name
    # Tất cả chiều đều ở mức cao -> total cao.
    assert v.total > 0.8


def test_score_vector_drawdown_cao_giam_diem_chieu_do():
    good = score_vector({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.3, "drawdown": 0.0})
    bad = score_vector({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.3, "drawdown": 0.25})
    assert bad.drawdown_fit < good.drawdown_fit
    assert bad.total < good.total


def test_score_vector_turnover_lech_target_giam_diem():
    on_target = score_vector({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.3, "drawdown": 0.1})
    off_target = score_vector({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.9, "drawdown": 0.1})
    assert off_target.turnover_fit < on_target.turnover_fit


def test_weakest_dimension_chon_chieu_thap_nhat():
    # turnover lệch xa nhất -> chiều yếu nhất là turnover_fit.
    v = score_vector({"sharpe": 1.8, "fitness": 1.4, "turnover": 0.95, "drawdown": 0.02})
    assert weakest_dimension(v) == "turnover_fit"


def test_weakest_dimension_sharpe_thap():
    v = score_vector({"sharpe": 0.1, "fitness": 1.4, "turnover": 0.3, "drawdown": 0.02})
    assert weakest_dimension(v) == "sharpe"


def test_weakest_dimension_restrict_chi_xet_chieu_chan():
    """restrict -> chỉ xét các chiều bị chặn, dù chiều khác thấp hơn tuyệt đối."""
    # sharpe thấp nhất tuyệt đối, nhưng chỉ fitness bị chặn -> phải trả fitness.
    v = score_vector({"sharpe": 0.1, "fitness": 0.5, "turnover": 0.3, "drawdown": 0.02})
    assert weakest_dimension(v, restrict={"fitness"}) == "fitness"


def test_weakest_dimension_restrict_rong_thi_xet_tat_ca():
    """restrict rỗng/None (alpha đã pass) -> hành vi cũ: chiều yếu nhất tuyệt đối."""
    v = score_vector({"sharpe": 0.1, "fitness": 1.4, "turnover": 0.3, "drawdown": 0.02})
    assert weakest_dimension(v, restrict=set()) == "sharpe"
    assert weakest_dimension(v, restrict=None) == "sharpe"


def test_blocking_dimensions_map_dung_chieu_fail():
    from src.scoring.filter import blocking_dimensions

    # fitness <= 1.0 và turnover ngoài khoảng -> 2 chiều bị chặn.
    blocked = blocking_dimensions(
        {"sharpe": 1.5, "fitness": 0.8, "turnover": 0.95, "drawdown": 0.05}
    )
    assert blocked == {"fitness", "turnover_fit"}
    # alpha đạt mọi ngưỡng -> rỗng.
    assert blocking_dimensions(
        {"sharpe": 1.5, "fitness": 1.4, "turnover": 0.3, "drawdown": 0.05}
    ) == set()


# ---------------------------------------- (1) pool-correlation thành chiều hạng nhất
def test_score_vector_khong_corr_thi_pool_fit_mac_dinh_1():
    """Không truyền pool_corr -> pool_fit = 1.0 (trực giao hoàn toàn, không phạt)."""
    v = score_vector({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.3, "drawdown": 0.1})
    assert v.pool_fit == 1.0
    assert "pool_fit" in v.dimensions()


def test_score_vector_corr_cao_giam_pool_fit_va_total():
    """Correlation với pool cao -> pool_fit thấp -> total giảm (corr vào objective)."""
    m = {"sharpe": 1.5, "fitness": 1.2, "turnover": 0.3, "drawdown": 0.1}
    base = score_vector(m)
    crowded = score_vector(m, pool_corr=0.65)
    assert crowded.pool_fit < base.pool_fit
    assert crowded.total < base.total


def test_with_pool_corr_cap_nhat_pool_fit_va_total_giu_chieu_khac():
    """with_pool_corr: cập nhật pool_fit + total cho vector đã tính, giữ nguyên chiều khác."""
    from src.scoring.vector import with_pool_corr

    v = score_vector({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.3, "drawdown": 0.1})
    v2 = with_pool_corr(v, 0.70)  # >= CORR_LIMIT -> pool_fit = 0
    assert v2.pool_fit == 0.0
    assert v2.total < v.total
    assert v2.sharpe == v.sharpe and v2.fitness == v.fitness


def test_weakest_dimension_nham_pool_fit_khi_crowded():
    """restrict = {pool_fit} (corr là thứ duy nhất đang chặn) -> nhắm pool_fit."""
    v = score_vector(
        {"sharpe": 1.8, "fitness": 1.4, "turnover": 0.3, "drawdown": 0.02}, pool_corr=0.68
    )
    assert weakest_dimension(v, restrict={"pool_fit"}) == "pool_fit"


def test_blocking_dimensions_them_pool_fit_khi_corr_vuot_nguong():
    from src.scoring.filter import blocking_dimensions

    m = {"sharpe": 1.5, "fitness": 1.4, "turnover": 0.3, "drawdown": 0.05}
    # metrics đạt hết, nhưng corr vượt ngưỡng -> pool_fit là chiều chặn.
    assert blocking_dimensions(m, pool_corr=0.80) == {"pool_fit"}
    # corr dưới ngưỡng -> không chặn.
    assert blocking_dimensions(m, pool_corr=0.50) == set()
    # không truyền corr -> tương thích ngược.
    assert blocking_dimensions(m) == set()


# ----------------------------------------------- (3) regime robustness thành chiều
def test_score_vector_khong_regime_thi_mac_dinh_1():
    v = score_vector({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.3, "drawdown": 0.1})
    assert v.regime_fit == 1.0
    assert "regime_fit" in v.dimensions()


def test_score_vector_regime_yeu_giam_total():
    m = {"sharpe": 1.5, "fitness": 1.2, "turnover": 0.3, "drawdown": 0.1}
    base = score_vector(m)
    fragile = score_vector(m, regime=0.2)
    assert fragile.regime_fit == 0.2
    assert fragile.total < base.total


def test_with_regime_fit_giu_pool_fit():
    from src.scoring.vector import with_regime_fit

    v = score_vector({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.3, "drawdown": 0.1}, pool_corr=0.5)
    v2 = with_regime_fit(v, 0.3)
    assert v2.regime_fit == 0.3
    assert v2.pool_fit == v.pool_fit  # giữ nguyên chiều pool
    assert v2.total < v.total


def test_score_vector_nhan_simulation_result_object():
    from src.simulation.simulator import SimulationResult

    r = SimulationResult(
        expression="rank(close)", status="passed",
        sharpe=1.5, fitness=1.2, turnover=0.3, drawdown=0.1,
    )
    v = score_vector(r)
    assert 0.0 <= v.total <= 1.0
