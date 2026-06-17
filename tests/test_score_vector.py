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


def test_score_vector_nhan_simulation_result_object():
    from src.simulation.simulator import SimulationResult

    r = SimulationResult(
        expression="rank(close)", status="passed",
        sharpe=1.5, fitness=1.2, turnover=0.3, drawdown=0.1,
    )
    v = score_vector(r)
    assert 0.0 <= v.total <= 1.0
