"""Pha 0 instrumentation: IdeaOutcome mang đủ trường chẩn đoán funnel.

Spec IMPROVEMENT_SPEC §3 Pha 0 yêu cầu mỗi ứng viên ghi được: chết ở stage nào
(stage_reached), vì check nào (fail_check), thuộc họ gì (family), depth biểu thức,
thời gian mỗi mốc (gen/backtest/sim), dedup_key, và local_sharpe tách khỏi brain.
"""

from __future__ import annotations

from src.pipeline.closed_loop import IdeaOutcome


def test_cac_truong_pha0_co_default_tuong_thich_nguoc():
    """IdeaOutcome cũ (chỉ trường bắt buộc) vẫn tạo được — trường Pha 0 có default."""
    o = IdeaOutcome(
        expr="close", canonical_hash="h", passed=False, wq_alpha_id=None,
        sharpe=None, fitness=None, turnover=None, self_corr=None,
        sims_used=0, stop_reason="local_floor",
    )
    assert o.stage_reached == ""
    assert o.fail_check == ""
    assert o.family == ""
    assert o.expr_depth is None
    assert o.gen_ms is None
    assert o.backtest_ms is None
    assert o.sim_ms is None
    assert o.dedup_key is None
    assert o.local_sharpe is None


def test_ghi_du_truong_chan_doan():
    o = IdeaOutcome(
        expr="rank(close)", canonical_hash="h", passed=True, wq_alpha_id="W",
        sharpe=1.5, fitness=1.1, turnover=0.3, self_corr=0.2, sims_used=1,
        stop_reason="local_tuned", stage_reached="passed", fail_check="",
        family="pv_reversal", expr_depth=4, gen_ms=12.0, backtest_ms=340.0,
        sim_ms=8000.0, dedup_key="dk1", local_sharpe=1.23,
    )
    assert o.stage_reached == "passed"
    assert o.family == "pv_reversal"
    assert o.expr_depth == 4
    assert o.local_sharpe == 1.23
    assert o.dedup_key == "dk1"
    # brain sharpe giữ nguyên trên trường sharpe (không đổi tên, tránh vỡ consumer)
    assert o.sharpe == 1.5
