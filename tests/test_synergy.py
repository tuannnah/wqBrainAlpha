"""Test SynergyScorer (Thành phần 1) — hàm mục tiêu pool-aware kiểu AlphaGen.

Yêu cầu hành vi:
  - Sim lỗi (status='error') -> -inf (vá bug 0.12 điểm cho rác).
  - Alpha chạy được -> base(score_vector) * originality^beta.
  - Alpha 'passed' được thêm vào pool -> ứng viên sau giống nó bị phạt độ độc đáo.
  - Alpha 'failed' (có metric thật, điểm thấp) -> hữu hạn, KHÔNG thêm vào pool.
"""

from __future__ import annotations

import math

from src.decorrelation.zoo import ReferenceZoo
from src.scoring.synergy import SynergyScorer
from src.simulation.simulator import SimulationResult


def _result(expr, status="passed", sharpe=2.0, fitness=1.5, turnover=0.3, drawdown=0.05):
    return SimulationResult(
        expression=expr, alpha_id="a1", status=status,
        sharpe=sharpe, fitness=fitness, turnover=turnover, drawdown=drawdown,
    )


def test_sim_loi_tra_am_vo_cuc():
    scorer = SynergyScorer(zoo=ReferenceZoo())
    res = SimulationResult(expression="zscore(dead_field)", status="error")
    assert scorer(res) == float("-inf")


def test_alpha_tot_diem_duong_va_huu_han():
    scorer = SynergyScorer(zoo=ReferenceZoo())  # zoo rỗng -> originality=1.0
    res = _result("group_neutralize(rank(implied_volatility_mean_30), sector)")
    val = scorer(res)
    assert val > 0
    assert math.isfinite(val)


def test_passed_them_vao_pool_va_phat_ung_vien_giong():
    scorer = SynergyScorer(zoo=ReferenceZoo(), beta=1.0)
    expr = "group_neutralize(rank(implied_volatility_mean_30), sector)"
    first = scorer(_result(expr))           # zoo rỗng -> originality 1.0
    second = scorer(_result(expr))          # đã có trong pool -> originality thấp
    assert second < first


def test_failed_khong_them_vao_pool():
    scorer = SynergyScorer(zoo=ReferenceZoo())
    expr = "group_neutralize(rank(scl12_buzz), industry)"
    scorer(_result(expr, status="failed", sharpe=0.4, fitness=0.3))
    assert len(scorer.zoo) == 0


def test_beta_cao_ep_doc_dao_manh_hon():
    # Pool có expr_a; expr_b chia sẻ phần lõi ts_mean(scl12_buzz, N) nhưng đổi
    # wrapper -> tương đồng một phần (originality ∈ (0,1)), nên beta mới có tác dụng.
    expr_a = "group_neutralize(rank(ts_mean(scl12_buzz, 20)), sector)"
    expr_b = "zscore(ts_mean(scl12_buzz, 5))"
    low = SynergyScorer(zoo=ReferenceZoo([expr_a]), beta=2.0)(_result(expr_b))
    high = SynergyScorer(zoo=ReferenceZoo([expr_a]), beta=0.5)(_result(expr_b))
    assert low < high  # beta lớn -> phạt nặng hơn khi kém độc đáo
