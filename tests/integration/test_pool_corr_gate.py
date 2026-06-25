"""Integration end-to-end KHÔNG mock: MiniBrainRepository (Phase 5, DB sqlite thật) ->
PoolCorrelation (Task 6.1) -> GateEvaluator.evaluate_with_pool (Task 6.2).

Chứng minh round-trip: pool rỗng cho self_corr=0.0 (pass); ghi alpha A THẬT vào pool
(qua upsert_expression + record_evaluation + save_pool_pnl, đúng FK evaluation_id) rồi
candidate B giống hệt A bị hard-fail self_corr; candidate C độc lập (rng khác) vẫn pass.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.gates import GateEvaluator
from src.backtest.metrics_local import AlphaMetrics
from src.backtest.pool_corr import PoolCorrelation
from src.local_types import Dates
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


def _dates(start: str, n: int) -> Dates:
    """Dãy ngày SORT tăng dần (datetime64[D]) bắt đầu từ `start`, độ dài `n`."""
    return (np.datetime64(start) + np.arange(n)).astype("datetime64[D]")


def _pnl(seed: int, n: int = 20) -> npt.NDArray[np.float64]:
    """PnL ngẫu nhiên (rng riêng theo seed) — seed khác nhau -> chuỗi độc lập."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=n)


def _good_metrics() -> AlphaMetrics:
    """AlphaMetrics pass mọi soft gate (tham khảo tests/unit/test_gates_pool_corr.py)."""
    return AlphaMetrics(
        sharpe=1.5, annual_return=0.20, turnover=0.30, max_drawdown=0.10,
        fitness=2.0, per_year_sharpe={2021: 1.2}, weight_concentration=0.05,
    )


def _make_repo() -> MiniBrainRepository:
    """Dựng DB sqlite in-memory THẬT (không mock) qua init_db, trả MiniBrainRepository."""
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    session_factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(session_factory)


def test_pool_round_trip_db_to_pool_corr_to_gate_evaluator() -> None:
    repo = _make_repo()
    dates = _dates("2021-01-01", 20)
    alpha_a_pnl = _pnl(seed=1)

    # 1) Pool rỗng (chưa ghi gì vào DB) -> self_corr=0.0 -> candidate A tự nó phải pass.
    pool_empty = PoolCorrelation(repo.load_pool())
    verdict_empty = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=alpha_a_pnl, candidate_dates=dates,
        pool_corr=pool_empty, depth=3, fields_ok=True,
    )
    assert verdict_empty.passed is True
    assert verdict_empty.hard_failures == []

    # 2) Alpha A pass -> ghi pool THẬT: tạo evaluation thật để có evaluation_id hợp lệ
    # (PoolPnlModel.evaluation_id là FK -> evaluations.id, không thể tự bịa số).
    expr_id = repo.upsert_expression("alpha_a", "hash_a", depth=3, complexity=3, fields={"close"})
    eval_id = repo.record_evaluation(
        expr_id, '{"delay":1}', "w1", _good_metrics(), 0.0, "passed", [], 1,
    )
    repo.save_pool_pnl(eval_id, dates, alpha_a_pnl)

    # 3) Candidate B giống hệt A (đọc lại pool từ DB) -> hard-fail self_corr (rho=1.0).
    pool_after = PoolCorrelation(repo.load_pool())
    verdict_b = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=alpha_a_pnl.copy(), candidate_dates=dates,
        pool_corr=pool_after, depth=3, fields_ok=True,
    )
    assert verdict_b.passed is False
    assert any("self_corr" in f for f in verdict_b.hard_failures)

    # 4) Candidate C độc lập (rng seed khác hẳn) -> vẫn pass với CÙNG pool_after.
    alpha_c_pnl = _pnl(seed=99)
    verdict_c = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=alpha_c_pnl, candidate_dates=dates,
        pool_corr=pool_after, depth=3, fields_ok=True,
    )
    assert verdict_c.passed is True
    assert verdict_c.hard_failures == []
