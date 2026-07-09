"""TDD cho repository.good_signals_for_combine — nguồn tín hiệu con tốt từ DB cho combiner."""

from __future__ import annotations

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.metrics_local import AlphaMetrics
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


def _repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    return MiniBrainRepository(sessionmaker(bind=engine, future=True, expire_on_commit=False))


def _metrics(fitness: float) -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.2, annual_return=0.1, turnover=0.2, max_drawdown=0.1,
        fitness=fitness, per_year_sharpe={}, weight_concentration=0.05,
    )


def _save(repo: MiniBrainRepository, expr: str, hsh: str, fitness: float, dates, pnl) -> None:
    expr_id = repo.upsert_expression(expr, hsh, depth=3, complexity=3, fields={"close"})
    eval_id = repo.record_evaluation(
        expr_id, '{"delay":1}', "w1", _metrics(fitness), 0.0, "passed", [], 1,
    )
    repo.save_pool_pnl(eval_id, dates, pnl)


def test_tra_ve_expr_pnl_fitness_sap_theo_fitness_giam():
    repo = _repo()
    dates = np.arange("2021-01-01", "2021-01-21", dtype="datetime64[D]")
    _save(repo, "rank(ts_delta(close, 5))", "h1", 0.8, dates, np.arange(20, dtype=np.float64))
    _save(repo, "rank(ts_delta(close, 20))", "h2", 1.5, dates, np.arange(20, dtype=np.float64) * 2)

    out = repo.good_signals_for_combine(limit=10)

    assert len(out) == 2
    # sắp theo fitness giảm dần -> h2 (1.5) trước h1 (0.8).
    assert out[0][0] == "rank(ts_delta(close, 20))"
    assert out[0][3] == 1.5
    assert out[1][3] == 0.8
    # PnL round-trip đúng.
    np.testing.assert_array_equal(out[0][2], np.arange(20, dtype=np.float64) * 2)
    assert out[0][1].dtype == np.dtype("datetime64[D]")


def test_ton_trong_limit():
    repo = _repo()
    dates = np.arange("2021-01-01", "2021-01-21", dtype="datetime64[D]")
    for i in range(5):
        _save(repo, f"rank(ts_delta(close, {i + 2}))", f"h{i}", float(i), dates,
              np.arange(20, dtype=np.float64) + i)
    out = repo.good_signals_for_combine(limit=3)
    assert len(out) == 3
    assert [r[3] for r in out] == [4.0, 3.0, 2.0]  # top-3 fitness


def test_db_rong_tra_list_rong():
    assert _repo().good_signals_for_combine() == []
