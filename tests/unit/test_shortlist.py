"""Test build_shortlist: rank theo fitness giảm dần + decorrelate (loại candidate có
max|rho| với cái đã chọn vượt ngưỡng), pool-aware qua PoolCorrelation."""

from __future__ import annotations

import numpy as np

from src.backtest.metrics_local import AlphaMetrics
from src.backtest.pool_corr import PoolCorrelation
from src.pipeline.shortlist import ShortlistCandidate, build_shortlist


def _dates(start: str, n: int) -> np.ndarray:
    return (np.datetime64(start) + np.arange(n)).astype("datetime64[D]")


def _metrics(fitness: float) -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.0, annual_return=0.1, turnover=0.3, max_drawdown=0.05,
        fitness=fitness, per_year_sharpe={2021: 1.0}, weight_concentration=0.05,
    )


def test_ranks_by_fitness_descending_when_uncorrelated() -> None:
    dates = _dates("2021-01-01", 20)
    rng = np.random.default_rng(0)
    low = ShortlistCandidate("low", _metrics(0.5), rng.normal(size=20), dates)
    high = ShortlistCandidate("high", _metrics(2.0), rng.normal(size=20), dates)
    out = build_shortlist([low, high], top_k=2, max_corr=0.7)
    assert [c.expr for c in out] == ["high", "low"]


def test_decorrelate_drops_high_correlation_pair() -> None:
    dates = _dates("2021-01-01", 20)
    base = np.linspace(0.01, 0.20, 20)
    a = ShortlistCandidate("a_best", _metrics(2.0), base.copy(), dates)
    b = ShortlistCandidate("b_dup", _metrics(1.5), base.copy() * 2.0, dates)
    c = ShortlistCandidate("c_diff", _metrics(1.0), -base.copy(), dates)
    out = build_shortlist([a, b, c], top_k=3, max_corr=0.7)
    names = [cand.expr for cand in out]
    assert "a_best" in names
    assert "b_dup" not in names  # rho=+1.0 với a_best
    assert "c_diff" not in names  # |rho|=1.0 (rho=-1) với a_best
    assert len(out) == 1


def test_respects_top_k_limit() -> None:
    dates = _dates("2021-01-01", 20)
    rng = np.random.default_rng(1)
    cands = [
        ShortlistCandidate(f"x{i}", _metrics(float(i)), rng.normal(size=20), dates)
        for i in range(5)
    ]
    out = build_shortlist(cands, top_k=2, max_corr=0.99)
    assert len(out) == 2
    assert out[0].expr == "x4"
    assert out[1].expr == "x3"


def test_pool_aware_drops_candidate_correlated_with_existing_pool() -> None:
    dates = _dates("2021-01-01", 20)
    pool_pnl = np.linspace(0.01, 0.20, 20)
    pool_corr = PoolCorrelation(pool={1: (dates, pool_pnl.copy())})
    dup = ShortlistCandidate("dup_pool", _metrics(2.0), pool_pnl.copy() * 3.0, dates)
    fresh = ShortlistCandidate("fresh", _metrics(1.0), -pool_pnl.copy(), dates)
    out = build_shortlist([dup, fresh], top_k=2, max_corr=0.7, pool_corr=pool_corr)
    names = [c.expr for c in out]
    assert "dup_pool" not in names
    assert "fresh" not in names  # |rho|=1.0 với pool


def test_empty_candidates_returns_empty_list() -> None:
    assert build_shortlist([], top_k=5, max_corr=0.7) == []


def test_pairwise_abs_rho_returns_none_on_insufficient_overlap() -> None:
    from src.pipeline.shortlist import _pairwise_abs_rho
    d1 = _dates("2021-01-01", 1)
    p1 = np.array([0.01])
    assert _pairwise_abs_rho(p1, d1, p1, d1) is None  # <2 điểm chung


def test_pairwise_abs_rho_returns_none_on_zero_variance() -> None:
    from src.pipeline.shortlist import _pairwise_abs_rho
    d = _dates("2021-01-01", 5)
    const = np.ones(5)
    assert _pairwise_abs_rho(const, d, const, d) is None  # phương sai 0


def test_does_not_mutate_input_list() -> None:
    dates = _dates("2021-01-01", 10)
    cands = [
        ShortlistCandidate("a", _metrics(1.0), np.ones(10), dates),
        ShortlistCandidate("b", _metrics(2.0), np.ones(10) * -1, dates),
    ]
    original = [c.expr for c in cands]
    build_shortlist(cands, top_k=2, max_corr=0.99)
    assert [c.expr for c in cands] == original
