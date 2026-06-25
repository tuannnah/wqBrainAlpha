"""Test selection.py: dominates đúng hướng tối ưu (sharpe max, penalty min),
fast_non_dominated_sort phân front đúng, crowding_distance giữ biên=inf, nsga2_select giữ
đúng số lượng và ưu tiên front tốt + đa dạng."""

from __future__ import annotations

import numpy as np

from src.gp.fitness_vec import FitnessVector
from src.gp.individual import Individual
from src.gp.selection import crowding_distance, dominates, fast_non_dominated_sort, nsga2_select
from src.lang.ast import Field


def _fv(sharpe=1.0, per_year=0.5, turn=0.0, complex_p=0.0, pool=0.0, pop=0.0) -> FitnessVector:
    return FitnessVector(
        sharpe_deflated=sharpe, per_year_min_sharpe=per_year, turnover_penalty=turn,
        complexity_penalty=complex_p, pool_corr_penalty=pool, pop_corr_penalty=pop,
    )


def _ind(fv: FitnessVector) -> Individual:
    ind = Individual(expr=Field("close"))
    ind.fitness = fv
    return ind


def test_dominates_higher_sharpe_lower_penalties_wins():
    better = _fv(sharpe=2.0, turn=0.0)
    worse = _fv(sharpe=1.0, turn=0.1)
    assert dominates(better, worse) is True
    assert dominates(worse, better) is False


def test_dominates_false_when_tradeoff_no_domination():
    a = _fv(sharpe=2.0, turn=0.2)  # sharpe cao hơn nhưng turnover penalty cao hơn
    b = _fv(sharpe=1.0, turn=0.0)
    assert dominates(a, b) is False
    assert dominates(b, a) is False


def test_fast_non_dominated_sort_front_zero_is_non_dominated():
    pop = [_ind(_fv(sharpe=2.0)), _ind(_fv(sharpe=1.0)), _ind(_fv(sharpe=0.5))]
    fronts = fast_non_dominated_sort(pop)
    assert pop[0] in fronts[0]  # sharpe cao nhất, mọi thứ khác bằng nhau -> không bị dominate


def test_fast_non_dominated_sort_covers_all_individuals():
    pop = [_ind(_fv(sharpe=s)) for s in [2.0, 1.5, 1.0, 0.5]]
    fronts = fast_non_dominated_sort(pop)
    total = sum(len(f) for f in fronts)
    assert total == len(pop)


def test_crowding_distance_boundary_individuals_are_infinite():
    front = [_ind(_fv(sharpe=s)) for s in [0.0, 1.0, 2.0, 3.0]]
    dist = crowding_distance(front)
    sharpes = sorted(front, key=lambda i: i.fitness.sharpe_deflated)
    assert dist[id(sharpes[0])] == float("inf")
    assert dist[id(sharpes[-1])] == float("inf")


def test_nsga2_select_returns_exact_count():
    rng = np.random.default_rng(0)
    pop = [_ind(_fv(sharpe=s, turn=abs(s - 1))) for s in np.linspace(0, 3, 12)]
    survivors = nsga2_select(pop, n_survivors=5, rng=rng)
    assert len(survivors) == 5


def test_nsga2_select_prefers_better_front_over_worse():
    rng = np.random.default_rng(1)
    dominant = _ind(_fv(sharpe=5.0, turn=0.0))
    dominated = _ind(_fv(sharpe=0.1, turn=0.5))
    survivors = nsga2_select([dominant, dominated], n_survivors=1, rng=rng)
    assert survivors == [dominant]


def test_nsga2_select_is_deterministic_for_same_seed():
    pop = [_ind(_fv(sharpe=s, turn=abs(s - 1))) for s in np.linspace(0, 3, 10)]
    s1 = nsga2_select(pop, n_survivors=4, rng=np.random.default_rng(7))
    s2 = nsga2_select(pop, n_survivors=4, rng=np.random.default_rng(7))
    assert [id(x) for x in s1] == [id(x) for x in s2]
