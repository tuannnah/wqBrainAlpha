"""NSGA-II (Deb et al. 2002) trên FitnessVector 6 chiều — correlation-aware multi-
objective selection (B13/R4): ngăn quần thể sụp vào "ngàn clone tương quan cao" bằng cách
giữ Pareto front + crowding distance (ưu tiên đa dạng) thay vì chỉ sắp theo 1 số Sharpe.
Hướng tối ưu: sharpe_deflated/per_year_min_sharpe MAXIMIZE; 4 penalty còn lại MINIMIZE
(xem fitness_vec.py).
"""

from __future__ import annotations

import numpy as np

from src.gp.fitness_vec import FitnessVector
from src.gp.individual import Individual

_MAXIMIZE_FIELDS = ("sharpe_deflated", "per_year_min_sharpe")
_ALL_FIELDS = (
    "sharpe_deflated", "per_year_min_sharpe", "turnover_penalty",
    "complexity_penalty", "pool_corr_penalty", "pop_corr_penalty",
)


def _as_minimize_vector(fv: FitnessVector) -> tuple[float, ...]:
    """6 số, tất cả theo hướng 'thấp hơn = tốt hơn' (âm hóa 2 chiều maximize)."""
    values: list[float] = [
        -fv.sharpe_deflated,
        -fv.per_year_min_sharpe,
        fv.turnover_penalty,
        fv.complexity_penalty,
        fv.pool_corr_penalty,
        fv.pop_corr_penalty,
    ]
    return tuple(values)


def dominates(a: FitnessVector, b: FitnessVector) -> bool:
    va, vb = _as_minimize_vector(a), _as_minimize_vector(b)
    not_worse_anywhere = all(x <= y for x, y in zip(va, vb))
    better_somewhere = any(x < y for x, y in zip(va, vb))
    return not_worse_anywhere and better_somewhere


def fast_non_dominated_sort(individuals: list[Individual]) -> list[list[Individual]]:
    n = len(individuals)
    dominated_count = [0] * n
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for i in range(n):
        fi = individuals[i].fitness
        assert fi is not None, "fast_non_dominated_sort yêu cầu mọi Individual đã eval"
        for j in range(n):
            if i == j:
                continue
            fj = individuals[j].fitness
            assert fj is not None
            if dominates(fi, fj):
                dominates_list[i].append(j)
            elif dominates(fj, fi):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front: list[int] = []
        for i in fronts[k]:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    fronts.pop()  # front rỗng cuối cùng (điều kiện dừng while)
    return [[individuals[i] for i in front] for front in fronts]


def _objective(ind: Individual, name: str) -> float:
    """Giá trị chiều `name` theo hướng minimize (âm hóa 2 chiều maximize)."""
    fv = ind.fitness
    assert fv is not None
    sign = -1.0 if name in _MAXIMIZE_FIELDS else 1.0
    value: float
    if name == "sharpe_deflated":
        value = fv.sharpe_deflated
    elif name == "per_year_min_sharpe":
        value = fv.per_year_min_sharpe
    elif name == "turnover_penalty":
        value = fv.turnover_penalty
    elif name == "complexity_penalty":
        value = fv.complexity_penalty
    elif name == "pool_corr_penalty":
        value = fv.pool_corr_penalty
    else:  # pop_corr_penalty
        value = fv.pop_corr_penalty
    return sign * value


def crowding_distance(front: list[Individual]) -> dict[int, float]:
    distances: dict[int, float] = {id(ind): 0.0 for ind in front}
    if len(front) <= 2:
        for ind in front:
            distances[id(ind)] = float("inf")
        return distances

    for name in _ALL_FIELDS:
        ordered = sorted(front, key=lambda ind: _objective(ind, name))
        values = [_objective(ind, name) for ind in ordered]
        span = values[-1] - values[0]
        distances[id(ordered[0])] = float("inf")
        distances[id(ordered[-1])] = float("inf")
        if span == 0:
            continue
        for i in range(1, len(ordered) - 1):
            distances[id(ordered[i])] += (values[i + 1] - values[i - 1]) / span

    return distances


def nsga2_select(
    individuals: list[Individual], n_survivors: int, rng: np.random.Generator,
) -> list[Individual]:
    fronts = fast_non_dominated_sort(individuals)
    survivors: list[Individual] = []

    for front in fronts:
        if len(survivors) + len(front) <= n_survivors:
            survivors.extend(front)
            continue

        remaining = n_survivors - len(survivors)
        if remaining <= 0:
            break
        distances = crowding_distance(front)
        # tie-break ngẫu nhiên (xác định theo rng) trước khi sort ổn định theo distance
        order = list(range(len(front)))
        rng.shuffle(order)
        shuffled = [front[i] for i in order]
        shuffled.sort(key=lambda ind: distances[id(ind)], reverse=True)
        survivors.extend(shuffled[:remaining])
        break

    return survivors
