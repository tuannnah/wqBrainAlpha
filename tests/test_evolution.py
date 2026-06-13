"""Test các phép biến đổi GA và vòng tiến hóa (mock simulator)."""

from __future__ import annotations

import random

from src.generation.ast_utils import iter_leaves, parse_expression, to_expression
from src.optimization.evolution import GeneticOptimizer
from src.simulation.pre_filter import PreFilter


class FakeSimulator:
    """Ghi lại mọi expression được simulate; trả chính expression đó."""

    def __init__(self):
        self.calls = []

    def simulate(self, expr):
        self.calls.append(expr)
        return expr


def _expr_scorer(expr: str) -> float:
    # Ưu tiên expression có nhiều 'rank' — deterministic để test sắp xếp.
    return float(expr.count("rank"))


def _make_optimizer(fields, seeds, rng, max_simulations=None):
    pf = PreFilter(known_operators=None, known_fields=None)
    sim = FakeSimulator()

    def seed_factory():
        expr = rng.choice(seeds)
        return GeneticOptimizer.expr_to_node(expr)

    opt = GeneticOptimizer(
        simulator=sim,
        prefilter=pf,
        seed_factory=seed_factory,
        fields=fields,
        scorer=_expr_scorer,
        population_size=6,
        generations=3,
        elite_size=2,
        max_simulations=max_simulations,
        rng=rng,
    )
    return opt, sim


def test_max_simulations_gioi_han_so_lan_mo_phong():
    rng = random.Random(5)
    seeds = [f"rank(f{i})" for i in range(30)]  # nhiều biểu thức khác nhau
    opt, sim = _make_optimizer([f"f{i}" for i in range(30)], seeds, rng, max_simulations=4)
    opt.run()
    # Không bao giờ gọi simulate quá trần đã đặt.
    assert len(sim.calls) <= 4
    assert opt.simulations_used <= 4


def test_crossover_tao_cay_hop_le():
    rng = random.Random(1)
    a = parse_expression("rank(ts_delta(close, 5))")
    b = parse_expression("ts_mean(open, 20)")
    opt, _ = _make_optimizer(["close", "open"], ["rank(close)"], rng)
    c1, c2 = opt.crossover(a, b)
    # Render lại không lỗi và parse được.
    parse_expression(to_expression(c1))
    parse_expression(to_expression(c2))


def test_mutate_field_chon_field_hop_le():
    rng = random.Random(2)
    opt, _ = _make_optimizer(["alpha_f", "beta_f"], ["rank(close)"], rng)
    tree = parse_expression("rank(close)")
    mutated = opt.mutate_field(tree)
    leaf_values = [lf.value for lf in iter_leaves(mutated) if isinstance(lf.value, str)]
    assert leaf_values and all(v in {"alpha_f", "beta_f"} for v in leaf_values)


def test_mutate_param_doi_so():
    rng = random.Random(3)
    opt, _ = _make_optimizer(["close"], ["rank(close)"], rng)
    opt.param_choices = [999]
    tree = parse_expression("ts_delta(close, 5)")
    mutated = opt.mutate_param(tree)
    nums = [lf.value for lf in iter_leaves(mutated) if isinstance(lf.value, (int, float))]
    assert 999 in nums


def test_run_cache_khong_simulate_trung():
    rng = random.Random(7)
    seeds = ["rank(close)", "rank(rank(open))", "ts_delta(close, 5)", "rank(volume)"]
    opt, sim = _make_optimizer(["close", "open", "volume"], seeds, rng)
    best = opt.run()

    assert len(best) == opt.population_size
    # Cache: không expression nào bị simulate hai lần.
    assert len(sim.calls) == len(set(sim.calls))
    # Có lịch sử cho mỗi generation.
    assert len(opt.history) == opt.generations


def test_run_history_ghi_best_score():
    rng = random.Random(11)
    seeds = ["rank(close)", "rank(rank(close))"]
    opt, _ = _make_optimizer(["close"], seeds, rng)
    opt.run()
    assert all(s.best_score >= s.avg_score for s in opt.history if s.avg_score != float("-inf"))


def test_on_generation_goi_moi_the_he():
    rng = random.Random(13)
    seeds = ["rank(close)", "rank(rank(open))"]
    opt, _ = _make_optimizer(["close", "open"], seeds, rng)
    seen = []
    opt.run(on_generation=seen.append)
    # Callback nhận đúng các GenerationStats theo thứ tự, khớp với history.
    assert [s.generation for s in seen] == [s.generation for s in opt.history]
    assert all(isinstance(s.best_expression, str) for s in seen)


def test_on_simulation_goi_moi_lan_mo_phong_that():
    rng = random.Random(17)
    seeds = [f"rank(f{i})" for i in range(20)]
    opt, sim = _make_optimizer([f"f{i}" for i in range(20)], seeds, rng)
    sims = []
    opt.run(on_simulation=lambda n, expr, score: sims.append((n, expr, score)))
    # Số lần callback bằng số simulate thật; bộ đếm tăng dần 1,2,3...
    assert [n for n, _, _ in sims] == list(range(1, len(sims) + 1))
    assert len(sims) == len(sim.calls) == opt.simulations_used


def test_on_simulation_ton_trong_max_simulations():
    rng = random.Random(19)
    seeds = [f"rank(f{i})" for i in range(20)]
    opt, _ = _make_optimizer(
        [f"f{i}" for i in range(20)], seeds, rng, max_simulations=2
    )
    sims = []
    opt.run(on_simulation=lambda n, expr, score: sims.append(n))
    assert opt.simulations_used <= 2
    assert sims == list(range(1, opt.simulations_used + 1))
