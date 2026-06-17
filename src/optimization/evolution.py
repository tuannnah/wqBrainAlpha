"""Genetic Algorithm tiến hóa cây AST của alpha.

Cá thể = một cây Node. Fitness = score từ simulation (cache theo expression để
không simulate trùng). Dùng GA tree tùy biến cho dễ kiểm soát/test; `deap` có
trong requirements cho các thử nghiệm mở rộng.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from loguru import logger

from src.generation.ast_utils import (
    BINARY_OPS,
    Leaf,
    Node,
    all_subtrees,
    iter_leaves,
    node_count,
    parse_expression,
    to_expression,
    tree_depth,
)
from src.scoring.scorer import score as default_score

NEG_INF = float("-inf")

DEFAULT_OPERATORS_BY_ARITY = {
    1: ["rank", "ts_rank", "zscore", "sign", "abs"],
    2: ["ts_mean", "ts_delta", "ts_std_dev", "ts_sum", "ts_max", "ts_min", "ts_zscore"],
}

DEFAULT_PARAM_CHOICES = [5, 10, 20, 40, 60, 120]
WRAP_GROUPS = ["market", "sector", "industry", "subindustry"]


@dataclass
class GenerationStats:
    generation: int
    best_score: float
    avg_score: float
    best_expression: str


@dataclass
class GeneticOptimizer:
    simulator: object
    prefilter: object
    seed_factory: object  # callable() -> Node
    fields: list[str]
    scorer: object = default_score
    operators_by_arity: dict = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_OPERATORS_BY_ARITY.items()})
    param_choices: list = field(default_factory=lambda: list(DEFAULT_PARAM_CHOICES))
    population_size: int = 30
    generations: int = 10
    crossover_rate: float = 0.4
    mutation_rate: float = 0.4
    elite_size: int = 2
    tournament_size: int = 3
    max_depth: int = 6
    max_nodes: int = 30
    max_simulations: int | None = None  # trần số lần simulate thật (None = không giới hạn)
    simulation_settings: dict | None = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        self._cache: dict[str, float] = {}
        self.history: list[GenerationStats] = []
        self._sim_count = 0
        self._on_simulation = None  # callback(n, expr, score) mỗi lần simulate thật

    @property
    def simulations_used(self) -> int:
        return self._sim_count

    def _budget_exhausted(self) -> bool:
        return self.max_simulations is not None and self._sim_count >= self.max_simulations

    # ----------------------------------------------------------- evaluation
    def evaluate(self, ind: Node) -> float:
        expr = to_expression(ind)
        if expr in self._cache:
            return self._cache[expr]
        if tree_depth(ind) > self.max_depth or node_count(ind) > self.max_nodes:
            self._cache[expr] = NEG_INF
            return NEG_INF
        ok, _ = self.prefilter.check(expr)
        if not ok:
            self._cache[expr] = NEG_INF
            return NEG_INF
        if self._budget_exhausted():
            # Hết ngân sách simulate — không gọi WQ thêm, coi như chưa đánh giá.
            return NEG_INF
        if self.simulation_settings is None:
            result = self.simulator.simulate(expr)
        else:
            result = self.simulator.simulate(expr, settings=self.simulation_settings)
        self._sim_count += 1
        value = self.scorer(result)
        self._cache[expr] = value
        if self._on_simulation is not None:
            self._on_simulation(self._sim_count, expr, value)
        return value

    # ----------------------------------------------------- genetic operators
    def crossover(self, a: Node, b: Node) -> tuple[Node, Node]:
        ca, cb = a.copy(), b.copy()
        nodes_a = [n for n in all_subtrees(ca) if isinstance(n, Node)]
        nodes_b = [n for n in all_subtrees(cb) if isinstance(n, Node)]
        if not nodes_a or not nodes_b:
            return ca, cb
        pa = self.rng.choice(nodes_a)
        pb = self.rng.choice(nodes_b)
        if pa.children and pb.children:
            ia = self.rng.randrange(len(pa.children))
            ib = self.rng.randrange(len(pb.children))
            pa.children[ia], pb.children[ib] = pb.children[ib], pa.children[ia]
        return ca, cb

    def mutate_field(self, ind: Node) -> Node:
        clone = ind.copy()
        leaves = [lf for lf in iter_leaves(clone) if isinstance(lf.value, str)]
        if leaves:
            self.rng.choice(leaves).value = self.rng.choice(self.fields)
        return clone

    def mutate_operator(self, ind: Node) -> Node:
        clone = ind.copy()
        funcs = [
            n
            for n in all_subtrees(clone)
            if isinstance(n, Node) and n.op not in BINARY_OPS and n.op != "neg"
        ]
        if funcs:
            target = self.rng.choice(funcs)
            arity = len(target.children)
            candidates = [
                o for o in self.operators_by_arity.get(arity, []) if o != target.op
            ]
            if candidates:
                target.op = self.rng.choice(candidates)
        return clone

    def mutate_param(self, ind: Node) -> Node:
        clone = ind.copy()
        nums = [lf for lf in iter_leaves(clone) if isinstance(lf.value, (int, float))]
        if nums:
            self.rng.choice(nums).value = self.rng.choice(self.param_choices)
        return clone

    def mutate_wrap(self, ind: Node) -> Node:
        clone = ind.copy()
        choice = self.rng.random()
        if choice < 0.34:
            return Node("rank", [clone])
        if choice < 0.67:
            return Node("group_neutralize", [clone, Leaf(self.rng.choice(WRAP_GROUPS))])
        return Node("neg", [clone])

    def mutate(self, ind: Node) -> Node:
        op = self.rng.choice(
            [self.mutate_field, self.mutate_operator, self.mutate_param, self.mutate_wrap]
        )
        return op(ind)

    # --------------------------------------------------------------- selection
    def _tournament(self, scored: list[tuple[Node, float]]) -> Node:
        contenders = self.rng.sample(scored, min(self.tournament_size, len(scored)))
        return max(contenders, key=lambda x: x[1])[0]

    # --------------------------------------------------------------------- run
    def _seed_population(self) -> list[Node]:
        pop = []
        for _ in range(self.population_size):
            pop.append(self.seed_factory())
        return pop

    def run(self, on_generation=None, on_simulation=None) -> list[Node]:
        """Chạy tiến hóa.

        on_generation(stats): gọi sau khi tổng kết mỗi thế hệ.
        on_simulation(n, expr, score): gọi mỗi lần simulate thật (qua evaluate).
        """
        self._on_simulation = on_simulation
        population = self._seed_population()

        for gen in range(self.generations):
            scored = [(ind, self.evaluate(ind)) for ind in population]
            scored.sort(key=lambda x: x[1], reverse=True)

            valid = [s for _, s in scored if s != NEG_INF]
            best = scored[0][1]
            avg = sum(valid) / len(valid) if valid else NEG_INF
            stats = GenerationStats(gen, best, avg, to_expression(scored[0][0]))
            self.history.append(stats)
            logger.info(
                "Gen {}: best={:.4f} avg={:.4f} expr={}",
                gen,
                best,
                avg,
                stats.best_expression,
            )
            if on_generation is not None:
                on_generation(stats)

            if self._budget_exhausted():
                logger.info(
                    "Đã đạt giới hạn {} simulation — dừng tiến hóa.", self.max_simulations
                )
                break

            elites = [ind.copy() for ind, _ in scored[: self.elite_size]]
            new_pop = elites
            while len(new_pop) < self.population_size:
                r = self.rng.random()
                if r < self.crossover_rate and len(scored) >= 2:
                    child, _ = self.crossover(self._tournament(scored), self._tournament(scored))
                elif r < self.crossover_rate + self.mutation_rate:
                    child = self.mutate(self._tournament(scored))
                else:
                    child = self.seed_factory()
                new_pop.append(child)
            population = new_pop

        final = [(ind, self.evaluate(ind)) for ind in population]
        final.sort(key=lambda x: x[1], reverse=True)
        return [ind for ind, _ in final]

    @staticmethod
    def expr_to_node(expr: str) -> Node:
        node = parse_expression(expr)
        return node if isinstance(node, Node) else Node("rank", [node])
