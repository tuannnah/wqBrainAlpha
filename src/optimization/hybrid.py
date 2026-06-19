"""Engine hybrid: LLM seed quần thể -> GA tiến hóa -> mỗi K thế hệ LLM refine top
alpha rồi bơm biến thể vào vòng. Chạy vô hạn đến khi LLM hết token (chỉ tắt phần
LLM, GA vẫn chạy) hoặc Ctrl+C.

Tái dùng: LLMAlphaGenerator (seed), AlphaRefiner (refine theo chiều yếu),
ReferenceZoo (khử tương quan biến thể), GeneticOptimizer (tìm kiếm).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from loguru import logger

from src.generation.ast_utils import Node, to_expression
from src.llm.hypothesis import Hypothesis
from src.llm.translator import AlphaCandidate
from src.optimization.evolution import GeneticOptimizer
from src.scoring.metrics import normalize
from src.scoring.scorer import score as default_score
from src.scoring.vector import score_vector, weakest_dimension


@dataclass
class HybridEngine:
    simulator: object            # .simulate(expr, settings=None) -> result
    prefilter: object            # .check(expr) -> (ok, reason)
    fields: list[str]
    llm_generator: object        # .generate_ideas(n), .generate(idea, n)
    refiner: object               # .refine(candidate, metrics, weak_dim) -> AlphaCandidate | None
    zoo: object                   # .originality(expr) -> float, .add(expr)
    template_generator: object = None  # fallback seed: .generate(count)
    scorer: object = default_score
    inject_every: int = 3
    refine_top: int = 2
    seed_ideas: int = 5
    per_idea: int = 2
    originality_min: float = 0.4
    population_size: int = 30
    generations: int | None = None
    max_simulations: int | None = None
    simulation_settings: dict | None = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        self._results: dict[str, object] = {}   # expr -> raw simulate result
        self._llm_disabled = False

    # --------------------------------------------------------------- seed pool
    def _seed_pool(self) -> list[str]:
        pool: list[str] = []
        try:
            ideas = self.llm_generator.generate_ideas(self.seed_ideas)
            for idea in ideas:
                pool.extend(self.llm_generator.generate(idea, self.per_idea))
        except Exception as exc:  # 402 / lỗi LLM -> tắt LLM, dùng fallback
            logger.warning("LLM seed lỗi ({}) — tắt LLM-in-loop, dùng fallback.", exc)
            self._llm_disabled = True
        # Khử trùng giữ thứ tự.
        pool = list(dict.fromkeys(p for p in pool if p))
        if not pool and self.template_generator is not None:
            pool = list(self.template_generator.generate(self.population_size))
        if not pool:
            pool = [f"rank({self.fields[0]})"] if self.fields else ["rank(close)"]
        return pool

    # ------------------------------------------------------------ inject hook
    def _build_inject(self):
        def inject(scored: list[tuple[Node, float]]) -> list[Node]:
            if self._llm_disabled:
                return []
            out: list[Node] = []
            for node, _score in scored[: self.refine_top]:
                expr = to_expression(node)
                result = self._results.get(expr)
                if result is None:
                    continue
                metrics = normalize(result)
                weak = weakest_dimension(score_vector(result))
                candidate = AlphaCandidate(
                    hypothesis=Hypothesis(), description="", expression=expr
                )
                try:
                    refined = self.refiner.refine(candidate, metrics, weak)
                except Exception as exc:  # 402 / lỗi LLM -> tắt phần LLM, GA chạy tiếp
                    logger.warning("LLM refine lỗi ({}) — tắt LLM-in-loop.", exc)
                    self._llm_disabled = True
                    return out
                if refined is None or not refined.expression:
                    continue
                new_expr = refined.expression
                ok, _reason = self.prefilter.check(new_expr)
                if not ok:
                    continue
                if self.zoo.originality(new_expr) < self.originality_min:
                    continue
                self.zoo.add(new_expr)
                out.append(GeneticOptimizer.expr_to_node(new_expr))
                logger.info("Bơm biến thể LLM vào quần thể: {} (chiều yếu={})", new_expr, weak)
            return out

        return inject

    # --------------------------------------------------------------------- run
    def run(self, on_generation=None, on_simulation=None, on_inject=None) -> list[Node]:
        pool = self._seed_pool()

        # Bọc simulator để bắt raw result theo expression (phục vụ inject).
        original_simulate = self.simulator.simulate

        def simulate_capture(expr, **kwargs):
            res = original_simulate(expr, **kwargs)
            self._results[expr] = res
            return res

        self.simulator.simulate = simulate_capture

        def seed_factory():
            return GeneticOptimizer.expr_to_node(self.rng.choice(pool))

        opt = GeneticOptimizer(
            simulator=self.simulator, prefilter=self.prefilter, seed_factory=seed_factory,
            fields=self.fields, scorer=self.scorer,
            population_size=self.population_size, generations=self.generations,
            max_simulations=self.max_simulations,
            simulation_settings=self.simulation_settings,
            inject=self._build_inject(), inject_every=self.inject_every,
            rng=self.rng,
        )
        try:
            best = opt.run(on_generation=on_generation, on_simulation=on_simulation)
        finally:
            self.simulator.simulate = original_simulate
        self.simulations_used = opt.simulations_used
        self.history = opt.history
        return best
