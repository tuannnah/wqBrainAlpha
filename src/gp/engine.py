"""GPEngine — vòng lặp tiến hóa MiniBrain ghép 6 building block Phase 7 với Phase 2/3/4/6
(Evaluator/Backtester/MetricsCalculator/GateEvaluator/PoolCorrelation) + persist mọi
kết quả qua MiniBrainRepository (Phase 5).

Stage separation (B5): tìm kiếm BARE SIGNAL CORE; neut/decay/trunc/scale/delay được áp
ngoài qua PortfolioConfig truyền vào constructor, KHÔNG bọc vào ``Individual.expr``.

Determinism (R8): mọi randomness đi qua ``np.random.default_rng(seed)`` inject; cùng seed +
cùng config phải cho cùng quần thể cuối. Không dùng ``np.random`` toàn cục.

Dependency rule (B1): module này được phép import lang/engine/backtest/storage/operators_local
nhưng KHÔNG import ``src.llm`` (seed LLM lấy qua ``all_seed_cores`` với dependency truyền vào).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src.backtest.backtester import Backtester, BacktestResult
from src.backtest.config import PortfolioConfig
from src.backtest.gates import GateEvaluator
from src.backtest.metrics_local import AlphaMetrics, MetricsCalculator
from src.backtest.pool_corr import PoolCorrelation
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData
from src.engine.evaluator import EvalContext, Evaluator
from src.engine.subexpr_cache import SubexprCache
from src.gp.fitness_vec import FitnessVector, from_metrics
from src.gp.individual import Individual
from src.gp.init import init_population
from src.gp.seeds import all_seed_cores
from src.gp.selection import nsga2_select
from src.gp.variation import (
    crossover,
    dedup_population,
    hoist_mutation,
    point_mutation,
    subtree_mutation,
)
from src.lang.registry import OperatorRegistry
from src.lang.visitors import (
    CanonicalHasher,
    ComplexityVisitor,
    DepthVisitor,
    FieldCollector,
    Serializer,
)
from src.storage.repository import MiniBrainRepository


@dataclass(frozen=True, slots=True)
class GPRunResult:
    """Kết quả một lần chạy GPEngine: quần thể cuối + cá thể tốt nhất + thống kê + seed.

    ``best_by_sharpe`` là cá thể có ``sharpe_deflated`` cao nhất trong quần thể cuối (chỉ
    xét cá thể đã đánh giá thành công); ``None`` nếu không có cá thể nào hợp lệ.
    """

    generations_run: int
    final_population: list[Individual]
    best_by_sharpe: Individual | None
    n_evaluated: int
    n_passed: int
    seed: int


class GPEngine:
    """Vòng lặp GP: init seed → đánh giá → biến đổi → chọn lọc → đánh giá → ... → kết quả.

    Mọi tham số tinh chỉnh nhận qua keyword-only để gọi rõ ràng; ``data``/``repo``/``config``/
    ``registry`` là phụ thuộc bắt buộc (positional). ``max_depth`` phải <= MAX_DEPTH cấu hình
    trong ``config/thresholds.py`` để không sinh cây vượt trần gate.
    """

    def __init__(
        self,
        data: MarketData,
        repo: MiniBrainRepository,
        config: PortfolioConfig,
        registry: OperatorRegistry,
        *,
        pop_size: int = 50,
        n_generations: int = 5,
        max_depth: int = 7,
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.3,
        seed: int = 42,
        seed_offset: int = 0,
        data_window: str = "default",
        with_llm_seeds: bool = False,
        n_jobs: int = 1,
    ) -> None:
        self.data = data
        self.repo = repo
        self.config = config
        self.registry = registry
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.seed_offset = seed_offset
        self.data_window = data_window
        self.with_llm_seeds = with_llm_seeds
        self.n_jobs = n_jobs

    def _evaluate_individual(
        self, ind: Individual, pool_corr: PoolCorrelation,
    ) -> tuple[FitnessVector | None, str, list[str], BacktestResult | None]:
        """Đánh giá một cá thể: eval signal → build danh mục → backtest → metrics → gate.

        Trả ``(fitness, status, fail_reasons, bt)``; việc persist do caller (``run``) đảm
        nhiệm. ``status`` là một trong: ``'passed'`` | ``'failed_gate'`` | ``'invalid'`` |
        ``'error'``. ``fail_reasons`` LUÔN là ``list[str]`` (rỗng khi pass). Quy ước bắt lỗi:

        - Eval AST hỏng (operator thiếu impl, kiểu sai) → ``'error'`` (fitness None, bt None).
        - Backtest/metrics ném exception → ``'error'``.
        - Gate hard-fail (depth/fields/self_corr/concentration) → ``'failed_gate'`` (vẫn có
          bt + fitness để cá thể còn tham gia chọn lọc, không bị loại khỏi quần thể).
        - Pass mọi hard gate → ``'passed'``.

        Lưu ý: ``Evaluator`` hiện gói lỗi parse-time vào exception runtime nên không có nhánh
        ``'invalid'`` riêng ở đây; ``'invalid'`` để dành cho cây sai cấu trúc registry (nếu
        tầng eval phân biệt sau này). ``SubexprCache`` tạo MỚI mỗi cá thể — tránh chia sẻ
        state cache giữa các lần eval khác nhau (B6)."""
        try:
            ctx = EvalContext(data=self.data, registry=self.registry, cache=SubexprCache())
            evaluator = Evaluator(ctx)
            signal = evaluator.evaluate(ind.expr)
        except Exception as exc:  # noqa: BLE001 — engine phải sống sót mọi lỗi cây
            return None, "error", [f"eval: {type(exc).__name__}: {exc}"], None

        try:
            weights = PortfolioBuilder().build(signal, self.config, self.data)
            bt = Backtester().run(weights, self.data)
        except Exception as exc:  # noqa: BLE001
            return None, "error", [f"backtest: {type(exc).__name__}: {exc}"], None

        try:
            metrics = MetricsCalculator().compute(bt, self.data)
        except Exception as exc:  # noqa: BLE001
            return None, "error", [f"metrics: {type(exc).__name__}: {exc}"], None

        depth = ind.expr.accept(DepthVisitor())
        fields = ind.expr.accept(FieldCollector())
        fields_ok = bool(fields) and fields.issubset(self.data.field_names())

        verdict = GateEvaluator().evaluate_with_pool(
            metrics, candidate_pnl=bt.daily_pnl, candidate_dates=self.data.dates,
            pool_corr=pool_corr, depth=depth, fields_ok=fields_ok,
        )
        # self_corr tính một lần (gate cũng tính nội bộ; ta cần lại cho fitness + persist).
        pool_rho, _worst_id = pool_corr.max_corr(bt.daily_pnl, self.data.dates)
        complexity = ind.expr.accept(ComplexityVisitor())
        # n_trials=1: chưa theo dõi số lần thử per-cá-thể nên không haircut deflation ở đây;
        # đa dạng quần thể đã do NSGA-II + pool_corr_penalty đảm nhiệm (xem fitness_vec).
        fv = from_metrics(
            metrics, complexity=complexity, pool_corr=pool_rho, pop_corr=0.0, n_trials=1,
        )

        if not verdict.passed:
            return fv, "failed_gate", list(verdict.hard_failures), bt
        return fv, "passed", [], bt

    def _config_json(self) -> str:
        """Khóa cấu hình stage cho cache/DB — ``sort_keys=True`` để canonical (Minor P5):
        cùng cấu hình PortfolioConfig phải sinh CÙNG chuỗi bất kể thứ tự key."""
        return json.dumps(
            {
                "neutralization": self.config.neutralization.name,
                "decay": self.config.decay,
                "truncation": self.config.truncation,
                "scale_book": self.config.scale_book,
                "delay": self.config.delay,
            },
            sort_keys=True,
        )

    def _persist(
        self,
        ind: Individual,
        status: str,
        fail_reasons: list[str],
        bt: BacktestResult | None,
        self_corr: float | None,
    ) -> None:
        """Upsert expression + ``record_evaluation`` (mọi outcome: pass/fail/seed — B11
        avoid-list) + ``save_pool_pnl`` khi pass. ``metrics`` tái lập từ ``bt`` cho trạng
        thái ``passed``/``failed_gate`` (gate đã chạy nên backtest hợp lệ); ``invalid``/
        ``error`` -> ``metrics=None`` (cột metric DB để trống)."""
        expr_string = ind.expr.accept(Serializer())
        canonical_hash = ind.expr.accept(CanonicalHasher())
        depth = ind.expr.accept(DepthVisitor())
        complexity = ind.expr.accept(ComplexityVisitor())
        fields = ind.expr.accept(FieldCollector())

        expr_id = self.repo.upsert_expression(
            expr_string, canonical_hash, depth, complexity, fields,
        )

        metrics_for_db: AlphaMetrics | None = None
        if bt is not None and status in {"passed", "failed_gate"}:
            metrics_for_db = MetricsCalculator().compute(bt, self.data)

        eval_id = self.repo.record_evaluation(
            expression_id=expr_id,
            config_json=self._config_json(),
            data_window=self.data_window,
            metrics=metrics_for_db,
            self_corr_max=self_corr,
            status=status,
            fail_reasons=fail_reasons,
            seed=self.seed,
        )

        if status == "passed" and bt is not None:
            self.repo.save_pool_pnl(eval_id, self.data.dates, bt.daily_pnl)

    def _evaluate_population(
        self, population: list[Individual], pool_corr: PoolCorrelation,
    ) -> tuple[int, int]:
        """Đánh giá + persist mọi cá thể CHƯA có fitness. Trả ``(n_evaluated, n_passed)``.
        Cá thể đã eval ở thế hệ trước (fitness != None, được NSGA-II giữ lại) bỏ qua."""
        n_evaluated = 0
        n_passed = 0
        for ind in population:
            if ind.fitness is not None:
                continue
            fv, status, reasons, bt = self._evaluate_individual(ind, pool_corr)
            self_corr: float | None = None
            if bt is not None:
                rho, _worst = pool_corr.max_corr(bt.daily_pnl, self.data.dates)
                self_corr = float(rho)
            self._persist(ind, status, reasons, bt, self_corr)
            ind.fitness = fv  # slots non-frozen — gán sau init (xem Individual Task 7.1)
            n_evaluated += 1
            if status == "passed":
                n_passed += 1
        return n_evaluated, n_passed

    def _make_offspring(
        self,
        evaluated: list[Individual],
        fields: tuple[str, ...],
        rng: np.random.Generator,
        generation: int,
    ) -> list[Individual]:
        """Sinh ``pop_size`` con từ các cá thể đã eval: crossover (lấy 2 cha) /
        mutation (point 0.4 / subtree 0.4 / hoist 0.2) / sao chép, theo
        ``crossover_rate``/``mutation_rate``. Con KHÔNG kế thừa fitness (=None)."""
        offspring: list[Individual] = []
        while len(offspring) < self.pop_size:
            u = rng.random()
            if u < self.crossover_rate and len(evaluated) >= 2:
                i, j = rng.choice(len(evaluated), size=2, replace=False)
                c1, c2 = crossover(
                    evaluated[int(i)].expr, evaluated[int(j)].expr, rng, self.max_depth,
                )
                offspring.append(Individual(expr=c1, generation=generation))
                if len(offspring) < self.pop_size:
                    offspring.append(Individual(expr=c2, generation=generation))
            elif u < self.crossover_rate + self.mutation_rate:
                parent = evaluated[int(rng.integers(0, len(evaluated)))]
                v = rng.random()
                if v < 0.4:
                    mutated = point_mutation(parent.expr, self.registry, rng, fields)
                elif v < 0.8:
                    mutated = subtree_mutation(
                        parent.expr, self.registry, rng, fields, self.max_depth,
                    )
                else:
                    mutated = hoist_mutation(parent.expr, rng, self.registry)
                offspring.append(Individual(expr=mutated, generation=generation))
            else:
                parent = evaluated[int(rng.integers(0, len(evaluated)))]
                offspring.append(Individual(expr=parent.expr, generation=generation))
        return offspring

    def run(self) -> GPRunResult:
        """Vòng lặp tiến hóa end-to-end (thuần Python, xác định theo ``seed``):

        1. ``rng = np.random.default_rng(seed)``; ``seeds = all_seed_cores(...)``.
        2. ``population = init_population(...)`` (ramped half-and-half + seeding).
        3. Lặp ``n_generations`` lần: đánh giá + persist toàn quần thể (pass/fail/seed),
           sinh offspring (crossover/mutation/sao chép), ``dedup_population`` theo
           canonical_hash, ``nsga2_select`` giữ ``pop_size`` cá thể (Pareto + crowding).
        4. Đánh giá thế hệ cuối (offspring chưa eval) → persist.
        5. Trả ``GPRunResult`` (quần thể cuối + best theo ``sharpe_deflated`` + thống kê).

        ``pool_corr`` được nạp lại từ DB ở đầu mỗi vòng (pool lớn dần khi có alpha pass)."""
        rng = np.random.default_rng(self.seed)
        fields = tuple(sorted(self.data.field_names()))
        seed_cores = all_seed_cores(with_llm=self.with_llm_seeds)
        population = init_population(
            registry=self.registry,
            rng=rng,
            population_size=self.pop_size,
            seed_cores=seed_cores,
            fields=fields,
            max_depth=self.max_depth,
            seed_offset=self.seed_offset,
        )

        total_eval = 0
        total_passed = 0
        for gen in range(self.n_generations):
            pool_corr = PoolCorrelation(pool=self.repo.load_pool())
            n_ev, n_pa = self._evaluate_population(population, pool_corr)
            total_eval += n_ev
            total_passed += n_pa

            evaluated = [i for i in population if i.fitness is not None]
            if not evaluated:
                break  # toàn quần thể error/invalid — không biến đổi được, dừng sớm
            offspring = self._make_offspring(evaluated, fields, rng, gen + 1)
            # Đánh giá offspring TRƯỚC chọn lọc (μ+λ): NSGA-II yêu cầu mọi cá thể có
            # fitness. pool_corr nạp lại vì pool có thể vừa lớn lên ở bước eval trên.
            pool_corr = PoolCorrelation(pool=self.repo.load_pool())
            n_ev, n_pa = self._evaluate_population(offspring, pool_corr)
            total_eval += n_ev
            total_passed += n_pa
            # Chỉ giữ cá thể có fitness để chọn lọc; loại error/invalid (đã persist fail).
            pool_for_select = [
                i for i in (population + offspring) if i.fitness is not None
            ]
            combined = dedup_population(pool_for_select, self.registry)
            population = nsga2_select(combined, self.pop_size, rng)

        # Đánh giá thế hệ cuối: offspring vừa được chọn nhưng chưa eval.
        pool_corr = PoolCorrelation(pool=self.repo.load_pool())
        n_ev, n_pa = self._evaluate_population(population, pool_corr)
        total_eval += n_ev
        total_passed += n_pa

        evaluated_final = [i for i in population if i.fitness is not None]
        best = max(
            evaluated_final,
            key=lambda i: i.fitness.sharpe_deflated,  # type: ignore[union-attr]
            default=None,
        )

        return GPRunResult(
            generations_run=self.n_generations,
            final_population=population,
            best_by_sharpe=best,
            n_evaluated=total_eval,
            n_passed=total_passed,
            seed=self.seed,
        )
