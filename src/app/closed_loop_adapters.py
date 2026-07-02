"""Adapter nối thành phần thật vào ClosedLoop (Phase 2). Tầng composition: được phép import
src.gp/src.llm/src.pipeline/src.lang (khác src/pipeline vốn cấm src.llm/src.gp theo B1).

- RefinementLoopRefiner: bọc RefinementLoop.run_from_seed (4A) → IdeaOutcome.
- GPIdeaSource: bọc generate_many (Phase 8) với seed GPEngine tăng dần → nguồn ý tưởng."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pipeline.closed_loop import ClosedLoop

from src.gp.engine import GPEngine
from src.lang.parser import parse
from src.lang.visitors import CanonicalHasher
from src.pipeline.closed_loop import IdeaOutcome, QuotaExhausted
from src.pipeline.runner import generate_many
from src.pipeline.shortlist import ShortlistCandidate
from src.simulation.simulator import AuthExpiredError


class RefinementLoopRefiner:
    """Bọc RefinementLoop: refine+sim một core (qua run_from_seed) → IdeaOutcome cho ClosedLoop."""

    def __init__(self, loop: object) -> None:
        self.loop = loop

    def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome:
        try:
            # result là Any: loop: object, run_from_seed dùng type: ignore[attr-defined]
            result: Any = self.loop.run_from_seed(candidate.expr)  # type: ignore[attr-defined]
        except AuthExpiredError as exc:
            # Best-effort: session chết / hết quota Brain → báo ClosedLoop dừng gọn.
            # (Tinh chỉnh nhận diện quota-ngày chính xác sau lần chạy thật đầu tiên.)
            raise QuotaExhausted(str(exc)) from exc
        best = result.best_candidate
        expr: str = best.expression if best is not None else candidate.expr
        canonical_hash = CanonicalHasher().visit(parse(expr))
        m: dict[str, Any] = result.best_metrics or {}
        return IdeaOutcome(
            expr=expr, canonical_hash=canonical_hash,
            passed=bool(result.best_passed),
            wq_alpha_id=result.best_alpha_id,
            sharpe=m.get("sharpe"), fitness=m.get("fitness"), turnover=m.get("turnover"),
            self_corr=result.best_self_corr,
            sims_used=result.sims_used,
            stop_reason=result.stop_reason,
        )


class GPIdeaSource:
    """Nguồn ý tưởng cho ClosedLoop: mỗi next_batch() chạy GPEngine với seed MỚI (tăng dần để
    đa dạng) rồi rút short-list qua generate_many. Pool decorrelate lấy từ repo.load_pool()."""

    def __init__(
        self, data: object, repo: object, config: object, registry: object, *,
        pop_size: int = 30, n_generations: int = 3, base_seed: int = 42,
        top_k: int = 10, max_corr: float = 0.70,
    ) -> None:
        # Lưu dưới Any để forward vào GPEngine/generate_many mà không cần cast cứng
        self._data: Any = data
        self._repo: Any = repo
        self._config: Any = config
        self._registry: Any = registry
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.base_seed = base_seed
        self.top_k = top_k
        self.max_corr = max_corr
        self._batch = 0

    def next_batch(self) -> list[ShortlistCandidate]:
        seed = self.base_seed + self._batch
        self._batch += 1
        engine = GPEngine(
            data=self._data, repo=self._repo, config=self._config, registry=self._registry,
            pop_size=self.pop_size, n_generations=self.n_generations, seed=seed,
        )
        pool: Any = self._repo.load_pool() or None
        # GPEngine.run() -> GPRunResult; Protocol _RunsGP đòi _GPRunResultLike với
        # list[_GPIndividualLike] — list là invariant nên cast qua Any để truyền qua.
        engine_any: Any = engine
        return generate_many(
            gp_engine=engine_any, cfg=self._config, data=self._data,
            top_k=self.top_k, max_corr=self.max_corr, pool=pool,
        )


def build_closed_loop(
    *, data: object, repo: object, config: object, registry: object, loop: object,
    region: str = "USA", universe: str = "TOP3000",
    pop_size: int = 30, n_generations: int = 3, base_seed: int = 42,
    top_k: int = 10, max_corr: float = 0.70,
    calibrate_every: int = 10, rho_bar: float = 0.5, max_ideas: int | None = None,
) -> "ClosedLoop":
    """Ráp vòng kín: GPIdeaSource (sinh ý tưởng) + RefinementLoopRefiner (AI refine+sim qua
    `loop`) + CalibrationTracker (ρ) + ClosedLoop. `loop` là RefinementLoop đã dựng (đăng nhập
    + Simulator thật) do composition root (main.py) truyền vào."""
    from src.pipeline.closed_loop import CalibrationTracker, ClosedLoop

    idea_source = GPIdeaSource(
        data, repo, config, registry, pop_size=pop_size, n_generations=n_generations,
        base_seed=base_seed, top_k=top_k, max_corr=max_corr,
    )
    refiner = RefinementLoopRefiner(loop)
    tracker = CalibrationTracker(repo, every=calibrate_every, rho_bar=rho_bar)  # type: ignore[arg-type]
    return ClosedLoop(
        idea_source=idea_source, refiner=refiner, repo=repo,  # type: ignore[arg-type]
        region=region, universe=universe, max_ideas=max_ideas,
        calibration_tracker=tracker,
    )
