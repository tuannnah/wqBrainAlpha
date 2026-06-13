"""Vòng lặp AI tham lam: giả thuyết → dịch → mô phỏng → tinh chỉnh chiều yếu (T2.14).

Greedy: luôn cải thiện alpha tốt nhất hiện có, nhắm chiều yếu nhất. Trần số
simulation cấu hình được; cache theo hash (cache hit không tính vào trần). Lưu
alpha pass vào DB (alpha zoo là view), ghi lại thất bại để tránh lặp.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from src.scoring.filter import passes as default_filter
from src.scoring.metrics import normalize
from src.scoring.vector import ScoreVector, score_vector, weakest_dimension


@dataclass
class LoopProgress:
    sims_used: int
    best_total: float
    phase: str          # hypothesis | seed | refine | done
    detail: str = ""


@dataclass
class LoopResult:
    best_candidate: object | None
    best_vector: ScoreVector | None
    history: list = field(default_factory=list)
    zoo_added: int = 0
    failures: list = field(default_factory=list)
    sims_used: int = 0


@dataclass
class _Eval:
    vector: ScoreVector
    metrics: dict
    alpha_id: str | None  # None khi cache hit (không lưu lại)
    passed: bool


class RefinementLoop:
    def __init__(
        self,
        hypothesis_gen,
        translator,
        refiner,
        simulator,
        prefilter,
        repo,
        region: str,
        universe: str,
        delay: int = 1,
        score_vector_fn=score_vector,
        hard_filter_fn=default_filter,
        max_simulations: int = 20,
        no_improve_patience: int = 3,
    ):
        self.hypothesis_gen = hypothesis_gen
        self.translator = translator
        self.refiner = refiner
        self.simulator = simulator
        self.prefilter = prefilter
        self.repo = repo
        self.region = region
        self.universe = universe
        self.delay = delay
        self.score_vector_fn = score_vector_fn
        self.hard_filter_fn = hard_filter_fn
        self.max_simulations = max_simulations
        self.no_improve_patience = no_improve_patience
        self.sims_used = 0
        self.zoo_added = 0

    # --------------------------------------------------------------- eval
    def _evaluate(self, candidate, parent_id: str | None) -> _Eval | None:
        expr = candidate.expression
        ok, reason = self.prefilter.check(expr)
        if not ok:
            self.repo.record_failure(expr, "syntax", reason, "llm")
            return None

        cached = self.repo.get_cached_simulation(expr)
        if cached is not None:
            vector = self.score_vector_fn(cached)
            return _Eval(vector, normalize(cached), None, cached.status == "passed")

        if self.sims_used >= self.max_simulations:
            return None  # hết trần sim, không gọi WQ thêm

        result = self.simulator.simulate(
            expr, settings={"region": self.region, "universe": self.universe, "delay": self.delay}
        )
        self.sims_used += 1
        vector = self.score_vector_fn(result)
        alpha_id = self.repo.save_alpha(
            expr,
            source="llm",
            hypothesis=candidate.hypothesis.to_dict(),
            description=candidate.description,
            parent_id=parent_id,
        )
        self.repo.save_simulation(
            result, region=self.region, universe=self.universe,
            score=vector.total, alpha_id=alpha_id,
        )
        ok_hard, reasons = self.hard_filter_fn(result)
        passed = result.status == "passed" and ok_hard
        if passed:
            self.zoo_added += 1
        else:
            self.repo.record_failure(expr, "low_score", "; ".join(reasons) or result.status, "llm")
        return _Eval(vector, normalize(result), alpha_id, passed)

    # ---------------------------------------------------------------- run
    def run(self, research_direction: str, on_progress=None) -> LoopResult:
        self.sims_used = 0
        self.zoo_added = 0
        history: list = []

        def emit(phase, best_total, detail=""):
            if on_progress:
                on_progress(LoopProgress(self.sims_used, best_total, phase, detail))

        emit("hypothesis", 0.0, research_direction)
        hypothesis = self.hypothesis_gen.generate(research_direction)
        seed = self.translator.translate(hypothesis)
        if seed is None:
            self.repo.record_failure("", "syntax", "không dịch được giả thuyết", "llm")
            return LoopResult(None, None, history, 0, self.repo.recent_failures(50), self.sims_used)

        best_ev = self._evaluate(seed, parent_id=None)
        if best_ev is None:
            return LoopResult(None, None, history, self.zoo_added, self.repo.recent_failures(50), self.sims_used)
        best_cand = seed
        history.append(
            {"step": 0, "action": "seed", "dimension": "-", "total": best_ev.vector.total,
             "expression": seed.expression, "accepted": True}
        )
        emit("seed", best_ev.vector.total, seed.expression)

        patience = 0
        step = 0
        while self.sims_used < self.max_simulations and patience < self.no_improve_patience:
            weak = weakest_dimension(best_ev.vector)
            cand = self.refiner.refine(best_cand, best_ev.metrics, weak)
            if cand is None:
                patience += 1
                continue
            ev = self._evaluate(cand, parent_id=best_ev.alpha_id)
            if ev is None:
                break  # hết trần sim giữa chừng
            step += 1
            improved = ev.vector.total > best_ev.vector.total + 1e-9
            history.append(
                {"step": step, "action": "refine", "dimension": weak, "total": ev.vector.total,
                 "expression": cand.expression, "accepted": improved}
            )
            if improved:
                best_cand, best_ev = cand, ev
                patience = 0
            else:
                patience += 1
            emit("refine", best_ev.vector.total, f"nhắm {weak}")

        logger.info(
            "Loop xong: {} sim, best total={:.3f}, zoo+{}",
            self.sims_used, best_ev.vector.total, self.zoo_added,
        )
        emit("done", best_ev.vector.total)
        return LoopResult(
            best_candidate=best_cand,
            best_vector=best_ev.vector,
            history=history,
            zoo_added=self.zoo_added,
            failures=self.repo.recent_failures(50),
            sims_used=self.sims_used,
        )
