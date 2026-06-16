"""Vòng lặp AI tham lam: giả thuyết → dịch → mô phỏng → tinh chỉnh chiều yếu (T2.14).

Greedy: luôn cải thiện alpha tốt nhất hiện có, nhắm chiều yếu nhất. Trần số
simulation cấu hình được; cache theo hash (cache hit không tính vào trần). Lưu
alpha pass vào DB (alpha zoo là view), ghi lại thất bại để tránh lặp.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from src.scoring.complexity import complexity_penalty
from src.scoring.filter import passes as default_filter
from src.scoring.metrics import normalize
from src.scoring.regularized import Penalties, PenaltyWeights, regularized_score
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
    effective_total: float = 0.0  # điểm dùng để so sánh best (điều chuẩn nếu bật, else = vector.total)


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
        zoo=None,
        min_originality: float = 0.2,
        aligner=None,
        min_alignment: float = 0.5,
        regularize: bool = False,
        penalty_lambda: float = 0.3,
        penalty_weights: PenaltyWeights | None = None,
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
        self.zoo = zoo
        self.min_originality = min_originality
        self.aligner = aligner
        self.min_alignment = min_alignment
        self.regularize = regularize
        self.penalty_lambda = penalty_lambda
        self.penalty_weights = penalty_weights or PenaltyWeights()
        self.sims_used = 0
        self.zoo_added = 0

    # --------------------------------------------------------------- eval
    def _effective_total(self, vector, expr, originality, alignment) -> float:
        """Điểm dùng để so sánh best. Bật regularize -> điểm điều chuẩn (T4.4),
        gộp phạt độ độc đáo/khớp giả thuyết/độ phức tạp; tắt -> total thô.
        Chiều không đo được (không zoo/aligner) coi như điểm tốt (phạt 0)."""
        if not self.regularize:
            return vector.total
        pen = Penalties.from_scores(
            originality=1.0 if originality is None else originality,
            alignment=1.0 if alignment is None else alignment,
            complexity=complexity_penalty(expr),
        )
        return regularized_score(
            vector.total, pen, weights=self.penalty_weights, lambda_=self.penalty_lambda
        )

    def _evaluate(self, candidate, parent_id: str | None) -> _Eval | None:
        expr = candidate.expression
        ok, reason = self.prefilter.check(expr)
        if not ok:
            self.repo.record_failure(expr, "syntax", reason, "llm")
            return None

        originality = None
        if self.zoo is not None:
            originality = self.zoo.originality(expr)
            if originality < self.min_originality:
                self.repo.record_failure(
                    expr, "duplicate",
                    f"độc đáo {originality:.2f} < ngưỡng {self.min_originality:.2f}",
                    "llm",
                )
                return None

        alignment = None
        if self.aligner is not None:
            align = self.aligner.score(candidate)
            alignment = align.value
            if align.value < self.min_alignment:
                self.repo.record_failure(
                    expr, "hypothesis_mismatch",
                    f"nhất quán {align.value:.2f} < ngưỡng {self.min_alignment:.2f}: {align.reason}",
                    "llm",
                )
                return None

        cached = self.repo.get_cached_simulation(expr)
        if cached is not None:
            vector = self.score_vector_fn(cached)
            eff = self._effective_total(vector, expr, originality, alignment)
            return _Eval(vector, normalize(cached), None, cached.status == "passed", eff)

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
        elif result.status == "error":
            # Sim lỗi/timeout ở phía WQ Brain: ghi lý do thật (đã kèm message) để
            # LLM tránh lặp lại biểu thức hỏng, thay vì chỉ ghi "error".
            detail = result.raw.get("error") if isinstance(result.raw, dict) else None
            self.repo.record_failure(expr, "sim_error", str(detail or result.status), "llm")
        else:
            self.repo.record_failure(expr, "low_score", "; ".join(reasons) or result.status, "llm")
        eff = self._effective_total(vector, expr, originality, alignment)
        return _Eval(vector, normalize(result), alpha_id, passed, eff)

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
            improved = ev.effective_total > best_ev.effective_total + 1e-9
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

    # ----------------------------------------------------------- MCTS (T6.1)
    def run_mcts(self, research_direction: str, iterations: int = 20, on_progress=None) -> LoopResult:
        """Thay vòng greedy bằng MCTS: giữ nhiều nhánh, UCB + lan ngược điểm.

        Tái dùng `_evaluate` (cache + trần sim + các pre-filter GĐ3/GĐ4) làm
        evaluate_fn, `refiner.refine` làm expand_fn, `weakest_dimension` chọn chiều
        nhắm. Trần sim của loop vẫn là giới hạn cứng (evaluate trả None khi hết)."""
        from src.llm.mcts import MCTSSearch

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

        seed_ev = self._evaluate(seed, parent_id=None)
        if seed_ev is None:
            return LoopResult(None, None, history, self.zoo_added, self.repo.recent_failures(50), self.sims_used)
        emit("seed", seed_ev.vector.total, seed.expression)

        def expand(candidate, metrics, weak):
            return self.refiner.refine(candidate, metrics, weak)

        def evaluate(candidate, parent_id):
            if candidate is None or self.sims_used >= self.max_simulations:
                return None
            ev = self._evaluate(candidate, parent_id=parent_id)
            if ev is not None:
                emit("mcts", ev.vector.total, candidate.expression)
            return ev

        search = MCTSSearch(
            expand, evaluate, weakest_dimension,
            max_iterations=iterations, none_patience=self.no_improve_patience,
        )
        result = search.search(seed, seed_ev)

        emit("done", result.best_eval.vector.total)
        logger.info(
            "MCTS xong: {} sim, best total={:.3f}, zoo+{}",
            self.sims_used, result.best_eval.vector.total, self.zoo_added,
        )
        return LoopResult(
            best_candidate=result.best_candidate,
            best_vector=result.best_eval.vector,
            history=result.history,
            zoo_added=self.zoo_added,
            failures=self.repo.recent_failures(50),
            sims_used=self.sims_used,
        )
