"""Vòng lặp AI tham lam: giả thuyết → dịch → mô phỏng → tinh chỉnh chiều yếu (T2.14).

Greedy: luôn cải thiện alpha tốt nhất hiện có, nhắm chiều yếu nhất. Trần số
simulation cấu hình được; cache theo hash (cache hit không tính vào trần). Lưu
alpha pass vào DB (alpha zoo là view), ghi lại thất bại để tránh lặp.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from src.scoring.complexity import complexity_penalty
from src.scoring.filter import blocking_dimensions
from src.scoring.filter import passes as default_filter
from src.scoring.metrics import normalize
from src.scoring.regularized import Penalties, PenaltyWeights, regularized_score
from src.scoring.vector import ScoreVector, score_vector, weakest_dimension, with_pool_corr
from src.simulation.config import SimConfig


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
    pool_corr: float | None = None  # self-corr với pool (đo sau sim); None = chưa/không đo


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
        # 0.35 = loại gần-trùng tới similarity 0.65 (0.20 cũ lọt tới 0.80 — quá lỏng).
        # AST chỉ là prefilter cấu trúc rẻ; corr-với-pool sau sim (gate 'crowded') mới
        # là ràng buộc nộp thật.
        min_originality: float = 0.35,
        aligner=None,
        min_alignment: float = 0.5,
        regularize: bool = False,
        penalty_lambda: float = 0.3,
        penalty_weights: PenaltyWeights | None = None,
        sim_config: SimConfig | None = None,
        pool_corr_fn=None,
        max_pool_corr: float = 0.70,
        oos_min_ratio: float | None = None,
    ):
        self.hypothesis_gen = hypothesis_gen
        self.translator = translator
        self.refiner = refiner
        self.simulator = simulator
        self.prefilter = prefilter
        self.repo = repo
        self.sim_config = sim_config or SimConfig.default(region=region, universe=universe, delay=delay)
        self.region = self.sim_config.region
        self.universe = self.sim_config.universe
        self.delay = self.sim_config.delay
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
        # Hàm đo self-correlation với pool (wq_alpha_id -> float|None). None = tắt gate.
        self.pool_corr_fn = pool_corr_fn
        self.max_pool_corr = max_pool_corr
        # Tỉ lệ OOS/IS sharpe tối thiểu để gắn passed. None = tắt (tương thích ngược).
        self.oos_min_ratio = oos_min_ratio
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

        # Kiểm cache TRƯỚC aligner: expr đã sim trước đây thì đã qua aligner rồi,
        # gọi lại chỉ tốn lượt LLM alignment (đắt) mà không đổi kết quả.
        config_key = self.sim_config.key()
        cached = self.repo.get_cached_simulation(expr, config_key=config_key)
        if cached is not None:
            vector = self.score_vector_fn(cached)
            eff = self._effective_total(vector, expr, originality, None)
            return _Eval(vector, normalize(cached), None, cached.status == "passed", eff)

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

        if self.sims_used >= self.max_simulations:
            return None  # hết trần sim, không gọi WQ thêm

        result = self.simulator.simulate(expr, settings=self.sim_config.to_settings())
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
            score=vector.total, alpha_id=alpha_id, config_key=config_key,
        )
        ok_hard, reasons = self.hard_filter_fn(result)
        passed = result.status == "passed" and ok_hard
        # Sim degenerate: status='error' HOẶC metric rỗng (sharpe & fitness None —
        # sim hoàn tất nhưng không ra metric, vd operator lỗi). Không phải alpha
        # điểm thấp, nên ghi sim_error để LLM tránh lặp, không thổi phồng low_score.
        metrics_missing = result.sharpe is None and result.fitness is None

        # Self-correlation với pool (ràng buộc chặn-nộp thật). Chỉ đo cho alpha ĐÃ đạt
        # metrics — alpha kém thì khỏi tốn lượt API. Vượt ngưỡng -> crowded: đẹp số
        # nhưng không nộp được -> loại khỏi zoo và đưa corr vào điểm để best né đỉnh đông.
        pool_corr = None
        gated = False  # bị chặn bởi ràng buộc ngoài-hard-filter (crowded/oos), đã ghi failure
        if passed and self.pool_corr_fn is not None and result.alpha_id:
            pool_corr = self.pool_corr_fn(result.alpha_id)
            if pool_corr is not None:
                vector = with_pool_corr(vector, pool_corr)
                if abs(pool_corr) >= self.max_pool_corr:
                    gated = True
                    passed = False
                    self.repo.record_failure(
                        expr, "crowded",
                        f"self-corr {pool_corr:.2f} >= ngưỡng {self.max_pool_corr:.2f}", "llm",
                    )

        # OOS gate (review 4): tinh chỉnh IS có hệ thống là một dạng overfit. Mỗi sim là
        # một "lần nhìn" IS; chỉ gắn passed khi OOS sharpe đạt tỉ lệ tối thiểu so với IS.
        if passed and self.oos_min_ratio is not None:
            from src.simulation.oos import oos_passes

            if not oos_passes(result, min_ratio=self.oos_min_ratio):
                gated = True
                passed = False
                self.repo.record_failure(
                    expr, "oos_fail",
                    f"OOS sharpe không đạt {self.oos_min_ratio:.2f}×IS", "llm",
                )

        if passed:
            self.zoo_added += 1
        elif result.status == "error" or metrics_missing:
            detail = result.raw.get("error") if isinstance(result.raw, dict) else None
            self.repo.record_failure(expr, "sim_error", str(detail or result.status), "llm")
        elif gated:
            pass  # đã ghi 'crowded'/'oos_fail' ở trên, không dán nhãn low_score chồng lên
        else:
            self.repo.record_failure(expr, "low_score", "; ".join(reasons) or result.status, "llm")
        eff = self._effective_total(vector, expr, originality, alignment)
        return _Eval(vector, normalize(result), alpha_id, passed, eff, pool_corr)

    # ---------------------------------------------------------------- run
    def run(self, research_direction: str, on_progress=None) -> LoopResult:
        self.sims_used = 0
        self.zoo_added = 0
        history: list = []

        def emit(phase, best_total, detail=""):
            if on_progress:
                on_progress(LoopProgress(self.sims_used, best_total, phase, detail))

        emit("hypothesis", 0.0, research_direction)
        palette = self.translator.field_palette(research_direction)
        hypothesis = self.hypothesis_gen.generate(research_direction, palette)
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
            # Nhắm chiều yếu nhất TRONG SỐ các chiều đang chặn hard filter (vd
            # fitness) để hướng refine về biên cần vượt; alpha đã đạt -> chiều yếu
            # nhất tuyệt đối như cũ.
            weak = weakest_dimension(
                best_ev.vector,
                restrict=blocking_dimensions(
                    best_ev.metrics, pool_corr=best_ev.pool_corr, max_pool_corr=self.max_pool_corr
                ),
            )
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
        palette = self.translator.field_palette(research_direction)
        hypothesis = self.hypothesis_gen.generate(research_direction, palette)
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
