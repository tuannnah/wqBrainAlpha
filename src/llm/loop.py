"""Vòng lặp AI tham lam: giả thuyết → dịch → mô phỏng → tinh chỉnh chiều yếu (T2.14).

Greedy: luôn cải thiện alpha tốt nhất hiện có, nhắm chiều yếu nhất. Trần số
simulation cấu hình được; cache theo hash (cache hit không tính vào trần). Lưu
alpha pass vào DB (alpha zoo là view), ghi lại thất bại để tránh lặp.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger

from src.backtest.config import PortfolioConfig
from src.backtest.gate import LocalGateVerdict, score_local_gate
from src.data.market_panel import MarketData
from src.scoring.complexity import complexity_penalty
from src.scoring.filter import blocking_dimensions
from src.scoring.filter import passes as default_filter
from src.scoring.metrics import normalize
from src.scoring.regularized import Penalties, PenaltyWeights, regularized_score
from src.scoring.vector import (
    ScoreVector,
    score_vector,
    weakest_dimension,
    with_pool_corr,
    with_regime_fit,
)
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
    # Vì sao vòng dừng: 'abandon' (referee bỏ hướng) | 'budget' (hết trần sim) |
    # 'patience' (hết kiên nhẫn) | 'no_seed' (không dịch được seed nào).
    stop_reason: str = ""


@dataclass
class _Eval:
    vector: ScoreVector
    metrics: dict
    alpha_id: str | None  # None khi cache hit (không lưu lại)
    passed: bool
    effective_total: float = 0.0  # điểm dùng để so sánh best (điều chuẩn nếu bật, else = vector.total)
    pool_corr: float | None = None  # self-corr với pool (đo sau sim); None = chưa/không đo
    regime_blocked: bool = False    # năm tệ nhất dưới ngưỡng -> refine nhắm regime_fit


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
        align_gate: bool = True,
        regularize: bool = False,
        penalty_lambda: float = 0.3,
        penalty_weights: PenaltyWeights | None = None,
        sim_config: SimConfig | None = None,
        pool_corr_fn=None,
        max_pool_corr: float = 0.70,
        oos_min_ratio: float | None = None,
        deflate_haircut: float = 0.0,
        pnl_fn=None,
        regime_min: float | None = None,
        regime_target: float = 1.0,
        improve_margin: float = 0.0,
        idea_generator=None,
        reseed_every: int = 0,
        referee=None,
        config_tuner=None,
        local_gate_fn: Callable[[str, PortfolioConfig, MarketData], LocalGateVerdict] | None = None,
        market_data: MarketData | None = None,
        local_gate_cfg: PortfolioConfig | None = None,
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
        # True = loại cứng candidate lệch giả thuyết TRƯỚC sim; False = chỉ tính điểm
        # alignment làm tín hiệu mềm (điểm điều chuẩn), không loại — tránh giết edge từ
        # conditioning không hiển nhiên (vd volume-gating). Review 5.
        self.align_gate = align_gate
        self.regularize = regularize
        self.penalty_lambda = penalty_lambda
        self.penalty_weights = penalty_weights or PenaltyWeights()
        # Hàm đo self-correlation với pool (wq_alpha_id -> float|None). None = tắt gate.
        self.pool_corr_fn = pool_corr_fn
        self.max_pool_corr = max_pool_corr
        # Tỉ lệ OOS/IS sharpe tối thiểu để gắn passed. None = tắt (tương thích ngược).
        self.oos_min_ratio = oos_min_ratio
        # Hệ số haircut điểm theo tỉ lệ budget sim đã dùng (deflated sharpe): candidate
        # nhìn càng muộn bị phạt càng nặng -> chống khai thác IS qua nhiều lần thử. 0 = tắt.
        self.deflate_haircut = deflate_haircut
        # Đo regime: pnl_fn(wq_alpha_id) -> [(date, daily_pnl)]; regime_min = sàn Sharpe
        # năm tệ nhất (None = tắt). regime_target chuẩn hoá regime_fit.
        self.pnl_fn = pnl_fn
        self.regime_min = regime_min
        self.regime_target = regime_target
        # Biên cải thiện tương đối tối thiểu để soán best (thay epsilon vô cùng nhỏ):
        # cải thiện vi mô thường là nhiễu IS, không đáng đổi best. 0 = giữ epsilon cũ.
        self.improve_margin = improve_margin
        # Re-seed diversity: idea_generator.generate_ideas(n) -> [direction]. Khi nhánh
        # refine stuck `reseed_every` vòng không cải thiện, sinh direction mới (LLM
        # re-seed) thay vì refine tiếp nhánh kẹt. 0/None = tắt (greedy thuần như cũ).
        self.idea_generator = idea_generator
        self.reseed_every = reseed_every
        # Trọng tài LLM (marathon): sau mỗi sim quyết refine_formula | tune_config |
        # abandon. None = giữ hành vi greedy cũ (heuristic patience quyết dừng hướng).
        # config_tuner: đề xuất decay/truncation/neutralization mới khi referee chọn
        # tune_config. Trần cứng (patience/max_sims) vẫn là giới hạn an toàn cuối cùng.
        self.referee = referee
        self.config_tuner = config_tuner
        # Local pre-filter BẮT BUỘC trước simulate (D9 — gỡ đường cũ "LLM->sim trực tiếp").
        # market_data=None -> gate bị bỏ qua (chưa wire data thật, Phase 3 MVP); có
        # market_data -> MỌI candidate phải pass local_gate_fn trước khi tốn sim.
        self.local_gate_fn = local_gate_fn or score_local_gate
        self.market_data = market_data
        self.local_gate_cfg = local_gate_cfg or PortfolioConfig()
        if self.market_data is None:
            logger.warning("local gate tắt: thiếu market_data")
        self.sims_used = 0
        self.zoo_added = 0

    # --------------------------------------------------------------- seed
    def seed_candidates(self, research_direction: str) -> list:
        """Tập seed khởi đầu = seed LLM (dịch từ giả thuyết) + NOVEL_ALPHAS.

        Salvage diversity của engine hybrid cũ (`_seed_pool` của nó) nhưng không có
        GA: seed LLM đứng đầu (giữ hành vi greedy hiện tại khi dịch được), NOVEL_ALPHAS
        làm sàn đa dạng/fallback để loop không sụp về một điểm nếu seed LLM bị loại
        trước sim. NOVEL được lọc qua `self.prefilter` ngay tại đây (giống seed pool
        cũ): field đã-học-chết/blacklist hoặc sai kiểu/scope không vào pool, nên
        fallback không tốn lượt đánh giá cho biểu thức chắc chắn hỏng."""
        from src.generation.novel_ideas import NOVEL_ALPHAS
        from src.llm.hypothesis import Hypothesis
        from src.llm.translator import AlphaCandidate

        pool: list = []
        palette = self.translator.field_palette(research_direction)
        hypothesis = self.hypothesis_gen.generate(research_direction, palette)
        seed = self.translator.translate(hypothesis)
        if seed is not None:
            pool.append(seed)
        for c in NOVEL_ALPHAS:
            if not self.prefilter.check(c.expression)[0]:
                continue
            novel_hyp = Hypothesis(economic_rationale=str(getattr(c, "rationale", "") or ""))
            pool.append(
                AlphaCandidate(novel_hyp, str(getattr(c, "hypothesis", "") or ""), c.expression)
            )
        return pool

    def _reseed_once(self, config=None):
        """Sinh một direction mới từ idea_generator rồi tạo+đánh giá seed cho nó.
        Trả (candidate, eval) nếu thành công, None nếu không sinh/dịch/đánh giá được."""
        if self.idea_generator is None:
            return None
        ideas = self.idea_generator.generate_ideas(1)
        if not ideas:
            return None
        direction = ideas[0]
        palette = self.translator.field_palette(direction)
        hypothesis = self.hypothesis_gen.generate(direction, palette)
        seed = self.translator.translate(hypothesis)
        if seed is None:
            return None
        ev = self._evaluate(seed, parent_id=None, config=config)
        if ev is None:
            return None
        return seed, ev

    # --------------------------------------------------------------- eval
    def _effective_total(self, vector, expr, originality, alignment) -> float:
        """Điểm dùng để so sánh best. Bật regularize -> điểm điều chuẩn (T4.4),
        gộp phạt độ độc đáo/khớp giả thuyết/độ phức tạp; tắt -> total thô.
        Chiều không đo được (không zoo/aligner) coi như điểm tốt (phạt 0)."""
        if self.regularize:
            pen = Penalties.from_scores(
                originality=1.0 if originality is None else originality,
                alignment=1.0 if alignment is None else alignment,
                complexity=complexity_penalty(expr),
            )
            base = regularized_score(
                vector.total, pen, weights=self.penalty_weights, lambda_=self.penalty_lambda
            )
        else:
            base = vector.total
        # Deflated sharpe: trừ haircut theo tỉ lệ budget sim đã tiêu (review 4b). Candidate
        # đánh giá muộn (sims_used cao) bị phạt nặng hơn -> phải vượt best cũ một biên thật.
        if self.deflate_haircut > 0 and self.max_simulations > 0:
            base -= self.deflate_haircut * (self.sims_used / self.max_simulations)
        return base

    def _evaluate(self, candidate, parent_id: str | None, config=None) -> _Eval | None:
        # config: cho phép đánh giá CÙNG biểu thức dưới cấu hình khác (referee tune_config).
        # None -> dùng cấu hình mặc định của loop (hành vi cũ).
        config = config or self.sim_config
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
        config_key = config.key()
        cached = self.repo.get_cached_simulation(expr, config_key=config_key)
        if cached is not None:
            vector = self.score_vector_fn(cached)
            eff = self._effective_total(vector, expr, originality, None)
            return _Eval(vector, normalize(cached), None, cached.status == "passed", eff)

        alignment = None
        if self.aligner is not None:
            align = self.aligner.score(candidate)
            alignment = align.value
            if self.align_gate and align.value < self.min_alignment:
                self.repo.record_failure(
                    expr, "hypothesis_mismatch",
                    f"nhất quán {align.value:.2f} < ngưỡng {self.min_alignment:.2f}: {align.reason}",
                    "llm",
                )
                return None

        # Local gate BẮT BUỘC (D9): chỉ chạy khi đã có market_data thật wire vào loop.
        # Local hard-fail -> bỏ NGAY, không tăng sims_used, không gọi simulator.
        if self.market_data is not None:
            verdict = self.local_gate_fn(expr, self.local_gate_cfg, self.market_data)
            if not verdict.passed:
                self.repo.record_failure(expr, "local_gate_fail", verdict.reason, "llm")
                return None

        if self.sims_used >= self.max_simulations:
            return None  # hết trần sim, không gọi WQ thêm

        result = self.simulator.simulate(expr, settings=config.to_settings())
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

        # Regime gate (review 3): metric tổng có thể đẹp nhờ vài năm tốt che một năm sập.
        # Tính Sharpe theo năm; năm tệ nhất dưới sàn -> mỏng manh theo regime -> loại.
        regime_blocked = False
        if passed and self.pnl_fn is not None and self.regime_min is not None and result.alpha_id:
            from src.scoring.regime import min_annual_sharpe, regime_fit, yearly_sharpe

            pnl = self.pnl_fn(result.alpha_id)
            if pnl:
                yearly = yearly_sharpe(pnl)
                vector = with_regime_fit(vector, regime_fit(yearly, target=self.regime_target))
                if min_annual_sharpe(yearly) < self.regime_min:
                    gated = True
                    regime_blocked = True
                    passed = False
                    self.repo.record_failure(
                        expr, "regime_fragile",
                        f"Sharpe năm tệ nhất {min_annual_sharpe(yearly):.2f} < sàn {self.regime_min:.2f}",
                        "llm",
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
        return _Eval(vector, normalize(result), alpha_id, passed, eff, pool_corr, regime_blocked)

    # ---------------------------------------------------------------- run
    def run(self, research_direction: str, on_progress=None) -> LoopResult:
        self.sims_used = 0
        self.zoo_added = 0
        history: list = []

        def emit(phase, best_total, detail=""):
            if on_progress:
                on_progress(LoopProgress(self.sims_used, best_total, phase, detail))

        # Cấu hình hiện hành của nhánh; referee có thể đổi qua tune_config trong khi chạy.
        current_config = self.sim_config

        emit("hypothesis", 0.0, research_direction)
        candidates = self.seed_candidates(research_direction)
        if not candidates:
            self.repo.record_failure("", "syntax", "không dịch được giả thuyết", "llm")
            return LoopResult(None, None, history, 0, self.repo.recent_failures(50),
                              self.sims_used, stop_reason="no_seed")

        # Thử lần lượt: seed LLM trước, NOVEL_ALPHAS làm fallback đa dạng. Dừng ở seed
        # đầu tiên đánh giá được (không bị loại trước sim / còn budget). Seed LLM hợp lệ
        # -> dừng ngay -> hành vi greedy không đổi; chỉ rơi xuống NOVEL khi seed LLM bị loại.
        best_ev = None
        best_cand = None
        for cand in candidates:
            best_ev = self._evaluate(cand, parent_id=None, config=current_config)
            if best_ev is not None:
                best_cand = cand
                break
        if best_ev is None:
            return LoopResult(None, None, history, self.zoo_added, self.repo.recent_failures(50),
                              self.sims_used, stop_reason="no_seed")
        history.append(
            {"step": 0, "action": "seed", "dimension": "-", "total": best_ev.vector.total,
             "expression": best_cand.expression, "accepted": True}
        )
        emit("seed", best_ev.vector.total, best_cand.expression)

        return self._refine_loop(best_cand, best_ev, research_direction, current_config, history, emit)

    # -------------------------------------------------- lõi refine dùng chung
    def _refine_loop(
        self, best_cand, best_ev, research_direction: str, current_config,
        history: list, emit,
    ) -> LoopResult:
        """Lõi vòng refine dùng chung bởi run() và run_from_seed: từ (best_cand, best_ev) đã
        có, lặp refine/tune/reseed tới patience/budget/abandon rồi trả LoopResult. Tách ra để
        seed có thể đến từ hypothesis (run) hoặc từ công thức cho sẵn (run_from_seed)."""
        patience = 0
        stuck = 0  # số vòng liên tiếp không cải thiện -> ngưỡng kích hoạt re-seed
        step = 0
        abandoned = False
        reseed_on = self.reseed_every > 0 and self.idea_generator is not None
        while self.sims_used < self.max_simulations and patience < self.no_improve_patience:
            # Trọng tài LLM (marathon): sau mỗi sim, quyết hành động kế tiếp cho hướng này.
            # abandon -> dừng hướng; tune_config -> đổi tham số, sim lại CÙNG biểu thức;
            # refine_formula (hoặc không có referee) -> rơi xuống nhánh refine biểu thức.
            if self.referee is not None:
                verdict = self.referee.judge(research_direction, history, best_ev.metrics)
                if verdict.action == "abandon":
                    abandoned = True
                    break
                if verdict.action == "tune_config" and self.config_tuner is not None:
                    new_config = self.config_tuner.tune(current_config, best_ev.metrics, verdict.reason)
                    if new_config.key() == current_config.key():
                        # Không đổi gì -> coi như một vòng không cải thiện (tránh kẹt vô hạn).
                        patience += 1
                        stuck += 1
                        continue
                    ev = self._evaluate(best_cand, parent_id=best_ev.alpha_id, config=new_config)
                    if ev is None:
                        break  # hết trần sim giữa chừng
                    step += 1
                    threshold = max(1e-9, self.improve_margin * abs(best_ev.effective_total))
                    improved = ev.effective_total > best_ev.effective_total + threshold
                    history.append(
                        {"step": step, "action": "tune_config", "dimension": "config",
                         "total": ev.vector.total, "expression": best_cand.expression,
                         "accepted": improved}
                    )
                    if improved:
                        best_ev = ev
                        current_config = new_config
                        patience = 0
                        stuck = 0
                    else:
                        patience += 1
                        stuck += 1
                    emit("tune", best_ev.vector.total,
                         f"decay={new_config.decay} trunc={new_config.truncation} neut={new_config.neutralization}")
                    continue

            # Re-seed: nhánh kẹt đủ `reseed_every` vòng -> sinh direction mới (LLM re-seed)
            # thay vì refine tiếp. Chỉ chuyển nhánh nếu seed mới tốt hơn best hiện tại.
            if reseed_on and stuck >= self.reseed_every:
                stuck = 0
                reseeded = self._reseed_once(config=current_config)
                if reseeded is not None:
                    new_cand, new_ev = reseeded
                    if new_ev.effective_total > best_ev.effective_total:
                        best_cand, best_ev = new_cand, new_ev
                        patience = 0
                    emit("seed", best_ev.vector.total, f"re-seed: {new_cand.expression}")
                continue

            # Nhắm chiều yếu nhất TRONG SỐ các chiều đang chặn hard filter (vd
            # fitness) để hướng refine về biên cần vượt; alpha đã đạt -> chiều yếu
            # nhất tuyệt đối như cũ.
            restrict = blocking_dimensions(
                best_ev.metrics, pool_corr=best_ev.pool_corr, max_pool_corr=self.max_pool_corr
            )
            if best_ev.regime_blocked:
                restrict.add("regime_fit")
            weak = weakest_dimension(best_ev.vector, restrict=restrict)
            cand = self.refiner.refine(best_cand, best_ev.metrics, weak)
            if cand is None:
                patience += 1
                stuck += 1
                continue
            ev = self._evaluate(cand, parent_id=best_ev.alpha_id, config=current_config)
            if ev is None:
                break  # hết trần sim giữa chừng
            step += 1
            threshold = max(1e-9, self.improve_margin * abs(best_ev.effective_total))
            improved = ev.effective_total > best_ev.effective_total + threshold
            history.append(
                {"step": step, "action": "refine", "dimension": weak, "total": ev.vector.total,
                 "expression": cand.expression, "accepted": improved}
            )
            if improved:
                best_cand, best_ev = cand, ev
                patience = 0
                stuck = 0
            else:
                patience += 1
                stuck += 1
            emit("refine", best_ev.vector.total, f"nhắm {weak}")

        if abandoned:
            stop_reason = "abandon"
        elif self.sims_used >= self.max_simulations:
            stop_reason = "budget"
        else:
            stop_reason = "patience"
        logger.info(
            "Loop xong ({}): {} sim, best total={:.3f}, zoo+{}",
            stop_reason, self.sims_used, best_ev.vector.total, self.zoo_added,
        )
        emit("done", best_ev.vector.total)
        return LoopResult(
            best_candidate=best_cand,
            best_vector=best_ev.vector,
            history=history,
            zoo_added=self.zoo_added,
            failures=self.repo.recent_failures(50),
            sims_used=self.sims_used,
            stop_reason=stop_reason,
        )

    def run_from_seed(self, expression: str, on_progress=None) -> LoopResult:
        """Như run() nhưng hạt giống là MỘT công thức FASTEXPR cho sẵn (vd core từ GPEngine),
        KHÔNG qua hypothesis_gen/translator. Phục vụ vòng kín 'GP trục, AI tăng cường'."""
        from src.llm.hypothesis import Hypothesis
        from src.llm.translator import AlphaCandidate

        self.sims_used = 0
        self.zoo_added = 0
        history: list = []

        def emit(phase, best_total, detail=""):
            if on_progress:
                on_progress(LoopProgress(self.sims_used, best_total, phase, detail))

        current_config = self.sim_config
        seed = AlphaCandidate(hypothesis=Hypothesis(), description=expression,
                              expression=expression)
        emit("seed", 0.0, expression)
        best_ev = self._evaluate(seed, parent_id=None, config=current_config)
        if best_ev is None:
            return LoopResult(None, None, history, self.zoo_added,
                              self.repo.recent_failures(50), self.sims_used,
                              stop_reason="no_seed")
        history.append(
            {"step": 0, "action": "seed", "dimension": "-", "total": best_ev.vector.total,
             "expression": expression, "accepted": True}
        )
        emit("seed", best_ev.vector.total, expression)
        return self._refine_loop(seed, best_ev, expression, current_config, history, emit)

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
