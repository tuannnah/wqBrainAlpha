"""Adapter nối thành phần thật vào ClosedLoop (Phase 2). Tầng composition: được phép import
src.gp/src.llm/src.pipeline/src.lang (khác src/pipeline vốn cấm src.llm/src.gp theo B1).

- RefinementLoopRefiner: bọc RefinementLoop.run_from_seed (4A) → IdeaOutcome.
- GPIdeaSource: bọc generate_many (Phase 8) với seed GPEngine tăng dần → nguồn ý tưởng."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pipeline.closed_loop import ClosedLoop

from config.thresholds import calibrated_floor
from src.backtest.gate import local_usable
from src.backtest.local_tuner import tune as _tune
from src.generation.alt_data_seeds import (
    ALT_DATA_CORES,
    neutralization_for_expr,
    pp_neutralization_for_expr,
)
from src.generation.fundamental_seeds import FUNDAMENTAL_CORES
from src.generation.combiner import SubSignal
from src.gp.engine import GPEngine
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import CanonicalHasher, DepthVisitor, FieldCollector, OperatorCollector
from src.reporting.diagnostics import (
    categorize_presim_reason,
    classify_family,
    fail_check_from_reasons,
)
from src.pipeline.closed_loop import IdeaOutcome, QuotaExhausted
from src.pipeline.combine_stage import combine_stage
from src.pipeline.runner import _score_one_full, generate_many
from src.pipeline.shortlist import ShortlistCandidate
from src.scoring.filter import passes as _default_filter
from src.scoring.vector import score_vector as _score_vector
from src.simulation.simulator import AuthExpiredError, QuotaExceededError

# Grouping field không tính vào giới hạn "3 field dữ liệu" của Power Pool (chỉ field dữ liệu
# thật, vd close/volume — field group như sector/industry chỉ dùng để neutralize).
_POWER_POOL_GROUPS = frozenset(
    {"country", "exchange", "market", "sector", "industry", "subindustry", "currency"}
)


def is_power_pool(expr: str, sharpe: float | None, self_corr: float | None, registry: Any) -> bool:
    """Đủ tiêu chí Power Pool (docs Brain): <=8 operator, <=3 field (trừ grouping),
    Sharpe>=1.0, self_corr None HOẶC <=0.5."""
    if sharpe is None or sharpe < 1.0:
        return False
    if self_corr is not None and abs(self_corr) > 0.5:
        return False
    node = parse(expr)
    n_ops = len(OperatorCollector().visit(node))
    n_fields = len(FieldCollector(registry).visit(node) - _POWER_POOL_GROUPS)
    return n_ops <= 8 and n_fields <= 3


# Mã categorize_presim_reason -> stage_reached tương ứng (Task 3, spec C2). PARSE/fallback
# dùng chung "presim" (không có bucket riêng — hiếm gặp, đủ để soi qua fail_check).
_PRESIM_STAGE_BY_CODE: dict[str, str] = {
    "OPERATOR_INVALID": "op_invalid",
    "FIELD_INVALID": "field_invalid",
    "DEPTH": "depth",
}


def _presim_reject_outcome(
    expr: str, canonical_hash: str, presim_reason: str, *, stop_reason: str, source: str,
) -> IdeaOutcome:
    """Ứng viên bị `PreFilter` loại TRƯỚC khi chạm Brain (Task 3, spec C2: đừng giả vờ
    'simmed/LOW_SHARPE' như bug cũ — CSV giấu bug operator/field bịa vì sim_ms≈0 nhưng
    stage_reached vẫn ghi 'simmed'). Outcome trung thực: sims_used=0 (chưa tốn quota Brain),
    is_brain_sim=False, stage/fail_check suy từ chính category của presim_reason."""
    code = categorize_presim_reason(presim_reason)
    stage = _PRESIM_STAGE_BY_CODE.get(code, "presim")
    try:
        depth = DepthVisitor().visit(parse(expr))
    except Exception:
        depth = None
    return IdeaOutcome(
        expr=expr, canonical_hash=canonical_hash, passed=False,
        wq_alpha_id=None, sharpe=None, fitness=None, turnover=None, self_corr=None,
        sims_used=0, stop_reason=stop_reason,
        stage_reached=stage, fail_check=code, family=classify_family(expr),
        expr_depth=depth, dedup_key=canonical_hash,
        presim_reason=presim_reason, is_brain_sim=False,
    )


class RefinementLoopRefiner:
    """Bọc RefinementLoop: refine+sim một core (qua run_from_seed) → IdeaOutcome cho ClosedLoop."""

    def __init__(self, loop: object) -> None:
        self.loop = loop

    def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome:
        try:
            # result là Any: loop: object, run_from_seed dùng type: ignore[attr-defined]
            result: Any = self.loop.run_from_seed(candidate.expr)  # type: ignore[attr-defined]
        except (AuthExpiredError, QuotaExceededError) as exc:
            # AuthExpiredError: session chết (401/403 lặp). QuotaExceededError: hết quota
            # simulation NGÀY thật (429 dai dẳng / X-Ratelimit-Remaining=0) — KHÁC lỗi auth,
            # phân biệt rõ ở Simulator để không bị coi nhầm là "sim lỗi" rồi cứ thử tiếp.
            # Cả hai đều báo ClosedLoop dừng gọn (không refine/sim thêm được nữa).
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


class LocalTunerRefiner:
    """Refiner vòng kín KHÔNG dùng LLM: tune tham số/config quanh core bằng eval local
    (Task 3 `tune`), rồi CHỈ sim Brain 1 lần cho cấu hình tốt nhất tìm được. Drop-in
    thay `RefinementLoopRefiner` — cùng protocol `refine_and_sim(candidate) -> IdeaOutcome`.

    Local Sharpe < `min_local_sharpe` (mặc định `PRE_SIM_LOCAL_SHARPE_FLOOR`) -> KHÔNG đốt
    quota Brain, trả outcome 0 sim ngay (chắc chắn rác theo hiệu chỉnh local≈Brain/1.28)."""

    def __init__(
        self, *, simulator, repo, data, local_config, sim_config,
        pool_corr_fn=None, min_local_sharpe: float = calibrated_floor(),
        hard_filter_fn=_default_filter, score_vector_fn=_score_vector,
        region: str = "USA", universe: str = "TOP3000", registry=None, tune_fn=None,
        max_pool_corr: float = 0.70, calib_repo=None,
        pp_allowed_neutralizations: frozenset[str] = frozenset(),
        neut_risk_factors: "list[str] | None" = None,
        calibration_tracker: object | None = None,
    ) -> None:
        self.simulator = simulator
        self.repo = repo
        self.data = data
        self.local_config = local_config
        self.sim_config = sim_config
        self.pool_corr_fn = pool_corr_fn
        self.min_local_sharpe = min_local_sharpe
        self.hard_filter_fn = hard_filter_fn
        self.score_vector_fn = score_vector_fn
        self.region = region
        self.universe = universe
        self.registry = registry
        self.max_pool_corr = max_pool_corr
        # Kho calibration (MiniBrainRepository): lưu local-eval của expr ĐÃ tune theo hash để
        # join local↔Brain (brain_local_sharpe_pairs) thu được ρ. None -> bỏ qua (test/không cần).
        self.calib_repo = calib_repo
        # Tập neutralization theo Power Pool Theme (rỗng -> đường non-theme, dùng group-neut cũ).
        self.pp_allowed_neutralizations = pp_allowed_neutralizations
        self._tune = tune_fn or _tune
        # Risk factor để tune bọc regression_neut hạ self-corr (Pha 3.1); None -> không thử.
        self.neut_risk_factors = neut_risk_factors
        # CalibrationTracker (src/pipeline/closed_loop.py) — cho biết ρ hiện tại giữa ranking
        # local và Brain có đáng tin không (Task 5). None -> hành vi cũ y nguyên (floor cứng).
        self.calibration_tracker = calibration_tracker

    def set_calibration_tracker(self, tracker: object) -> None:
        """Gắn CalibrationTracker SAU khi khởi tạo — dùng khi `build_closed_loop` dựng tracker
        sau refiner (cùng object mà ClosedLoop cập nhật last_rho mỗi `maybe_calibrate`)."""
        self.calibration_tracker = tracker

    def _luu_local_eval_calibration(self, tr, canonical_hash: str) -> None:
        """Ghi ExpressionModel + EvaluationModel cho expr đã tune (khớp hash với record_brain_sim
        mà ClosedLoop ghi sau đó) -> CalibrationTracker có cặp (local_sharpe, brain_sharpe)."""
        import json

        from src.lang.registry import default_registry
        from src.lang.visitors import ComplexityVisitor, DepthVisitor, FieldCollector

        reg = self.registry or default_registry()
        node = parse(tr.best_expr)
        expr_id = self.calib_repo.upsert_expression(
            tr.best_expr, canonical_hash, DepthVisitor().visit(node),
            ComplexityVisitor().visit(node), FieldCollector(reg).visit(node),
        )
        self.calib_repo.record_evaluation(
            expr_id,
            config_json=json.dumps(
                {"decay": tr.best_config.decay, "truncation": tr.best_config.truncation}
            ),
            data_window="default", metrics=tr.local_metrics, self_corr_max=None,
            status="local_tuned", fail_reasons=[], seed=None,
        )

    def _is_alt_data(self, expr: str) -> bool:
        """Expr dùng field NGOÀI panel local -> không chấm/tune local được -> phải sim thẳng.
        `data` không có `field_names()` (test fake/object()) -> coi như local (giữ hành vi cũ)."""
        try:
            return not local_usable(expr, self.data)
        except Exception:
            return False

    def _sim_direct(self, candidate: ShortlistCandidate) -> IdeaOutcome:
        """Nhánh alt-data: BỎ tune/floor local (panel không có field), sim Brain 1 lần với
        neutralization chọn theo category dataset (docs WQ), rồi chấm/lưu như đường tune."""
        expr = candidate.expr
        canonical_hash = CanonicalHasher().visit(parse(expr))
        if self.pp_allowed_neutralizations:
            neut = pp_neutralization_for_expr(
                expr, self.pp_allowed_neutralizations, self.registry
            )
        else:
            neut = neutralization_for_expr(expr, self.registry)
        sim_cfg = self.sim_config.with_overrides(neutralization=neut)
        try:
            result = self.simulator.simulate(expr, settings=sim_cfg.to_settings())
        except (AuthExpiredError, QuotaExceededError) as exc:
            raise QuotaExhausted(str(exc)) from exc
        if result.presim_reason is not None:
            # Chưa chạm Brain (PreFilter loại) -> outcome trung thực, KHÔNG _finalize (nó luôn
            # gán sims_used=1/stage='simmed', đúng cho sim thật nhưng SAI ở đây).
            return _presim_reject_outcome(
                expr, canonical_hash, result.presim_reason,
                stop_reason="presim_reject", source="alt_data",
            )
        return self._finalize(
            result, expr, canonical_hash, sim_cfg,
            stop_reason="alt_data_direct", source="alt_data", description="alt-data direct",
        )

    def _finalize(
        self, result, expr: str, canonical_hash: str, sim_cfg, *,
        stop_reason: str, source: str, description: str,
        local_sharpe: float | None = None, backtest_ms: float | None = None,
        sim_ms: float | None = None,
    ) -> IdeaOutcome:
        """Chấm điểm + lưu DB + xét Power Pool cho 1 kết quả sim Brain — DÙNG CHUNG cho đường
        tune (local_tuned) và đường alt-data (alt_data_direct) để không lặp logic."""
        vector = self.score_vector_fn(result)
        alpha_id = self.repo.save_alpha(
            expr, source=source, hypothesis={}, description=description, parent_id=None,
        )
        self.repo.save_simulation(
            result, region=self.region, universe=self.universe,
            score=vector.total, alpha_id=alpha_id, config_key=sim_cfg.key(),
        )
        ok_hard, _reasons = self.hard_filter_fn(result)
        passed = result.status == "passed" and ok_hard
        registry = self.registry or default_registry()
        # Cấu trúc Power Pool (chưa xét self_corr): Sharpe>=1.0, <=8 op, <=3 field, turnover hợp lệ.
        turnover_ok = result.turnover is not None and 0.01 <= result.turnover <= 0.70
        pp_structural = (
            result.status != "error"
            and result.sharpe is not None
            and turnover_ok
            and is_power_pool(expr, result.sharpe, None, registry)
        )
        # Chỉ đo self-corr (tốn 1 lệnh Brain API) khi alpha CÓ THỂ nộp được: đã passed Regular,
        # HOẶC đạt cấu trúc Power Pool. Alpha yếu không bao giờ nộp được nên khỏi tốn corr-check.
        self_corr = None
        if self.pool_corr_fn is not None and result.alpha_id and (passed or pp_structural):
            self_corr = self.pool_corr_fn(result.alpha_id)
            if passed and self_corr is not None and abs(self_corr) >= self.max_pool_corr:
                passed = False
        # power_pool_eligible ĐỘC LẬP với `passed` Regular: cấu trúc đạt + self_corr<=0.5.
        power_pool = pp_structural and is_power_pool(expr, result.sharpe, self_corr, registry)
        # Instrumentation Pha 0: stage đã sim; fail_check suy từ reasons hard_filter (KHÔNG vứt
        # _reasons như trước) + self-corr; family/depth để phân bố funnel.
        if passed:
            fail_check = ""
        elif self_corr is not None and abs(self_corr) >= self.max_pool_corr:
            fail_check = "SELF_CORR"
        else:
            fail_check = fail_check_from_reasons(_reasons)
        return IdeaOutcome(
            expr=expr, canonical_hash=canonical_hash, passed=passed,
            wq_alpha_id=result.alpha_id, sharpe=result.sharpe, fitness=result.fitness,
            turnover=result.turnover, self_corr=self_corr, sims_used=1,
            stop_reason=stop_reason, power_pool_eligible=power_pool,
            sim_settings=sim_cfg.to_settings(), source=source,
            stage_reached="passed" if passed else "simmed", fail_check=fail_check,
            family=classify_family(expr), expr_depth=DepthVisitor().visit(parse(expr)),
            dedup_key=canonical_hash, local_sharpe=local_sharpe,
            backtest_ms=backtest_ms, sim_ms=sim_ms, is_brain_sim=True,
        )

    def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome:
        # Depth guard (Pha 1.3): loại biểu thức quá sâu TRƯỚC mọi backtest (rẻ->đắt). Cây trần
        # đã > MAX_DEPTH thì cộng wrapper stack Brain (decay/neut/scale) chắc chắn vượt trần
        # depth -> WQ loại; backtest nó chỉ tốn thời gian vô ích. Sửa depth = làm phẳng core,
        # KHÔNG swap field (theo skill) — việc đó ở tầng generation, không phải ở đây.
        from config.thresholds import MAX_DEPTH

        try:
            _cand_node = parse(candidate.expr)
            _cand_depth = DepthVisitor().visit(_cand_node)
        except Exception:
            _cand_depth = None
        if _cand_depth is not None and _cand_depth > MAX_DEPTH:
            return IdeaOutcome(
                expr=candidate.expr,
                canonical_hash=CanonicalHasher().visit(_cand_node), passed=False,
                wq_alpha_id=None, sharpe=None, fitness=None, turnover=None,
                self_corr=None, sims_used=0, stop_reason="depth",
                stage_reached="depth", fail_check="DEPTH",
                family=classify_family(candidate.expr), expr_depth=_cand_depth,
                dedup_key=CanonicalHasher().visit(_cand_node), is_brain_sim=False,
            )
        # Seed alt-data (field ngoài panel local) đi thẳng Brain — không tune/floor local được.
        if self._is_alt_data(candidate.expr):
            return self._sim_direct(candidate)
        # Giai đoạn 1 (không mạng): coordinate descent quanh core + config, đánh giá bằng
        # đúng đường backtest local (Evaluator -> PortfolioBuilder -> Backtester -> Metrics).
        import time

        _t0 = time.perf_counter()
        _tune_kw = {"registry": self.registry}
        if self.neut_risk_factors:
            _tune_kw["neut_risk_factors"] = self.neut_risk_factors
        tr = self._tune(candidate.expr, self.local_config, self.data, **_tune_kw)
        backtest_ms = (time.perf_counter() - _t0) * 1000.0
        canonical_hash = CanonicalHasher().visit(parse(tr.best_expr))
        node = parse(tr.best_expr)
        _depth = DepthVisitor().visit(node)
        _family = classify_family(tr.best_expr)
        # Lưu local-eval để calibration ρ khớp hash với Brain sim ghi sau (chỉ khi có kho + metrics).
        if self.calib_repo is not None and tr.local_metrics is not None:
            self._luu_local_eval_calibration(tr, canonical_hash)
        # Sàn floor HIỆU LỰC phụ thuộc ρ (Task 5): ρ toàn cục đo độ tin ranking local so Brain.
        # ρ < rho_bar (VD 0.36 log thật) nghĩa ranking local hết tin -> floor local_sharpe chỉ
        # là NHIỄU, vừa giết oan ứng viên tốt (local thấp/Brain tốt) vừa không đáng dùng để lọc
        # -> TẮT floor (0.0), để Brain tự phân xử thay vì local. ρ≥rho_bar hoặc chưa đo được
        # (None) -> giữ floor calibrated như cũ (hành vi mặc định không đổi khi không có tracker).
        # TODO(per-family ρ): brain_local_sharpe_pairs() (repository.py:558) chưa gắn nhãn family
        # cho từng cặp -> chỉ làm được ρ TOÀN CỤC ở đây; ρ theo family cần schema DB mới (cột
        # family trên bảng lưu cặp local/Brain sharpe) + validate bằng chạy thật, ngoài scope task này.
        # local_untrusted tính MỘT LẦN, dùng chung cho CẢ floor lẫn gate sub_universe bên dưới
        # (Finding 1, follow-up a404874): sub_universe_ok cũng gate trên tr.local_metrics.sharpe —
        # ĐÚNG local-panel metric mà ρ vừa tuyên bố không tin. Nếu chỉ tắt floor mà vẫn để
        # sub_universe_ok chặn thì ρ-bypass bị vô hiệu hoá một nửa: ứng viên tốt Brain/xấu local
        # vẫn bị giết oan ở gate thứ hai, đúng vấn đề mà Task 5 định sửa. getattr(...) cho CẢ
        # last_rho lẫn rho_bar (Finding 2) để tracker duck-typed thiếu rho_bar không văng
        # AttributeError; thiếu rho_bar mặc định = "chưa đủ dữ liệu để phán KHÔNG tin" -> not
        # untrusted (dùng -inf làm ngưỡng để last_rho < rho_bar luôn False).
        effective_floor = self.min_local_sharpe
        tracker = self.calibration_tracker
        last_rho = getattr(tracker, "last_rho", None) if tracker is not None else None
        rho_bar = getattr(tracker, "rho_bar", float("-inf")) if tracker is not None else float("-inf")
        local_untrusted = last_rho is not None and last_rho < rho_bar
        if local_untrusted:
            effective_floor = 0.0
        if tr.local_sharpe < effective_floor:
            # Dưới sàn pre-sim CALIBRATED: KHÔNG gọi simulator -> sims_used=0, không tốn quota
            # Brain. Ghi cả local_sharpe ĐẠT ĐƯỢC và NGƯỠNG áp dụng vào stop_reason để audit
            # (Pha 4: floor là calibrated_floor(target/1.28), không còn hằng cứng).
            return IdeaOutcome(
                expr=tr.best_expr, canonical_hash=canonical_hash, passed=False,
                wq_alpha_id=None, sharpe=None, fitness=None, turnover=None,
                self_corr=None, sims_used=0,
                stop_reason=f"local_floor(<{effective_floor:.2f})",
                stage_reached="local_floor", fail_check="LOW_SHARPE", family=_family,
                expr_depth=_depth, dedup_key=canonical_hash, local_sharpe=tr.local_sharpe,
                backtest_ms=backtest_ms, is_brain_sim=False,
            )

        # Proxy robustness sub-universe (xấp xỉ sub-universe test của Brain): winner phải giữ
        # Sharpe khi giới hạn về nhóm mã thanh khoản nhất -> không đạt thì KHÔNG đốt quota Brain.
        # Chỉ chạy khi có local_metrics thật (tr.local_metrics is not None) -> test refiner cũ
        # dùng eval_fn giả (data=object(), không backtest thật) không bị vỡ vì gate này.
        # KHÔNG chạy khi local_untrusted (ρ thấp): gate này gate trên tr.local_metrics.sharpe —
        # cùng loại metric local mà ρ vừa nói không tin, chạy nó sẽ giết oan y hệt floor phía
        # trên -> phải bỏ qua ĐỒNG BỘ với floor, không chỉ tắt floor một mình.
        from src.backtest.sub_universe import sub_universe_ok

        registry = self.registry or default_registry()
        if not local_untrusted and tr.local_metrics is not None and not sub_universe_ok(
            parse(tr.best_expr), tr.best_config, self.data, registry,
            full_sharpe=tr.local_metrics.sharpe,
        ):
            return IdeaOutcome(
                expr=tr.best_expr, canonical_hash=canonical_hash, passed=False,
                wq_alpha_id=None, sharpe=None, fitness=None, turnover=None,
                self_corr=None, sims_used=0, stop_reason="sub_universe",
                stage_reached="sub_universe", fail_check="LOW_SUB_UNIVERSE_SHARPE",
                family=_family, expr_depth=_depth, dedup_key=canonical_hash,
                local_sharpe=tr.local_sharpe, backtest_ms=backtest_ms, is_brain_sim=False,
            )

        # Giai đoạn 2 (1 lần gọi mạng): sim Brain đúng config tốt nhất tune() tìm được —
        # bao gồm CẢ neutralization đã sweep (Task 1: MARKET/SECTOR), không chỉ decay/
        # truncation (thiếu neutralization ở đây từng khiến Brain sim luôn chạy default
        # SUBINDUSTRY dù local đã tune ra config tốt hơn). `.name` của enum Neutralization
        # (MARKET/SECTOR, chữ hoa) khớp thẳng VALID_NEUTRALIZATIONS của SimConfig.
        sim_cfg = self.sim_config.with_overrides(
            decay=tr.best_config.decay, truncation=tr.best_config.truncation,
            neutralization=tr.best_config.neutralization.name,
        )
        _ts = time.perf_counter()
        try:
            result = self.simulator.simulate(tr.best_expr, settings=sim_cfg.to_settings())
        except (AuthExpiredError, QuotaExceededError) as exc:
            # Cùng lý do như RefinementLoopRefiner: session chết hoặc hết quota ngày ->
            # ClosedLoop cần dừng gọn, không coi là "sim lỗi" rồi thử thêm ứng viên khác.
            raise QuotaExhausted(str(exc)) from exc
        sim_ms = (time.perf_counter() - _ts) * 1000.0

        if result.presim_reason is not None:
            # Chưa chạm Brain (PreFilter loại winner của tune()) -> outcome trung thực, KHÔNG
            # _finalize (nó luôn gán sims_used=1/stage='simmed', đúng cho sim thật nhưng SAI
            # ở đây — đây chính là bug spec C2: CSV giấu bug operator vì sim_ms≈0.7ms).
            return _presim_reject_outcome(
                tr.best_expr, canonical_hash, result.presim_reason,
                stop_reason="presim_reject", source="gp_local_tuner",
            )
        return self._finalize(
            result, tr.best_expr, canonical_hash, sim_cfg,
            stop_reason="local_tuned", source="gp_local_tuner", description="local-tuned motif",
            local_sharpe=tr.local_sharpe, backtest_ms=backtest_ms, sim_ms=sim_ms,
        )


# Core price/volume ĐÃ KIỂM CHỨNG trên Brain (commit d481fe1: intraday mean-reversion
# close↔vwap/open, Sharpe ~1.5+ và qua HẾT is.checks). Seed thẳng để LocalTuner tune quanh
# thay vì để GP random pha loãng — bằng chứng live: GP thường sinh biến thể yếu (local<0.5).
VERIFIED_CORES: tuple[str, ...] = (
    # Tổ hợp hai kênh intraday (biến thể đạt Sharpe cao nhất ~1.57).
    "add(multiply(2, multiply(-1, ts_mean(subtract(close, vwap), 10))), "
    "multiply(-1, ts_mean(subtract(close, open), 5)))",
    "add(multiply(1, multiply(-1, ts_mean(subtract(close, vwap), 10))), "
    "multiply(-1, ts_mean(subtract(close, open), 5)))",
    # Từng kênh riêng (đơn giản hơn -> hợp cấu trúc Power Pool: ít op/field).
    "multiply(-1, ts_mean(subtract(close, vwap), 10))",
    "multiply(-1, ts_mean(subtract(close, vwap), 20))",
    "multiply(-1, ts_mean(subtract(close, open), 5))",
    "multiply(-1, ts_mean(subtract(close, open), 10))",
)


class CuratedIdeaSource:
    """Yield các core ĐÃ KIỂM CHỨNG ở batch ĐẦU, rồi ủy quyền cho nguồn fallback (GP). Đưa hạt
    giống mạnh vào pipeline trước để LocalTuner tune quanh -> chạm alpha đạt chuẩn nộp nhanh,
    thay vì phụ thuộc GP random tình cờ sinh đúng cấu trúc thắng."""

    def __init__(self, *, fallback, cores: tuple[str, ...] = VERIFIED_CORES) -> None:
        self._fallback = fallback
        self._cores = tuple(cores)
        self._served_curated = False
        # Task 4: giống GPIdeaSource — lọc core của CHÍNH mình theo họ đã đóng, đồng thời ủy
        # quyền xuống fallback để cả chuỗi (GP ở cuối) cũng học tín hiệu này.
        self._saturated: set[str] = set()

    def set_saturated_families(self, fams: "set[str] | frozenset[str]") -> None:
        self._saturated = set(fams)
        if hasattr(self._fallback, "set_saturated_families"):
            self._fallback.set_saturated_families(fams)

    def next_batch(self):
        if not self._served_curated:
            self._served_curated = True
            import numpy as np

            empty = np.zeros(0, dtype=np.float64)
            dates = np.zeros(0, dtype="datetime64[ns]")
            cores = [e for e in self._cores if classify_family(e) not in self._saturated]
            if cores:
                return [
                    ShortlistCandidate(expr=e, metrics=None, pnl=empty, dates=dates)
                    for e in cores
                ]
            # Toàn bộ core curated thuộc họ đã đóng -> rơi thẳng xuống fallback thay vì trả
            # rỗng (rỗng ở đây KHÔNG có nghĩa "cạn ý tưởng", chỉ là curated không còn gì hợp).
            return self._fallback.next_batch()
        return self._fallback.next_batch()


class AltDataIdeaSource:
    """Yield các core ALT-DATA (option8/socialmedia8… — field ngoài panel local) ở batch ĐẦU,
    rồi ủy quyền fallback. Giống CuratedIdeaSource nhưng cho seed đi THẲNG Brain: refiner nhận
    diện qua `local_usable == False` và sim thẳng (không tune local). Mở rộng khỏi họ price/
    volume đã bão hòa -> alpha mới ít trùng self-corr (đòn bẩy chất lượng chính)."""

    def __init__(self, *, fallback, cores: tuple[str, ...] = ALT_DATA_CORES) -> None:
        self._fallback = fallback
        self._cores = tuple(cores)
        self._served = False
        # Task 4: cùng cơ chế lọc + ủy quyền như CuratedIdeaSource.
        self._saturated: set[str] = set()

    def set_saturated_families(self, fams: "set[str] | frozenset[str]") -> None:
        self._saturated = set(fams)
        if hasattr(self._fallback, "set_saturated_families"):
            self._fallback.set_saturated_families(fams)

    def next_batch(self):
        if not self._served:
            self._served = True
            import numpy as np

            empty = np.zeros(0, dtype=np.float64)
            dates = np.zeros(0, dtype="datetime64[ns]")
            cores = [e for e in self._cores if classify_family(e) not in self._saturated]
            if cores:
                return [
                    ShortlistCandidate(expr=e, metrics=None, pnl=empty, dates=dates)
                    for e in cores
                ]
            return self._fallback.next_batch()
        return self._fallback.next_batch()


class GPIdeaSource:
    """Nguồn ý tưởng cho ClosedLoop: mỗi next_batch() chạy GPEngine với seed MỚI (tăng dần để
    đa dạng) rồi rút short-list qua generate_many. Pool decorrelate lấy từ repo.load_pool()."""

    def __init__(
        self, data: object, repo: object, config: object, registry: object, *,
        pop_size: int = 30, n_generations: int = 3, base_seed: int = 42,
        top_k: int = 10, max_corr: float = 0.70, max_empty_retries: int = 8,
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
        self.max_empty_retries = max_empty_retries
        self._batch = 0
        # Task 4: họ đã đóng (ClosedLoop báo qua on_family_closed) -> lọc bỏ candidate cùng họ
        # TRƯỚC khi trả, tránh sinh mãi pv_reversal rồi bị ClosedLoop loại sau (tốn ~2 phút/batch).
        self._saturated: set[str] = set()

    def set_saturated_families(self, fams: "set[str] | frozenset[str]") -> None:
        """Nhận tập họ vừa đóng (ClosedLoop truyền TOÀN BỘ closed_families mỗi lần, tích luỹ
        dần) -> thay thế set hiện tại (không union thủ công vì nguồn đã là snapshot đầy đủ)."""
        self._saturated = set(fams)

    def _run_one_batch(self) -> list[ShortlistCandidate]:
        seed = self.base_seed + self._batch
        seed_offset = self._batch * self.pop_size
        self._batch += 1
        engine = GPEngine(
            data=self._data, repo=self._repo, config=self._config, registry=self._registry,
            pop_size=self.pop_size, n_generations=self.n_generations, seed=seed,
            seed_offset=seed_offset,
        )
        pool: Any = self._repo.load_pool() or None
        # GPEngine.run() -> GPRunResult; Protocol _RunsGP đòi _GPRunResultLike với
        # list[_GPIndividualLike] — list là invariant nên cast qua Any để truyền qua.
        engine_any: Any = engine
        return generate_many(
            gp_engine=engine_any, cfg=self._config, data=self._data,
            top_k=self.top_k, max_corr=self.max_corr, pool=pool,
        )

    def next_batch(self) -> list[ShortlistCandidate]:
        # Một quần thể GP (1 seed) có thể tình cờ 0 ứng viên qua gate/decorrelate — đừng
        # vội kết luận "cạn ý tưởng" (no_more_ideas) chỉ vì 1 seed xui. Thử tới
        # max_empty_retries lô (seed khác nhau) rồi mới trả rỗng thật sự.
        for _ in range(max(1, self.max_empty_retries)):
            batch = self._run_one_batch()
            if self._saturated:
                # Lọc SAU sinh (đủ rẻ — sinh đã xong, lọc chỉ là classify_family theo chuỗi):
                # nếu lọc hết sạch, coi như lô này "rỗng" -> thử seed khác thay vì trả candidate
                # thuộc họ đã đóng.
                batch = [c for c in batch if classify_family(c.expr) not in self._saturated]
            if batch:
                return batch
        return []


class CombinerIdeaSource:
    """Nối tiếp mỗi batch bằng các ALPHA GHÉP: gom tín hiệu con của batch (có PnL local) +
    kho alpha tốt trong DB, chọn greedy khử tương quan (spec 2026-07-09), dựng biểu thức
    add(rank(...)) trọng số đều, chấm local, chỉ THÊM combo qua gate & vượt tín hiệu con tốt
    nhất. Combo = tổ hợp tín hiệu ít tương quan -> Sharpe ~√N (Grinold–Kahn) -> dễ chạm ngưỡng
    nộp hơn alpha đơn. Bọc quanh fallback (GP/curated/alt-data) — combo không thay thế batch
    gốc mà bổ sung; refiner tune 1 neutralization cho combo (biểu thức đã tước group_neutralize)."""

    def __init__(
        self, *, fallback, data: object, repo: object, config: object, registry: object,
        tau: float = 0.30, n_min: int = 2, n_max: int = 4, max_combos: int = 5,
        db_limit: int = 50,
    ) -> None:
        self._fallback = fallback
        self._data: Any = data
        self._repo: Any = repo
        self._config: Any = config
        self._registry: Any = registry
        self.tau = tau
        self.n_min = n_min
        self.n_max = n_max
        self.max_combos = max_combos
        self.db_limit = db_limit
        # Task 4: combo tự dựng cũng có thể rơi vào họ đã đóng (vd toàn tín hiệu con pv_reversal
        # ghép lại) -> lọc chính combo của mình, đồng thời ủy quyền xuống fallback.
        self._saturated: set[str] = set()

    def set_saturated_families(self, fams: "set[str] | frozenset[str]") -> None:
        self._saturated = set(fams)
        if hasattr(self._fallback, "set_saturated_families"):
            self._fallback.set_saturated_families(fams)

    def _score_fn(self, pool):
        def score(expr: str):
            return _score_one_full(expr, self._config, self._data, pool)
        return score

    def _signals(self, batch: list[ShortlistCandidate]) -> list[SubSignal]:
        """Tín hiệu con ứng viên = candidate batch CÓ PnL local (curated/alt-data pnl rỗng bị
        bỏ) + kho alpha tốt DB. score = fitness để xếp seed."""
        sigs: list[SubSignal] = []
        for c in batch:
            if c.metrics is not None and getattr(c.pnl, "size", 0) > 0:
                sigs.append(SubSignal(c.expr, c.pnl, c.dates, c.metrics.fitness, source="run"))
        for expr, dates, pnl, fitness in self._repo.good_signals_for_combine(limit=self.db_limit):
            sigs.append(SubSignal(expr, pnl, dates, fitness, source="db"))
        return sigs

    def next_batch(self) -> list[ShortlistCandidate]:
        batch = self._fallback.next_batch()
        if not batch:
            return batch
        signals = self._signals(batch)
        if len(signals) < self.n_min:
            return batch
        pool: Any = self._repo.load_pool() or None
        combos = combine_stage(
            signals, self._score_fn(pool), tau=self.tau, n_min=self.n_min,
            n_max=self.n_max, max_combos=self.max_combos, registry=self._registry,
        )
        if self._saturated:
            combos = [c for c in combos if classify_family(c.expr) not in self._saturated]
        return batch + combos


def build_closed_loop(
    *, data: object, repo: object, config: object, registry: object, loop: object,
    region: str = "USA", universe: str = "TOP3000",
    pop_size: int = 30, n_generations: int = 3, base_seed: int = 42,
    top_k: int = 10, max_corr: float = 0.70,
    calibrate_every: int = 10, rho_bar: float = 0.5, max_ideas: int | None = None,
    refiner: object | None = None, curated_seeds: bool = True,
    include_alt_data: bool = True, alpha_logger: object | None = None,
    include_combiner: bool = True, session_summary: object | None = None,
    include_fundamental: bool = True, max_per_family: int | None = 8,
    idea_generator: object | None = None,
) -> "ClosedLoop":
    """Ráp vòng kín: GPIdeaSource (sinh ý tưởng) + refiner (mặc định RefinementLoopRefiner
    bọc `loop` AI thật; truyền `refiner` tường minh — vd LocalTunerRefiner (Task 4) — để bỏ
    qua LLM refine, chỉ tune local rồi sim Brain 1 lần) + CalibrationTracker (ρ) + ClosedLoop.
    `loop` là RefinementLoop đã dựng (đăng nhập + Simulator thật) do composition root
    (main.py) truyền vào; không dùng tới khi đã truyền `refiner` tường minh."""
    from src.pipeline.closed_loop import CalibrationTracker, ClosedLoop

    idea_source: object = GPIdeaSource(
        data, repo, config, registry, pop_size=pop_size, n_generations=n_generations,
        base_seed=base_seed, top_k=top_k, max_corr=max_corr,
    )
    # Thử core price/volume ĐÃ KIỂM CHỨNG (Brain ~1.5+) TRƯỚC, rồi mới tới GP random — hạt
    # giống mạnh vào pipeline sớm để LocalTuner tune quanh -> chạm alpha đạt chuẩn nộp nhanh.
    if curated_seeds:
        idea_source = CuratedIdeaSource(fallback=idea_source)
    # Alt-data đặt NGOÀI CÙNG -> phục vụ ở batch đầu (trước cả curated PV) để phiên ngắn/
    # --max-ideas nhỏ vẫn chạm alt-data (đòn bẩy độ mới), không bị PV core nuốt hết quota.
    # Alt-data + fundamental: field ngoài panel local -> refiner sim thẳng Brain. GỘP cores vào
    # MỘT batch đầu (không bọc lồng nhiều tầng) để phiên ngắn (--max-ideas nhỏ) chạm cả hai họ
    # mới cùng lúc (IMPROVEMENT_SPEC §2.1: thoát cụm PV/VWAP bão hòa bằng nhiều họ orthogonal).
    direct_cores: tuple[str, ...] = ()
    if include_alt_data:
        direct_cores += ALT_DATA_CORES
    if include_fundamental:
        direct_cores += FUNDAMENTAL_CORES
    if direct_cores:
        idea_source = AltDataIdeaSource(fallback=idea_source, cores=direct_cores)
    # Combiner bọc NGOÀI CÙNG: nối tiếp mỗi batch (sau curated/alt-data) bằng alpha ghép từ
    # chính tín hiệu con batch đó + kho DB -> tự động chạy sau mỗi run (spec 2026-07-09).
    if include_combiner:
        idea_source = CombinerIdeaSource(
            fallback=idea_source, data=data, repo=repo, config=config, registry=registry,
        )
    if refiner is None:
        refiner = RefinementLoopRefiner(loop)
    tracker = CalibrationTracker(repo, every=calibrate_every, rho_bar=rho_bar)  # type: ignore[arg-type]
    # Task 5: nối tracker vào refiner (nếu refiner biết đọc ρ, vd LocalTunerRefiner) — CÙNG một
    # object mà ClosedLoop cập nhật last_rho mỗi maybe_calibrate() nên refiner luôn thấy ρ mới nhất.
    if hasattr(refiner, "set_calibration_tracker"):
        refiner.set_calibration_tracker(tracker)  # type: ignore[attr-defined]
    # Dedup key = canonical hash đã fold scale dương (Pha 1.2). Tiêm vào ClosedLoop để dedup
    # TRƯỚC refine bắt cả biến thể scale; parse lỗi -> fallback chuỗi thô (không chặn oan).
    _hasher = CanonicalHasher(registry if registry is not None else default_registry())

    def _dedup_key(expr: str) -> str:
        try:
            return _hasher.visit(parse(expr))
        except Exception:
            return expr

    # Family-aware budget (Pha 2.2): classify_family suy họ từ field/cấu trúc; ClosedLoop đóng
    # họ khi cạn max_per_family mà 0 pass -> chuyển ngân sách sang họ orthogonal (yield).
    from src.reporting.diagnostics import classify_family

    # Nối exhaustion guard -> ĐƯỜNG THẬT (Task 4, sửa bug Pha 2.3): idea_source LÀ chuỗi
    # generator chạy thật (GPIdeaSource/CuratedIdeaSource/AltDataIdeaSource/CombinerIdeaSource
    # dựng ở trên) — bug cũ chỉ nối on_family_closed khi truyền `idea_generator` (LLM re-seed
    # riêng, mặc định None ở đường research) nên tín hiệu đóng họ KHÔNG BAO GIỜ tới chuỗi thật,
    # cứ sinh mãi pv_reversal rồi bị ClosedLoop loại sau khi đã tốn ~2 phút/batch. `idea_source`
    # (wrapper ngoài cùng) LUÔN có `set_saturated_families` (mọi wrapper đều tự lọc + ủy quyền
    # xuống fallback) nên gọi thẳng, không cần hasattr guard; vẫn gọi thêm idea_generator nếu có
    # để giữ tương thích ngược với đường LLM re-seed (test Pha 2.3).
    def on_family_closed(fams: set[str]) -> None:
        idea_source.set_saturated_families(fams)  # type: ignore[attr-defined]
        if idea_generator is not None and hasattr(idea_generator, "set_saturated_families"):
            idea_generator.set_saturated_families(fams)

    return ClosedLoop(
        idea_source=idea_source, refiner=refiner, repo=repo,  # type: ignore[arg-type]
        region=region, universe=universe, max_ideas=max_ideas,
        calibration_tracker=tracker, alpha_logger=alpha_logger,
        session_summary=session_summary, dedup_key_fn=_dedup_key,
        family_fn=classify_family, max_per_family=max_per_family,
        on_family_closed=on_family_closed,
    )
