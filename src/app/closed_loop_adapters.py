"""Adapter nối thành phần thật vào ClosedLoop (Phase 2). Tầng composition: được phép import
src.gp/src.llm/src.pipeline/src.lang (khác src/pipeline vốn cấm src.llm/src.gp theo B1).

- RefinementLoopRefiner: bọc RefinementLoop.run_from_seed (4A) → IdeaOutcome.
- GPIdeaSource: bọc generate_many (Phase 8) với seed GPEngine tăng dần → nguồn ý tưởng."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

    from src.pipeline.closed_loop import ClosedLoop

from loguru import logger

from config.thresholds import (
    ALT_SWEEP_MIN_ABS_SHARPE,
    COMBINER_MIN_BRAIN_SHARPE,
    DEGENERATE_SHARPE,
    DEGENERATE_TURNOVER,
    SUBMIT_FITNESS_REF,
    SUBMIT_SHARPE_REF,
    calibrated_floor,
)
from src.backtest.gate import local_usable
from src.backtest.local_tuner import tune as _tune
from src.generation.alt_data_seeds import (
    ALT_DATA_CORES,
    neutralization_for_expr,
    pp_neutralization_for_expr,
)
from src.generation.field_verification import filter_seeds_by_verified_fields
from src.generation.frontier_seeds import FRONTIER_CORES
from src.generation.near_miss_variants import NearMissVariantSource
from src.generation.fundamental_seeds import FUNDAMENTAL_CORES
from src.generation.hypothesis_seeds import HYPOTHESIS_CORES
from src.generation.combiner import SubSignal
from src.gp.engine import GPEngine
from src.gp.parallel_eval import khoi_tao_worker
from src.lang.ast import Call, Constant
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import (
    CanonicalHasher,
    DepthVisitor,
    FieldCollector,
    OperatorCollector,
    Serializer,
)
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
    backtest_ms: float | None = None, sim_ms: float | None = None,
) -> IdeaOutcome:
    """Ứng viên bị `PreFilter` loại TRƯỚC khi chạm Brain (Task 3, spec C2: đừng giả vờ
    'simmed/LOW_SHARPE' như bug cũ — CSV giấu bug operator/field bịa vì sim_ms≈0 nhưng
    stage_reached vẫn ghi 'simmed'). Outcome trung thực: sims_used=0 (chưa tốn quota Brain),
    is_brain_sim=False, stage/fail_check suy từ chính category của presim_reason. Giữ `source`
    (nhánh nào sinh) + timing đã tính (backtest_ms nếu đã tune, sim_ms lần gọi pre-check) để
    CSV không bỏ trống cột chẩn đoán — chính là dữ liệu Task 3 cần để soi presim-reject."""
    code = categorize_presim_reason(presim_reason)
    stage = _PRESIM_STAGE_BY_CODE.get(code, "presim")
    try:
        depth = DepthVisitor().visit(parse(expr))
    except Exception:
        depth = None
    return IdeaOutcome(
        expr=expr, canonical_hash=canonical_hash, passed=False,
        wq_alpha_id=None, sharpe=None, fitness=None, turnover=None, self_corr=None,
        sims_used=0, stop_reason=stop_reason, source=source,
        stage_reached=stage, fail_check=code, family=classify_family(expr),
        expr_depth=depth, dedup_key=canonical_hash,
        presim_reason=presim_reason, is_brain_sim=False,
        backtest_ms=backtest_ms, sim_ms=sim_ms,
    )


def _flip_sign(expr: str) -> str:
    """Đảo dấu biểu thức bằng AST (Task 5, KHÔNG xử lý chuỗi): nếu gốc đã là dạng
    `multiply(-1, X)` thì BÓC thành X (tránh bọc chồng multiply(-1, multiply(-1, X))); ngược
    lại BỌC `multiply(-1, <expr>)`. Dùng khi sim core alt-data ra sharpe quá âm — bằng chứng
    thật: seed social từng SAI DẤU (Sharpe -0.48 lẽ ra +0.48 nếu được flip thay vì vứt thẳng
    hypothesis kinh tế đằng sau nó)."""
    node = parse(expr)
    if (
        isinstance(node, Call) and node.op == "multiply" and len(node.args) == 2
        and isinstance(node.args[0], Constant) and node.args[0].value == -1
    ):
        flipped = node.args[1]
    else:
        flipped = Call(op="multiply", args=(Constant(value=-1.0), node))
    return Serializer().visit(flipped)


def _neutralization_for(expr: str, pp_allowed: frozenset[str], registry: Any) -> str:
    """Chọn neutralization cho 1 biểu thức alt-data — DÙNG CHUNG giữa `LocalTunerRefiner.
    _sim_direct` (sim từng candidate) và `AltDataIdeaSource` (Task 6: tiền-sim CẢ BATCH qua
    `simulate_many` TRƯỚC khi `_sim_direct` chạy) để 2 nơi tính RA ĐÚNG CÙNG MỘT settings cho
    cùng 1 expr — tách hàm này để không có 2 bản logic có thể trôi lệch nhau theo thời gian."""
    if pp_allowed:
        return pp_neutralization_for_expr(expr, pp_allowed, registry)
    return neutralization_for_expr(expr, registry)


def _submit_score(sharpe: float | None, fitness: float | None) -> float:
    """Điểm-nộp (cùng công thức combine_stage._submit_score, Task 2 Fix 4):
    min(sharpe/SUBMIT_SHARPE_REF, fitness/SUBMIT_FITNESS_REF) — đo một kết quả sim tiến GẦN
    NGƯỠNG NỘP thật tới đâu trên CẢ HAI trục. sharpe/fitness None (sim lỗi/presim) -> -inf,
    không bao giờ được chọn làm 'tốt nhất' khi còn ứng viên có số liệu thật."""
    if sharpe is None or fitness is None:
        return float("-inf")
    return min(sharpe / SUBMIT_SHARPE_REF, fitness / SUBMIT_FITNESS_REF)


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
        alt_sweep_budget: int = 2,
        presim_cache: "dict[str, Any] | None" = None,
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
        # Risk factor để tune bọc vector_neut hạ self-corr (Pha 3.1; đổi từ regression_neut —
        # account không có op đó trong catalog live, xem Task 1); None -> không thử.
        self.neut_risk_factors = neut_risk_factors
        # CalibrationTracker (src/pipeline/closed_loop.py) — cho biết ρ hiện tại giữa ranking
        # local và Brain có đáng tin không (Task 5). None -> hành vi cũ y nguyên (floor cứng).
        self.calibration_tracker = calibration_tracker
        # Ngân sách mini-sweep alt-data (Task 5): số sim THÊM tối đa sau sim core (flip dấu/
        # decay khác) cho MỘT hypothesis — mặc định 2, tức tối đa 1 + alt_sweep_budget sim thật
        # cho một expr alt-data. 0 = tắt sweep (giữ hành vi cũ: đúng 1 sim/ý tưởng).
        self.alt_sweep_budget = alt_sweep_budget
        # Task 6: cache kết quả sim CORE đã chạy TRƯỚC (multi-sim gộp batch, xem
        # AltDataIdeaSource) — khoá bằng expr thô. `_sim_direct` đọc (pop) cache TRƯỚC khi tự
        # gọi `simulator.simulate()`, tránh sim lại lần 2 cho core đã sim trong batch multi-sim.
        # None/rỗng (mặc định) -> hành vi CŨ y nguyên (luôn tự sim, tương thích ngược).
        self.presim_cache = presim_cache

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
        """Nhánh alt-data: BỎ tune/floor local (panel không có field), sim Brain core 1 lần với
        neutralization chọn theo category dataset (docs WQ), rồi mini-sweep CÓ KỶ LUẬT (Task 5):
        mỗi hypothesis kinh tế đáng được cứu bằng ≤ `alt_sweep_budget` sim thêm thay vì vứt sau
        1 sim (bằng chứng: seed social từng SAI DẤU, analyst revision 1-shot 0.64 rồi bỏ).
        Sim core sharpe <= -ALT_SWEEP_MIN_ABS_SHARPE -> thử flip dấu; sharpe >= +ngưỡng nhưng
        chưa pass -> thử decay khác quanh best-so-far. Dừng khi hết ngân sách hoặc
        `status == 'passed'`. Outcome cuối = sim có điểm-nộp cao nhất trong TOÀN BỘ lần đã sim."""
        expr = candidate.expr
        neut = _neutralization_for(expr, self.pp_allowed_neutralizations, self.registry)
        sim_cfg = self.sim_config.with_overrides(neutralization=neut)
        # Task 6: core đã được sim TRƯỚC theo batch (AltDataIdeaSource.next_batch gọi
        # simulate_many 1 lần cho cả nhóm) -> dùng lại kết quả, KHÔNG sim lại lần 2. `pop` để
        # cache không phình lên qua nhiều batch (mỗi core chỉ tiêu thụ đúng 1 lần).
        cached = self.presim_cache.pop(expr, None) if self.presim_cache else None
        if cached is not None:
            result = cached
        else:
            try:
                result = self.simulator.simulate(expr, settings=sim_cfg.to_settings())
            except (AuthExpiredError, QuotaExceededError) as exc:
                raise QuotaExhausted(str(exc)) from exc
        if result.presim_reason is not None:
            # Chưa chạm Brain (PreFilter loại) -> outcome trung thực, KHÔNG sweep (biểu thức
            # gốc đã sai cấu trúc thì flip dấu/đổi decay cũng vô nghĩa) và KHÔNG _finalize (nó
            # luôn gán sims_used=1/stage='simmed', đúng cho sim thật nhưng SAI ở đây).
            return _presim_reject_outcome(
                expr, CanonicalHasher().visit(parse(expr)), result.presim_reason,
                stop_reason="presim_reject", source="alt_data",
            )
        # Mini-sweep: mỗi phần tử là (expr, sim_cfg, result) của MỘT sim THẬT đã chạy.
        attempts: list[tuple[str, Any, Any]] = [(expr, sim_cfg, result)]
        sims_used = 1
        budget_left = self.alt_sweep_budget
        cur_expr, cur_cfg, cur_result = expr, sim_cfg, result
        # Finding reviewer #1 (CRITICAL): toggle decay qua lại (4<->8) không nhớ config đã thử
        # -> ở lần thứ 3 có thể quay lại ĐÚNG (expr, decay) của sim #1 -> sim TRÙNG y hệt, đốt
        # quota + tạo alpha trùng vô ích. Nhớ tập (expr, cfg.key()) đã sim trong vòng sweep này
        # để chặn biến thể trùng thay vì sim lại.
        tried: set[tuple[str, str]] = {(cur_expr, cur_cfg.key())}
        try:
            while budget_left > 0 and cur_result.status != "passed":
                sharpe = cur_result.sharpe
                if sharpe is None:
                    break  # không đủ tín hiệu để quyết định sweep tiếp -> dừng an toàn
                if sharpe <= -ALT_SWEEP_MIN_ABS_SHARPE:
                    next_expr, next_cfg = _flip_sign(cur_expr), cur_cfg
                    logger.info(
                        "Sweep alt-data: sharpe {:.2f} quá âm -> thử FLIP DẤU: {!r}",
                        sharpe, next_expr,
                    )
                elif sharpe >= ALT_SWEEP_MIN_ABS_SHARPE:
                    next_decay = 8 if cur_cfg.decay == 4 else 4
                    next_expr, next_cfg = cur_expr, cur_cfg.with_overrides(decay=next_decay)
                    logger.info(
                        "Sweep alt-data: sharpe {:.2f} dương nhưng chưa pass -> thử decay {}->{}",
                        sharpe, cur_cfg.decay, next_decay,
                    )
                else:
                    break  # |sharpe| < ngưỡng -> chưa đủ tín hiệu để biết nên flip hay đổi decay
                next_key = (next_expr, next_cfg.key())
                if next_key in tried:
                    logger.info(
                        "Sweep alt-data: biến thể {!r} TRÙNG cấu hình đã sim -> dừng sweep, "
                        "không đốt thêm quota.", next_key,
                    )
                    break
                tried.add(next_key)
                try:
                    next_result = self.simulator.simulate(
                        next_expr, settings=next_cfg.to_settings(),
                    )
                except (AuthExpiredError, QuotaExceededError) as exc:
                    raise QuotaExhausted(str(exc)) from exc
                budget_left -= 1
                if next_result.presim_reason is not None:
                    # Biến thể sweep (hiếm) bị PreFilter chặn -> không phải sim thật, dừng sweep
                    # thay vì cố đưa presim reject vào so điểm-nộp (sharpe/fitness None).
                    break
                sims_used += 1
                attempts.append((next_expr, next_cfg, next_result))
                cur_expr, cur_cfg, cur_result = next_expr, next_cfg, next_result
        except QuotaExhausted:
            # Finding #5 (review): hết quota GIỮA sweep -> không còn "best" để _finalize (đã
            # ném exception, hàm sẽ thoát ngay ở re-raise dưới) nhưng MỌI attempt trong
            # `attempts` (kể cả sim #1) ĐÃ sim THẬT, đã đốt quota + tạo alpha thật trên Brain —
            # phải persist qua `_persist_sweep_attempt_thua` (không _finalize: tránh gọi thêm
            # pool_corr_fn có thể tốn quota) TRƯỚC KHI re-raise, nếu không attempt đã sim mất
            # trắng dấu vết audit/calibration (alpha mồ côi trên Brain).
            for a_expr, a_cfg, a_result in attempts:
                self._persist_sweep_attempt_thua(a_expr, a_cfg, a_result)
            raise
        best_idx, (best_expr, best_cfg, best_result) = max(
            enumerate(attempts), key=lambda ia: _submit_score(ia[1][2].sharpe, ia[1][2].fitness)
        )
        # Finding reviewer (Important): sim Brain THẬT không thắng vẫn ĐÃ đốt quota và tạo
        # alpha thật trên platform (wq_alpha_id) -> phải có bản ghi local (save_alpha +
        # save_simulation) để không mất dữ liệu calibration/audit (alpha mồ côi trên Brain).
        # Outcome trả về VẪN chỉ 1 (best) — chỉ persist, không tạo thêm IdeaOutcome.
        for i, (a_expr, a_cfg, a_result) in enumerate(attempts):
            if i == best_idx:
                continue  # bản thắng do _finalize lưu (kèm chấm Power Pool/self-corr đầy đủ)
            self._persist_sweep_attempt_thua(a_expr, a_cfg, a_result)
        return self._finalize(
            best_result, best_expr, CanonicalHasher().visit(parse(best_expr)), best_cfg,
            stop_reason="alt_data_direct", source="alt_data", description="alt-data direct",
            sims_used=sims_used,
        )

    def _persist_sweep_attempt_thua(self, expr: str, sim_cfg, result) -> None:
        """Lưu 1 attempt sweep THUA (sim Brain thật nhưng không được chọn làm outcome cuối) —
        cùng repo/pattern `_finalize` dùng (save_alpha + save_simulation, score qua
        score_vector_fn) nhưng KHÔNG chấm Power Pool/self-corr (không đáng tốn corr-check cho
        bản thua) và KHÔNG tạo IdeaOutcome (hợp đồng refine_and_sim: 1 outcome duy nhất)."""
        vector = self.score_vector_fn(result)
        alpha_id = self.repo.save_alpha(
            expr, source="alt_data", hypothesis={},
            description="alt-data sweep attempt (thua)", parent_id=None,
        )
        self.repo.save_simulation(
            result, region=self.region, universe=self.universe,
            score=vector.total, alpha_id=alpha_id, config_key=sim_cfg.key(),
        )
        logger.info(
            "Sweep alt-data: attempt THUA đã lưu — expr={!r} wq_alpha_id={} sharpe={}",
            expr if len(expr) <= 80 else expr[:77] + "...", result.alpha_id, result.sharpe,
        )

    def _finalize(
        self, result, expr: str, canonical_hash: str, sim_cfg, *,
        stop_reason: str, source: str, description: str,
        local_sharpe: float | None = None, backtest_ms: float | None = None,
        sim_ms: float | None = None, sims_used: int = 1,
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
            turnover=result.turnover, self_corr=self_corr, sims_used=sims_used,
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
        # Gate "degenerate position" (Task 4, backtest-cheap): turnover local GẦN NHƯ 0 VÀ
        # |sharpe| local GẦN NHƯ 0 ĐỒNG THỜI -> vị thế suy biến/gần hằng số (backtest local
        # KHÔNG suy biến hoàn toàn nhưng vẫn lộ dấu hiệu vô nghĩa mà rule AST structural
        # (src/lang/meaningfulness.py) không chắc bắt được, vd base khác sign(...) hoặc field
        # trộn lẫn). Chạy TRƯỚC gate floor/ρ-untrusted (Task 5) — đây là backstop CẤU TRÚC độc
        # lập với việc có tin ranking local hay không, nên KHÔNG được phép bị ρ-bypass tắt như
        # floor thường. Bằng chứng thật (log 07-12): các biểu thức dạng này đốt sim Brain thật
        # ra đúng Sharpe 0.00/turnover 0.00.
        if tr.local_metrics is not None:
            m = tr.local_metrics
            if m.turnover < DEGENERATE_TURNOVER and abs(m.sharpe) < DEGENERATE_SHARPE:
                return IdeaOutcome(
                    expr=tr.best_expr, canonical_hash=canonical_hash, passed=False,
                    wq_alpha_id=None, sharpe=None, fitness=None, turnover=None,
                    self_corr=None, sims_used=0, stop_reason="degenerate_position",
                    stage_reached="degenerate", fail_check="DEGENERATE_POSITION",
                    family=_family, expr_depth=_depth, dedup_key=canonical_hash,
                    local_sharpe=tr.local_sharpe, backtest_ms=backtest_ms, is_brain_sim=False,
                )
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
                backtest_ms=backtest_ms, sim_ms=sim_ms,
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

    def __init__(
        self, *, fallback, cores: tuple[str, ...] = VERIFIED_CORES,
        avoided_hashes: "set[str] | None" = None,
        dedup_key_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._fallback = fallback
        self._cores = tuple(cores)
        self._served_curated = False
        # Task 4: giống GPIdeaSource — lọc core của CHÍNH mình theo họ đã đóng, đồng thời ủy
        # quyền xuống fallback để cả chuỗi (GP ở cuối) cũng học tín hiệu này.
        self._saturated: set[str] = set()
        # Task 6 (RC7 lean fix): lọc core ĐÃ Brain-sim & lưu tried_hashes phiên trước tại
        # NGUỒN — tránh phục vụ lại core bão hoà, tốn slot batch curated + gây dedup-block
        # log spam ở ClosedLoop.run. Cả hai None -> không lọc gì (tương thích ngược).
        self._avoided_hashes = avoided_hashes
        self._dedup_key_fn = dedup_key_fn

    def set_saturated_families(self, fams: "set[str] | frozenset[str]") -> None:
        self._saturated = set(fams)
        if hasattr(self._fallback, "set_saturated_families"):
            self._fallback.set_saturated_families(fams)

    def set_gp_budget_exhausted(self, flag: bool) -> None:
        if hasattr(self._fallback, "set_gp_budget_exhausted"):
            self._fallback.set_gp_budget_exhausted(flag)

    def reseed_epoch(self) -> None:
        """B1: uỷ quyền xuống fallback (cùng pattern set_saturated_families/
        set_gp_budget_exhausted) — CuratedIdeaSource tự nó không có epoch/seed, việc reseed
        thật xảy ra ở GPIdeaSource cuối chuỗi."""
        if hasattr(self._fallback, "reseed_epoch"):
            self._fallback.reseed_epoch()

    def next_batch(self):
        if not self._served_curated:
            self._served_curated = True
            import numpy as np

            empty = np.zeros(0, dtype=np.float64)
            dates = np.zeros(0, dtype="datetime64[ns]")
            cores = [e for e in self._cores if classify_family(e) not in self._saturated]
            cores = _drop_saturated_cores(
                cores, dedup_key_fn=self._dedup_key_fn,
                avoided_hashes=self._avoided_hashes, label="Curated",
            )
            if cores:
                return [
                    ShortlistCandidate(
                        expr=e, metrics=None, pnl=empty, dates=dates, origin="curated",
                    )
                    for e in cores
                ]
            # Toàn bộ core curated thuộc họ đã đóng (hoặc đã bị lọc bởi avoided_hashes) -> rơi
            # thẳng xuống fallback thay vì trả rỗng (rỗng ở đây KHÔNG có nghĩa "cạn ý tưởng").
            return self._fallback.next_batch()
        return self._fallback.next_batch()


def _drop_saturated_cores(
    cores: "list[str]",
    *,
    dedup_key_fn,
    avoided_hashes,
    label: str,
) -> "list[str]":
    """Lọc bỏ core đã sim & bão hoà (dedup_key nằm trong avoided_hashes cross-session).

    Log 1 dòng INFO tóm tắt SỐ LƯỢNG thay vì 1 dòng/core — kho seed trực tiếp đã cạn sau
    nhiều phiên nên mỗi lần khởi động dội ~60 dòng lặp lại (phàn nàn user 2026-07-18);
    chi tiết từng core hạ xuống DEBUG. avoided_hashes/dedup_key_fn None -> pass-through
    (tương thích đường chưa inject)."""
    if avoided_hashes is None or dedup_key_fn is None:
        return list(cores)
    kept: list[str] = []
    skipped: list[str] = []
    for e in cores:
        (skipped if dedup_key_fn(e) in avoided_hashes else kept).append(e)
    if skipped:
        logger.info(
            "{}: bỏ {} core đã sim & bão hoà, không phục vụ lại (chi tiết ở mức DEBUG).",
            label, len(skipped),
        )
        for e in skipped:
            logger.debug("{}: core bão hoà: {!r}", label, e)
    return kept


def _filter_known_fields(
    cores: tuple[str, ...], known_fields: "frozenset[str] | set[str] | None", registry: Any,
) -> tuple[str, ...]:
    """Field-validity guard (RC1/RC2 fix idea-generator): lọc BỎ core mà `FieldCollector`
    thu được field KHÔNG nằm trong `known_fields` (catalog cache thật của account) TRƯỚC khi
    core được yield đi sim — chặn đúng lỗi tốn quota/sai dấu nghiêm trọng nhất của dự án (field
    bịa gửi thẳng Brain). `known_fields=None` -> KHÔNG lọc gì (tương thích ngược, catalog chưa
    load). Field nhóm (sector/industry/market…, `_POWER_POOL_GROUPS`) luôn coi là hợp lệ vì
    không phải field DỮ LIỆU (dù thực tế cores hiện tại không dùng group_neutralize nên hiếm
    khi chạm nhánh này — vẫn trừ ra để đúng tinh thần "chỉ lọc field dữ liệu thật, không lọc oan
    pseudo-field", tránh false-positive nếu sau này có core dùng group).
    Core parse lỗi (hiếm, bug soạn thảo) -> LOẠI + log, không để lọt lên Brain."""
    if known_fields is None:
        return cores
    kept: list[str] = []
    for expr in cores:
        try:
            fields = FieldCollector(registry).visit(parse(expr)) - _POWER_POOL_GROUPS
        except Exception as exc:
            logger.info("Field guard: bỏ qua core (parse lỗi) {!r}: {}", expr, exc)
            continue
        missing = fields - set(known_fields)
        if missing:
            logger.info(
                "Field guard: bỏ qua core {!r} — field không có trong catalog cache: {}",
                expr, sorted(missing),
            )
            continue
        kept.append(expr)
    return tuple(kept)


class AltDataIdeaSource:
    """Yield các core ALT-DATA (option8/socialmedia8… — field ngoài panel local) ở batch ĐẦU,
    rồi ủy quyền fallback. Giống CuratedIdeaSource nhưng cho seed đi THẲNG Brain: refiner nhận
    diện qua `local_usable == False` và sim thẳng (không tune local). Mở rộng khỏi họ price/
    volume đã bão hòa -> alpha mới ít trùng self-corr (đòn bẩy chất lượng chính).

    `known_fields` (field-validity guard): khi truyền (khác None), MỌI core có field không
    nằm trong tập này bị lọc bỏ ở constructor (một lần, không lặp lại mỗi next_batch) — core đó
    KHÔNG BAO GIỜ được yield ra, tức KHÔNG BAO GIỜ chạm Brain sim.

    `simulator`/`sim_config`/`pp_allowed_neutralizations`/`presim_cache` (Task 6, tất cả tùy
    chọn, mặc định None -> TẮT tính năng, hành vi cũ y nguyên): khi đủ 4 tham số này, batch core
    ĐẦU TIÊN được sim CẢ NHÓM 1 lần qua `simulator.simulate_many` (thay vì mỗi core đợi
    `LocalTunerRefiner._sim_direct` tự sim tuần tự sau này) — kết quả ghi vào `presim_cache`
    (dict dùng CHUNG với refiner, khoá bằng expr thô) để `_sim_direct` đọc lại thay vì sim lần 2.
    `avoided_hashes`/`dedup_key_fn`: lọc core ĐÃ Brain-sim & lưu tried_hashes phiên trước TRƯỚC
    khi đưa vào batch multi-sim (giống CuratedIdeaSource) — quan trọng để KHÔNG gửi core trùng
    (đã sim phiên trước) vào payload multi-sim, vừa lãng phí 1 slot mảng (tối đa 10, xem
    `Simulator.MULTI_SIM_MAX`) vừa tốn quota vô ích (ClosedLoop.run() dù sao cũng sẽ tự bỏ core
    trùng qua `seen`, nhưng lúc đó quota multi-sim đã tốn rồi).

    `presim_cap` (review Finding #1): trần số core được sim TRƯỚC theo batch — build_closed_loop
    wire = `max_ideas` của phiên. Không có trần này, phiên `--max-ideas` nhỏ sẽ sim cả nhóm
    (vd 6 core) rồi ClosedLoop chỉ tiêu thụ 2: 4 kết quả sim THẬT còn lại bị vứt (không
    `_finalize`, không vào avoided_hashes) -> PHIÊN SAU sim lại đúng các core đó, lãng phí quota
    lặp vô hạn. Core vượt trần vẫn được yield làm candidate bình thường (đi đường sim đơn trong
    `_sim_direct` nếu tới lượt — KHÔNG mất). None (mặc định) = không trần."""

    def __init__(
        self, *, fallback, cores: tuple[str, ...] = ALT_DATA_CORES,
        known_fields: "frozenset[str] | set[str] | None" = None, registry: Any = None,
        avoided_hashes: "set[str] | None" = None,
        dedup_key_fn: Callable[[str], str] | None = None,
        simulator: Any = None, sim_config: Any = None,
        pp_allowed_neutralizations: frozenset[str] = frozenset(),
        presim_cache: "dict[str, Any] | None" = None,
        presim_cap: int | None = None,
    ) -> None:
        self._fallback = fallback
        self._registry = registry if registry is not None else default_registry()
        self._cores = _filter_known_fields(tuple(cores), known_fields, self._registry)
        self._served = False
        # Task 4: cùng cơ chế lọc + ủy quyền như CuratedIdeaSource.
        self._saturated: set[str] = set()
        self._avoided_hashes = avoided_hashes
        self._dedup_key_fn = dedup_key_fn
        self._simulator = simulator
        self._sim_config = sim_config
        self._pp_allowed_neutralizations = pp_allowed_neutralizations
        self._presim_cache = presim_cache
        self._presim_cap = presim_cap

    def set_saturated_families(self, fams: "set[str] | frozenset[str]") -> None:
        self._saturated = set(fams)
        if hasattr(self._fallback, "set_saturated_families"):
            self._fallback.set_saturated_families(fams)

    def set_gp_budget_exhausted(self, flag: bool) -> None:
        if hasattr(self._fallback, "set_gp_budget_exhausted"):
            self._fallback.set_gp_budget_exhausted(flag)

    def reseed_epoch(self) -> None:
        """B1: uỷ quyền xuống fallback, cùng pattern set_saturated_families/
        set_gp_budget_exhausted (xem CuratedIdeaSource.reseed_epoch)."""
        if hasattr(self._fallback, "reseed_epoch"):
            self._fallback.reseed_epoch()

    def _presim_batch(self, cores: list[str]) -> None:
        """Task 6: sim NHÓM `cores` (cắt trần `presim_cap` — xem docstring class, Finding #1)
        1 lần qua `simulate_many` (thay vì N lần tuần tự sau này trong `_sim_direct`), ghi kết
        quả vào `self._presim_cache` (khoá = expr thô — cùng khoá `_sim_direct` dùng để tra
        lại). CHỈ chạy khi đủ cấu hình (simulator/sim_config/cache) VÀ nhóm sau khi cắt trần
        còn ≥2 core (khớp giới hạn API: mảng multi-sim cần ≥2 phần tử — 1 core thì tự
        `_sim_direct` sim đường đơn như cũ, khỏi tốn thêm 1 lần round-trip vô ích).

        LỖI BẤT KỲ (multi-sim hỏng, mất mạng, quota cạn...) -> log warning, KHÔNG cache gì —
        `_sim_direct` sẽ tự sim tuần tự từng core như hành vi CŨ (an toàn, không chết phiên;
        quota cạn thật vẫn được `_sim_direct` phát hiện + báo QuotaExhausted đúng như trước,
        chỉ trễ hơn 1 nhịp vì phải chạm Brain lại ở đường đơn)."""
        if self._simulator is None or self._sim_config is None or self._presim_cache is None:
            return
        # Finding #1: chỉ sim trước tối đa `presim_cap` core (= max_ideas phiên) — sim vượt
        # trần sẽ bị ClosedLoop vứt kết quả (không _finalize/avoided_hashes) rồi PHIÊN SAU sim
        # lại, lãng phí quota lặp. Core bị cắt vẫn nằm trong batch yield (không mất).
        batch_cores = cores if self._presim_cap is None else cores[: self._presim_cap]
        if len(batch_cores) < 2:
            return
        jobs = [
            (expr, self._sim_config.with_overrides(
                neutralization=_neutralization_for(expr, self._pp_allowed_neutralizations, self._registry)
            ).to_settings())
            for expr in batch_cores
        ]
        try:
            results = self._simulator.simulate_many(jobs)
        except Exception as exc:  # noqa: BLE001 - fallback cố ý bắt rộng, xem docstring.
            logger.warning(
                "AltDataIdeaSource: multi-sim batch lỗi ({}) — fallback sim tuần tự từng core.",
                exc,
            )
            return
        for expr, result in zip(batch_cores, results):
            self._presim_cache[expr] = result

    def next_batch(self):
        if not self._served:
            self._served = True
            import numpy as np

            empty = np.zeros(0, dtype=np.float64)
            dates = np.zeros(0, dtype="datetime64[ns]")
            cores = [e for e in self._cores if classify_family(e) not in self._saturated]
            cores = _drop_saturated_cores(
                cores, dedup_key_fn=self._dedup_key_fn,
                avoided_hashes=self._avoided_hashes, label="AltData",
            )
            if cores:
                self._presim_batch(cores)
                return [
                    ShortlistCandidate(
                        expr=e, metrics=None, pnl=empty, dates=dates, origin="alt_data",
                    )
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
        top_k: int = 10, max_corr: float = 0.70, max_empty_retries: int = 2,
        field_groups: "tuple[tuple[str, ...], ...] | None" = None,
        n_jobs: int = 1,
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
        # C1: song song hoá backtest thuần — n_jobs=1 (mặc định) KHÔNG tạo pool, GPEngine đi
        # đường tuần tự y hệt trước C1 (CLI/menu nối n_jobs thật ở task sau, ngoài phạm vi
        # C1). n_jobs>1: dựng ProcessPoolExecutor MỘT LẦN ở đây, sống xuyên suốt mọi batch/
        # GPEngine của phiên (không tạo/hủy pool mỗi batch — khởi động worker process tốn
        # thời gian). initializer nạp data/config/registry MỘT LẦN mỗi worker (không pickle
        # lại mỗi task, xem khoi_tao_worker).
        self.n_jobs = n_jobs
        self._executor: "ProcessPoolExecutor | None" = None
        if n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor

            self._executor = _ProcessPoolExecutor(
                max_workers=n_jobs, initializer=khoi_tao_worker,
                initargs=(data, config, registry),
            )
        # B1: nhóm field theo dataset (dataset ÍT field trước — proxy "ít dùng"), để xoay
        # sang nhóm khác mỗi epoch reseed (xem reseed_epoch/_run_one_batch bên dưới). None =
        # không xoay (dùng toàn bộ field mọi epoch, tương thích ngược).
        self.field_groups = field_groups
        # B1: đếm epoch hiện tại (0 = epoch gốc, dùng toàn bộ field như cũ). Tăng dần mỗi lần
        # reseed_epoch() được gọi (ClosedLoop gọi khi batch rỗng nhưng chưa chắc đã cạn hẳn).
        self._epoch = 0
        # Task 4: họ đã đóng (ClosedLoop báo qua on_family_closed) -> lọc bỏ candidate cùng họ
        # TRƯỚC khi trả, tránh sinh mãi pv_reversal rồi bị ClosedLoop loại sau (tốn ~2 phút/batch).
        self._saturated: set[str] = set()
        # A1: ClosedLoop báo trần sim GP/phiên đã chạm -> next_batch bỏ hẳn tiến hoá (xem
        # set_gp_budget_exhausted bên dưới).
        self._gp_budget_exhausted = False
        # A3: cache in-memory cấp phiên (canonical_hash -> kết quả eval THUẦN), CHIA SẺ xuyên
        # mọi GPEngine dựng ở _run_one_batch -> biểu thức trùng giữa các seed/batch không bị
        # backtest lại từ đầu. Cap 5000 entry tránh phình bộ nhớ vô hạn trong phiên dài.
        self._eval_cache: "dict[str, tuple]" = {}

    def close(self) -> None:
        """C1: đóng pool process (nếu có) — gọi khi phiên ClosedLoop kết thúc. An toàn gọi
        nhiều lần / khi n_jobs=1 (không có pool để đóng, no-op)."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def set_saturated_families(self, fams: "set[str] | frozenset[str]") -> None:
        """Nhận tập họ vừa đóng (ClosedLoop truyền TOÀN BỘ closed_families mỗi lần, tích luỹ
        dần) -> thay thế set hiện tại (không union thủ công vì nguồn đã là snapshot đầy đủ)."""
        self._saturated = set(fams)

    def set_gp_budget_exhausted(self, flag: bool) -> None:
        """A1: ClosedLoop báo trần sim GP/phiên đã chạm (True) — next_batch bỏ hẳn tiến hoá
        (kết quả trước giờ vẫn bị vứt ở gate gp_budget, bỏ chạy = không mất gì, tiết kiệm
        3–14 phút/batch). Epoch reseed (B1) gọi lại với False để mở lại."""
        self._gp_budget_exhausted = flag

    def reseed_epoch(self) -> None:
        """B1: mở epoch mới khi ClosedLoop báo batch rỗng (chưa chắc đã cạn tuyệt đối) — đổi
        seed (base_seed += 10_000, đủ xa để không trùng lô seed cũ), reset batch counter (epoch
        mới bắt đầu lại từ seed_offset=0), MỞ LẠI ngân sách sim GP (gp_budget_exhausted=False —
        trần cũ thuộc epoch trước, epoch mới là một đợt tìm kiếm mới). GIỮ NGUYÊN
        `_saturated` (họ đã đóng vẫn đóng — không "quên" bài học trong phiên) và `_eval_cache`
        (backtest thuần là hàm xác định của (expr, config, data), không phụ thuộc epoch)."""
        self._epoch += 1
        self.base_seed += 10_000
        self._batch = 0
        self._gp_budget_exhausted = False

    def _run_one_batch(self) -> list[ShortlistCandidate]:
        seed = self.base_seed + self._batch
        seed_offset = self._batch * self.pop_size
        self._batch += 1
        # A3: cache đã phình quá cỡ -> clear để tránh phình vô hạn trong phiên dài (đủ để cache
        # hit trong đa số cửa sổ batch gần nhau, không cần bền vững cả phiên).
        if len(self._eval_cache) > 5000:
            self._eval_cache.clear()
        # B1: epoch 0 (gốc) luôn dùng TOÀN BỘ field như cũ (fields_override=None). Từ epoch 1
        # trở đi (đã reseed ít nhất 1 lần) mới xoay sang nhóm field ưu tiên tiếp theo — nếu
        # field_groups không được tiêm (composition root không xác định được dataset), giữ
        # nguyên None mọi epoch.
        fields_override = (
            self.field_groups[self._epoch % len(self.field_groups)]
            if self.field_groups and self._epoch > 0
            else None
        )
        engine = GPEngine(
            data=self._data, repo=self._repo, config=self._config, registry=self._registry,
            pop_size=self.pop_size, n_generations=self.n_generations, seed=seed,
            seed_offset=seed_offset,
            # A2: truyền họ đã đóng xuống GPEngine -> lọc TRƯỚC backtest trong GP thay vì chỉ
            # lọc SAU sinh ở next_batch() (defense in depth — lọc sau vẫn giữ nguyên bên dưới).
            saturated_families=self._saturated,
            # A3: cache backtest thuần CHIA SẺ xuyên mọi GPEngine dựng trong phiên này.
            eval_cache=self._eval_cache,
            # C1: pool process CHIA SẺ (nếu n_jobs>1) — mọi GPEngine của phiên dùng CHUNG một
            # executor sống xuyên batch (không tạo/hủy pool mỗi batch).
            n_jobs=self.n_jobs,
            executor=self._executor,
            # B1: nhóm field ưu tiên của epoch hiện tại (None = toàn bộ field, epoch 0 hoặc
            # không có field_groups).
            fields_override=fields_override,
            # B2: two-stage sampling CHỈ có ý nghĩa ở epoch 0 (toàn bộ field, dataset đông/ít
            # field cùng cạnh tranh). Từ epoch 1 trở đi, fields_override đã thu về MỘT nhóm
            # duy nhất (xem trên) -> two-stage thừa (chỉ 1 nhóm thì mọi field trong đó vẫn
            # uniform phẳng như cũ), nên truyền None để không đổi hành vi.
            field_groups=self.field_groups if self._epoch == 0 else None,
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
        # A1: trần ngân sách sim GP/phiên đã chạm -> bỏ hẳn tiến hoá (không dựng GPEngine),
        # kết quả trước giờ vẫn bị ClosedLoop vứt ở gate gp_budget nên bỏ chạy không mất gì.
        if self._gp_budget_exhausted:
            return []
        # Một quần thể GP (1 seed) có thể tình cờ 0 ứng viên qua gate/decorrelate — đừng
        # vội kết luận "cạn ý tưởng" (no_more_ideas) chỉ vì 1 seed xui. Thử tới
        # max_empty_retries lô (seed khác nhau) rồi mới trả rỗng thật sự.
        # A4: từ khi A2 chuyển lọc họ-đóng + degenerate vào TRONG tiến hoá (trước backtest,
        # xem saturated_families ở _run_one_batch), lô rỗng không còn là "xui vì lọc sau-sinh"
        # (default cũ 8 là để bù rủi ro đó) — giờ nó gần như luôn nghĩa là cạn ý tưởng thật sự,
        # nên default hạ xuống 2 lần thử seed khác là đủ chống nhiễu ngẫu nhiên còn lại.
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
        # Review fix: trần số expr Brain-proven lấy từ DB mỗi batch — DB tích luỹ vô hạn,
        # không giới hạn thì _signals backtest ngày càng nhiều expr mỗi next_batch().
        self.db_limit = db_limit
        # Review fix: cache PnL local của expr nguồn "db" — panel local BẤT BIẾN trong phiên
        # nên backtest cùng expr luôn ra cùng PnL; không cache thì MỖI next_batch() (vòng
        # while closed-loop) backtest lại TOÀN BỘ danh sách (~20s/expr) vô ích. Giá trị None
        # = kết quả ÂM (không local-usable / backtest lỗi / 0 PnL) — cũng cache để không thử
        # lại. Sống theo đời CombinerIdeaSource (1 phiên).
        self._db_pnl_cache: dict[str, tuple[Any, Any] | None] = {}
        # Task 4: combo tự dựng cũng có thể rơi vào họ đã đóng (vd toàn tín hiệu con pv_reversal
        # ghép lại) -> lọc chính combo của mình, đồng thời ủy quyền xuống fallback.
        self._saturated: set[str] = set()
        # Task 7: instrumentation — thống kê CUỘC GỌI next_batch() gần nhất (n_run/n_db/
        # total tín hiệu, có bị skip vì < n_min hay không, số combo sinh ra). Log qua loguru
        # NGOÀI ra để soi trực tiếp trong CSV/console; `last_stats` cho test/tool đọc lại mà
        # không cần bắt log — CombinerIdeaSource từng "0 combo" âm thầm vì _signals() vứt
        # oan candidate curated pnl rỗng (xem _signals) mà không ai biết TẠI SAO.
        self.last_stats: dict[str, int | bool] = {}

    def set_saturated_families(self, fams: "set[str] | frozenset[str]") -> None:
        self._saturated = set(fams)
        if hasattr(self._fallback, "set_saturated_families"):
            self._fallback.set_saturated_families(fams)

    def set_gp_budget_exhausted(self, flag: bool) -> None:
        if hasattr(self._fallback, "set_gp_budget_exhausted"):
            self._fallback.set_gp_budget_exhausted(flag)

    def reseed_epoch(self) -> None:
        """B1: uỷ quyền xuống fallback, cùng pattern set_saturated_families/
        set_gp_budget_exhausted (xem CuratedIdeaSource.reseed_epoch)."""
        if hasattr(self._fallback, "reseed_epoch"):
            self._fallback.reseed_epoch()

    def _score_fn(self, pool):
        def score(expr: str):
            return _score_one_full(expr, self._config, self._data, pool)
        return score

    def _score_fn_factory(self, others: list[SubSignal]):
        """Fix 2 (Task 2): pool chấm gate combo = PnL local của CHÍNH các tín hiệu Brain-
        proven NGOÀI combo (dict tự đánh số — `SubSignal` không có eval_id), KHÔNG PHẢI
        `repo.load_pool()` (1321+ eval LOCAL bão hòa). Đo được (logs/diag_combiner_20260712.md,
        20260713.md): self-corr THẬT của Brain cho vùng price/volume này chỉ 0.40-0.46, trong
        khi pool 1321+ eval local giết oan combo với self_corr đo được 0.70-0.86 — proxy local
        sai, không phản ánh pool alpha Brain thật của account. `combine_stage` gọi lại hàm này
        cho MỖI combo với `others` đã loại chính thành phần combo đó -> khử luôn tự-so."""
        pool: Any = {i: (s.dates, s.pnl) for i, s in enumerate(others)} or None
        return self._score_fn(pool)

    def _local_backtest(self, expr: str) -> Any:
        """Backtest local qua `_score_one_full` NẾU expr local-usable (field nằm trong
        panel); trả `_ScoreResult` hoặc None nếu không dùng được (alt-data ngoài panel,
        parse/eval lỗi, hoặc backtest ra 0 ngày PnL). Dùng chung cho cả tín hiệu "run"
        (candidate batch chưa có pnl, kiểu curated/alt-data) lẫn "db" (Fix 1 Task 2:
        `brain_proven_signals` chỉ trả expr+sharpe, KHÔNG có PnL sẵn — phải tự backtest)."""
        try:
            # data thiếu field_names() thật (fake/object() trong test cũ) -> coi như
            # local-usable, GIỮ hành vi cũ (khớp `_is_alt_data` của LocalTunerRefiner).
            usable = local_usable(expr, self._data)
        except Exception:
            usable = True
        if not usable:
            return None
        try:
            res = _score_one_full(expr, self._config, self._data)
        except Exception as exc:
            # Backtest lỗi (config/data giả trong test, hoặc expr rơi vào nhánh lỗi hiếm)
            # -> bỏ qua candidate này, KHÔNG crash cả next_batch vì 1 ứng viên xấu.
            logger.debug("CombinerIdeaSource._local_backtest: backtest lỗi cho {!r}: {}", expr, exc)
            return None
        if res.pnl.size == 0:
            return None
        return res

    def _signals(self, batch: list[ShortlistCandidate]) -> list[SubSignal]:
        """Tín hiệu con ứng viên = candidate batch CÓ PnL local + kho alpha Brain-proven DB.

        Candidate batch CHƯA có pnl local (kiểu curated/alt-data: `metrics=None`, `pnl`
        rỗng — `CuratedIdeaSource`/`AltDataIdeaSource` yield thẳng core KHÔNG tự backtest)
        KHÔNG còn bị vứt thẳng như trước (bug gốc khiến combo luôn = 0 khi DB rỗng): nếu
        expr local-usable (field nằm trong panel) thì backtest NGAY bằng đúng đường
        `_score_one_full` (parse->eval->portfolio->backtest->metrics — CÙNG đường tuner/
        generate_many dùng, không tự chế lại) để lấy pnl/fitness làm tín hiệu con. Alt-data
        thật (field ngoài panel local, vd option8/socialmedia8) vẫn bị loại — không có cách
        chấm local. score = fitness (local, đo TRỰC TIẾP từ backtest run này) để xếp seed.

        Nguồn "db" (Task 2 Fix 1, thay `good_signals_for_combine`): calibration đo được
        ρ=0.308 giữa fitness LOCAL và sharpe Brain (`logs/diag_combiner_20260712.md`) — xếp
        theo fitness local chọn toàn GP junk, các core Brain-proven KHÔNG có mặt. Lấy
        `repo.brain_proven_signals(COMBINER_MIN_BRAIN_SHARPE)` (expr, sharpe Brain THẬT),
        backtest local từng expr để lấy PnL (đo tương quan) nhưng SCORE = sharpe Brain (không
        phải fitness local vừa đo) — vì sharpe Brain mới là thước đo đáng tin theo calibration.
        Expr không local-usable (không chấm nổi tương quan cục bộ) bị bỏ."""
        sigs: list[SubSignal] = []
        for c in batch:
            if c.metrics is not None and getattr(c.pnl, "size", 0) > 0:
                sigs.append(SubSignal(c.expr, c.pnl, c.dates, c.metrics.fitness, source="run"))
                continue
            res = self._local_backtest(c.expr)
            if res is None:
                continue
            sigs.append(SubSignal(c.expr, res.pnl, res.dates, res.metrics.fitness, source="run"))
        for expr, sharpe in self._repo.brain_proven_signals(
            COMBINER_MIN_BRAIN_SHARPE, limit=self.db_limit,
        ):
            # Cache theo expr (review fix): chỉ backtest expr CHƯA gặp; query DB vẫn chạy
            # mỗi batch (rẻ) nên sharpe luôn tươi — cache chỉ giữ (pnl, dates) bất biến.
            if expr not in self._db_pnl_cache:
                res = self._local_backtest(expr)
                self._db_pnl_cache[expr] = None if res is None else (res.pnl, res.dates)
            cached = self._db_pnl_cache[expr]
            if cached is None:  # kết quả âm đã cache — không local-usable/backtest lỗi
                continue
            pnl, dates = cached
            sigs.append(SubSignal(expr, pnl, dates, sharpe, source="db"))
        return sigs

    def next_batch(self) -> list[ShortlistCandidate]:
        batch = self._fallback.next_batch()
        if not batch:
            return batch
        signals = self._signals(batch)
        n_run = sum(1 for s in signals if s.source == "run")
        n_db = sum(1 for s in signals if s.source == "db")
        if len(signals) < self.n_min:
            self.last_stats = {
                "n_run_signals": n_run, "n_db_signals": n_db,
                "total_signals": len(signals), "skipped": True, "n_combos": 0,
            }
            logger.info(
                "CombinerIdeaSource: n_run={} n_db={} total={} < n_min={} -> bỏ qua (0 combo)",
                n_run, n_db, len(signals), self.n_min,
            )
            return batch
        # Fix 2 (Task 2): score_fn cũ (pool=None) chỉ là fallback tương thích chữ ký —
        # score_fn_factory (pool = tín hiệu Brain-proven NGOÀI combo) ưu tiên dùng thật sự,
        # KHÔNG còn gọi repo.load_pool() (1321+ eval LOCAL bão hòa, xem _score_fn_factory).
        # Fix 4: drop_stats mới, riêng cho lần gọi này — combine_stage mutate in-place.
        drop_stats: dict[str, int] = {}
        combos = combine_stage(
            signals, self._score_fn(None), tau=self.tau, n_min=self.n_min,
            n_max=self.n_max, max_combos=self.max_combos, registry=self._registry,
            score_fn_factory=self._score_fn_factory, drop_stats=drop_stats,
        )
        if self._saturated:
            combos = [c for c in combos if classify_family(c.expr) not in self._saturated]
        self.last_stats = {
            "n_run_signals": n_run, "n_db_signals": n_db,
            "total_signals": len(signals), "skipped": False, "n_combos": len(combos),
            # Fix 4: gộp drop_stats vào last_stats (mặc định 0 nếu tầng đó không rớt gì) để
            # tool/test đọc lại chẩn đoán "TẠI SAO 0 combo" mà không cần bắt log.
            "depth": drop_stats.get("depth", 0), "gate": drop_stats.get("gate", 0),
            "not_better": drop_stats.get("not_better", 0),
            "greedy_empty": drop_stats.get("greedy_empty", 0),
        }
        logger.info(
            "CombinerIdeaSource: n_run={} n_db={} total={} -> {} combo",
            n_run, n_db, len(signals), len(combos),
        )
        logger.info(
            "Combiner drop: depth={} gate={} not_better={} greedy_empty={}",
            self.last_stats["depth"], self.last_stats["gate"],
            self.last_stats["not_better"], self.last_stats["greedy_empty"],
        )
        return batch + combos


def _gather_direct_cores(
    include_alt_data: bool, include_fundamental: bool,
    include_hypothesis: bool, include_frontier: bool,
) -> tuple[str, ...]:
    """Gom core đường sim-thẳng theo cờ — tách hàm để test wire không cần dựng cả loop.
    Frontier đặt SAU kho cũ: kho cũ đã bão hoà (saturation skip bỏ qua nhanh), frontier
    là nguồn mới chiếm phần lớn batch đầu."""
    cores: tuple[str, ...] = ()
    if include_alt_data:
        cores += ALT_DATA_CORES
    if include_fundamental:
        cores += FUNDAMENTAL_CORES
    if include_hypothesis:
        cores += HYPOTHESIS_CORES
    if include_frontier:
        cores += FRONTIER_CORES
    return cores


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
    idea_generator: object | None = None, include_hypothesis: bool = True,
    known_fields: "frozenset[str] | set[str] | None" = None,
    max_gp_sims: int | None = 3, include_frontier: bool = True,
    verified_fields: "frozenset[str] | None" = None,
) -> "ClosedLoop":
    """Ráp vòng kín: GPIdeaSource (sinh ý tưởng) + refiner (mặc định RefinementLoopRefiner
    bọc `loop` AI thật; truyền `refiner` tường minh — vd LocalTunerRefiner (Task 4) — để bỏ
    qua LLM refine, chỉ tune local rồi sim Brain 1 lần) + CalibrationTracker (ρ) + ClosedLoop.
    `loop` là RefinementLoop đã dựng (đăng nhập + Simulator thật) do composition root
    (main.py) truyền vào; không dùng tới khi đã truyền `refiner` tường minh.

    `known_fields`: catalog field cache thật của account (vd `set(_cached_symbols(...)[0])`
    ở main.py) — field-validity guard (RC1/RC2): core alt-data/fundamental/hypothesis nào tham
    chiếu field KHÔNG nằm trong tập này bị lọc bỏ, KHÔNG BAO GIỜ gửi lên Brain sim (chỉ log).
    None (mặc định) -> KHÔNG lọc gì (tương thích ngược khi chưa/không load được catalog).

    `max_gp_sims`: trần sim Brain THẬT/phiên riêng cho candidate origin "gp" (Task 3) — GP là
    nguồn nhiễu (2 phiên gần nhất đốt ~10 sim ra toàn Sharpe ≤0.31, calibration ρ=0.308 không
    đủ tin để lọc trước). Mặc định 3; ưu tiên quota còn lại cho curated/alt_data/combiner
    (không bị cap này). None = không cap (tương thích ngược).

    `verified_fields` (WS3 T3.3, cardinal rule #1): tập field ĐÃ verify LIVE (vd từ
    `src.generation.field_verification.load_latest_verified_fields(Path("logs"))`, do CLI/
    main.py load rồi truyền vào — build_closed_loop KHÔNG tự đọc file, giữ hàm test được).
    Core alt-data/frontier/fundamental/hypothesis dùng field KHÔNG nằm trong tập này bị loại
    TRƯỚC khi tới AltDataIdeaSource/frontier_reserve (không bao giờ chạm sim), kèm log WARNING.
    None (mặc định) -> FAIL-OPEN, không lọc gì (tương thích ngược, khớp quyết định T3.3: thiếu
    bằng chứng verify không nên chặn oan seed thật)."""
    from src.pipeline.closed_loop import CalibrationTracker, ClosedLoop, compute_avoided_hashes

    # Dedup key = canonical hash đã fold scale dương (Pha 1.2). Định nghĩa TRƯỚC khi dựng
    # CuratedIdeaSource (Task 6) để wrapper này pha loãng đúng core ĐÃ Brain-sim & lưu
    # tried_hashes phiên trước — cùng không gian hash với ClosedLoop.run/_dedup_key ở dưới.
    _hasher = CanonicalHasher(registry if registry is not None else default_registry())

    def _dedup_key(expr: str) -> str:
        try:
            return _hasher.visit(parse(expr))
        except Exception:
            return expr

    # Task 6 (RC7 lean fix) + Finding #2 (review): nạp avoided-hashes MỘT LẦN tại build time —
    # DÙNG CHUNG `compute_avoided_hashes` với `ClosedLoop.run` (UNION avoided_hashes ∪
    # avoided_hashes_original ∪ dedup(avoided_exprs)) để lọc core tại NGUỒN (Curated/
    # AltDataIdeaSource) KHỚP đúng tập `seen` mà run() dùng để chặn refine trùng — trước đây
    # chỉ nạp avoided_hashes_original(), bỏ lọt core đã Brain-sim-fail phiên trước (chỉ có
    # trong avoided_hashes()/avoided_exprs()) khiến nó lọt qua lọc nguồn, đốt quota thật rồi
    # mới bị `seen` ở run() chặn — không còn dấu vết audit.
    avoided_hashes: "set[str] | None" = compute_avoided_hashes(repo, _dedup_key)

    # dataset_of_fields (nếu repo có — MiniBrainRepository) phục vụ CẢ combo cùng-dataset của
    # NearMissVariantSource (dưới) LẪN xoay nhóm field ưu tiên theo epoch reseed (B1) — lấy
    # sớm để dùng chung, tránh định nghĩa trùng. Chữ ký thật (MiniBrainRepository.
    # dataset_of_fields) nhận `set[str]` field_ids, trả {field_id: dataset_id} (field không có
    # trong catalog hoặc dataset_id NULL thì vắng mặt trong map).
    _ds_fn = getattr(repo, "dataset_of_fields", None)
    # B1: nhóm field theo dataset cho xoay epoch (ưu tiên originality — dataset ÍT field lên
    # trước, proxy "ít dùng"). repo không có dataset_of_fields (test giả) hoặc data không có
    # field_names() -> None, không xoay (mọi epoch dùng toàn bộ field như cũ).
    field_groups: "tuple[tuple[str, ...], ...] | None" = None
    if callable(_ds_fn) and hasattr(data, "field_names"):
        try:
            _mapping = _ds_fn(set(data.field_names()))  # {field: dataset}
            _by_ds: dict[str, list[str]] = {}
            for f, ds in _mapping.items():
                _by_ds.setdefault(ds or "khac", []).append(f)
            if len(_by_ds) >= 2:
                # dataset ÍT field trước (proxy "ít dùng"), pv lớn xuống cuối
                field_groups = tuple(
                    tuple(sorted(fs)) for _, fs in sorted(_by_ds.items(), key=lambda kv: len(kv[1]))
                )
        except Exception:
            field_groups = None

    idea_source: object = GPIdeaSource(
        data, repo, config, registry, pop_size=pop_size, n_generations=n_generations,
        base_seed=base_seed, top_k=top_k, max_corr=max_corr, field_groups=field_groups,
    )
    # Thử core price/volume ĐÃ KIỂM CHỨNG (Brain ~1.5+) TRƯỚC, rồi mới tới GP random — hạt
    # giống mạnh vào pipeline sớm để LocalTuner tune quanh -> chạm alpha đạt chuẩn nộp nhanh.
    # Task 6: truyền avoided_hashes+dedup_key_fn để prune core ĐÃ bão hoà tại NGUỒN, tránh
    # tốn slot batch curated + dedup-block log spam ở ClosedLoop.run cho core biết trước sẽ
    # bị chặn.
    if curated_seeds:
        idea_source = CuratedIdeaSource(
            fallback=idea_source, avoided_hashes=avoided_hashes, dedup_key_fn=_dedup_key,
        )
    # Refiner resolve TRƯỚC AltDataIdeaSource (Task 6): cần simulator/sim_config/pp_allowed_
    # neutralizations của refiner để AltDataIdeaSource tự tính ĐÚNG settings cho batch multi-sim
    # (dùng chung `_neutralization_for`, xem docstring AltDataIdeaSource).
    if refiner is None:
        refiner = RefinementLoopRefiner(loop)
    # Task 6: cache dùng CHUNG giữa AltDataIdeaSource (ghi, sau khi simulate_many) và refiner
    # (đọc trong _sim_direct) — chỉ gắn khi refiner có thuộc tính `presim_cache` (LocalTunerRefiner;
    # RefinementLoopRefiner/refiner giả trong test không có -> giữ nguyên hành vi cũ, KHÔNG batch).
    _presim_cache: "dict[str, Any] | None" = None
    if hasattr(refiner, "presim_cache"):
        _presim_cache = {}
        refiner.presim_cache = _presim_cache  # type: ignore[attr-defined]
    # Near-miss variant expander (bằng chứng log 2026-07-16: 389 core alt-data bão hoà sau
    # 1 sim/core -> vòng kín rơi về GP nhiễu best Sharpe 0.68 suốt ~6h): chen GIỮA alt-data
    # và curated/GP — khi kho core cạn, sinh biến thể window/wrapper quanh near-miss Brain-sim
    # (Sharpe [0.6, 1.0)) thay vì nhảy thẳng về GP. Lọc avoid-hashes cùng không gian _dedup_key.
    # dataset_of_fields (nếu repo có — MiniBrainRepository) bật thêm combo CÙNG-DATASET:
    # ghép cặp near-miss chung dataset thành rank(add(a,b)) (bài học KP9Aw3lj 2026-07-16:
    # combo 2 field order_flow_imb thắng Sharpe 1.03 vs biến thể đơn lẻ <=0.9). `_ds_fn` đã
    # lấy sớm hơn (đầu hàm, cùng chỗ tính field_groups cho B1) — dùng lại, không định nghĩa
    # trùng.
    idea_source = NearMissVariantSource(
        repo=repo, fallback=idea_source,
        dedup_key_fn=_dedup_key, avoided_hashes=avoided_hashes,
        dataset_of_fields_fn=_ds_fn if callable(_ds_fn) else None,
    )
    # Alt-data đặt NGOÀI CÙNG -> phục vụ ở batch đầu (trước cả curated PV) để phiên ngắn/
    # --max-ideas nhỏ vẫn chạm alt-data (đòn bẩy độ mới), không bị PV core nuốt hết quota.
    # CHỈ alt-data "thật" (option8/socialmedia8...) đi đường AltDataIdeaSource (giữ nguyên cơ
    # chế multi-sim theo batch, Task 6) — fundamental/hypothesis/frontier tách ra `frontier_
    # pool_cores` bên dưới, nạp vào `ClosedLoop.frontier_reserve` (WS3 T3.1) thay vì dồn hết
    # vào CÙNG một batch đầu rồi cạn: trước T3.1, alt_data+fundamental+hypothesis+frontier
    # GỘP vào một `direct_cores` duy nhất, AltDataIdeaSource dump TOÀN BỘ ở batch #1 rồi mọi
    # batch SAU đó (GP/curated) toàn pv_reversal suốt phần còn lại phiên (PROGRESS Session 16)
    # — đây chính là vấn đề T3.1 sửa: rút DẦN từ reserve mỗi batch thay vì dump 1 lần.
    direct_cores: tuple[str, ...] = _gather_direct_cores(include_alt_data, False, False, False)
    # WS3 T3.3 (cardinal rule #1): loại core dùng field CHƯA verify LIVE trước khi tới
    # AltDataIdeaSource — verified_fields=None (không bằng chứng) -> fail-open, không lọc gì.
    direct_cores = filter_seeds_by_verified_fields(
        direct_cores, verified_fields, registry if registry is not None else default_registry(),
    )
    if direct_cores:
        idea_source = AltDataIdeaSource(
            fallback=idea_source, cores=direct_cores,
            known_fields=known_fields, registry=registry,
            avoided_hashes=avoided_hashes, dedup_key_fn=_dedup_key,
            simulator=getattr(refiner, "simulator", None),
            sim_config=getattr(refiner, "sim_config", None),
            pp_allowed_neutralizations=getattr(refiner, "pp_allowed_neutralizations", frozenset()),
            presim_cache=_presim_cache,
            # Finding #1: trần batch = trần ý tưởng phiên — không sim trước nhiều hơn số
            # candidate ClosedLoop sẽ thực sự tiêu thụ (kết quả vượt trần bị vứt, phiên sau
            # sim lại -> lãng phí quota lặp). None = không trần (phiên không giới hạn).
            presim_cap=max_ideas,
        )
    # WS3 T3.1: frontier/fundamental/hypothesis (field NGOÀI panel local, cùng đi thẳng
    # `_sim_direct` như alt-data qua `local_usable(...)==False`) — lọc known_fields (cùng guard
    # RC1/RC2 dùng cho AltDataIdeaSource) rồi avoided_hashes (không nạp lại core đã Brain-sim
    # phiên trước) TRƯỚC khi đóng gói thành `ShortlistCandidate` cho `ClosedLoop.frontier_
    # reserve` — origin="alt_data" để refiner nhận diện + route giống hệt alt-data thật.
    _registry_for_reserve = registry if registry is not None else default_registry()
    frontier_pool_cores: tuple[str, ...] = _gather_direct_cores(
        False, include_fundamental, include_hypothesis, include_frontier,
    )
    frontier_pool_cores = _filter_known_fields(
        frontier_pool_cores, known_fields, _registry_for_reserve,
    )
    # WS3 T3.3 (cardinal rule #1): cùng guard verified_fields áp cho reserve non-PV.
    frontier_pool_cores = filter_seeds_by_verified_fields(
        frontier_pool_cores, verified_fields, _registry_for_reserve,
    )
    frontier_pool_cores = tuple(
        _drop_saturated_cores(
            list(frontier_pool_cores), dedup_key_fn=_dedup_key,
            avoided_hashes=avoided_hashes, label="FrontierReserve",
        )
    )
    frontier_reserve: list[ShortlistCandidate] = []
    if frontier_pool_cores:
        import numpy as _np

        _empty_pnl = _np.zeros(0, dtype=_np.float64)
        _empty_dates = _np.zeros(0, dtype="datetime64[ns]")
        frontier_reserve = [
            ShortlistCandidate(
                expr=e, metrics=None, pnl=_empty_pnl, dates=_empty_dates, origin="alt_data",
            )
            for e in frontier_pool_cores
        ]
    # Combiner bọc NGOÀI CÙNG: nối tiếp mỗi batch (sau curated/alt-data) bằng alpha ghép từ
    # chính tín hiệu con batch đó + kho DB -> tự động chạy sau mỗi run (spec 2026-07-09).
    if include_combiner:
        idea_source = CombinerIdeaSource(
            fallback=idea_source, data=data, repo=repo, config=config, registry=registry,
        )
    tracker = CalibrationTracker(repo, every=calibrate_every, rho_bar=rho_bar)  # type: ignore[arg-type]
    # Task 5: nối tracker vào refiner (nếu refiner biết đọc ρ, vd LocalTunerRefiner) — CÙNG một
    # object mà ClosedLoop cập nhật last_rho mỗi maybe_calibrate() nên refiner luôn thấy ρ mới nhất.
    if hasattr(refiner, "set_calibration_tracker"):
        refiner.set_calibration_tracker(tracker)  # type: ignore[attr-defined]
    # Dedup key = canonical hash đã fold scale dương (Pha 1.2), `_dedup_key` đã định nghĩa ở
    # trên (dùng chung cho CuratedIdeaSource + ClosedLoop) — tiêm vào ClosedLoop để dedup
    # TRƯỚC refine bắt cả biến thể scale; parse lỗi -> fallback chuỗi thô (không chặn oan).
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

    # A1: nối tín hiệu "trần ngân sách sim GP đã chạm" -> chuỗi idea_source THẬT (mọi wrapper
    # đều có set_gp_budget_exhausted, ủy quyền xuống tận GPIdeaSource, cùng pattern
    # on_family_closed ở trên) — GPIdeaSource bỏ hẳn tiến hoá thay vì chạy xong 3–14 phút/batch
    # rồi mới bị vứt ở gate gp_budget trong ClosedLoop.run().
    def on_gp_budget_exhausted(flag: bool) -> None:
        idea_source.set_gp_budget_exhausted(flag)  # type: ignore[attr-defined]

    # B1: khi batch rỗng (chưa chắc đã cạn tuyệt đối), ClosedLoop gọi callback này để mở epoch
    # mới thay vì dừng ngay — seed mới (base_seed += 10_000) + xoay nhóm field ưu tiên
    # (field_groups, nếu wiring dataset thành công ở trên) + MỞ LẠI ngân sách sim GP cho epoch
    # mới, GIỮ NGUYÊN saturated_families/eval_cache (đã tiêm trực tiếp vào GPIdeaSource,
    # reseed_epoch() không đụng tới). `idea_source` (wrapper ngoài cùng) LUÔN có reseed_epoch
    # (ủy quyền xuống tận GPIdeaSource, cùng pattern on_gp_budget_exhausted ở trên) — gọi thẳng,
    # không cần hasattr guard. Luôn trả True (composition root luôn nối được reseed_epoch thật;
    # rỗng NGAY SAU epoch mới thì tự ClosedLoop.run() coi là cạn tuyệt đối, không cần callback
    # tự báo False).
    def on_epoch_reseed() -> bool:
        idea_source.reseed_epoch()  # type: ignore[attr-defined]
        return True

    # Yêu cầu 2026-07-18: cuối phiên ClosedLoop tự chấm PP-ready (cấu trúc + theme +
    # description) và in khối "⭐ PP SẴN SÀNG NỘP" — bọc select_power_pool_candidates
    # (module-level, chỉ cần session_factory, không cần client) tại composition root để
    # pipeline không import tầng submission (B1). Repo thiếu session_factory (fake tối
    # giản trong test) -> không wire, tương thích ngược.
    pp_ready_fn = None
    _sf = getattr(repo, "session_factory", None)
    if _sf is not None:
        from src.submission.manager import select_power_pool_candidates

        def pp_ready_fn():  # type: ignore[no-redef]
            return select_power_pool_candidates(_sf)

    return ClosedLoop(
        idea_source=idea_source, refiner=refiner, repo=repo,  # type: ignore[arg-type]
        region=region, universe=universe, max_ideas=max_ideas,
        calibration_tracker=tracker, alpha_logger=alpha_logger,
        session_summary=session_summary, dedup_key_fn=_dedup_key,
        family_fn=classify_family, max_per_family=max_per_family,
        on_family_closed=on_family_closed, max_gp_sims=max_gp_sims,
        pp_ready_fn=pp_ready_fn,
        on_gp_budget_exhausted=on_gp_budget_exhausted,
        on_epoch_reseed=on_epoch_reseed,
        frontier_reserve=frontier_reserve,
    )
