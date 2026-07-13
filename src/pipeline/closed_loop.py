"""ClosedLoop — orchestrator vòng kín AI + MiniBrain (thuần logic, network-agnostic).

Lặp: lấy ý tưởng (GP→short-list, qua `idea_source`) → với mỗi ý tưởng gọi `refiner.
refine_and_sim` (bọc RefinementLoop thật ở phase wiring) → ghi kết quả SIM qua
`repo.record_brain_sim` (cầu DB Phase 1) → tránh trùng → dừng gọn khi Brain hết quota
(`QuotaExhausted`) hoặc cạn ý tưởng.

Dependency rule B1: KHÔNG import `src.llm`/`src.gp`/`src.simulation` — mọi dependency
injected qua Protocol structural; việc dựng cụ thể nằm ở `main.py`/adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from loguru import logger

from src.calibration.stats import spearman
from src.pipeline.shortlist import ShortlistCandidate
from src.storage.repository import MiniBrainRepository


def _short(expr: str, n: int = 70) -> str:
    """Rút gọn biểu thức cho log 1 dòng (bỏ khoảng trắng thừa, cắt đuôi)."""
    s = " ".join(str(expr).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt(x: float | None) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


class QuotaExhausted(Exception):
    """Refiner ném khi Brain hết quota SIM — ClosedLoop dừng vòng gọn, persist mọi thứ."""


@dataclass(frozen=True, slots=True)
class IdeaOutcome:
    """Kết quả refine+sim một ý tưởng (refiner-protocol trả về)."""

    expr: str
    canonical_hash: str
    passed: bool
    wq_alpha_id: str | None
    sharpe: float | None
    fitness: float | None
    turnover: float | None
    self_corr: float | None
    sims_used: int
    stop_reason: str
    # Cờ Power Pool (docs Brain): Sharpe>=1.0, <=8 operator, <=3 field (trừ grouping),
    # self_corr<=0.5. Default False để tương thích ngược với nơi tạo IdeaOutcome cũ.
    power_pool_eligible: bool = False
    # Settings Brain thật đã dùng khi sim (dict từ SimConfig.to_settings) — None nếu ý tưởng bị
    # gate local chặn (0 sim). Nguồn ý tưởng (alt_data/gp_local_tuner...) để soi độ lặp công thức.
    sim_settings: dict | None = None
    source: str | None = None
    # --- Instrumentation Pha 0 (IMPROVEMENT_SPEC §3): trả lời "chết ở đâu, vì sao, tốn bao lâu".
    # stage_reached: chặng cuối cùng ứng viên đi tới (idea|syntax|depth|dedup|local_floor|
    #   sub_universe|simmed|corr_checked|passed). fail_check: mã check thất bại (LOW_SHARPE/
    #   SELF_CORR/DEPTH/DUP/...). family: họ nhân tố. expr_depth: độ sâu cây. *_ms: thời gian
    #   mỗi mốc. dedup_key: canonical hash (đã fold). local_sharpe: Sharpe backtest local (TÁCH
    #   khỏi `sharpe` vốn là Brain sharpe — spec: đừng gộp local/brain vào 1 cột).
    stage_reached: str = ""
    fail_check: str = ""
    family: str = ""
    expr_depth: int | None = None
    gen_ms: float | None = None
    backtest_ms: float | None = None
    sim_ms: float | None = None
    dedup_key: str | None = None
    local_sharpe: float | None = None
    # --- Task 3 (spec C2): phân biệt "chưa chạm Brain" (pre-sim reject) khỏi "sim thật rớt".
    # presim_reason: reason gốc từ SimulationResult.presim_reason (None nếu đã sim thật, kể cả
    #   khi sim đó lỗi). is_brain_sim: True <=> outcome này ĐÃ gọi Brain thật (mặc định True để
    #   tương thích ngược — mọi call site cũ dựng IdeaOutcome đều từ 1 sim thật hoặc chưa sim gì
    #   cả nhưng không set presim_reason, nên default True không làm sai lệch outcome cũ).
    presim_reason: str | None = None
    is_brain_sim: bool = True


@dataclass(frozen=True, slots=True)
class ClosedLoopReport:
    """Thống kê một lần chạy ClosedLoop + lý do dừng."""

    ideas_tried: int
    sims_used: int
    n_passed: int
    n_abandoned: int
    stop_reason: str
    rho_sharpe: float | None = None


class CalibrationTracker:
    """Theo dõi độ tin ranking local: sau mỗi `every` sim, tính lại Spearman ρ giữa local
    sharpe và Brain sharpe (trên các expression đã có cả hai). ρ < `rho_bar` -> cảnh báo
    (ranking local có thể không còn đáng tin -> nên điều tra data/operator fidelity)."""

    def __init__(
        self, repo: MiniBrainRepository, *, every: int = 10, rho_bar: float = 0.5,
    ) -> None:
        self.repo = repo
        self.every = every
        self.rho_bar = rho_bar
        self.last_rho: float | None = None
        self._last_mark = 0

    def maybe_calibrate(self, sims_total: int) -> float | None:
        """Tính ρ nếu `sims_total` đã qua mốc bội số `every` kể từ lần trước; ngược lại None.
        ρ tính qua `spearman` trên `brain_local_sharpe_pairs()` (NaN nếu < 2 cặp)."""
        if sims_total < self._last_mark + self.every:
            return None
        self._last_mark = sims_total - (sims_total % self.every)
        pairs = self.repo.brain_local_sharpe_pairs()
        if len(pairs) < 2:
            self.last_rho = None
            return None
        local = np.array([p[0] for p in pairs], dtype=np.float64)
        brain = np.array([p[1] for p in pairs], dtype=np.float64)
        rho = spearman(local, brain)
        self.last_rho = rho
        if not np.isnan(rho) and rho < self.rho_bar:
            logger.warning(
                "Calibration ρ={:.3f} < bar {:.2f} — ranking local kém tin", rho, self.rho_bar,
            )
        return rho


class _GeneratesIdeas(Protocol):
    def next_batch(self) -> list[ShortlistCandidate]: ...


class _RefinesIdea(Protocol):
    def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome: ...


class ClosedLoop:
    """Vòng kín: lấy ý tưởng → refine+sim mỗi cái → persist kết quả SIM → dừng khi hết
    quota / cạn ý tưởng. Thuần điều phối; mọi dependency injected."""

    def __init__(
        self,
        idea_source: _GeneratesIdeas,
        refiner: _RefinesIdea,
        repo: MiniBrainRepository,
        *,
        region: str = "USA",
        universe: str = "TOP3000",
        max_ideas: int | None = None,
        calibration_tracker: CalibrationTracker | None = None,
        alpha_logger=None,
        session_summary=None,
        dedup_key_fn=None,
        family_fn=None,
        max_per_family: int | None = None,
        on_family_closed=None,
        max_gp_sims: int | None = 3,
    ) -> None:
        self.idea_source = idea_source
        self.refiner = refiner
        self.repo = repo
        self.region = region
        self.universe = universe
        self.max_ideas = max_ideas
        self.calibration_tracker = calibration_tracker
        # Logger CSV mọi ý tưởng (Task 2 RunAlphaLogger); None -> bỏ qua, tương thích ngược.
        self.alpha_logger = alpha_logger
        # Thu funnel cuối phiên (Pha 0 SessionSummary); None -> bỏ qua, tương thích ngược.
        self.session_summary = session_summary
        # Hàm chuẩn hoá expr -> dedup key (canonical fold, Pha 1.2). None -> identity (dùng
        # chuỗi thô như cũ). Giữ B1: ClosedLoop KHÔNG import src.lang; composition root
        # (main/adapter) tiêm CanonicalHasher thật vào đây.
        self.dedup_key_fn = dedup_key_fn or (lambda expr: expr)
        # Family-aware budget + exhaustion guard (Pha 2.2). family_fn: expr -> nhãn họ (None =
        # tắt, không giới hạn theo họ). max_per_family: trần ứng viên/họ/phiên; khi một họ đã
        # refine >= trần mà 0 pass -> ĐÓNG họ trong phiên (chuyển ngân sách sang họ khác). Họ có
        # >=1 pass thì không đóng (còn tiềm năng). Giữ B1: family_fn tiêm từ composition root.
        self.family_fn = family_fn
        self.max_per_family = max_per_family
        # Callback khi một họ bị đóng (Pha 2.3): nhận set họ bão hoà -> composition root nối
        # tới idea_generator.set_saturated_families để LLM tránh tái sinh. None -> bỏ qua.
        self.on_family_closed = on_family_closed
        # Cap ngân sách sim Brain riêng cho candidate origin "gp" (Task 3): 2 phiên chạy thật
        # gần nhất GP đốt ~10 sim (≈50% quota) ra toàn Sharpe ≤0.31 — calibration ρ=0.308 nên
        # floor local không lọc nổi rác GP TRƯỚC khi chạm Brain. Cap cứng rẻ hơn cải thiện
        # calibration: candidate origin "gp" vượt trần KHÔNG refine+sim (0 sim thật), ưu tiên
        # quota còn lại cho seed đã kiểm chứng (curated/alt_data) và alpha ghép (combiner) —
        # các nguồn này KHÔNG bị cap (YAGNI: chỉ GP là nguồn nhiễu đã có bằng chứng thật).
        # None = không cap (tương thích ngược cho ai đang set max_gp_sims=None tường minh).
        self.max_gp_sims = max_gp_sims

    def run(self) -> ClosedLoopReport:
        """Lặp: next_batch → mỗi ý tưởng refine_and_sim → record_brain_sim → đếm. Dừng khi
        batch rỗng (cạn ý tưởng), đạt max_ideas, hoặc refiner ném QuotaExhausted (hết quota
        Brain). Bỏ qua expr đã thấy trong phiên (tránh refine trùng)."""
        ideas_tried = 0
        sims_used = 0
        n_passed = 0
        n_abandoned = 0
        # Task 3: đếm SỐ SIM Brain thật (cộng dồn outcome.sims_used) đã dùng bởi candidate
        # origin "gp" — cap ngân sách riêng cho GP, không đụng curated/alt_data/combiner.
        gp_sims_used = 0
        # seen chứa DEDUP KEY (canonical fold Pha 1.2), không phải chuỗi thô. Nạp avoid-list
        # bền: ưu tiên hash cross-session (avoided_hashes) nếu repo có; fallback chuỗi thô
        # (đưa qua dedup_key_fn để cùng không gian key).
        seen: set[str] = set()
        avoided_hashes = getattr(self.repo, "avoided_hashes", None)
        if callable(avoided_hashes):
            seen |= set(avoided_hashes())
        # Hash GỐC (pre-tune) đã thử ở phiên trước (Task 6 fix) — không gian hash NÀY khớp
        # đúng với `key` tính ở dưới (dedup_key_fn trên cand.expr TRƯỚC tune), khác
        # avoided_hashes() ở trên vốn lấy canonical_hash SAU tune từ BrainSimLinkModel nên
        # không bao giờ khớp candidate mới sinh ra ở phiên sau. Guard getattr+callable: repo
        # fake/cũ thiếu method này vẫn chạy được (tương thích ngược).
        avoided_hashes_original = getattr(self.repo, "avoided_hashes_original", None)
        if callable(avoided_hashes_original):
            seen |= set(avoided_hashes_original())
        seen |= {self.dedup_key_fn(e) for e in self.repo.avoided_exprs()}
        # Thu thập expr đạt Power Pool nhưng KHÔNG đạt Regular — để tóm tắt cuối phiên (không
        # đổi ClosedLoopReport public: chỉ log, tránh rủi ro cho consumer đang đọc report).
        power_pool_only: list[str] = []
        # Family budget state (Pha 2.2): đếm số ứng viên đã refine + số pass mỗi họ.
        fam_tried: dict[str, int] = {}
        fam_passed: dict[str, int] = {}
        closed_families: set[str] = set()

        def _report(stop_reason: str) -> ClosedLoopReport:
            if power_pool_only:
                # RC8 fix: "Power Pool eligible" chỉ là cờ CẤU TRÚC (is_power_pool: Sharpe≥1.0,
                # ≤8 operator, ≤3 field, self_corr≤0.5) — KHÔNG phải xác nhận nộp được. Các
                # alpha này KHÔNG đạt ngưỡng Regular (Sharpe cần ~1.58+, xem
                # config/thresholds.py IS_LADDER) và tool này CHƯA có đường nộp Power Pool tự
                # động (không auth/sim/submit thật ở đây). Nêu rõ hành động tiếp theo: người
                # dùng phải tự xem lại từng alpha trên WQ Brain và tự quyết định nộp hay không.
                logger.info(
                    "⭐ Tóm tắt Power Pool: {} ứng viên đạt CẤU TRÚC Power Pool (Sharpe≥1.0, "
                    "≤8 operator, ≤3 field, self_corr≤0.5) nhưng KHÔNG đạt ngưỡng Regular "
                    "(Sharpe cần ~1.58+). Đây KHÔNG phải xác nhận nộp được — tool này hiện "
                    "CHƯA có đường nộp Power Pool tự động. Hành động tiếp theo: tự xem lại "
                    "từng alpha bên dưới trên WQ Brain và tự quyết định có nộp qua nhánh Power "
                    "Pool hay không (đạt cấu trúc ≠ được chấp nhận): {}",
                    len(power_pool_only),
                    ", ".join(_short(e, 40) for e in power_pool_only),
                )
            return ClosedLoopReport(
                ideas_tried, sims_used, n_passed, n_abandoned, stop_reason,
                rho_sharpe=self.calibration_tracker.last_rho if self.calibration_tracker else None,
            )

        while True:
            logger.info("⏳ Sinh batch ý tưởng (GP + decorrelate)…")
            _gen_t0 = time.perf_counter()
            batch = self.idea_source.next_batch()
            gen_batch_ms = (time.perf_counter() - _gen_t0) * 1000.0
            if not batch:
                logger.info("Cạn ý tưởng (batch rỗng) — dừng vòng kín.")
                return _report("no_more_ideas")
            # Chi phí sinh (GP/decorrelate) là của CẢ batch; phân bổ đều cho mỗi ứng viên để
            # điền gen_ms (refiner không biết chi phí này). Xấp xỉ đủ tốt cho funnel timing.
            gen_ms_each = gen_batch_ms / len(batch)
            logger.info("📦 Batch: {} ứng viên qua sàng lọc/decorrelate.", len(batch))
            for cand in batch:
                if self.max_ideas is not None and ideas_tried >= self.max_ideas:
                    logger.info("Đạt trần max_ideas={} — dừng.", self.max_ideas)
                    return _report("no_more_ideas")
                key = self.dedup_key_fn(cand.expr)
                if key in seen:
                    logger.info("↩︎ Bỏ ý tưởng trùng phiên/avoid-list: {}", _short(cand.expr))
                    if self.session_summary is not None:
                        self.session_summary.record_dup_blocked()
                    continue
                seen.add(key)
                # Family budget: bỏ candidate thuộc họ đã ĐÓNG (đã cạn ngân sách mà 0 pass).
                fam = self.family_fn(cand.expr) if self.family_fn is not None else None
                if fam is not None and fam in closed_families:
                    logger.info("↩︎ Bỏ ý tưởng thuộc họ đã đóng [{}]: {}", fam, _short(cand.expr))
                    if self.session_summary is not None:
                        self.session_summary.record_dup_blocked()
                    continue
                origin = getattr(cand, "origin", "gp")
                # Task 3: candidate origin "gp" đã chạm trần ngân sách sim GP/phiên -> KHÔNG
                # refine+sim (0 sim thật) — outcome trung thực để funnel CSV/session_summary
                # ghi rõ ứng viên bị chặn vì ngân sách, không phải bị `continue` âm thầm như
                # dedup/family-closed. Candidate origin khác (curated/alt_data/combiner) không
                # bị cap này -> luôn đi tiếp đường refine+sim bình thường.
                if (
                    origin == "gp"
                    and self.max_gp_sims is not None
                    and gp_sims_used >= self.max_gp_sims
                ):
                    logger.info(
                        "🚧 Ứng viên GP #{} chạm trần ngân sách sim ({} sim) — bỏ qua, ưu "
                        "tiên quota cho seed/combiner: {}",
                        ideas_tried + 1, self.max_gp_sims, _short(cand.expr),
                    )
                    outcome = IdeaOutcome(
                        expr=cand.expr, canonical_hash=key, passed=False, wq_alpha_id=None,
                        sharpe=None, fitness=None, turnover=None, self_corr=None,
                        sims_used=0, stop_reason="gp_budget", source=origin,
                        stage_reached="gp_budget", fail_check="GP_BUDGET",
                        family=fam or "", dedup_key=key, is_brain_sim=False,
                    )
                else:
                    logger.info(
                        "🔎 Ý tưởng #{}: refine+sim {}", ideas_tried + 1, _short(cand.expr),
                    )
                    try:
                        outcome = self.refiner.refine_and_sim(cand)
                    except QuotaExhausted:
                        logger.info(
                            "Hết quota Brain — dừng vòng kín ({} sim đã dùng).", sims_used,
                        )
                        return _report("quota")
                # Điền gen_ms (chi phí sinh batch phân bổ) nếu refiner chưa set — refiner đo
                # backtest_ms/sim_ms, còn gen_ms thuộc tầng ClosedLoop (Fix gap Pha 0).
                if getattr(outcome, "gen_ms", None) is None:
                    outcome = replace(outcome, gen_ms=gen_ms_each)
                self.repo.record_brain_sim(
                    canonical_hash=outcome.canonical_hash, expr_string=outcome.expr,
                    wq_alpha_id=outcome.wq_alpha_id, region=self.region,
                    universe=self.universe, sharpe=outcome.sharpe, fitness=outcome.fitness,
                    turnover=outcome.turnover, self_corr=outcome.self_corr,
                    status="passed" if outcome.passed else "failed",
                )
                # Persist hash GỐC (pre-tune, `key` tính ở trên TRƯỚC khi refiner tune) để
                # phiên SAU pre-check khớp đúng không gian hash (Task 6 fix). CHỈ ghi khi
                # outcome đã thực sự sim Brain (`is_brain_sim`) — sim tốn thật là bằng chứng
                # thật (pass hay fail đều đáng nhớ, khớp semantics `seen.add(key)` trong-phiên
                # ở trên vốn cũng chặn trùng bất kể pass/fail). Outcome bị gate LOCAL (local_floor/
                # presim_reject/depth/sub_universe...) KHÔNG có bằng chứng Brain thật gì cả — cấm
                # vĩnh viễn sẽ làm cạn seed novelty (RC1: ALT_DATA/FUNDAMENTAL core bị gate 1 lần
                # là mất luôn cơ hội thử lại ở phiên sau, kể cả khi conditioning/config đã đổi).
                # Guard: repo fake/cũ thiếu method vẫn chạy được (tương thích ngược).
                if outcome.is_brain_sim:
                    record_avoided_hash = getattr(self.repo, "record_avoided_hash", None)
                    if callable(record_avoided_hash):
                        record_avoided_hash(key)
                ideas_tried += 1
                # Log CSV mọi ý tưởng có outcome (kể cả bị gate 0-sim) — Task 3.
                if self.alpha_logger is not None:
                    self.alpha_logger.log(ideas_tried, outcome)
                if self.session_summary is not None:
                    self.session_summary.record(outcome)
                sims_used += outcome.sims_used
                # Đếm theo SỐ SIM thật (outcome.sims_used), KHÔNG phải số candidate (review
                # fix): với --refiner llm, 1 candidate gp có thể đốt nhiều sim (patience loop)
                # — đếm theo candidate thì cap 3 vẫn có thể cho lọt 15 sim thật. Outcome bị
                # gate local/gp_budget có sims_used=0 nên tự nhiên không cộng gì.
                if origin == "gp":
                    gp_sims_used += outcome.sims_used
                if outcome.passed:
                    n_passed += 1
                else:
                    n_abandoned += 1
                    if getattr(outcome, "power_pool_eligible", False):
                        power_pool_only.append(outcome.expr)
                # Family budget (Pha 2.2): cập nhật đếm + đóng họ nếu đã cạn ngân sách mà 0 pass.
                if fam is not None:
                    fam_tried[fam] = fam_tried.get(fam, 0) + 1
                    if outcome.passed:
                        fam_passed[fam] = fam_passed.get(fam, 0) + 1
                    if (
                        self.max_per_family is not None
                        and fam_tried[fam] >= self.max_per_family
                        and fam_passed.get(fam, 0) == 0
                    ):
                        closed_families.add(fam)
                        logger.info(
                            "🚪 Đóng họ [{}]: đã thử {} ứng viên, 0 pass — chuyển ngân sách.",
                            fam, fam_tried[fam],
                        )
                        if self.on_family_closed is not None:
                            self.on_family_closed(set(closed_families))
                # Kết quả 1 dòng: 0 sim = bị gate local chặn trước khi đốt quota Brain. In kèm
                # lý do (stop_reason) để phân biệt local_floor (Sharpe/turnover) vs sub_universe.
                if outcome.sims_used == 0:
                    logger.info("   → ⚠ bị gate local chặn [{}] (0 sim Brain).", outcome.stop_reason)
                else:
                    logger.info(
                        "   → {} Sharpe={} fit={} TO={} self_corr={} ({} sim)",
                        "✅ PASSED" if outcome.passed else "✗ failed",
                        _fmt(outcome.sharpe), _fmt(outcome.fitness), _fmt(outcome.turnover),
                        _fmt(outcome.self_corr), outcome.sims_used,
                    )
                if getattr(outcome, "power_pool_eligible", False):
                    # RC8 fix: nêu rõ đây là cờ CẤU TRÚC, không phải "đã nộp được" — alpha vẫn
                    # dưới ngưỡng Regular và chưa có đường nộp Power Pool tự động trong tool này.
                    logger.info(
                        "   ⭐ Đạt CẤU TRÚC Power Pool (Sharpe≥1.0, ≤8 op, ≤3 field, "
                        "self_corr≤0.5) — KHÔNG phải đã nộp được, cần tự xem lại trên WQ Brain."
                    )
                logger.info(
                    "   Σ {} ý tưởng / {} sim / {} pass / {} bỏ.",
                    ideas_tried, sims_used, n_passed, n_abandoned,
                )
                if self.calibration_tracker is not None:
                    self.calibration_tracker.maybe_calibrate(sims_used)
