"""ClosedLoop — orchestrator vòng kín AI + MiniBrain (thuần logic, network-agnostic).

Lặp: lấy ý tưởng (GP→short-list, qua `idea_source`) → với mỗi ý tưởng gọi `refiner.
refine_and_sim` (bọc RefinementLoop thật ở phase wiring) → ghi kết quả SIM qua
`repo.record_brain_sim` (cầu DB Phase 1) → tránh trùng → dừng gọn khi Brain hết quota
(`QuotaExhausted`) hoặc cạn ý tưởng.

Dependency rule B1: KHÔNG import `src.llm`/`src.gp`/`src.simulation` — mọi dependency
injected qua Protocol structural; việc dựng cụ thể nằm ở `main.py`/adapter."""

from __future__ import annotations

from dataclasses import dataclass
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

    def run(self) -> ClosedLoopReport:
        """Lặp: next_batch → mỗi ý tưởng refine_and_sim → record_brain_sim → đếm. Dừng khi
        batch rỗng (cạn ý tưởng), đạt max_ideas, hoặc refiner ném QuotaExhausted (hết quota
        Brain). Bỏ qua expr đã thấy trong phiên (tránh refine trùng)."""
        ideas_tried = 0
        sims_used = 0
        n_passed = 0
        n_abandoned = 0
        seen: set[str] = set()
        seen |= self.repo.avoided_exprs()
        # Thu thập expr đạt Power Pool nhưng KHÔNG đạt Regular — để tóm tắt cuối phiên (không
        # đổi ClosedLoopReport public: chỉ log, tránh rủi ro cho consumer đang đọc report).
        power_pool_only: list[str] = []

        def _report(stop_reason: str) -> ClosedLoopReport:
            if power_pool_only:
                logger.info(
                    "⭐ Tóm tắt Power Pool: {} ứng viên đạt Power Pool nhưng KHÔNG đạt "
                    "Regular (đáng cân nhắc nộp qua nhánh Power Pool): {}",
                    len(power_pool_only),
                    ", ".join(_short(e, 40) for e in power_pool_only),
                )
            return ClosedLoopReport(
                ideas_tried, sims_used, n_passed, n_abandoned, stop_reason,
                rho_sharpe=self.calibration_tracker.last_rho if self.calibration_tracker else None,
            )

        while True:
            logger.info("⏳ Sinh batch ý tưởng (GP + decorrelate)…")
            batch = self.idea_source.next_batch()
            if not batch:
                logger.info("Cạn ý tưởng (batch rỗng) — dừng vòng kín.")
                return _report("no_more_ideas")
            logger.info("📦 Batch: {} ứng viên qua sàng lọc/decorrelate.", len(batch))
            for cand in batch:
                if self.max_ideas is not None and ideas_tried >= self.max_ideas:
                    logger.info("Đạt trần max_ideas={} — dừng.", self.max_ideas)
                    return _report("no_more_ideas")
                if cand.expr in seen:
                    logger.info("↩︎ Bỏ ý tưởng trùng phiên/avoid-list: {}", _short(cand.expr))
                    continue
                seen.add(cand.expr)
                logger.info("🔎 Ý tưởng #{}: refine+sim {}", ideas_tried + 1, _short(cand.expr))
                try:
                    outcome = self.refiner.refine_and_sim(cand)
                except QuotaExhausted:
                    logger.info("Hết quota Brain — dừng vòng kín ({} sim đã dùng).", sims_used)
                    return _report("quota")
                self.repo.record_brain_sim(
                    canonical_hash=outcome.canonical_hash, expr_string=outcome.expr,
                    wq_alpha_id=outcome.wq_alpha_id, region=self.region,
                    universe=self.universe, sharpe=outcome.sharpe, fitness=outcome.fitness,
                    turnover=outcome.turnover, self_corr=outcome.self_corr,
                    status="passed" if outcome.passed else "failed",
                )
                ideas_tried += 1
                # Log CSV mọi ý tưởng có outcome (kể cả bị gate 0-sim) — Task 3.
                if self.alpha_logger is not None:
                    self.alpha_logger.log(ideas_tried, outcome)
                sims_used += outcome.sims_used
                if outcome.passed:
                    n_passed += 1
                else:
                    n_abandoned += 1
                    if getattr(outcome, "power_pool_eligible", False):
                        power_pool_only.append(outcome.expr)
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
                    logger.info("   ⭐ Power Pool eligible (Sharpe≥1.0, ≤8 op, ≤3 field, self_corr≤0.5)")
                logger.info(
                    "   Σ {} ý tưởng / {} sim / {} pass / {} bỏ.",
                    ideas_tried, sims_used, n_passed, n_abandoned,
                )
                if self.calibration_tracker is not None:
                    self.calibration_tracker.maybe_calibrate(sims_used)
