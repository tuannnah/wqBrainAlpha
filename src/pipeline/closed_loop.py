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

from src.pipeline.shortlist import ShortlistCandidate
from src.storage.repository import MiniBrainRepository


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


@dataclass(frozen=True, slots=True)
class ClosedLoopReport:
    """Thống kê một lần chạy ClosedLoop + lý do dừng."""

    ideas_tried: int
    sims_used: int
    n_passed: int
    n_abandoned: int
    stop_reason: str


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
    ) -> None:
        self.idea_source = idea_source
        self.refiner = refiner
        self.repo = repo
        self.region = region
        self.universe = universe
        self.max_ideas = max_ideas

    def run(self) -> ClosedLoopReport:
        """Lặp: next_batch → mỗi ý tưởng refine_and_sim → record_brain_sim → đếm. Dừng khi
        batch rỗng (cạn ý tưởng), đạt max_ideas, hoặc refiner ném QuotaExhausted (hết quota
        Brain). Bỏ qua expr đã thấy trong phiên (tránh refine trùng)."""
        ideas_tried = 0
        sims_used = 0
        n_passed = 0
        n_abandoned = 0
        seen: set[str] = set()

        while True:
            batch = self.idea_source.next_batch()
            if not batch:
                return ClosedLoopReport(ideas_tried, sims_used, n_passed, n_abandoned,
                                        "no_more_ideas")
            for cand in batch:
                if self.max_ideas is not None and ideas_tried >= self.max_ideas:
                    return ClosedLoopReport(ideas_tried, sims_used, n_passed, n_abandoned,
                                            "no_more_ideas")
                if cand.expr in seen:
                    continue
                seen.add(cand.expr)
                try:
                    outcome = self.refiner.refine_and_sim(cand)
                except QuotaExhausted:
                    return ClosedLoopReport(ideas_tried, sims_used, n_passed, n_abandoned,
                                            "quota")
                self.repo.record_brain_sim(
                    canonical_hash=outcome.canonical_hash, expr_string=outcome.expr,
                    wq_alpha_id=outcome.wq_alpha_id, region=self.region,
                    universe=self.universe, sharpe=outcome.sharpe, fitness=outcome.fitness,
                    turnover=outcome.turnover, self_corr=outcome.self_corr,
                    status="passed" if outcome.passed else "failed",
                )
                ideas_tried += 1
                sims_used += outcome.sims_used
                if outcome.passed:
                    n_passed += 1
                else:
                    n_abandoned += 1
