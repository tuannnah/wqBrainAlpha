"""Submission Manager: chọn alpha đạt ngưỡng, check correlation, nộp WQ Brain."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from loguru import logger

from src.storage.models import AlphaModel, SimulationModel, SubmissionModel


@dataclass
class Candidate:
    wq_alpha_id: str
    expression: str
    sharpe: float | None
    fitness: float | None
    score: float | None


@dataclass
class SubmissionResult:
    wq_alpha_id: str
    status: str  # submitted/rejected/error
    detail: str = ""
    self_correlation: float | None = None


class SubmissionManager:
    MIN_SHARPE = 1.5
    MIN_FITNESS = 1.2
    DAILY_QUOTA = 10

    def __init__(
        self,
        client,
        session_factory,
        correlation_checker,
        min_sharpe: float | None = None,
        min_fitness: float | None = None,
        daily_quota: int | None = None,
    ):
        self.client = client
        self.session_factory = session_factory
        self.correlation = correlation_checker
        self.min_sharpe = min_sharpe if min_sharpe is not None else self.MIN_SHARPE
        self.min_fitness = min_fitness if min_fitness is not None else self.MIN_FITNESS
        self.daily_quota = daily_quota if daily_quota is not None else self.DAILY_QUOTA

    # --------------------------------------------------------------- selection
    def select_candidates(self) -> list[Candidate]:
        session = self.session_factory()
        try:
            submitted = {
                row[0]
                for row in session.query(SubmissionModel.alpha_id)
                .filter(SubmissionModel.status == "submitted")
                .all()
            }
            rows = (
                session.query(SimulationModel, AlphaModel)
                .join(AlphaModel, SimulationModel.alpha_id == AlphaModel.id)
                .filter(SimulationModel.status == "passed")
                .filter(SimulationModel.sharpe >= self.min_sharpe)
                .filter(SimulationModel.fitness >= self.min_fitness)
                .filter(SimulationModel.wq_alpha_id.isnot(None))
                .order_by(SimulationModel.score.desc())
                .all()
            )
        finally:
            session.close()

        candidates: list[Candidate] = []
        seen: set[str] = set()
        for sim, alpha in rows:
            if sim.wq_alpha_id in submitted or sim.wq_alpha_id in seen:
                continue
            seen.add(sim.wq_alpha_id)
            candidates.append(
                Candidate(sim.wq_alpha_id, alpha.expression, sim.sharpe, sim.fitness, sim.score)
            )
        return candidates

    # ------------------------------------------------------------------ submit
    def submit(self, wq_alpha_id: str) -> SubmissionResult:
        corr = self.correlation.max_self_correlation(wq_alpha_id)
        if corr > self.correlation.max_self_corr:
            result = SubmissionResult(
                wq_alpha_id, "rejected", f"self-corr {corr:.3f} > {self.correlation.max_self_corr}", corr
            )
            self._record(result)
            return result

        try:
            resp = self.client.post(f"/alphas/{wq_alpha_id}/submit")
        except Exception as exc:  # noqa: BLE001 - không để pipeline crash
            result = SubmissionResult(wq_alpha_id, "error", str(exc), corr)
            self._record(result)
            return result

        if resp.status_code in (200, 201):
            result = SubmissionResult(wq_alpha_id, "submitted", "ok", corr)
        else:
            result = SubmissionResult(wq_alpha_id, "error", f"HTTP {resp.status_code}", corr)
        self._record(result)
        return result

    def run_daily(self, dry_run: bool = True) -> list[Candidate]:
        """Chọn ≤ quota alpha tốt nhất, không trùng correlation. Nộp nếu không dry-run."""
        candidates = self.select_candidates()
        selected: list[Candidate] = []
        for cand in candidates:
            if len(selected) >= self.daily_quota:
                break
            if not self.correlation.is_acceptable(cand.wq_alpha_id):
                logger.info("Bỏ {} vì self-correlation cao", cand.wq_alpha_id)
                continue
            selected.append(cand)
            if not dry_run:
                self.submit(cand.wq_alpha_id)
        return selected

    # ------------------------------------------------------------------ record
    def _record(self, result: SubmissionResult) -> None:
        session = self.session_factory()
        try:
            session.add(
                SubmissionModel(
                    id=uuid.uuid4().hex,
                    alpha_id=result.wq_alpha_id,
                    status=result.status,
                    self_correlation=result.self_correlation,
                    detail=result.detail,
                )
            )
            session.commit()
        finally:
            session.close()
