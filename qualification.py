"""Chính sách đánh giá Alpha đạt chuẩn và quality gate chọn Alpha cha."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class QualificationResult:
    qualified: bool
    parent_eligible: bool
    reasons: List[str]
    sharpe_ratio: float
    fitness_ratio: float


class QualificationPolicy:
    def __init__(self, sharpe_threshold, fitness_threshold, turnover_min,
                 turnover_hard_limit, quality_gate_ratio):
        self.sharpe_threshold = sharpe_threshold
        self.fitness_threshold = fitness_threshold
        self.turnover_min = turnover_min
        self.turnover_hard_limit = turnover_hard_limit
        self.quality_gate_ratio = quality_gate_ratio

    def evaluate(self, simulation):
        if simulation.status != "COMPLETED":
            return QualificationResult(False, False, [simulation.status], 0.0, 0.0)

        metrics = simulation.metrics or {}
        sharpe = float(metrics.get("sharpe", 0) or 0)
        fitness = float(metrics.get("fitness", 0) or 0)
        turnover = float(metrics.get("turnover", 0) or 0)
        failed_checks = [
            item.get("name")
            for item in (simulation.checks or [])
            if item.get("result") != "PASS"
        ]

        qualified = (
            sharpe >= self.sharpe_threshold
            and fitness >= self.fitness_threshold
            and self.turnover_min <= turnover <= self.turnover_hard_limit
            and not failed_checks
        )
        parent_eligible = (
            not failed_checks
            and turnover <= self.turnover_hard_limit
            and sharpe / self.sharpe_threshold >= self.quality_gate_ratio
            and fitness / self.fitness_threshold >= self.quality_gate_ratio
        )

        return QualificationResult(
            qualified=qualified,
            parent_eligible=qualified or parent_eligible,
            reasons=self._reasons(sharpe, fitness, turnover, failed_checks),
            sharpe_ratio=sharpe / self.sharpe_threshold,
            fitness_ratio=fitness / self.fitness_threshold,
        )

    def _reasons(self, sharpe, fitness, turnover, failed_checks):
        reasons = []
        if sharpe < self.sharpe_threshold:
            reasons.append(f"sharpe {sharpe:.3f} < {self.sharpe_threshold}")
        if fitness < self.fitness_threshold:
            reasons.append(f"fitness {fitness:.3f} < {self.fitness_threshold}")
        if turnover > self.turnover_hard_limit:
            reasons.append(f"turnover {turnover:.3f} > {self.turnover_hard_limit}")
        if turnover < self.turnover_min:
            reasons.append(f"turnover {turnover:.3f} < {self.turnover_min}")
        if failed_checks:
            reasons.append("failed checks: " + ", ".join(map(str, failed_checks)))
        if not reasons:
            reasons.append("đạt toàn bộ tiêu chí")
        return reasons
