import unittest

from qualification import QualificationPolicy
from research_models import SimulationResult


def completed(sharpe, fitness, turnover, checks=None):
    return SimulationResult(
        worldquant_alpha_id="a1",
        status="COMPLETED",
        metrics={"sharpe": sharpe, "fitness": fitness, "turnover": turnover,
                 "margin": 0.03},
        checks=checks if checks is not None else [
            {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"}
        ],
    )


class QualificationPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = QualificationPolicy(
            sharpe_threshold=1.5,
            fitness_threshold=1.0,
            turnover_min=0.01,
            turnover_hard_limit=0.9,
            quality_gate_ratio=0.8,
        )

    def test_near_threshold_is_parent_but_not_qualified(self):
        result = self.policy.evaluate(completed(1.25, 0.85, 0.4))
        self.assertFalse(result.qualified)
        self.assertTrue(result.parent_eligible)

    def test_fully_qualified(self):
        result = self.policy.evaluate(completed(1.7, 1.2, 0.4))
        self.assertTrue(result.qualified)
        self.assertTrue(result.parent_eligible)

    def test_turnover_above_hard_limit_is_not_eligible(self):
        result = self.policy.evaluate(completed(1.7, 1.2, 0.95))
        self.assertFalse(result.qualified)
        self.assertFalse(result.parent_eligible)

    def test_failed_check_blocks_qualification_and_parent(self):
        result = self.policy.evaluate(completed(1.7, 1.2, 0.4, checks=[
            {"name": "CONCENTRATED_WEIGHT", "result": "FAIL"},
        ]))
        self.assertFalse(result.qualified)
        self.assertFalse(result.parent_eligible)

    def test_non_completed_simulation_is_rejected(self):
        failed = SimulationResult(
            worldquant_alpha_id=None,
            status="FAILED",
            error_code="COMPILE_ERROR",
        )
        result = self.policy.evaluate(failed)
        self.assertFalse(result.qualified)
        self.assertFalse(result.parent_eligible)
        self.assertIn("FAILED", result.reasons)


if __name__ == "__main__":
    unittest.main()
