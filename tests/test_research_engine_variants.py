import tempfile
import unittest
from pathlib import Path

from research_engine import IMPROVEMENT_DIRECTIONS, ParentCandidate
from tests.engine_fakes import (
    FakeLlm,
    FakeWorldQuant,
    bad_result,
    build_engine,
    context,
)


class ResearchEngineVariantTest(unittest.TestCase):
    def _make_parent(self, store, run_id, expression, order):
        idea_id = store.create_idea(run_id, '{"title": "i"}', "DEEPSEEK")
        hyp_id = store.create_hypothesis(idea_id, "h", "r", ["pv1"], ["close"])
        alpha_id = store.create_alpha(
            run_id, hyp_id, expression, f"h-{order}", f"f-{order}",
            {"region": "USA"}, ["pv1"], None, 0, None,
        )
        return ParentCandidate(
            alpha_id=alpha_id, expression=expression, hypothesis="h",
            metrics={}, sharpe_ratio=0.9, fitness_ratio=0.9, turnover=0.4,
            qualified=False, order=order,
        ), idea_id

    def test_creates_five_targeted_variants_per_parent_only_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            llm = FakeLlm(root_batches=[])
            worldquant = FakeWorldQuant(results=[bad_result()] * 10)
            engine, store = build_engine(
                Path(temp_dir) / "r.sqlite", llm=llm, worldquant=worldquant
            )
            try:
                parent_one, idea_id = self._make_parent(
                    store, engine.run_id, "rank(close)", 1
                )
                parent_two, _ = self._make_parent(
                    store, engine.run_id, "rank(returns)", 2
                )
                idea = {"id": idea_id}

                engine._run_variants(idea, context(), [parent_one, parent_two])

                self.assertEqual(llm.variant_request_count, 10)
                directions = [call.direction for call in llm.variant_calls]
                self.assertEqual(list(directions[:5]), list(IMPROVEMENT_DIRECTIONS))

                alphas = store.list_alphas(engine.run_id)
                variants = [a for a in alphas if a["generation"] == 1]
                self.assertEqual(len(variants), 10)
                parent_ids = {parent_one.alpha_id, parent_two.alpha_id}
                self.assertTrue(all(v["parent_id"] in parent_ids for v in variants))

                # không có biến thể thế hệ hai (parent trỏ vào một biến thể)
                variant_ids = {v["id"] for v in variants}
                self.assertFalse(any(a["parent_id"] in variant_ids for a in alphas))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
