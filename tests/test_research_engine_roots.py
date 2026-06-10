import tempfile
import unittest
from pathlib import Path

from tests.engine_fakes import (
    FakeLlm,
    FakeValidator,
    FakeWorldQuant,
    bad_result,
    build_engine,
    make_drafts,
    parent_eligible_result,
)


class ResearchEngineRootTest(unittest.TestCase):
    def test_three_bad_batches_exhaust_idea_and_create_next_idea(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            llm = FakeLlm(
                ideas=[{"title": "one"}, {"title": "two"}],
                root_batches=[
                    make_drafts("b1"),
                    make_drafts("b2"),
                    make_drafts("b3"),
                    make_drafts("b4"),
                ],
            )
            worldquant = FakeWorldQuant(
                results=[bad_result()] * 15 + [parent_eligible_result()] * 5
            )
            engine, store = build_engine(
                Path(temp_dir) / "r.sqlite", llm=llm, worldquant=worldquant
            )
            try:
                outcome = engine.run_until_iteration_boundary()

                ideas = store.list_ideas(engine.run_id)
                self.assertEqual(ideas[0]["status"], "EXHAUSTED")
                self.assertEqual(worldquant.simulation_count, 20)
                self.assertEqual(llm.root_batch_count, 4)
                self.assertEqual(outcome.status, "PARENTS_READY")
                self.assertEqual(len(outcome.parents), 2)
            finally:
                store.close()

    def test_invalid_drafts_are_recorded_but_not_simulated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            drafts = make_drafts("only", count=5)
            # đánh dấu 2 biểu thức là không hợp lệ -> không được simulate
            invalid = {drafts[0].expression, drafts[1].expression}
            llm = FakeLlm(ideas=[{"title": "one"}], root_batches=[drafts])
            worldquant = FakeWorldQuant(results=[parent_eligible_result()] * 3)
            engine, store = build_engine(
                Path(temp_dir) / "r.sqlite",
                llm=llm,
                worldquant=worldquant,
                validator=FakeValidator(invalid_expressions=invalid),
            )
            try:
                outcome = engine.run_until_iteration_boundary()
                # lô 1: 5 draft, 2 invalid -> chỉ 3 được simulate
                self.assertEqual(worldquant.simulation_count, 3)
                self.assertEqual(outcome.status, "PARENTS_READY")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
