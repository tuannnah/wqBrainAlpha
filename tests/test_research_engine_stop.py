import tempfile
import unittest
from pathlib import Path

from tests.engine_fakes import (
    FakeLlm,
    FakeWorldQuant,
    build_engine,
    make_drafts,
    qualified_result,
)


class StopAfterFirstSim:
    def __init__(self):
        self._stop = False

    def stop_requested(self):
        return self._stop

    def trigger(self):
        self._stop = True


class OneShotWorldQuant:
    def __init__(self, control):
        self.control = control
        self.simulation_count = 0

    def simulate_alpha(self, payload):
        self.simulation_count += 1
        self.control.trigger()
        return qualified_result()


class ResearchEngineStopTest(unittest.TestCase):
    def test_stops_after_ten_new_qualified_alphas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            llm = FakeLlm(
                ideas=[{"title": "i"}],
                root_batches=[make_drafts("a")],
            )
            worldquant = FakeWorldQuant(always_qualified=True)
            engine, store = build_engine(
                Path(temp_dir) / "r.sqlite", llm=llm, worldquant=worldquant,
                target=10,
            )
            try:
                outcome = engine.run()

                self.assertEqual(outcome.status, "TARGET_REACHED")
                self.assertEqual(store.count_qualified_for_run(outcome.run_id), 10)
                self.assertEqual(len(store.list_pending_review()), 10)
            finally:
                store.close()

    def test_quit_finishes_current_simulation_but_starts_no_new_work(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            control = StopAfterFirstSim()
            llm = FakeLlm(ideas=[{"title": "i"}], root_batches=[make_drafts("a")])
            worldquant = OneShotWorldQuant(control)
            engine, store = build_engine(
                Path(temp_dir) / "r.sqlite", llm=llm, worldquant=worldquant,
                control=control,
            )
            try:
                outcome = engine.run()

                self.assertEqual(outcome.status, "STOPPED_BY_USER")
                self.assertEqual(worldquant.simulation_count, 1)
                self.assertEqual(llm.requests_after_stop, 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
