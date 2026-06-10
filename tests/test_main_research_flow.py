import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from main import ResearchDependencies, run_application


def snapshot(snapshot_id, label):
    return SimpleNamespace(
        snapshot_id=snapshot_id, label=label,
        path=Path(f"{snapshot_id}.sqlite"),
    )


def fake_dependencies(ready_snapshots=None):
    engine = Mock()
    engine_factory = Mock(return_value=engine)
    synchronizer = Mock()
    synchronizer.create_snapshot.return_value = SimpleNamespace(
        snapshot_id="new-id", path=Path("new.sqlite"), status="READY"
    )
    store = Mock()
    store.list_pending_review.return_value = [
        {"alpha_id": 1, "worldquant_alpha_id": "wq-1", "status": "PENDING_REVIEW"}
    ]
    store.get_alpha.return_value = {"expression": "rank(close)"}
    deps = ResearchDependencies(
        store=store, synchronizer=synchronizer,
        ready_snapshots=ready_snapshots or [], engine_factory=engine_factory,
        control=Mock(),
    )
    deps.engine = engine
    return deps


class MainResearchFlowTest(unittest.TestCase):
    def test_create_snapshot_then_start_engine(self):
        input_func = Mock(side_effect=["1", "USA tháng 6"])
        dependencies = fake_dependencies()

        run_application(
            "user@example.com", "secret",
            input_func=input_func, dependencies=dependencies,
        )

        dependencies.synchronizer.create_snapshot.assert_called_once_with(
            "USA tháng 6"
        )
        dependencies.engine_factory.assert_called_once()
        dependencies.engine.run.assert_called_once()

    def test_selects_only_ready_snapshot_for_same_account(self):
        input_func = Mock(side_effect=["2", "1"])
        dependencies = fake_dependencies(
            ready_snapshots=[snapshot("id-1", "Old DB")]
        )

        run_application(
            "user@example.com", "secret",
            input_func=input_func, dependencies=dependencies,
        )

        self.assertEqual(
            dependencies.engine_factory.call_args.kwargs["snapshot_id"], "id-1"
        )
        dependencies.engine.run.assert_called_once()

    def test_review_option_prints_pending_without_running(self):
        input_func = Mock(side_effect=["3"])
        dependencies = fake_dependencies()

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            run_application(
                "user@example.com", "secret",
                input_func=input_func, dependencies=dependencies,
            )

        output = buffer.getvalue()
        self.assertIn("wq-1", output)
        self.assertIn("rank(close)", output)
        dependencies.engine_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
