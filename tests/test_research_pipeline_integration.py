import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from candidate_selector import CandidateSelector
from expression_validator import ExpressionValidator
from logging_setup import create_run_logger
from metadata_store import MetadataStore
from qualification import QualificationPolicy
from research_config import ResearchConfig
from research_engine import ResearchEngine
from research_models import AlphaDraft, Scope
from research_store import ResearchStore
from tests.engine_fakes import FakeControl, FakeWorldQuant


SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "delay": 1,
    "universe": "TOP3000",
    "neutralization": "SUBINDUSTRY",
}


def _draft(hypothesis, expression):
    return AlphaDraft(
        hypothesis=hypothesis,
        rationale="r",
        expression=expression,
        dataset_ids=["pv1"],
        field_ids=["close"],
        operator_names=["rank"],
        settings=dict(SETTINGS),
        generation=0,
    )


VARIANT_POOL = [
    "ts_delta(returns, 3)",
    "rank(returns)",
    "rank(ts_mean(close, 4))",
    "ts_delta(ts_mean(returns, 2), 6)",
    "group_neutralize(ts_mean(close, 5), subindustry)",
    "ts_mean(rank(returns), 9)",
    "ts_delta(rank(close), 8)",
    "rank(vec_avg(news_vector))",
]


class FakeGenerator:
    def __init__(self):
        self.variant_n = 0

    def create_idea(self, catalog, lessons):
        return (
            {"title": "reversal", "field_keywords": ["close", "returns"],
             "dataset_keywords": ["price"]},
            {"prompt_tokens": 10, "completion_tokens": 5},
        )

    def generate_root_alphas(self, idea, context, count, lessons):
        drafts = [
            _draft("h1", "rank(ts_delta(close, 5))"),
            _draft("h2", "ts_mean(returns, 10)"),
            _draft("h3", "vec_avg(news_vector)"),
            _draft("h4", "group_neutralize(rank(close), subindustry)"),
            _draft("h5", "rank(unknown_field)"),  # invalid: field không tồn tại
        ]
        return drafts, {"prompt_tokens": 20, "completion_tokens": 10}

    def generate_variant(self, parent, direction, context):
        expression = VARIANT_POOL[self.variant_n % len(VARIANT_POOL)]
        self.variant_n += 1
        draft = replace(
            _draft(f"v{self.variant_n}", expression),
            generation=1,
            parent_id=parent.alpha_id,
            improvement_direction=direction,
        )
        return draft, {"prompt_tokens": 5, "completion_tokens": 2}


def build_metadata(path):
    store = MetadataStore.create(path, "snap-int", "Integration")
    scope = Scope("EQUITY", "USA", 1, "TOP3000")
    scope_id = store.upsert_scope(scope)
    store.upsert_dataset({
        "id": "pv1", "name": "Price Volume", "description": "price and volume",
        "category": {"id": "pv"},
    }, scope_id)
    fields = [
        ("close", "close price", "MATRIX"),
        ("returns", "daily returns", "MATRIX"),
        ("news_vector", "news sentiment vector", "VECTOR"),
    ]
    for field_id, description, field_type in fields:
        store.upsert_data_field({
            "id": field_id, "dataset": {"id": "pv1"},
            "description": description, "type": field_type,
        }, scope_id)
    for name in ["rank", "ts_delta", "ts_mean", "vec_avg", "group_neutralize"]:
        store.upsert_operator({"name": name, "definition": f"{name}(x)"})
    store.complete_snapshot()
    return store


class ResearchPipelineIntegrationTest(unittest.TestCase):
    def test_end_to_end_with_fake_services(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            metadata = build_metadata(temp / "snap.sqlite")
            research = ResearchStore.create(temp / "research.sqlite")
            config = ResearchConfig(**{
                **ResearchConfig().__dict__, "similarity_threshold": 1.0,
            })
            worldquant = FakeWorldQuant(always_qualified=True)
            control = FakeControl()
            engine = ResearchEngine(
                snapshot_id="snap-int",
                store=research,
                selector=CandidateSelector(metadata, research, config),
                llm=FakeGenerator(),
                validator=ExpressionValidator(metadata, research, config),
                worldquant=worldquant,
                policy=QualificationPolicy(
                    config.sharpe_threshold, config.fitness_threshold,
                    config.turnover_min, config.turnover_hard_limit,
                    config.quality_gate_ratio,
                ),
                control=control,
                config=config,
            )
            run_id = research.start_run("snap-int", {"target": 10})
            engine.run_id = run_id
            logger = create_run_logger(run_id, temp, research, stream=io.StringIO())
            engine.logger = logger

            try:
                outcome = engine.run()

                self.assertEqual(outcome.status, "TARGET_REACHED")

                ideas = research.list_ideas(run_id)
                self.assertEqual(len(ideas), 1)

                alphas = research.list_alphas(run_id)
                # biểu thức invalid không bao giờ tạo Alpha hay được simulate
                self.assertFalse(any(a["expression"] == "rank(unknown_field)" for a in alphas))
                self.assertEqual(worldquant.simulation_count, len(alphas))

                pending = research.list_pending_review()
                self.assertEqual(len(pending), 10)
                self.assertTrue(all(item["status"] == "PENDING_REVIEW" for item in pending))

                variants = [a for a in alphas if a["generation"] == 1]
                self.assertTrue(variants)
                gen0_ids = {a["id"] for a in alphas if a["generation"] == 0}
                self.assertTrue(all(v["parent_id"] in gen0_ids for v in variants))

                hypotheses = research.connection.execute(
                    "SELECT COUNT(*) AS c FROM hypotheses"
                ).fetchone()["c"]
                self.assertGreaterEqual(hypotheses, 5)

                tokens = research.connection.execute(
                    "SELECT SUM(prompt_tokens) AS t FROM llm_requests"
                ).fetchone()["t"]
                self.assertGreater(tokens, 0)
                self.assertTrue(research.list_events(run_id))

                self.assertEqual(worldquant.submit_calls, 0)
            finally:
                logger.close()
                research.close()
                metadata.close()


if __name__ == "__main__":
    unittest.main()
