import tempfile
import unittest
from pathlib import Path

from expression_parser import parse_expression
from expression_validator import ExpressionValidator
from research_config import ResearchConfig
from research_models import AlphaDraft, Scope
from research_store import ResearchStore


def draft(expression, settings=None):
    return AlphaDraft(
        hypothesis="h",
        rationale="r",
        expression=expression,
        dataset_ids=["news"],
        field_ids=["news_vector", "close"],
        operator_names=["rank", "vec_avg"],
        settings=settings or {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "neutralization": "SUBINDUSTRY",
        },
    )


class _Context:
    dataset_ids = ["news"]
    scope = Scope("EQUITY", "USA", 1, "TOP3000")
    fields = [
        {"id": "news_vector", "dataset_id": "news", "field_type": "VECTOR"},
        {"id": "close", "dataset_id": "news", "field_type": "MATRIX"},
    ]
    operators = [{"name": "rank"}, {"name": "vec_avg"}]


class ExpressionValidatorTest(unittest.TestCase):
    def test_rejects_unknown_field_and_vector_without_reducer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            research = ResearchStore.create(Path(temp_dir) / "r.sqlite")
            try:
                validator = ExpressionValidator(None, research, ResearchConfig())
                context = _Context()

                unknown = validator.validate(draft("rank(missing_field)"), context)
                vector = validator.validate(draft("rank(news_vector)"), context)
                reduced = validator.validate(draft("rank(vec_avg(news_vector))"), context)

                self.assertIn("UNKNOWN_FIELD", unknown.error_codes)
                self.assertIn("VECTOR_REDUCER_REQUIRED", vector.error_codes)
                self.assertTrue(reduced.is_valid, reduced.error_codes)
            finally:
                research.close()

    def test_rejects_unknown_operator_and_syntax_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            research = ResearchStore.create(Path(temp_dir) / "r.sqlite")
            try:
                validator = ExpressionValidator(None, research, ResearchConfig())
                context = _Context()

                unknown_op = validator.validate(draft("zscore(close)"), context)
                broken = validator.validate(draft("rank(close"), context)

                self.assertIn("UNKNOWN_OPERATOR", unknown_op.error_codes)
                self.assertIn("SYNTAX_ERROR", broken.error_codes)
            finally:
                research.close()

    def test_rejects_exact_duplicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            research = ResearchStore.create(Path(temp_dir) / "r.sqlite")
            try:
                existing = "rank(vec_avg(news_vector))"
                parsed = parse_expression(existing)
                run_id = research.start_run("snap", {})
                idea_id = research.create_idea(run_id, "i", "DEEPSEEK")
                hyp_id = research.create_hypothesis(idea_id, "h", "r", ["news"], ["x"])
                research.create_alpha(
                    run_id, hyp_id, existing, parsed.expression_hash,
                    parsed.fingerprint, {"region": "USA"}, ["news"], None, 0, None,
                )

                validator = ExpressionValidator(None, research, ResearchConfig())
                result = validator.validate(draft(existing), _Context())

                self.assertIn("DUPLICATE_EXPRESSION", result.error_codes)
            finally:
                research.close()


if __name__ == "__main__":
    unittest.main()
