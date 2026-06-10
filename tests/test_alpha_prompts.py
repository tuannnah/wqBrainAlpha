import json
import unittest

from alpha_prompts import (
    PromptResponseError,
    build_root_alpha_prompt,
    build_variant_prompt,
    map_root_response,
    map_variant_response,
)
from candidate_selector import CandidateContext
from research_models import Scope


def context():
    return CandidateContext(
        dataset_ids=["pv1"],
        scope=Scope("EQUITY", "USA", 1, "TOP3000"),
        fields=[
            {"id": "close", "dataset_id": "pv1", "field_type": "MATRIX"},
            {"id": "returns", "dataset_id": "pv1", "field_type": "MATRIX"},
        ],
        operators=[{"name": "rank"}, {"name": "ts_delta"}],
    )


def alpha_obj(hypothesis, expression):
    return {
        "hypothesis": hypothesis,
        "rationale": "r",
        "expression": expression,
        "dataset_ids": ["pv1"],
        "field_ids": ["close"],
        "operator_names": ["rank"],
        "settings": {"region": "USA"},
    }


class AlphaPromptTest(unittest.TestCase):
    def test_root_prompt_requires_distinct_hypotheses_and_allowed_ids(self):
        system, payload = build_root_alpha_prompt(
            {"title": "Reversal", "field_keywords": ["close"]},
            context(),
            count=5,
            lessons=[],
        )

        self.assertIn("exactly 5", system)
        self.assertIn("distinct hypothesis", system)
        self.assertEqual(
            {item["id"] for item in payload["allowed_fields"]},
            {"close", "returns"},
        )
        self.assertNotIn("api_key", json.dumps(payload))

    def test_variant_prompt_has_one_improvement_direction(self):
        parent = {
            "alpha_id": 7,
            "expression": "rank(close)",
            "hypothesis": "h",
            "metrics": {"sharpe": 1.2},
        }
        system, payload = build_variant_prompt(parent, "REDUCE_TURNOVER", context())
        self.assertEqual(payload["improvement_direction"], "REDUCE_TURNOVER")
        self.assertIn("exactly one", system)

    def test_map_root_rejects_duplicate_hypotheses(self):
        data = {"alphas": [
            alpha_obj("same", "rank(close)"),
            alpha_obj("same", "ts_delta(close, 5)"),
        ]}
        with self.assertRaises(PromptResponseError):
            map_root_response(data, context())

    def test_map_root_builds_drafts(self):
        data = {"alphas": [alpha_obj("h1", "rank(close)"), alpha_obj("h2", "rank(returns)")]}
        drafts = map_root_response(data, context())
        self.assertEqual(len(drafts), 2)
        self.assertEqual(drafts[0].generation, 0)
        self.assertIsNone(drafts[0].parent_id)

    def test_map_variant_sets_lineage(self):
        data = {"alphas": [alpha_obj("h1", "rank(ts_delta(close, 5))")]}
        draft = map_variant_response(data, parent_alpha_id=7, direction="CHANGE_TIME_WINDOW")
        self.assertEqual(draft.generation, 1)
        self.assertEqual(draft.parent_id, 7)
        self.assertEqual(draft.improvement_direction, "CHANGE_TIME_WINDOW")

    def test_map_requires_keys(self):
        with self.assertRaises(PromptResponseError):
            map_root_response({"alphas": [{"hypothesis": "x"}]}, context())


if __name__ == "__main__":
    unittest.main()
