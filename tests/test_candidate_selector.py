import tempfile
import unittest
from pathlib import Path

from candidate_selector import CandidateContext, CandidateSelector
from metadata_store import MetadataStore
from research_config import ResearchConfig
from research_models import Scope
from research_store import ResearchStore


class CandidateSelectorTest(unittest.TestCase):
    KEYWORD_DESCRIPTIONS = [
        "earnings per share growth",
        "operating cash flow margin",
        "accrual ratio quality",
        "cash flow to debt",
        "earnings surprise standardized",
    ]

    def _build_metadata(self, path):
        store = MetadataStore.create(path, "snap-1", "Fundamentals")
        scope = Scope("EQUITY", "USA", 1, "TOP3000")
        scope_id = store.upsert_scope(scope)
        store.upsert_dataset({
            "id": "fundamental6",
            "name": "Fundamentals",
            "description": "Fundamental accounting data",
            "category": {"id": "fundamental"},
        }, scope_id)
        for index, description in enumerate(self.KEYWORD_DESCRIPTIONS):
            store.upsert_data_field({
                "id": f"kw_{index}",
                "dataset": {"id": "fundamental6"},
                "description": description,
                "type": "MATRIX",
            }, scope_id)
        for index in range(25):
            store.upsert_data_field({
                "id": f"generic_{index}",
                "dataset": {"id": "fundamental6"},
                "description": f"generic metric number {index}",
                "type": "MATRIX",
            }, scope_id)
        store.upsert_operator({
            "name": "rank", "definition": "rank(x)", "scope": ["REGULAR"],
        })
        store.upsert_operator({
            "name": "ts_delta", "definition": "ts_delta(x, d)", "scope": ["REGULAR"],
        })
        store.complete_snapshot()
        return store

    def test_builds_small_catalog_and_selects_fields_by_keywords(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = self._build_metadata(Path(temp_dir) / "snap.sqlite")
            research = ResearchStore.create(Path(temp_dir) / "research.sqlite")
            config = ResearchConfig()
            try:
                selector = CandidateSelector(metadata, research, config)

                catalog = selector.build_dataset_catalog(limit=8)
                context = selector.select_context(idea={
                    "title": "Earnings quality",
                    "field_keywords": ["earnings", "cash flow", "accrual"],
                    "dataset_keywords": ["fundamental"],
                })

                self.assertIsInstance(context, CandidateContext)
                self.assertLessEqual(len(catalog), 8)
                self.assertGreaterEqual(len(context.fields), config.candidate_fields_min)
                self.assertLessEqual(len(context.fields), config.candidate_fields_max)
                self.assertEqual(context.scope, Scope("EQUITY", "USA", 1, "TOP3000"))
                self.assertTrue(all(
                    field["dataset_id"] in context.dataset_ids
                    for field in context.fields
                ))
                self.assertTrue(context.operators)
            finally:
                metadata.close()
                research.close()


if __name__ == "__main__":
    unittest.main()
