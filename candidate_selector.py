"""Chọn cục bộ dataset, data field và operator làm context cho DeepSeek."""

from dataclasses import dataclass
from typing import List

from research_models import Scope


MAX_SELECTED_DATASETS = 3


class CandidateSelectionError(RuntimeError):
    """Không tìm được tổ hợp dataset có scope chung hợp lệ."""


@dataclass(frozen=True)
class CandidateContext:
    dataset_ids: List[str]
    scope: Scope
    fields: List[dict]
    operators: List[dict]


class CandidateSelector:
    def __init__(self, metadata_store, research_store, config):
        self.metadata_store = metadata_store
        self.research_store = research_store
        self.config = config

    def build_dataset_catalog(self, limit):
        catalog = self.metadata_store.dataset_catalog(limit=limit)
        usage = self.research_store.dataset_usage_counts()
        catalog.sort(key=lambda item: (usage.get(item["id"], 0), -item["field_count"]))
        return catalog[:limit]

    def select_context(self, idea):
        datasets, scope = self._select_datasets(idea)
        fields = self._search_and_fill_fields(
            idea.get("field_keywords", []),
            [item["id"] for item in datasets],
        )
        field_types = {field["field_type"] for field in fields}
        operators = self.metadata_store.operators_for_types(field_types)
        return CandidateContext(
            dataset_ids=[item["id"] for item in datasets],
            scope=scope,
            fields=fields,
            operators=operators,
        )

    # -- Dataset selection -------------------------------------------------

    def _select_datasets(self, idea):
        catalog = self.build_dataset_catalog(
            limit=max(self.config.candidate_fields_max, MAX_SELECTED_DATASETS)
        )
        ranked = self._prefer_keyword_matches(catalog, idea.get("dataset_keywords", []))

        selected = []
        common_scope = None
        for dataset in ranked:
            scopes = set(self.metadata_store.scope_for_dataset(dataset["id"]))
            if not scopes:
                continue
            candidate_common = scopes if common_scope is None else common_scope & scopes
            if not candidate_common:
                continue
            selected.append(dataset)
            common_scope = candidate_common
            if len(selected) >= MAX_SELECTED_DATASETS:
                break

        if not selected or not common_scope:
            raise CandidateSelectionError(
                "Không có dataset nào dùng được với scope chung."
            )

        scope = sorted(
            common_scope,
            key=lambda s: (s.instrument_type, s.region, s.delay, s.universe),
        )[0]
        return selected, scope

    @staticmethod
    def _prefer_keyword_matches(catalog, dataset_keywords):
        keywords = [keyword.lower() for keyword in dataset_keywords]
        if not keywords:
            return catalog

        def matches(dataset):
            text = " ".join(filter(None, [
                dataset.get("name"),
                dataset.get("description"),
                dataset.get("category_id"),
            ])).lower()
            return any(keyword in text for keyword in keywords)

        preferred = [dataset for dataset in catalog if matches(dataset)]
        rest = [dataset for dataset in catalog if not matches(dataset)]
        return preferred + rest

    # -- Field selection ---------------------------------------------------

    def _search_and_fill_fields(self, field_keywords, dataset_ids):
        minimum = self.config.candidate_fields_min
        maximum = self.config.candidate_fields_max

        selected = {}
        for keyword in field_keywords:
            for field in self.metadata_store.search_fields(keyword, maximum, dataset_ids):
                selected.setdefault(field["id"], field)
            if len(selected) >= maximum:
                break

        if len(selected) < minimum:
            for field in self.metadata_store.fields_in_datasets(dataset_ids, maximum):
                selected.setdefault(field["id"], field)
                if len(selected) >= minimum:
                    break

        return list(selected.values())[:maximum]
