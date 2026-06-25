"""Test seeds.py: seed cores tu families/novel_ideas parse thanh Node hop le, seed LLM
qua fake hypothesis_gen/translator (khong mang thuc), all_seed_cores gop dung + fail-fast
khi with_llm=True thieu dependency."""

from __future__ import annotations

import pytest

from src.gp.seeds import (
    all_seed_cores,
    seed_cores_from_families,
    seed_cores_from_llm,
    seed_cores_from_novel_ideas,
)
from src.lang.ast import Node


def test_seed_cores_from_novel_ideas_returns_parsed_nodes():
    nodes = seed_cores_from_novel_ideas()
    assert len(nodes) > 0
    assert all(isinstance(n, Node) for n in nodes)


def test_seed_cores_from_families_returns_parsed_nodes():
    nodes = seed_cores_from_families()
    assert len(nodes) > 0
    assert all(isinstance(n, Node) for n in nodes)


def test_seed_cores_from_llm_uses_injected_fakes_no_network():
    class _FakeHypothesis:
        def to_dict(self):
            return {}

    class _FakeHypothesisGen:
        def generate(self, direction, palette=None):
            return _FakeHypothesis()

    class _FakeCandidate:
        expression = "rank(close)"

    class _FakeTranslator:
        def translate(self, hypothesis):
            return _FakeCandidate()

    nodes = seed_cores_from_llm(
        _FakeHypothesisGen(), _FakeTranslator(), research_directions=["momentum mới"],
    )
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)


def test_seed_cores_from_llm_skips_none_candidate():
    class _FakeHypothesisGen:
        def generate(self, direction, palette=None):
            return object()

    class _FakeTranslatorRejects:
        def translate(self, hypothesis):
            return None  # translator từ chối

    nodes = seed_cores_from_llm(
        _FakeHypothesisGen(), _FakeTranslatorRejects(), research_directions=["x"],
    )
    assert nodes == []


def test_all_seed_cores_without_llm_combines_families_and_novel():
    nodes = all_seed_cores(with_llm=False)
    expected_min = len(seed_cores_from_families()) + len(seed_cores_from_novel_ideas())
    assert len(nodes) == expected_min


def test_all_seed_cores_with_llm_true_requires_dependencies():
    with pytest.raises(ValueError):
        all_seed_cores(with_llm=True)  # thiếu hypothesis_gen/translator/research_directions
