"""Hat giong GP: cores (signal thuan, khong config wrapper) tu 3 nguon -- families.py
(khung cong thuc kinh dien), novel_ideas.py (10 alpha dataset it nguoi khai thac), va tuy
chon LLM (hypothesis -> translator). Day la "ramped half-and-half + SEEDING" cua B13 --
GP ngau nhien thuan lang phi danh gia tren cay vo nghia; seed kinh nghiem huong tim kiem
toi cau truc co gia thuyet kinh te. CHI tra Node (chua boc Individual) -- init.py (Task
7.4) la noi ghep seed vao quan the ban dau.

Luu y registry: ta dung ``parse`` (validate=True) de chi giu seed hop le theo registry
operator THAT (Phase 2), nen import side-effect ``src.operators_local`` o module-level de
nap toan bo operator vao REGISTRY truoc khi parse -- neu khong, registry chi co tap toi
thieu Phase 1 va gan nhu moi seed se bi loc bo. Seed dung operator chua dang ky (vd
``ts_min``) se parse loi va bi bo qua co log -- chap nhan duoc, khong sap toan bo seeding.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import src.operators_local  # noqa: F401  (side-effect: nap operator that vao REGISTRY)
from src.lang.ast import Node
from src.lang.parser import ParseError, parse

logger = logging.getLogger(__name__)


@runtime_checkable
class _HypothesisGenLike(Protocol):
    """Giao dien toi gian cho generator gia thuyet (src/llm/hypothesis.py)."""

    def generate(self, direction: str) -> object: ...


@runtime_checkable
class _TranslatorLike(Protocol):
    """Giao dien toi gian cho translator gia thuyet -> AlphaCandidate (src/llm/translator.py).

    Tra ``None`` khi tu choi (field khong hop le); nguoc lai tra object co ``.expression``.
    """

    def translate(self, hypothesis: object) -> object | None: ...


def _parse_all(expressions: list[str], *, source: str) -> list[Node]:
    """Parse tung bieu thuc; bo qua (log warning) cai nao parse loi -- 1 seed loi khong
    duoc lam sap toan bo qua trinh seeding."""
    nodes: list[Node] = []
    for expr in expressions:
        try:
            nodes.append(parse(expr))
        except ParseError as exc:
            logger.warning("seed tu %s parse loi, bo qua: %r (%s)", source, expr, exc)
    return nodes


def seed_cores_from_families() -> list[Node]:
    """Seed tu cac khung cong thuc kinh dien (src/generation/families.py).

    Ham export that la ``generate_candidates`` (gop moi ho + loai trung) -- KHAC ten gia
    dinh ``generate_family_candidates`` trong brief; da xac minh bang grep va dung ten that.
    """
    from src.generation.families import generate_candidates

    candidates = generate_candidates()
    return _parse_all([c.expression for c in candidates], source="families")


def seed_cores_from_novel_ideas() -> list[Node]:
    """Seed tu 10 alpha dataset it nguoi khai thac (src/generation/novel_ideas.py)."""
    from src.generation.novel_ideas import NOVEL_ALPHAS

    return _parse_all([c.expression for c in NOVEL_ALPHAS], source="novel_ideas")


def seed_cores_from_llm(
    hypothesis_gen: _HypothesisGenLike,
    translator: _TranslatorLike,
    research_directions: list[str],
) -> list[Node]:
    """Seed tu LLM: hypothesis_gen.generate(direction) -> Hypothesis ->
    translator.translate(hypothesis) -> AlphaCandidate | None -> parse. Khong catch loi
    mang/LLM o day -- caller (Task 7.7/CLI) quyet dinh retry/timeout."""
    nodes: list[Node] = []
    for direction in research_directions:
        hypothesis = hypothesis_gen.generate(direction)
        candidate = translator.translate(hypothesis)
        if candidate is None:
            logger.info("LLM seed bi translator tu choi cho huong: %s", direction)
            continue
        expression = candidate.expression  # type: ignore[attr-defined]
        try:
            nodes.append(parse(expression))
        except ParseError as exc:
            logger.warning("LLM seed parse loi, bo qua: %r (%s)", expression, exc)
    return nodes


def all_seed_cores(
    *,
    with_llm: bool = False,
    hypothesis_gen: _HypothesisGenLike | None = None,
    translator: _TranslatorLike | None = None,
    research_directions: list[str] | None = None,
) -> list[Node]:
    """Gop toan bo seed: families + novel_ideas luon chay (re, khong mang); LLM tuy chon,
    fail-fast neu with_llm=True ma thieu dependency (tranh am tham bo qua phan LLM khi
    caller tuong da bat)."""
    nodes = seed_cores_from_families() + seed_cores_from_novel_ideas()
    if with_llm:
        if hypothesis_gen is None or translator is None or not research_directions:
            raise ValueError(
                "with_llm=True can hypothesis_gen, translator, research_directions day du"
            )
        nodes += seed_cores_from_llm(hypothesis_gen, translator, research_directions)
    return nodes
