"""Hạt giống GP: cores (signal thuần, không config wrapper) từ 3 nguồn -- families.py
(khung công thức kinh điển), novel_ideas.py (10 alpha dataset ít người khai thác), và tùy
chọn LLM (hypothesis -> translator). Đây là "ramped half-and-half + SEEDING" của B13 --
GP ngẫu nhiên thuần lãng phí đánh giá trên cây vô nghĩa; seed kinh nghiệm hướng tìm kiếm
tới cấu trúc có giả thuyết kinh tế. CHỈ trả Node (chưa bọc Individual) -- init.py (Task
7.4) là nơi ghép seed vào quần thể ban đầu.

Lưu ý registry: ta dùng ``parse`` (validate=True) để chỉ giữ seed hợp lệ theo registry
operator THẬT (Phase 2), nên import side-effect ``src.operators_local`` ở module-level để
nạp toàn bộ operator vào REGISTRY trước khi parse -- nếu không, registry chỉ có tập tối
thiểu Phase 1 và gần như mọi seed sẽ bị lọc bỏ. Seed dùng operator chưa đăng ký (vd
``ts_min``) sẽ parse lỗi và bị bỏ qua có log -- chấp nhận được, không sập toàn bộ seeding.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import src.operators_local  # noqa: F401  (side-effect: nạp operator thật vào REGISTRY)
from src.lang.ast import Node
from src.lang.parser import ParseError, parse

logger = logging.getLogger(__name__)


@runtime_checkable
class _HypothesisGenLike(Protocol):
    """Giao diện tối giản cho generator giả thuyết (src/llm/hypothesis.py)."""

    def generate(self, direction: str) -> object: ...


@runtime_checkable
class _TranslatorLike(Protocol):
    """Giao diện tối giản cho translator giả thuyết -> AlphaCandidate (src/llm/translator.py).

    Trả ``None`` khi từ chối (field không hợp lệ); ngược lại trả object có ``.expression``.
    """

    def translate(self, hypothesis: object) -> object | None: ...


def _parse_all(expressions: list[str], *, source: str) -> list[Node]:
    """Parse từng biểu thức; bỏ qua (log warning) cái nào parse lỗi -- 1 seed lỗi không
    được làm sập toàn bộ quá trình seeding."""
    nodes: list[Node] = []
    for expr in expressions:
        try:
            nodes.append(parse(expr))
        except ParseError as exc:
            logger.warning("seed từ %s parse lỗi, bỏ qua: %r (%s)", source, expr, exc)
    return nodes


def seed_cores_from_families() -> list[Node]:
    """Seed từ các khung công thức kinh điển (src/generation/families.py).

    Hàm export thật là ``generate_candidates`` (gộp mọi họ + loại trùng) -- KHÁC tên giả
    định ``generate_family_candidates`` trong brief; đã xác minh bằng grep và dùng tên thật.
    """
    from src.generation.families import generate_candidates

    candidates = generate_candidates()
    return _parse_all([c.expression for c in candidates], source="families")


def seed_cores_from_novel_ideas() -> list[Node]:
    """Seed từ alpha dataset ít người khai thác (src/generation/novel_ideas.py) — gồm cả
    v1 (khung alt-dataset kinh điển) lẫn v2 (cấu trúc gap/gate/residual chống self-corr)."""
    from src.generation.novel_ideas import all_novel_alphas

    return _parse_all([c.expression for c in all_novel_alphas()], source="novel_ideas")


def seed_cores_from_llm(
    hypothesis_gen: _HypothesisGenLike,
    translator: _TranslatorLike,
    research_directions: list[str],
) -> list[Node]:
    """Seed từ LLM: hypothesis_gen.generate(direction) -> Hypothesis ->
    translator.translate(hypothesis) -> AlphaCandidate | None -> parse. Không catch lỗi
    mạng/LLM ở đây -- caller (Task 7.7/CLI) quyết định retry/timeout."""
    nodes: list[Node] = []
    for direction in research_directions:
        hypothesis = hypothesis_gen.generate(direction)
        candidate = translator.translate(hypothesis)
        if candidate is None:
            logger.info("LLM seed bị translator từ chối cho hướng: %s", direction)
            continue
        expression = candidate.expression  # type: ignore[attr-defined]
        try:
            nodes.append(parse(expression))
        except ParseError as exc:
            logger.warning("LLM seed parse lỗi, bỏ qua: %r (%s)", expression, exc)
    return nodes


def all_seed_cores(
    *,
    with_llm: bool = False,
    hypothesis_gen: _HypothesisGenLike | None = None,
    translator: _TranslatorLike | None = None,
    research_directions: list[str] | None = None,
) -> list[Node]:
    """Gộp toàn bộ seed: families + novel_ideas luôn chạy (rẻ, không mạng); LLM tùy chọn,
    fail-fast nếu with_llm=True mà thiếu dependency (tránh âm thầm bỏ qua phần LLM khi
    caller tưởng đã bật)."""
    nodes = seed_cores_from_families() + seed_cores_from_novel_ideas()
    if with_llm:
        if hypothesis_gen is None or translator is None or not research_directions:
            raise ValueError(
                "with_llm=True cần hypothesis_gen, translator, research_directions đầy đủ"
            )
        nodes += seed_cores_from_llm(hypothesis_gen, translator, research_directions)
    return nodes
