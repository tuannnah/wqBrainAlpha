"""Salvage diversity của Engine B vào A: NOVEL_ALPHAS trộn vào seed pool của
RefinementLoop (Task 1a). Trước đây diversity nằm ở `_seed_pool()` của HybridEngine;
sau khi gỡ B, A phải kế thừa để không sụp về một seed LLM duy nhất."""

from __future__ import annotations

from sqlalchemy import create_engine

from src.generation.novel_ideas import NOVEL_ALPHAS
from src.llm.hypothesis import Hypothesis
from src.llm.loop import RefinementLoop
from src.llm.translator import AlphaCandidate
from src.simulation.pre_filter import PreFilter
from src.simulation.simulator import SimulationResult
from src.storage.db import init_db, make_session_factory
from src.storage.repository import AlphaRepository
from tests.fakes import FakeSimulator


def _repo():
    engine = init_db(
        create_engine(
            "sqlite:///:memory:", future=True, connect_args={"check_same_thread": False}
        )
    )
    return AlphaRepository(make_session_factory(engine))


class _FakeHyp:
    def generate(self, direction, palette=None):
        return Hypothesis("o", "b", "r", "s")


class _FakeTranslator:
    def field_palette(self, text):
        return []

    def translate(self, hyp):
        return AlphaCandidate(hyp, "mô tả gốc", "rank(close)")


def _loop(**kw):
    return RefinementLoop(
        hypothesis_gen=_FakeHyp(),
        translator=_FakeTranslator(),
        refiner=None,
        simulator=FakeSimulator(
            results=lambda e: SimulationResult(
                expression=e, status="passed", sharpe=1.5, fitness=1.2,
                turnover=0.3, drawdown=0.1, raw={},
            )
        ),
        prefilter=PreFilter(known_operators=None, known_fields=None),
        repo=_repo(),
        region="USA",
        universe="TOP3000",
        **kw,
    )


def test_seed_includes_novel_alphas():
    """Tập seed của một RefinementLoop mới chứa >=1 entry từ NOVEL_ALPHAS."""
    loop = _loop()
    pool = loop.seed_candidates("hướng nghiên cứu X")
    exprs = {c.expression for c in pool}
    novel = {c.expression for c in NOVEL_ALPHAS}
    assert exprs & novel, "seed pool phải trộn ít nhất 1 alpha NOVEL"


def test_seed_uu_tien_seed_llm_dau_tien():
    """Seed LLM (dịch từ giả thuyết) vẫn đứng đầu — NOVEL chỉ là fallback đa dạng,
    không thay thế hành vi greedy hiện tại khi LLM dịch được."""
    loop = _loop()
    pool = loop.seed_candidates("X")
    assert pool[0].expression == "rank(close)"
    assert all(isinstance(c, AlphaCandidate) for c in pool)
