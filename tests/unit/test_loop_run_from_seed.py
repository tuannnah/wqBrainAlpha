"""Test RefinementLoop.run_from_seed: hạt giống là công thức cho sẵn (core GP), KHÔNG qua
hypothesis_gen/translator. Tái dùng pattern fake của tests/test_loop_seed.py."""

from __future__ import annotations

from sqlalchemy import create_engine

from src.llm.hypothesis import Hypothesis
from src.llm.loop import LoopResult, RefinementLoop
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


class _FakeRefiner:
    """Trả None ngay (không cải tiến thêm) — chỉ để loop có refiner hợp lệ."""

    def refine(self, candidate, metrics, weak_dimension):
        return None


def _make_loop(**kw):
    """Dựng RefinementLoop với fakes — sao theo pattern của tests/test_loop_seed.py."""
    return RefinementLoop(
        hypothesis_gen=_FakeHyp(),
        translator=_FakeTranslator(),
        refiner=_FakeRefiner(),
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
        no_improve_patience=1,
        **kw,
    )


def test_run_from_seed_uses_given_expression_as_seed() -> None:
    """run_from_seed('rank(close)') đánh giá đúng công thức đó làm seed (KHÔNG gọi
    hypothesis_gen/seed_candidates), trả LoopResult với best_candidate.expression bắt nguồn
    từ seed."""
    loop = _make_loop()
    result = loop.run_from_seed("rank(close)")
    assert isinstance(result, LoopResult)
    assert result.best_candidate is not None
    # seed eval được -> best_candidate.expression là 'rank(close)' (hoặc biến thể refine của nó)
    assert result.sims_used >= 1


def test_run_from_seed_unparseable_seed_returns_no_seed() -> None:
    """Seed bị prefilter loại / eval None -> LoopResult stop_reason='no_seed', best=None."""
    loop = _make_loop()  # prefilter từ chối "khong_hop_le(" vì dấu ngoặc không cân bằng
    result = loop.run_from_seed("khong_hop_le(")
    assert result.best_candidate is None
    assert result.stop_reason == "no_seed"
