"""Re-seed diversity định kỳ (Task 1b): thay cho "inject mỗi K gen" của GA cũ.

Khi nhánh refine stuck `reseed_every` vòng không cải thiện, loop sinh một direction
MỚI từ idea_generator (LLM re-seed) thay vì tiếp tục refine nhánh kẹt. Không tái lập GA."""

from __future__ import annotations

from sqlalchemy import create_engine

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


def _prefilter():
    return PreFilter(known_operators={"rank"}, known_fields={"close"})


class _FakeHyp:
    def generate(self, direction, palette=None):
        return Hypothesis("o", "b", "r", "s")


class _FakeTranslator:
    def field_palette(self, text):
        return []

    def translate(self, hyp):
        return AlphaCandidate(hyp, "mô tả", "rank(close)")


class _NoneRefiner:
    """Không đề xuất được gì -> ép nhánh stuck (patience tăng)."""

    def refine(self, candidate, metrics, weak_dimension):
        return None


class _IdeaGen:
    def __init__(self):
        self.calls = 0

    def generate_ideas(self, n):
        self.calls += 1
        return ["hướng mới từ dataset chưa dùng"]


def _sim():
    return FakeSimulator(
        results=lambda e: SimulationResult(
            expression=e, alpha_id="wq-" + e, status="passed", sharpe=1.5,
            fitness=1.2, turnover=0.3, drawdown=0.1, raw={},
        )
    )


def _loop(idea_gen, **kw):
    return RefinementLoop(
        hypothesis_gen=_FakeHyp(),
        translator=_FakeTranslator(),
        refiner=_NoneRefiner(),
        simulator=_sim(),
        prefilter=_prefilter(),
        repo=_repo(),
        region="USA",
        universe="TOP3000",
        idea_generator=idea_gen,
        **kw,
    )


def test_reseed_triggers_new_direction():
    """Bật reseed_every: nhánh stuck đủ N vòng -> gọi generate_ideas sinh direction mới."""
    idea_gen = _IdeaGen()
    loop = _loop(idea_gen, max_simulations=10, no_improve_patience=5, reseed_every=2)
    loop.run("hướng gốc")
    assert idea_gen.calls >= 1


def test_reseed_tat_mac_dinh_khong_goi_generate_ideas():
    """reseed_every=0 (mặc định) -> KHÔNG re-seed, hành vi greedy không đổi."""
    idea_gen = _IdeaGen()
    loop = _loop(idea_gen, max_simulations=10, no_improve_patience=3)
    loop.run("hướng gốc")
    assert idea_gen.calls == 0


def test_reseed_khong_idea_generator_thi_bo_qua():
    """Không truyền idea_generator -> reseed_every bị bỏ qua (tương thích ngược)."""
    loop = RefinementLoop(
        hypothesis_gen=_FakeHyp(),
        translator=_FakeTranslator(),
        refiner=_NoneRefiner(),
        simulator=_sim(),
        prefilter=_prefilter(),
        repo=_repo(),
        region="USA",
        universe="TOP3000",
        max_simulations=10,
        no_improve_patience=3,
        reseed_every=2,
    )
    res = loop.run("hướng gốc")
    assert res.best_candidate is not None  # vẫn chạy bình thường, không lỗi
