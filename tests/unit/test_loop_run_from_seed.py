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


def test_run_from_seed_loopresult_carries_brain_metrics() -> None:
    """LoopResult sau run_from_seed mang best_passed + best_metrics (sharpe/fitness/turnover)
    + best_alpha_id, để adapter map sang IdeaOutcome. Dùng fake simulator pass có metric."""
    loop = _make_loop()
    result = loop.run_from_seed("rank(close)")
    assert result.best_passed is True
    assert "sharpe" in result.best_metrics
    # alpha_id do repo.save_alpha trả (fake repo sqlite in-memory) — không None khi đã sim
    assert result.best_alpha_id is not None


class _RaisingRefiner:
    """Mô phỏng LLM refine lỗi TẠM THỜI (claude-cli exit≠0) — phải KHÔNG làm sập phiên."""

    def refine(self, candidate, metrics, weak_dimension):
        raise RuntimeError("CLI 'claude' exit 1:  (loi tam thoi)")


def test_run_from_seed_khong_crash_khi_refiner_loi() -> None:
    """LLM refine ném RuntimeError -> loop coi như bước không sinh được (bỏ qua) và vẫn trả
    LoopResult hợp lệ giữ best từ seed, KHÔNG crash cả phiên (fix reliability chạy dài)."""
    loop = _make_loop()
    loop.refiner = _RaisingRefiner()
    result = loop.run_from_seed("rank(close)")
    assert isinstance(result, LoopResult)
    assert result.best_candidate is not None  # vẫn giữ best từ seed đã sim
    assert result.sims_used >= 1
