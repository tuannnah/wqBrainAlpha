"""Test tích hợp end-to-end (fake, không mạng/LLM): ClosedLoop chạy qua LocalTunerRefiner
thay cho RefinementLoopRefiner cũ — xác nhận wiring Task 5 (cờ --refiner) hoạt động đúng
với PROTOCOL refine_and_sim(candidate) -> IdeaOutcome mà ClosedLoop mong đợi."""

from __future__ import annotations

import numpy as np

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.app.closed_loop_adapters import LocalTunerRefiner
from src.backtest.config import PortfolioConfig
from src.pipeline.closed_loop import ClosedLoop, IdeaOutcome
from src.pipeline.shortlist import ShortlistCandidate
from src.simulation.config import SimConfig
from src.simulation.simulator import SimulationResult
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


@pytest.fixture
def repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    return MiniBrainRepository(sessionmaker(bind=engine, future=True, expire_on_commit=False))


class _Repo:
    """Fake repo tối thiểu: gộp vai trò AlphaRepository (save_alpha/save_simulation, do
    LocalTunerRefiner gọi) + MiniBrainRepository (avoided_exprs/record_brain_sim/load_pool,
    do ClosedLoop/GPIdeaSource gọi) — trong wiring thật đây là HAI object khác nhau
    (xem main.py: loop.repo là AlphaRepository, repo truyền ClosedLoop là MiniBrainRepository),
    nhưng test này chỉ cần một fake gộp đủ method để không crash."""

    def save_alpha(self, *a, **k):
        return "a1"

    def save_simulation(self, *a, **k):
        return None

    def record_brain_sim(self, *a, **k):
        return None

    def load_pool(self):
        return []

    def avoided_exprs(self):
        return set()


class _OneShotSource:
    """Nguồn ý tưởng trả đúng 1 batch (1 ứng viên) rồi cạn ([] lần sau)."""

    def __init__(self):
        self._done = False

    def next_batch(self):
        if self._done:
            return []
        self._done = True
        return [ShortlistCandidate(expr="rank(ts_delta(close, 5))", metrics=None,
                                   pnl=np.zeros(3),
                                   dates=np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]"))]


class _Sim:
    """Simulator giả: luôn trả kết quả pass Sharpe/fitness cao, không gọi mạng thật."""

    def __init__(self):
        self.calls = 0

    def simulate(self, expr, settings=None):
        self.calls += 1
        return SimulationResult(expression=expr, alpha_id="wq-1", status="passed",
                                sharpe=1.7, fitness=1.2, turnover=0.3, raw={})


def test_closed_loop_chay_qua_local_refiner():
    """ClosedLoop chạy trọn 1 ý tưởng qua LocalTunerRefiner (tune local + đúng 1 sim Brain
    cho config tốt nhất) — không đụng RefinementLoop/LLM nào."""
    from src.backtest.local_tuner import TuneResult

    def fake_tune(expr, cfg, data, **kw):
        # Giả lập coordinate descent (Task 2/3) đã tune xong: local_sharpe=1.6 > sàn
        # PRE_SIM_LOCAL_SHARPE_FLOOR (0.5) -> LocalTunerRefiner phải đi tiếp tới sim Brain.
        return TuneResult(best_expr=expr, best_config=PortfolioConfig(decay=4, truncation=0.08),
                          local_sharpe=1.6)

    refiner = LocalTunerRefiner(
        simulator=_Sim(), repo=_Repo(), data=object(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(), tune_fn=fake_tune,
    )
    cl = ClosedLoop(idea_source=_OneShotSource(), refiner=refiner, repo=_Repo(),
                    region="USA", universe="TOP3000", max_ideas=1)
    report = cl.run()
    assert report.sims_used == 1     # đúng 1 sim Brain (không refine lặp qua LLM)
    assert report.ideas_tried == 1


def test_build_closed_loop_dung_refiner_duoc_truyen_vao(small_panel, repo) -> None:  # noqa: ANN001
    """`build_closed_loop(..., refiner=...)` phải DÙNG ĐÚNG refiner truyền vào (vd
    LocalTunerRefiner) thay vì luôn mặc định RefinementLoopRefiner(loop) — đây chính là
    wiring cốt lõi của Task 5 (cờ --refiner). loop=None hợp lệ vì refiner tường minh
    khiến build_closed_loop KHÔNG cần đụng tới loop để dựng RefinementLoopRefiner."""
    from src.app.closed_loop_adapters import build_closed_loop
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.lang.registry import default_registry

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)

    class _StubRefiner:
        def __init__(self) -> None:
            self.calls = 0

        def refine_and_sim(self, candidate):
            self.calls += 1
            return IdeaOutcome(expr=candidate.expr, canonical_hash="h", passed=False,
                               wq_alpha_id=None, sharpe=None, fitness=None, turnover=None,
                               self_corr=None, sims_used=0, stop_reason="stub")

    stub = _StubRefiner()
    cl = build_closed_loop(data=small_panel, repo=repo, config=cfg,
                           registry=default_registry(), loop=None,
                           pop_size=6, n_generations=0, top_k=3, max_ideas=1,
                           refiner=stub)
    assert cl.refiner is stub        # refiner truyền vào được dùng nguyên vẹn, không bị bỏ qua
    cl.run()
    assert stub.calls >= 1           # refiner truyền vào thực sự được gọi khi run()
