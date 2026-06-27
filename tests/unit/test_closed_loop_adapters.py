"""Test adapter vòng kín: RefinementLoopRefiner map LoopResult->IdeaOutcome; GPIdeaSource bọc
generate_many với seed tăng dần. RefinementLoopRefiner test bằng fake loop (không AI/sim thật);
GPIdeaSource test trên small_panel + DB in-memory."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.operators_local  # noqa: F401
from src.app.closed_loop_adapters import GPIdeaSource, RefinementLoopRefiner
from src.backtest.config import Neutralization, PortfolioConfig
from src.lang.registry import default_registry
from src.pipeline.closed_loop import IdeaOutcome
from src.pipeline.shortlist import ShortlistCandidate
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


class _FakeLoopResult:
    def __init__(self) -> None:
        self.best_candidate = type("C", (), {"expression": "rank(close)"})()
        self.best_passed = True
        self.best_alpha_id = "WQ42"
        self.best_metrics = {"sharpe": 1.6, "fitness": 1.3, "turnover": 0.2}
        self.best_self_corr = 0.35
        self.sims_used = 3
        self.stop_reason = "patience"


class _FakeLoop:
    def __init__(self) -> None:
        self.seeds: list[str] = []

    def run_from_seed(self, expression: str, on_progress=None) -> _FakeLoopResult:
        self.seeds.append(expression)
        return _FakeLoopResult()


def _cand(expr: str) -> ShortlistCandidate:
    from src.backtest.metrics_local import AlphaMetrics
    m = AlphaMetrics(sharpe=1.0, annual_return=0.1, turnover=0.3, max_drawdown=0.05,
                     fitness=1.0, per_year_sharpe={2021: 1.0}, weight_concentration=0.05)
    d = (np.datetime64("2021-01-01") + np.arange(5)).astype("datetime64[D]")
    return ShortlistCandidate(expr=expr, metrics=m, pnl=np.ones(5), dates=d)


def test_refiner_maps_loopresult_to_ideaoutcome() -> None:
    refiner = RefinementLoopRefiner(_FakeLoop())
    outcome = refiner.refine_and_sim(_cand("rank(close)"))
    assert isinstance(outcome, IdeaOutcome)
    assert outcome.passed is True
    assert outcome.wq_alpha_id == "WQ42"
    assert outcome.sharpe == 1.6
    assert outcome.fitness == 1.3
    assert outcome.turnover == 0.2
    assert outcome.self_corr == 0.35
    assert outcome.sims_used == 3
    assert outcome.stop_reason == "patience"
    assert outcome.canonical_hash  # tính được từ expr (parse+CanonicalHasher), không rỗng


def test_refiner_seeds_loop_with_candidate_expr() -> None:
    loop = _FakeLoop()
    RefinementLoopRefiner(loop).refine_and_sim(_cand("ts_mean(close, 5)"))
    assert loop.seeds == ["ts_mean(close, 5)"]  # seed loop bằng đúng expr candidate


@pytest.fixture
def repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    return MiniBrainRepository(sessionmaker(bind=engine, future=True, expire_on_commit=False))


def test_gp_idea_source_yields_candidates_and_advances_seed(small_panel, repo) -> None:  # noqa: ANN001
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    src = GPIdeaSource(small_panel, repo, cfg, default_registry(),
                       pop_size=6, n_generations=0, base_seed=42, top_k=5, max_corr=0.99)
    b1 = src.next_batch()
    b2 = src.next_batch()
    assert all(isinstance(c, ShortlistCandidate) for c in b1)
    assert isinstance(b2, list)  # batch 2 dùng seed khác (42 -> 43), không crash
