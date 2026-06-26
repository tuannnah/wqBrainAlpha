"""Test orchestrator ClosedLoop bằng fake (không mạng/AI/sim). Kiểm luồng: lấy ý tưởng →
refine+sim mỗi cái → record_brain_sim → tránh trùng → dừng khi hết quota / cạn ý tưởng."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.metrics_local import AlphaMetrics
from src.pipeline.closed_loop import (
    ClosedLoop,  # noqa: F401  — dùng ở test Task 2+
    ClosedLoopReport,
    IdeaOutcome,
    QuotaExhausted,
)
from src.pipeline.shortlist import ShortlistCandidate
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


def _cand(expr: str) -> ShortlistCandidate:
    m = AlphaMetrics(sharpe=1.0, annual_return=0.1, turnover=0.3, max_drawdown=0.05,
                     fitness=1.0, per_year_sharpe={2021: 1.0}, weight_concentration=0.05)
    dates = (np.datetime64("2021-01-01") + np.arange(5)).astype("datetime64[D]")
    return ShortlistCandidate(expr=expr, metrics=m, pnl=np.ones(5), dates=dates)


@pytest.fixture
def repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    sf = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(sf)


def test_idea_outcome_and_report_are_frozen() -> None:
    o = IdeaOutcome(expr="close", canonical_hash="h", passed=True, wq_alpha_id="W",
                    sharpe=1.0, fitness=1.0, turnover=0.2, self_corr=0.3, sims_used=1,
                    stop_reason="passed")
    with pytest.raises(Exception):  # FrozenInstanceError  # noqa: PT011
        o.passed = False  # type: ignore[misc]
    r = ClosedLoopReport(ideas_tried=0, sims_used=0, n_passed=0, n_abandoned=0,
                         stop_reason="no_more_ideas")
    with pytest.raises(Exception):  # noqa: PT011
        r.sims_used = 9  # type: ignore[misc]


def test_quota_exhausted_is_exception() -> None:
    assert issubclass(QuotaExhausted, Exception)
