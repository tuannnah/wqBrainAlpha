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


class _FakeIdeaSource:
    """Trả các batch cố định rồi cạn ([] -> ClosedLoop dừng)."""

    def __init__(self, batches: list[list[ShortlistCandidate]]) -> None:
        self._batches = list(batches)

    def next_batch(self) -> list[ShortlistCandidate]:
        return self._batches.pop(0) if self._batches else []


class _FakeRefiner:
    """Trả IdeaOutcome theo map expr->outcome; expr không có map -> failed mặc định.
    Nếu expr nằm trong `quota_on` -> ném QuotaExhausted (giả lập Brain hết quota)."""

    def __init__(self, outcomes: dict[str, IdeaOutcome], quota_on: set[str] | None = None) -> None:
        self._outcomes = outcomes
        self._quota_on = quota_on or set()
        self.calls: list[str] = []

    def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome:
        self.calls.append(candidate.expr)
        if candidate.expr in self._quota_on:
            raise QuotaExhausted("het quota")
        return self._outcomes.get(
            candidate.expr,
            IdeaOutcome(expr=candidate.expr, canonical_hash="h_" + candidate.expr,
                        passed=False, wq_alpha_id=None, sharpe=None, fitness=None,
                        turnover=None, self_corr=None, sims_used=1, stop_reason="patience"),
        )


def _passed(expr: str) -> IdeaOutcome:
    return IdeaOutcome(expr=expr, canonical_hash="h_" + expr, passed=True,
                       wq_alpha_id="WQ_" + expr, sharpe=1.5, fitness=1.2, turnover=0.2,
                       self_corr=0.3, sims_used=2, stop_reason="passed")


def test_run_persists_each_outcome_and_counts(repo) -> None:  # noqa: ANN001
    src = _FakeIdeaSource([[_cand("close"), _cand("open")]])
    refiner = _FakeRefiner({"close": _passed("close")})  # open -> failed mặc định
    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo)
    report = loop.run()
    assert isinstance(report, ClosedLoopReport)
    assert report.ideas_tried == 2
    assert report.n_passed == 1
    assert report.n_abandoned == 1
    assert report.sims_used == 3  # 2 (close passed) + 1 (open failed)
    assert report.stop_reason == "no_more_ideas"
    sims = repo.load_brain_sims()
    assert len(sims) == 2
    assert {s.status for s in sims} == {"passed", "failed"}


def test_run_stops_on_quota_exhausted(repo) -> None:  # noqa: ANN001
    src = _FakeIdeaSource([[_cand("a"), _cand("b"), _cand("c")]])
    refiner = _FakeRefiner({"a": _passed("a")}, quota_on={"b"})  # b -> hết quota
    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo)
    report = loop.run()
    assert report.stop_reason == "quota"
    assert report.ideas_tried == 1   # chỉ 'a' xong; 'b' ném quota trước khi tính
    assert refiner.calls == ["a", "b"]  # 'c' không bao giờ được gọi
    assert len(repo.load_brain_sims()) == 1  # chỉ 'a' kịp ghi


def test_run_skips_duplicate_expr_within_session(repo) -> None:  # noqa: ANN001
    src = _FakeIdeaSource([[_cand("dup"), _cand("dup")]])
    refiner = _FakeRefiner({"dup": _passed("dup")})
    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo)
    report = loop.run()
    assert refiner.calls == ["dup"]  # lần 2 bị bỏ qua
    assert report.ideas_tried == 1


def test_run_stops_on_empty_batch(repo) -> None:  # noqa: ANN001
    src = _FakeIdeaSource([])  # cạn ngay
    refiner = _FakeRefiner({})
    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo)
    report = loop.run()
    assert report.ideas_tried == 0
    assert report.stop_reason == "no_more_ideas"


def test_run_respects_max_ideas(repo) -> None:  # noqa: ANN001
    # idea_source vô hạn (mỗi batch 1 ý tưởng mới) -> max_ideas chặn.
    class _Infinite:
        def __init__(self) -> None:
            self.i = 0

        def next_batch(self) -> list[ShortlistCandidate]:
            self.i += 1
            return [_cand(f"x{self.i}")]

    loop = ClosedLoop(idea_source=_Infinite(), refiner=_FakeRefiner({}), repo=repo,
                      max_ideas=3)
    report = loop.run()
    assert report.ideas_tried == 3
    assert report.stop_reason == "no_more_ideas"
