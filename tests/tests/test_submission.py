"""Test CorrelationChecker và SubmissionManager."""

from __future__ import annotations

from sqlalchemy import create_engine

from src.storage.db import init_db, make_session_factory
from src.storage.models import AlphaModel, SimulationModel, SubmissionModel
from src.submission.correlation import CorrelationChecker
from src.submission.manager import SubmissionManager
from tests.fakes import FakeClient, FakeResponse


def _engine():
    return create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})


def _seed(session_factory):
    session = session_factory()
    try:
        def add(alpha_id, wq_id, sharpe, fitness, score, status="passed"):
            session.add(AlphaModel(id=alpha_id, expression=f"rank({alpha_id})", source="ga"))
            session.add(
                SimulationModel(
                    id="s_" + alpha_id,
                    alpha_id=alpha_id,
                    wq_alpha_id=wq_id,
                    region="USA",
                    universe="TOP3000",
                    sharpe=sharpe,
                    fitness=fitness,
                    score=score,
                    status=status,
                )
            )

        add("a1", "WQ1", 2.0, 1.5, 0.9)  # đạt
        add("a2", "WQ2", 1.6, 1.3, 0.8)  # đạt
        add("a3", "WQ3", 1.0, 1.3, 0.5)  # sharpe thấp -> loại
        add("a4", "WQ4", 2.0, 1.0, 0.7)  # fitness thấp -> loại
        add("a5", "WQ5", 1.9, 1.4, 0.95, status="failed")  # failed -> loại
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------- correlation
def test_extract_max_format_records():
    payload = {
        "schema": {"properties": [{"name": "alpha"}, {"name": "correlation"}]},
        "records": [["x", 0.3], ["y", -0.65], ["z", 0.5]],
    }
    assert CorrelationChecker._extract_max(payload) == 0.65


def test_extract_max_format_simple():
    assert CorrelationChecker._extract_max({"max": 0.42}) == 0.42


def test_is_acceptable():
    client = FakeClient()
    client.queue_get(FakeResponse(200, json_data={"max": 0.5}))
    checker = CorrelationChecker(client)
    assert checker.is_acceptable("WQ1") is True


# ------------------------------------------------------------------ selection
class FakeCorr:
    def __init__(self, value=0.1, max_self_corr=0.7):
        self.value = value
        self.max_self_corr = max_self_corr

    def max_self_correlation(self, _):
        return self.value

    def is_acceptable(self, _):
        return self.value <= self.max_self_corr


def test_select_candidates_chi_lay_dat_nguong_sap_theo_score():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    mgr = SubmissionManager(FakeClient(), sf, FakeCorr())
    cands = mgr.select_candidates()
    ids = [c.wq_alpha_id for c in cands]
    assert ids == ["WQ1", "WQ2"]  # đúng 2 alpha đạt, sắp theo score giảm dần


def test_submit_reject_khi_correlation_cao():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    mgr = SubmissionManager(FakeClient(), sf, FakeCorr(value=0.95))
    result = mgr.submit("WQ1")
    assert result.status == "rejected"

    session = sf()
    try:
        sub = session.query(SubmissionModel).filter_by(alpha_id="WQ1").one()
        assert sub.status == "rejected"
    finally:
        session.close()


def test_submit_thanh_cong():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_post(FakeResponse(201))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1))
    result = mgr.submit("WQ1")
    assert result.status == "submitted"


def test_run_daily_dry_run_khong_ghi_submission():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    mgr = SubmissionManager(FakeClient(), sf, FakeCorr(value=0.1), daily_quota=1)
    selected = mgr.run_daily(dry_run=True)
    assert len(selected) == 1  # tôn trọng quota
    assert selected[0].wq_alpha_id == "WQ1"

    session = sf()
    try:
        assert session.query(SubmissionModel).count() == 0  # dry-run không nộp
    finally:
        session.close()
