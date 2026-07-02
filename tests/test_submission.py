"""Test CorrelationChecker và SubmissionManager."""

from __future__ import annotations

import json

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

        add("a1", "WQ1", 2.0, 1.5, 0.9)  # WQ tự PASS toàn bộ check -> đạt
        add("a2", "WQ2", 1.6, 1.3, 0.8)  # WQ tự PASS -> đạt
        add("a3", "WQ3", 1.0, 1.3, 0.5, status="failed")  # WQ tự FAIL (vd LOW_SHARPE) -> loại
        add("a4", "WQ4", 2.0, 1.0, 0.7, status="failed")  # WQ tự FAIL (vd LOW_FITNESS) -> loại
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


def test_submission_model_co_cot_properties():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    try:
        session.add(
            SubmissionModel(
                id="sub1", alpha_id="WQ1", status="properties_set",
                tags='["PowerPoolSelected"]', regular_desc="Idea: ...",
            )
        )
        session.commit()
        row = session.query(SubmissionModel).filter_by(id="sub1").one()
        assert row.tags == '["PowerPoolSelected"]'
        assert row.regular_desc == "Idea: ..."
        assert row.properties_set_at is None
    finally:
        session.close()


# --------------------------------------------------------------- set_properties
def test_set_properties_insert_row_moi_khi_chua_tung_submit():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_patch(FakeResponse(200, json_data={"id": "WQ1"}))
    mgr = SubmissionManager(client, sf, FakeCorr())

    result = mgr.set_properties("WQ1", tags=["PowerPoolSelected"], regular_desc="Idea: " + "x" * 100)
    assert result.status == "ok"

    method, path, kwargs = client.calls[-1]
    assert method == "PATCH"
    assert path == "/alphas/WQ1"
    payload = kwargs["json"]
    assert payload["tags"] == ["PowerPoolSelected"]
    assert payload["regular"] == {"description": "Idea: " + "x" * 100}
    assert "name" not in payload  # None -> không đưa vào payload

    session = sf()
    try:
        row = session.query(SubmissionModel).filter_by(alpha_id="WQ1").one()
        assert row.status == "properties_set"
        assert row.tags == '["PowerPoolSelected"]'
        assert row.properties_set_at is not None
    finally:
        session.close()


def test_set_properties_update_row_da_submit():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_post(FakeResponse(201))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1))
    mgr.submit("WQ1")  # tạo sẵn 1 row status=submitted

    client.queue_patch(FakeResponse(200, json_data={"id": "WQ1"}))
    mgr.set_properties("WQ1", tags=["t1"])

    session = sf()
    try:
        rows = session.query(SubmissionModel).filter_by(alpha_id="WQ1").all()
        assert len(rows) == 1  # KHÔNG insert thêm row mới
        assert rows[0].status == "submitted"  # giữ nguyên status gốc
        assert rows[0].tags == '["t1"]'
    finally:
        session.close()


def test_set_properties_goi_lai_cung_payload_thi_bo_qua():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_patch(FakeResponse(200, json_data={"id": "WQ1"}))
    mgr = SubmissionManager(client, sf, FakeCorr())

    r1 = mgr.set_properties("WQ1", tags=["a"], regular_desc="mo ta")
    assert r1.status == "ok"
    n_calls_before = len(client.calls)

    r2 = mgr.set_properties("WQ1", tags=["a"], regular_desc="mo ta")
    assert r2.status == "unchanged"
    assert len(client.calls) == n_calls_before  # không gọi PATCH thêm


def test_set_properties_loi_http_khong_crash():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_patch(FakeResponse(500, text="server error"))
    mgr = SubmissionManager(client, sf, FakeCorr())

    result = mgr.set_properties("WQ1", tags=["a"])
    assert result.status == "error"

    session = sf()
    try:
        row = session.query(SubmissionModel).filter_by(alpha_id="WQ1").one()
        assert row.status == "properties_set"
        assert row.properties_set_at is None  # lỗi -> không đánh dấu đã set thành công
    finally:
        session.close()


# ------------------------------------------------------- power pool auto-tag
def test_submit_tu_gan_tag_power_pool_khi_du_dieu_kien():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    hyp = {
        "observation": "Gia co phieu dao chieu sau chuoi giam manh trong ngan han lien tuc.",
        "background": "Ly thuyet mean-reversion tren thi truong von ngan han duoc ung ho rong rai.",
        "economic_rationale": "Nha dau tu phan ung thai qua roi dieu chinh lai theo thoi gian giao dich.",
        "implementation_spec": "Dung field close, cua so 5 ngay, chuan hoa bang toan tu rank toan thi truong.",
    }
    session.add(AlphaModel(
        id="a1", expression="rank(add(close, open))", source="ga", hypothesis=json.dumps(hyp),
    ))
    session.add(SimulationModel(
        id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=1.5, status="passed",
    ))
    session.commit()
    session.close()

    client = FakeClient()
    client.queue_post(FakeResponse(201))
    client.queue_patch(FakeResponse(200, json_data={"id": "WQ1"}))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1))

    result = mgr.submit("WQ1")
    assert result.status == "submitted"

    patch_calls = [c for c in client.calls if c[0] == "PATCH"]
    assert len(patch_calls) == 1
    payload = patch_calls[0][2]["json"]
    assert payload["tags"] == ["PowerPoolSelected"]
    assert "Idea:" in payload["regular"]["description"]


def test_submit_khong_gan_tag_khi_khong_du_dieu_kien_power_pool():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    session.add(AlphaModel(id="a1", expression="rank(close)", source="ga"))
    session.add(SimulationModel(
        id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=0.5, status="passed",  # Sharpe < 1.0 -> không đạt Power Pool
    ))
    session.commit()
    session.close()

    client = FakeClient()
    client.queue_post(FakeResponse(201))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1))

    result = mgr.submit("WQ1")
    assert result.status == "submitted"
    assert not any(c[0] == "PATCH" for c in client.calls)


def test_submit_khong_gan_tag_khi_thieu_hypothesis():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    session.add(AlphaModel(id="a1", expression="rank(close)", source="ga"))  # không có hypothesis
    session.add(SimulationModel(
        id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=1.5, status="passed",
    ))
    session.commit()
    session.close()

    client = FakeClient()
    client.queue_post(FakeResponse(201))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1))

    result = mgr.submit("WQ1")
    assert result.status == "submitted"
    assert not any(c[0] == "PATCH" for c in client.calls)
