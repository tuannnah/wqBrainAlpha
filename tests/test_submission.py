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
def _queue_submit_success(client):
    """Xếp sẵn 2 GET cho kịch bản submit thành công (Bug 1): poll /submit không FAIL nào,
    rồi GET /alphas/{id} xác nhận dateSubmitted -> submit() mới dám báo 'submitted'."""
    client.queue_get(FakeResponse(200, json_data={"is": {"checks": []}}, text="non-empty"))
    client.queue_get(FakeResponse(200, json_data={"dateSubmitted": "2026-07-14T00:00:00Z"}))


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


def test_select_candidates_loai_alpha_duoi_nguong_nop_that_du_status_passed():
    """Bug 2 (bằng chứng thật 2026-07-14): alpha KP9nwpEg Sharpe 1.41/fitness 0.99,
    `status='passed'` lúc sim (WQ tự PASS `is.checks` lúc đó) nhưng KHÔNG đạt ngưỡng NỘP
    THẬT (Sharpe>=1.58, fitness>=1.0) -> select_candidates() KHÔNG được chọn, tránh tốn
    quota nộp cho alpha chắc chắn bị 403 REJECTED."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    try:
        session.add(AlphaModel(id="kp9", expression="rank(close)", source="ga"))
        session.add(
            SimulationModel(
                id="s_kp9", alpha_id="kp9", wq_alpha_id="KP9nwpEg", region="USA",
                universe="TOP3000", sharpe=1.41, fitness=0.99, score=0.99, status="passed",
            )
        )
        session.commit()
    finally:
        session.close()

    mgr = SubmissionManager(FakeClient(), sf, FakeCorr())
    ids = [c.wq_alpha_id for c in mgr.select_candidates()]
    assert "KP9nwpEg" not in ids


def test_run_daily_khong_chon_alpha_duoi_nguong_nop_that():
    """Cùng bằng chứng KP9nwpEg — run_daily() (dùng select_candidates() ở khâu chọn) cũng
    không được chọn ứng viên dưới ngưỡng Sharpe/fitness nộp thật."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    try:
        session.add(AlphaModel(id="kp9", expression="rank(close)", source="ga"))
        session.add(
            SimulationModel(
                id="s_kp9", alpha_id="kp9", wq_alpha_id="KP9nwpEg", region="USA",
                universe="TOP3000", sharpe=1.41, fitness=0.99, score=0.99, status="passed",
            )
        )
        session.commit()
    finally:
        session.close()

    mgr = SubmissionManager(FakeClient(), sf, FakeCorr(value=0.1))
    selected = mgr.run_daily(dry_run=True)
    assert "KP9nwpEg" not in [c.wq_alpha_id for c in selected]


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
    # Bug 1: sau POST 201, manager phải POLL GET /alphas/{id}/submit rồi XÁC NHẬN thêm
    # bằng GET /alphas/{id} (dateSubmitted) trước khi dám báo "submitted".
    _queue_submit_success(client)
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1))
    result = mgr.submit("WQ1")
    assert result.status == "submitted"


# ------------------------------------------------------------- Bug 1: poll bất đồng bộ
# Bằng chứng thật 2026-07-14 (alpha KP9nwpEg): POST /alphas/{id}/submit trả 200 NGAY
# nhưng đó KHÔNG phải kết quả — WQ tính bất đồng bộ, phải poll GET cùng path (body rỗng +
# Retry-After trong lúc tính, 403 kèm is.checks khi bị từ chối thật).


def test_submit_poll_tra_ve_403_sau_2_lan_body_rong_thi_rejected():
    """(a) GET /submit trả 200 body rỗng 2 lần (đang tính) rồi 403 với is.checks liệt kê
    LOW_SHARPE/LOW_FITNESS -> rejected, detail liệt kê đúng check FAIL kiểu 'LOW_SHARPE 1.41<1.58'."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_post(FakeResponse(200))
    client.queue_get(FakeResponse(200, text="", headers={"Retry-After": "1"}))
    client.queue_get(FakeResponse(200, text="", headers={"Retry-After": "1"}))
    client.queue_get(FakeResponse(
        403,
        json_data={"is": {"checks": [
            {"name": "LOW_SHARPE", "result": "FAIL", "limit": 1.58, "value": 1.41},
            {"name": "LOW_FITNESS", "result": "FAIL", "limit": 1.0, "value": 0.99},
            {"name": "LOW_TURNOVER", "result": "PASS", "limit": 0.01, "value": 0.2908},
        ]}},
        text="non-empty",
    ))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1), sleep_func=lambda _s: None)
    result = mgr.submit("WQ1")

    assert result.status == "rejected"
    assert "LOW_SHARPE 1.41<1.58" in result.detail
    assert "LOW_FITNESS 0.99<1.0" in result.detail
    assert "LOW_TURNOVER" not in result.detail  # check PASS -> không liệt

    session = sf()
    try:
        sub = session.query(SubmissionModel).filter_by(alpha_id="WQ1").one()
        assert sub.status == "rejected"
        assert "LOW_SHARPE" in sub.detail
    finally:
        session.close()


def test_submit_poll_xong_roi_xac_nhan_dateSubmitted_thi_submitted():
    """(b) Poll /submit trả 200 kèm is.checks không FAIL nào -> chưa vội tin, phải xác nhận
    thêm GET /alphas/{id} thấy dateSubmitted mới báo submitted."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_post(FakeResponse(200))
    client.queue_get(FakeResponse(200, json_data={"is": {"checks": [
        {"name": "LOW_SHARPE", "result": "PASS", "limit": 1.58, "value": 2.0},
    ]}}, text="non-empty"))
    client.queue_get(FakeResponse(200, json_data={"dateSubmitted": "2026-07-14T00:00:00Z", "stage": "OS"}))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1), sleep_func=lambda _s: None)
    result = mgr.submit("WQ1")
    assert result.status == "submitted"


def test_submit_poll_khong_fail_nhung_chua_xac_nhan_duoc_thi_unknown():
    """2xx + is.checks không FAIL nào, nhưng GET /alphas/{id} lại KHÔNG có dateSubmitted/stage
    OS -> KHÔNG được đoán bừa là submitted, phải trả 'unknown' kèm detail rõ."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_post(FakeResponse(200))
    client.queue_get(FakeResponse(200, json_data={"is": {"checks": []}}, text="non-empty"))
    client.queue_get(FakeResponse(200, json_data={"dateSubmitted": None, "stage": "IS"}))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1), sleep_func=lambda _s: None)
    result = mgr.submit("WQ1")
    assert result.status == "unknown"
    assert result.detail  # có lời giải thích, không rỗng


def test_submit_poll_qua_han_thi_pending():
    """(c) Poll mãi vẫn body rỗng, tới khi vượt SUBMIT_POLL_TIMEOUT -> pending (KHÔNG ghi
    submitted khi chưa biết kết quả thật)."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_post(FakeResponse(200))
    client.queue_get(FakeResponse(200, text="", headers={"Retry-After": "1"}))
    # time_func: lần 1 set deadline (0 + TIMEOUT), lần 2 (sau vòng poll đầu) đã vượt hạn.
    times = iter([0.0, 10_000.0])
    mgr = SubmissionManager(
        client, sf, FakeCorr(value=0.1), sleep_func=lambda _s: None, time_func=lambda: next(times),
    )
    result = mgr.submit("WQ1")
    assert result.status == "pending"

    session = sf()
    try:
        sub = session.query(SubmissionModel).filter_by(alpha_id="WQ1").one()
        assert sub.status == "pending"  # không được ghi 'submitted' khi chưa xác nhận
    finally:
        session.close()


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
    _queue_submit_success(client)
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
    _queue_submit_success(client)
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
    _queue_submit_success(client)
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
    _queue_submit_success(client)
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1))

    result = mgr.submit("WQ1")
    assert result.status == "submitted"
    assert not any(c[0] == "PATCH" for c in client.calls)


def test_max_self_correlation_poll_body_rong_roi_lay_duoc():
    """WQ trả 200 body rỗng + Retry-After trong lúc tính -> poll tới khi có JSON.
    Trước fix: resp.json() trên body rỗng -> JSONDecodeError làm sập submit."""
    client = FakeClient()
    client.queue_get(FakeResponse(200, text="", headers={"Retry-After": "1"}))
    client.queue_get(FakeResponse(200, text="", headers={"Retry-After": "1"}))
    client.queue_get(FakeResponse(200, json_data={"max": 0.48}, text='{"max":0.48}'))
    checker = CorrelationChecker(client)
    val = checker.max_self_correlation("WQ1", sleep_fn=lambda _s: None)
    assert val == 0.48
    assert sum(1 for c in client.calls if c[0] == "GET") == 3  # đã poll 3 lần


def test_max_self_correlation_poll_qua_han_coi_rui_ro_cao():
    """Tính mãi không xong (luôn rỗng) -> sau max_polls trả 1.0 (an toàn: coi corr cao)."""
    client = FakeClient()
    for _ in range(5):
        client.queue_get(FakeResponse(200, text="", headers={"Retry-After": "1"}))
    checker = CorrelationChecker(client)
    val = checker.max_self_correlation("WQ1", max_polls=5, sleep_fn=lambda _s: None)
    assert val == 1.0
