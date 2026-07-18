"""Test đường nộp pure Power Pool (SubmissionManager.select_power_pool_candidates /
submit_power_pool).

Bằng chứng nền (2026-07-15, alpha KP92dQAx): alpha Sharpe 1.71 chỉ FAIL LOW_FITNESS +
LOW_2Y_SHARPE (không đạt Regular) nhưng đạt CẤU TRÚC Power Pool, khớp theme
"USA/D1 Power Pool July`26 2" (MATCHES_THEMES PASS sau khi set description) — docs
power-pool-alphas.md: pure Power Pool nộp được nếu khớp theme + mô tả >=100 ký tự."""

from __future__ import annotations

import json
from datetime import date

from sqlalchemy import create_engine

from src.scoring.power_pool_theme import PowerPoolThemeWeek
from src.storage.db import init_db, make_session_factory
from src.storage.models import AlphaModel, DataFieldModel, SimulationModel, SubmissionModel
from src.submission.manager import SubmissionManager
from tests.fakes import FakeClient, FakeResponse

ON_DATE = date(2026, 7, 15)
# Lịch theme cố định cho test (không phụ thuộc ngày chạy test): USA/D1/TOP1000, loại pv1.
CALENDAR = [
    PowerPoolThemeWeek(
        date(2026, 7, 12), date(2026, 7, 26),
        name="USA/D1 Power Pool July`26 2",
        region="USA", delay=1, universe="TOP1000", datasets_excluded=("pv1",),
    )
]

HYPO = json.dumps({
    "observation": "Tỷ lệ khối lượng bán khống trên tổng khối lượng giảm dần trong một tháng.",
    "background": "Short covering giảm áp lực bên bán, giá thường phục hồi sau đó.",
    "economic_rationale": "Vị thế bán khống đóng lại tạo lực mua ròng, đẩy giá lên.",
    "implementation_spec": "divide(reported_short_sale_share_quantity, reported_total_trade_share_quantity) rồi ts_delta 22 phiên.",
})


def _engine():
    return create_engine(
        "sqlite:///:memory:", future=True, connect_args={"check_same_thread": False}
    )


def _settings(universe="TOP1000", delay=1, neutralization="SUBINDUSTRY"):
    return json.dumps({"settings": {
        "region": "USA", "universe": universe, "delay": delay,
        "neutralization": neutralization, "decay": 4,
    }})


def _add_sim(session, alpha_id, wq_id, expr, sharpe, fitness, failed_checks,
             raw_result=None, hypothesis=None, status="failed"):
    session.add(AlphaModel(id=alpha_id, expression=expr, source="gp", hypothesis=hypothesis))
    session.add(SimulationModel(
        id="s_" + alpha_id, alpha_id=alpha_id, wq_alpha_id=wq_id, region="USA",
        universe="TOP1000", sharpe=sharpe, fitness=fitness, score=sharpe, status=status,
        failed_checks=json.dumps(failed_checks),
        raw_result=raw_result or _settings(),
    ))


def _seed_fields(session):
    """Map field -> dataset để chấm datasets_excluded của theme."""
    for fid, ds in [
        ("reported_short_sale_share_quantity", "us_short_sale"),
        ("reported_total_trade_share_quantity", "us_short_sale"),
        ("close", "pv1"),
        ("vwap", "pv1"),
    ]:
        session.add(DataFieldModel(id=fid, region="USA", universe="TOP1000", delay=1, dataset_id=ds))


PP_EXPR = ("multiply(-1, ts_delta(divide(reported_short_sale_share_quantity, "
           "add(1, reported_total_trade_share_quantity)), 22))")


class FakeCorr:
    def __init__(self, value=0.1, max_self_corr=0.7):
        self.value = value
        self.max_self_corr = max_self_corr

    def max_self_correlation(self, _):
        return self.value

    def is_acceptable(self, _):
        return self.value <= self.max_self_corr


def _mgr(sf, client=None):
    return SubmissionManager(
        client or FakeClient(), sf, FakeCorr(value=0.1), sleep_func=lambda _s: None
    )


def test_select_power_pool_chon_dung_pure_pp_va_loai_regular_yeu_cau_khac():
    """Chọn alpha KHÔNG đạt Regular nhưng đạt cấu trúc PP + chỉ fail check Regular-only;
    loại: alpha đạt Regular (đi run_daily), Sharpe<1.0, fail check ngoài tập Regular-only."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    try:
        _seed_fields(session)
        # pure PP đúng nghĩa (mẫu KP92dQAx)
        _add_sim(session, "a1", "PP1", PP_EXPR, 1.71, 0.61,
                 ["LOW_FITNESS", "LOW_2Y_SHARPE"], hypothesis=HYPO, status="passed")
        # đạt Regular -> KHÔNG thuộc đường pure PP
        _add_sim(session, "a2", "REG1", PP_EXPR.replace("22", "44"), 1.7, 1.2, [], status="passed")
        # Sharpe dưới 1.0 -> loại
        _add_sim(session, "a3", "PP2", PP_EXPR.replace("22", "10"), 0.9, 0.5, ["LOW_SHARPE"])
        # fail HIGH_TURNOVER — PP cũng đòi turnover PASS -> loại
        _add_sim(session, "a4", "PP3", PP_EXPR.replace("22", "5"), 1.3, 0.5,
                 ["LOW_SHARPE", "HIGH_TURNOVER"])
        session.commit()
    finally:
        session.close()

    cands = _mgr(sf).select_power_pool_candidates(on_date=ON_DATE, calendar=CALENDAR)
    ids = [c.wq_alpha_id for c in cands]
    assert ids == ["PP1"]
    assert cands[0].theme_ok is True
    assert cands[0].description  # dựng được từ hypothesis 4 phần


def test_select_power_pool_theme_loai_dataset_pv1_va_universe_lech():
    """Theme USA/D1/TOP1000 loại pv1: alpha dùng close/vwap (pv1) hoặc sim TOP3000 phải có
    theme_ok=False kèm lý do — không âm thầm coi là nộp được."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    try:
        _seed_fields(session)
        _add_sim(session, "a1", "PV", "multiply(-1, ts_mean(subtract(close, vwap), 10))",
                 1.6, 0.8, ["LOW_FITNESS"], hypothesis=HYPO)
        _add_sim(session, "a2", "U3K", PP_EXPR, 1.71, 0.61, ["LOW_FITNESS"],
                 raw_result=_settings(universe="TOP3000"), hypothesis=HYPO)
        session.commit()
    finally:
        session.close()

    cands = _mgr(sf).select_power_pool_candidates(on_date=ON_DATE, calendar=CALENDAR)
    by_id = {c.wq_alpha_id: c for c in cands}
    assert by_id["PV"].theme_ok is False
    assert any("pv1" in r for r in by_id["PV"].theme_reasons)
    assert by_id["U3K"].theme_ok is False
    assert any("TOP1000" in r for r in by_id["U3K"].theme_reasons)


def test_submit_power_pool_dry_run_khong_goi_api():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    try:
        _seed_fields(session)
        _add_sim(session, "a1", "PP1", PP_EXPR, 1.71, 0.61, ["LOW_FITNESS"],
                 hypothesis=HYPO, status="passed")
        session.commit()
    finally:
        session.close()

    client = FakeClient()
    outcomes = _mgr(sf, client).submit_power_pool(
        dry_run=True, on_date=ON_DATE, calendar=CALENDAR
    )
    assert len(outcomes) == 1
    cand, result = outcomes[0]
    assert cand.wq_alpha_id == "PP1" and result is None
    assert client.calls == []  # dry-run tuyệt đối không chạm API


def test_submit_power_pool_nop_that_set_desc_truoc_roi_submit():
    """Nộp thật: PATCH description (bắt buộc >=100 ký tự theo docs) TRƯỚC, rồi POST /submit
    (dùng lại submit() sẵn có — poll + xác nhận dateSubmitted). Quota mặc định 1 pure PP/lần."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    try:
        _seed_fields(session)
        _add_sim(session, "a1", "PP1", PP_EXPR, 1.71, 0.61, ["LOW_FITNESS"],
                 hypothesis=HYPO, status="passed")
        # ứng viên thứ 2 hợp lệ nhưng ngoài quota 1/ngày
        _add_sim(session, "a2", "PP9", PP_EXPR.replace("22", "44"), 1.5, 0.6,
                 ["LOW_FITNESS"], hypothesis=HYPO)
        session.commit()
    finally:
        session.close()

    client = FakeClient()
    client.queue_patch(FakeResponse(200))  # set_properties description
    client.queue_post(FakeResponse(201))  # POST /submit
    client.queue_get(FakeResponse(200, json_data={"is": {"checks": []}}, text="non-empty"))
    client.queue_get(FakeResponse(200, json_data={"dateSubmitted": "2026-07-15T00:00:00Z"}))

    outcomes = _mgr(sf, client).submit_power_pool(
        dry_run=False, on_date=ON_DATE, calendar=CALENDAR
    )
    submitted = [(c, r) for c, r in outcomes if r is not None]
    assert len(submitted) == 1
    assert submitted[0][0].wq_alpha_id == "PP1"  # top Sharpe trước
    assert submitted[0][1].status == "submitted"
    patch_calls = [c for c in client.calls if c[0] == "PATCH"]
    assert patch_calls and "PP1" in patch_calls[0][1]

    session = sf()
    try:
        assert session.query(SubmissionModel).filter_by(
            alpha_id="PP1", status="submitted"
        ).count() == 1
    finally:
        session.close()


def test_select_power_pool_fallback_description_tu_bang_submissions():
    """Alpha không có hypothesis nhưng đã từng set description qua set_properties (bảng
    submissions.regular_desc, >=100 ký tự — trường hợp KP92dQAx 2026-07-15 set tay qua API)
    -> dùng lại bản đó, không skip oan."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    desc = "Idea: " + "x" * 120
    session = sf()
    try:
        _seed_fields(session)
        _add_sim(session, "a1", "PP1", PP_EXPR, 1.71, 0.61, ["LOW_FITNESS"], hypothesis=None)
        session.add(SubmissionModel(
            id="prop1", alpha_id="PP1", status="properties_set", regular_desc=desc,
        ))
        session.commit()
    finally:
        session.close()

    cands = _mgr(sf).select_power_pool_candidates(on_date=ON_DATE, calendar=CALENDAR)
    assert len(cands) == 1
    assert cands[0].description == desc
    assert cands[0].skip_reason == ""


def test_submit_power_pool_thieu_hypothesis_khong_nop_va_neu_ro_ly_do():
    """Không có hypothesis VÀ field ngoài kho frontier (không có hypothesis category để
    fallback) -> không dựng được mô tả Idea/Rationale (bắt buộc theo docs) -> KHÔNG nộp,
    skip_reason nêu rõ; không âm thầm nộp alpha thiếu mô tả."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    # Field KHÔNG thuộc FRONTIER_CATEGORY_BY_FIELD và dataset không bị theme loại trừ.
    expr = "multiply(-1, ts_delta(some_niche_field, 22))"
    session = sf()
    try:
        _seed_fields(session)
        session.add(DataFieldModel(
            id="some_niche_field", region="USA", universe="TOP1000", delay=1,
            dataset_id="niche42",
        ))
        _add_sim(session, "a1", "PP1", expr, 1.71, 0.61, ["LOW_FITNESS"], hypothesis=None)
        session.commit()
    finally:
        session.close()

    client = FakeClient()
    outcomes = _mgr(sf, client).submit_power_pool(
        dry_run=False, on_date=ON_DATE, calendar=CALENDAR
    )
    assert len(outcomes) == 1
    cand, result = outcomes[0]
    assert result is None
    assert "mô tả" in cand.skip_reason
    assert client.calls == []


def test_select_power_pool_fallback_mo_ta_tu_frontier_hypothesis():
    """Alpha dùng field frontier nhưng hypothesis rỗng '{}' (thực tế DB 2026-07-18:
    LLdLVX0a — mọi alpha đường sim-thẳng/near-miss đều '{}') -> tự dựng mô tả từ
    FRONTIER_HYPOTHESES của category, hết skip oan "thiếu mô tả"."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    expr = "ts_rank(multiply(-1, ts_mean(firm_vol_imbalance, 5)), 66)"
    session = sf()
    try:
        _seed_fields(session)
        session.add(DataFieldModel(
            id="firm_vol_imbalance", region="USA", universe="TOP1000", delay=1,
            dataset_id="order_flow_imb",
        ))
        _add_sim(session, "a1", "OFI1", expr, 1.08, 0.33, [], hypothesis="{}")
        session.commit()
    finally:
        session.close()

    cands = _mgr(sf).select_power_pool_candidates(on_date=ON_DATE, calendar=CALENDAR)
    assert len(cands) == 1
    assert cands[0].description is not None
    assert cands[0].description.startswith("Idea: ")
    assert cands[0].skip_reason == ""
