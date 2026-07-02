"""Test Simulator end-to-end với FakeClient."""

from __future__ import annotations

import pytest

from src.simulation.rate_limiter import RateLimiter
from src.simulation.simulator import (
    AuthExpiredError,
    QuotaExceededError,
    Simulator,
    extract_event_fields,
    extract_invalid_field,
    extract_rejected_field,
)
from tests.fakes import FakeClient, FakeResponse


def _no_sleep_limiter() -> RateLimiter:
    return RateLimiter(min_delay=0, sleep_func=lambda *_: None, time_func=lambda: 0.0)


def test_simulate_luong_day_du():
    client = FakeClient()
    # POST /simulations → 201 + Location
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-1"}))
    # Poll lần 1: chưa xong; lần 2: COMPLETE kèm alpha id
    client.queue_get(FakeResponse(200, json_data={"status": "RUNNING"}, headers={"Retry-After": "0"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-9"}))
    # GET /alphas/alpha-9 → metrics
    client.queue_get(
        FakeResponse(
            200,
            json_data={
                "is": {
                    "sharpe": 1.8,
                    "fitness": 1.3,
                    "turnover": 0.25,
                    "returns": 0.12,
                    "drawdown": 0.08,
                    "margin": 0.0015,
                    "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
                }
            },
        )
    )

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")

    assert result.status == "passed"
    assert result.alpha_id == "alpha-9"
    assert result.sharpe == 1.8
    assert result.fitness == 1.3
    assert result.metrics()["turnover"] == 0.25


def test_simulate_failed_khi_check_fail():
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-2"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-x"}))
    client.queue_get(
        FakeResponse(
            200,
            json_data={"is": {"sharpe": 0.2, "checks": [{"name": "LOW_SHARPE", "result": "FAIL"}]}},
        )
    )

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")
    assert result.status == "failed"


def test_simulate_error_giu_message_giai_thich():
    """status=ERROR kèm message → message phải được nêu trong raw['error'] để chẩn đoán."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-3"}))
    client.queue_get(
        FakeResponse(
            200,
            json_data={
                "status": "ERROR",
                "message": "Datafield 'foo_bar' is not supported in this region.",
            },
        )
    )

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(foo_bar)")

    assert result.status == "error"
    assert "foo_bar" in result.raw["error"]
    assert "ERROR" in result.raw["error"]


def test_extract_invalid_field():
    """Trích đúng field id từ thông điệp lỗi WQ; trả None với lỗi khác."""
    msg = (
        "status=ERROR: Invalid data field mdl77_2gdna_cfroi. "
        "<linkToCommonErrorMessages>Learn more</linkToCommonErrorMessages>"
    )
    assert extract_invalid_field(msg) == "mdl77_2gdna_cfroi"
    assert extract_invalid_field("status=ERROR: Invalid number of inputs : 3") is None
    assert extract_invalid_field("") is None


def test_simulate_goi_callback_khi_invalid_field():
    """WQ báo Invalid data field -> simulator gọi on_invalid_field với đúng field id."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-4"}))
    client.queue_get(
        FakeResponse(
            200,
            json_data={
                "status": "ERROR",
                "message": "Invalid data field mdl77_2gdna_cfroi. Learn more",
            },
        )
    )
    recorded: list[str] = []
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None,
        time_func=lambda: 0.0, on_invalid_field=recorded.append,
    )
    result = sim.simulate("zscore(mdl77_2gdna_cfroi)")
    assert result.status == "error"
    assert recorded == ["mdl77_2gdna_cfroi"]


def test_simulate_khong_goi_callback_voi_loi_khac():
    """Lỗi không phải invalid-field -> không gọi callback."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-5"}))
    client.queue_get(
        FakeResponse(
            200,
            json_data={"status": "ERROR", "message": "Invalid number of inputs : 3"},
        )
    )
    recorded: list[str] = []
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None,
        time_func=lambda: 0.0, on_invalid_field=recorded.append,
    )
    sim.simulate("rank(a, b, c)")
    assert recorded == []


def test_extract_event_fields():
    """Trích field 'event' là input trực tiếp của operator bị WQ báo lỗi event."""
    err = "status=ERROR: Operator ts_zscore does not support event inputs. Learn more"
    expr = "divide(ts_zscore(aggregate_option_open_interest_2, 20), abs_avg_pct_move_announcements_12)"
    assert extract_event_fields(err, expr) == ["aggregate_option_open_interest_2"]

    err2 = "Operator subtract does not support event inputs."
    expr2 = "subtract(actual_cashflow_per_share_value_quarterly, actual_eps_value)"
    assert set(extract_event_fields(err2, expr2)) == {
        "actual_cashflow_per_share_value_quarterly",
        "actual_eps_value",
    }
    # số literal không tính; lỗi khác -> rỗng
    assert extract_event_fields("Operator add does not support event inputs.", "add(x, 0.001)") == ["x"]
    assert extract_event_fields("Invalid number of inputs : 3", "rank(a, b, c)") == []


def test_extract_event_fields_bo_qua_ten_group():
    """Arg group (sector/industry/...) của group_* KHÔNG phải data field -> không
    bị trích làm field chết (gốc rễ 'sector' lọt blacklist sai)."""
    err = "Operator group_neutralize does not support event inputs."
    fields = extract_event_fields(err, "group_neutralize(nws18_bee, sector)")
    assert "sector" not in fields
    assert fields == ["nws18_bee"]


def test_simulate_blacklist_event_field():
    """WQ báo 'does not support event inputs' -> blacklist field event qua callback."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-6"}))
    client.queue_get(
        FakeResponse(
            200,
            json_data={
                "status": "ERROR",
                "message": "Operator add does not support event inputs. Learn more",
            },
        )
    )
    recorded: list[str] = []
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None,
        time_func=lambda: 0.0, on_invalid_field=recorded.append,
    )
    sim.simulate("add(composite_sentiment_score_2, 0.001)")
    assert recorded == ["composite_sentiment_score_2"]


def test_simulate_raise_khi_auth_loi_lien_tiep():
    """401 lặp lại (re-auth thất bại) -> raise AuthExpiredError để dừng, khỏi phí quota."""
    client = FakeClient()
    for _ in range(3):
        client.queue_post(
            FakeResponse(401, text='{"detail":"Incorrect authentication credentials."}')
        )
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    assert sim.simulate("rank(close)").status == "error"  # lần 1
    assert sim.simulate("rank(close)").status == "error"  # lần 2
    with pytest.raises(AuthExpiredError):
        sim.simulate("rank(close)")  # lần 3 -> dừng


def test_simulate_raise_quota_exceeded_khi_429_dai_dang():
    """POST /simulations vẫn 429 (WQBrainClient đã tự retry theo Retry-After nội bộ nhưng vẫn
    bị chặn) -> hết quota ngày, KHÔNG phải lỗi auth -> raise QuotaExceededError riêng để
    ClosedLoop dừng gọn thay vì cứ coi là sim lỗi rồi thử ý tưởng khác."""
    client = FakeClient()
    client.queue_post(FakeResponse(429, text="Too Many Requests", headers={"Retry-After": "5"}))
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    with pytest.raises(QuotaExceededError):
        sim.simulate("rank(close)")


def test_simulate_raise_quota_exceeded_khi_ratelimit_remaining_ve_0():
    """Header X-Ratelimit-Remaining=0 trên response lỗi -> hết quota simulation ngày (theo tài
    liệu WQ: 'X-Ratelimit-Remaining: Remaining simulations for the day')."""
    client = FakeClient()
    client.queue_post(
        FakeResponse(400, text="bad request", headers={"X-Ratelimit-Remaining": "0"})
    )
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    with pytest.raises(QuotaExceededError):
        sim.simulate("rank(close)")


def test_simulate_khong_raise_quota_khi_con_remaining():
    """X-Ratelimit-Remaining > 0 -> KHÔNG phải hết quota, xử lý như lỗi thường (không raise)."""
    client = FakeClient()
    client.queue_post(
        FakeResponse(400, text="bad request", headers={"X-Ratelimit-Remaining": "42"})
    )
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    assert sim.simulate("rank(close)").status == "error"


def test_auth_counter_reset_khi_loi_khac():
    """Lỗi không phải auth xen giữa -> reset bộ đếm, không tích lũy tới ngưỡng."""
    client = FakeClient()
    client.queue_post(FakeResponse(401, text="Incorrect authentication credentials."))
    client.queue_post(FakeResponse(401, text="Incorrect authentication credentials."))
    client.queue_post(FakeResponse(500, text="server error"))  # reset
    client.queue_post(FakeResponse(401, text="Incorrect authentication credentials."))
    client.queue_post(FakeResponse(401, text="Incorrect authentication credentials."))
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    for _ in range(5):
        assert sim.simulate("rank(close)").status == "error"  # không raise


def test_extract_rejected_field():
    """Trích field id từ reason tiền-kiểm 'Field/hằng không tồn tại: X'; None nếu khác."""
    assert extract_rejected_field("Field/hằng không tồn tại: foo_bar") == "foo_bar"
    assert extract_rejected_field("Độ sâu > 6") is None
    assert extract_rejected_field("") is None


def test_pre_sim_validator_chan_truoc_khi_goi_api():
    """validator trả (False, reason) -> KHÔNG gọi API, trả error, ghi field qua callback."""
    client = FakeClient()  # không queue post nào: nếu gọi post sẽ IndexError
    recorded: list[str] = []
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0,
        on_invalid_field=recorded.append,
        pre_sim_validator=lambda e: (False, "Field/hằng không tồn tại: foo_bar"),
    )
    result = sim.simulate("rank(foo_bar)")
    assert all(c[0] != "POST" for c in client.calls)  # API không bị gọi
    assert result.status == "error"
    assert "pre-sim reject" in result.raw["error"]
    assert recorded == ["foo_bar"]


def test_pre_sim_validator_ok_thi_van_goi_api():
    """validator pass -> đi tiếp tới POST như cũ."""
    client = FakeClient()
    client.queue_post(FakeResponse(500, text="server error"))  # lỗi để dừng sớm sau POST
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0,
        pre_sim_validator=lambda e: (True, "ok"),
    )
    sim.simulate("rank(close)")
    assert any(c[0] == "POST" for c in client.calls)


def test_field_chet_giua_phien_bi_loai_ngay():
    """Recorder loại field khỏi known_fields -> lần sau cùng phiên bị tiền-kiểm chặn."""
    from src.simulation.pre_filter import PreFilter

    pf = PreFilter(known_fields={"close", "dead_fld"}, known_operators={"rank"})

    def recorder(field_id):
        pf.known_fields.discard(field_id)

    # trước khi loại: dead_fld còn hợp lệ
    assert pf.check("rank(dead_fld)")[0] is True
    recorder("dead_fld")  # mô phỏng WQ báo dead_fld chết giữa phiên
    # sau khi loại: bị tiền-kiểm chặn, không tốn sim
    assert pf.check("rank(dead_fld)")[0] is False


def test_simulate_thieu_location_tra_error():
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={}))
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")
    assert result.status == "error"
