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


def test_simulate_luu_ten_check_bi_fail():
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-3"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-y"}))
    client.queue_get(
        FakeResponse(
            200,
            json_data={
                "is": {
                    "sharpe": 0.2,
                    "checks": [
                        {"name": "LOW_SHARPE", "result": "FAIL"},
                        {"name": "LOW_FITNESS", "result": "FAIL"},
                        {"name": "LOW_TURNOVER", "result": "PASS"},
                    ],
                }
            },
        )
    )

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")
    assert result.failed_checks == ["LOW_SHARPE", "LOW_FITNESS"]


def test_simulate_failed_checks_rong_khi_toan_bo_pass():
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-4"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-z"}))
    client.queue_get(
        FakeResponse(200, json_data={"is": {"sharpe": 1.8, "checks": [{"name": "LOW_SHARPE", "result": "PASS"}]}})
    )

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")
    assert result.failed_checks == []


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


def test_pre_sim_validator_chan_gan_presim_reason():
    """Task 3 (spec C2): reject tiền-kiểm phải mang `presim_reason` = reason gốc để caller
    phân biệt 'chưa chạm Brain' khỏi 1 sim thật rớt (status='error' dùng chung cho cả 2 trước
    đây làm mất thông tin này)."""
    client = FakeClient()
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0,
        pre_sim_validator=lambda e: (False, "Operator không tồn tại: fake_op"),
    )
    result = sim.simulate("fake_op(close)")
    assert result.status == "error"
    assert result.presim_reason == "Operator không tồn tại: fake_op"


def test_sim_that_khong_gan_presim_reason():
    """Sim thật đi qua API (không bị tiền-kiểm chặn) -> presim_reason PHẢI None, kể cả khi
    sim đó rớt (status='error' vì lỗi Brain thật) — tránh lẫn với pre-sim reject."""
    client = FakeClient()
    client.queue_post(FakeResponse(500, text="server error"))
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0,
        pre_sim_validator=lambda e: (True, "ok"),
    )
    result = sim.simulate("rank(close)")
    assert result.status == "error"
    assert result.presim_reason is None


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


def test_simulate_tra_error_khi_poll_qua_deadline():
    """Poll mãi RUNNING và thời gian vượt TIMEOUT_SECONDS -> _poll raise 'timeout khi
    poll', simulate nuốt lỗi và trả status='error' (không treo vô hạn)."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-x"}))
    client.queue_get(
        FakeResponse(200, json_data={"status": "RUNNING"}, headers={"Retry-After": "0"})
    )
    # time_func: lần đầu (tính deadline) = 0; lần sau vượt -> chạm deadline ngay.
    times = iter([0.0] + [10.0**9] * 8)
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None,
        time_func=lambda: next(times),
    )
    result = sim.simulate("rank(close)")
    assert result.status == "error"
    assert "timeout" in str(result.raw).lower()


def test_timeout_seconds_du_dai_cho_wq_cham():
    """TIMEOUT_SECONDS phải >= 600s: WQ Brain đôi lúc xử lý sim > 5 phút, deadline 300s
    cũ khiến mọi sim bị bỏ oan trước khi COMPLETE."""
    assert Simulator.TIMEOUT_SECONDS >= 600.0


def test_simulate_chu_dong_sleep_khi_ratelimit_remaining_ve_0():
    """POST /simulations thành công nhưng X-Ratelimit-Remaining=0 -> CHỦ ĐỘNG sleep theo
    X-Ratelimit-Reset TRƯỚC khi bị 429 (phòng ngừa, khác _is_quota_exhausted xử lý SAU khi
    đã bị chặn)."""
    client = FakeClient()
    client.queue_post(
        FakeResponse(
            201,
            headers={
                "Location": "/simulations/sim-7",
                "X-Ratelimit-Remaining": "0",
                "X-Ratelimit-Reset": "5",
            },
        )
    )
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-9"}))
    client.queue_get(FakeResponse(200, json_data={"is": {"sharpe": 1.0, "checks": []}}))

    sleeps: list[float] = []
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=sleeps.append, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")

    assert sleeps == [5.0]  # đúng 1 lần sleep chủ động, không lẫn sleep của poll (COMPLETE ngay)
    assert result.status == "passed"


def test_simulate_khong_sleep_chu_dong_khi_con_remaining():
    """X-Ratelimit-Remaining > 0 -> KHÔNG sleep chủ động (chỉ phòng ngừa khi thật sự cạn)."""
    client = FakeClient()
    client.queue_post(
        FakeResponse(
            201,
            headers={"Location": "/simulations/sim-8", "X-Ratelimit-Remaining": "10"},
        )
    )
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-10"}))
    client.queue_get(FakeResponse(200, json_data={"is": {"sharpe": 1.0, "checks": []}}))

    sleeps: list[float] = []
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=sleeps.append, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")

    assert sleeps == []
    assert result.status == "passed"


def _metrics_response(sharpe, alpha_id):
    return FakeResponse(
        200,
        json_data={"is": {"sharpe": sharpe, "fitness": 1.0, "checks": []}},
    )


def test_simulate_many_post_1_lan_mang_3_payload_dung_thu_tu():
    """simulate_many với 3 job hợp lệ -> ĐÚNG 1 lần POST /simulations với body là MẢNG 3
    phần tử, poll cha trả children, GET từng con -> 3 SimulationResult đúng thứ tự."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/parent-1"}))
    # Poll cha: children xuất hiện ngay ở lần poll đầu.
    client.queue_get(
        FakeResponse(200, json_data={"status": "COMPLETE", "children": ["c1", "c2", "c3"]})
    )
    # Poll TẤT CẢ con trước (c1, c2, c3), rồi mới fetch metrics từng alpha theo thứ tự exprs.
    for alpha_id in ["alpha-c1", "alpha-c2", "alpha-c3"]:
        client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": alpha_id}))
    for sharpe, alpha_id in [(1.1, "alpha-c1"), (1.2, "alpha-c2"), (1.3, "alpha-c3")]:
        client.queue_get(_metrics_response(sharpe, alpha_id))

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    jobs = [("rank(a)", None), ("rank(b)", None), ("rank(c)", None)]
    results = sim.simulate_many(jobs)

    post_calls = [c for c in client.calls if c[0] == "POST"]
    assert len(post_calls) == 1
    assert isinstance(post_calls[0][2]["json"], list)
    assert len(post_calls[0][2]["json"]) == 3
    assert [r.expression for r in results] == ["rank(a)", "rank(b)", "rank(c)"]
    assert [r.alpha_id for r in results] == ["alpha-c1", "alpha-c2", "alpha-c3"]
    assert [r.sharpe for r in results] == [1.1, 1.2, 1.3]
    assert all(r.status == "passed" for r in results)


def test_simulate_many_presim_reject_khong_chiem_slot_payload():
    """1 job bị pre_sim_validator chặn -> KHÔNG vào mảng payload gửi Brain (chỉ 2 job còn lại
    được POST); job bị chặn trả presim_reason ngay tại đúng vị trí trong kết quả."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/parent-2"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "children": ["c1", "c2"]}))
    for alpha_id in ["alpha-c1", "alpha-c2"]:
        client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": alpha_id}))
    for sharpe, alpha_id in [(0.9, "alpha-c1"), (1.0, "alpha-c2")]:
        client.queue_get(_metrics_response(sharpe, alpha_id))

    def validator(expr):
        if expr == "rank(bad)":
            return False, "Field/hằng không tồn tại: bad"
        return True, "ok"

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None,
        time_func=lambda: 0.0, pre_sim_validator=validator,
    )
    jobs = [("rank(a)", None), ("rank(bad)", None), ("rank(c)", None)]
    results = sim.simulate_many(jobs)

    post_calls = [c for c in client.calls if c[0] == "POST"]
    assert len(post_calls) == 1
    assert len(post_calls[0][2]["json"]) == 2  # job giữa bị chặn -> KHÔNG chiếm slot
    assert results[1].presim_reason == "Field/hằng không tồn tại: bad"
    assert results[1].status == "error"
    # Thứ tự kết quả vẫn khớp jobs gốc (giữ đúng vị trí bị chặn).
    assert [r.expression for r in results] == ["rank(a)", "rank(bad)", "rank(c)"]
    assert results[0].alpha_id == "alpha-c1"
    assert results[2].alpha_id == "alpha-c2"


def test_simulate_many_429_raise_quota_exceeded_giong_duong_don():
    """POST /simulations (multi) vẫn 429 -> raise QuotaExceededError giống hệt đường đơn,
    KHÔNG fallback sim tuần tự (hết quota thì tuần tự cũng hết quota)."""
    client = FakeClient()
    client.queue_post(FakeResponse(429, text="Too Many Requests", headers={"Retry-After": "5"}))
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    jobs = [("rank(a)", None), ("rank(b)", None)]
    with pytest.raises(QuotaExceededError):
        sim.simulate_many(jobs)
    # KHÔNG có POST fallback tuần tự nào được thử thêm.
    assert len([c for c in client.calls if c[0] == "POST"]) == 1


def test_simulate_many_loi_khac_fallback_tuan_tu():
    """Multi-sim lỗi KHÔNG PHẢI quota/auth (vd thiếu Location header) -> fallback: sim TUẦN
    TỰ từng job qua đường đơn, không chết phiên, vẫn trả đủ kết quả đúng thứ tự."""
    client = FakeClient()
    # POST multi-sim: 201 nhưng THIẾU Location -> _post_multi_sim raise SimulationError.
    client.queue_post(FakeResponse(201, headers={}))
    # Fallback tuần tự: mỗi job đi lại đường đơn đầy đủ (POST -> poll -> fetch).
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/single-1"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-1"}))
    client.queue_get(_metrics_response(1.5, "alpha-1"))
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/single-2"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-2"}))
    client.queue_get(_metrics_response(1.6, "alpha-2"))

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    jobs = [("rank(a)", None), ("rank(b)", None)]
    results = sim.simulate_many(jobs)

    post_calls = [c for c in client.calls if c[0] == "POST"]
    assert len(post_calls) == 3  # 1 multi-sim hỏng + 2 sim đơn fallback
    assert [r.status for r in results] == ["passed", "passed"]
    assert [r.sharpe for r in results] == [1.5, 1.6]


def test_simulate_many_chunk_qua_10_thanh_nhieu_request():
    """11 job hợp lệ -> chia 2 chunk (10 + 1): chunk đầu multi-sim (mảng 10), chunk cuối dùng
    đường đơn (mảng 1 phần tử KHÔNG hợp lệ với API multi-sim theo tài liệu 2..10)."""
    client = FakeClient()
    # Chunk 1: 10 job -> multi-sim.
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/parent-big"}))
    children = [f"c{i}" for i in range(10)]
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "children": children}))
    for i, cid in enumerate(children):  # poll TẤT CẢ con trước...
        client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": f"alpha-{i}"}))
    for i in range(10):  # ...rồi fetch metrics theo thứ tự exprs
        client.queue_get(_metrics_response(1.0 + i * 0.01, f"alpha-{i}"))
    # Chunk 2: 1 job còn lại -> đường đơn.
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/single-11"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-10"}))
    client.queue_get(_metrics_response(2.0, "alpha-10"))

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    jobs = [(f"rank(f{i})", None) for i in range(11)]
    results = sim.simulate_many(jobs)

    post_calls = [c for c in client.calls if c[0] == "POST"]
    assert len(post_calls) == 2
    assert len(post_calls[0][2]["json"]) == 10  # chunk đầu: mảng 10 phần tử
    assert isinstance(post_calls[1][2]["json"], dict)  # chunk cuối: object đơn (không phải mảng)
    assert len(results) == 11
    assert results[-1].sharpe == 2.0


def test_simulate_many_children_dao_thu_tu_van_map_dung_theo_echo_regular():
    """Finding Important #2: tài liệu Brain KHÔNG đảm bảo thứ tự `children` == thứ tự payload
    (cả docs lẫn wqb-mcp đều chỉ GIẢ ĐỊNH). Response mỗi child echo lại request data (field
    `regular` = biểu thức gốc, brain-api.md:336) -> code phải ĐỐI CHIẾU echo với expr kỳ vọng:
    children bị đảo thứ tự -> vẫn map đúng kết quả về job gốc + log WARNING, không gán nhầm
    sharpe/alpha_id cho biểu thức khác một cách âm thầm."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/parent-swap"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "children": ["c1", "c2"]}))
    # Brain trả children ĐẢO: c1 thực ra là sim của rank(b), c2 của rank(a) — echo `regular` nói thật.
    client.queue_get(
        FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-b", "regular": "rank(b)"})
    )
    client.queue_get(
        FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-a", "regular": "rank(a)"})
    )
    # Fetch metrics theo thứ tự exprs SAU khi map: rank(a)->alpha-a trước, rank(b)->alpha-b sau.
    client.queue_get(_metrics_response(1.1, "alpha-a"))
    client.queue_get(_metrics_response(2.2, "alpha-b"))

    from loguru import logger as _logger

    warnings: list[str] = []
    sink_id = _logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    try:
        sim = Simulator(
            client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None,
            time_func=lambda: 0.0,
        )
        results = sim.simulate_many([("rank(a)", None), ("rank(b)", None)])
    finally:
        _logger.remove(sink_id)

    assert [r.expression for r in results] == ["rank(a)", "rank(b)"]
    assert [r.alpha_id for r in results] == ["alpha-a", "alpha-b"]  # KHÔNG bị tráo
    assert [r.sharpe for r in results] == [1.1, 2.2]
    assert any("thứ tự" in w for w in warnings)  # có cảnh báo thứ tự children lệch


def test_simulate_many_echo_lech_khong_match_duoc_thi_error_thay_vi_gan_bua():
    """Echo `regular` lệch mà KHÔNG đối chiếu được biểu thức nào -> job đó nhận status='error'
    thay vì gán bừa kết quả của job khác (thà mất 1 kết quả còn hơn ghi sai sharpe/alpha_id)."""
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/parent-bad"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "children": ["c1", "c2"]}))
    # CẢ HAI child đều echo 'rank(b)' (bất thường) -> rank(a) không có child nào khớp.
    client.queue_get(
        FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-b1", "regular": "rank(b)"})
    )
    client.queue_get(
        FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-b2", "regular": "rank(b)"})
    )
    # Chỉ rank(b) fetch metrics (match child ĐẦU TIÊN chưa dùng: alpha-b1).
    client.queue_get(_metrics_response(1.5, "alpha-b1"))

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    results = sim.simulate_many([("rank(a)", None), ("rank(b)", None)])

    assert results[0].expression == "rank(a)"
    assert results[0].status == "error"          # không gán bừa alpha-b2 cho rank(a)
    assert results[0].alpha_id is None
    assert results[1].alpha_id == "alpha-b1"
    assert results[1].sharpe == 1.5


def test_simulate_many_rong_tra_rong_khong_goi_api():
    """jobs rỗng -> trả [] ngay, KHÔNG gọi API nào."""
    client = FakeClient()
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    assert sim.simulate_many([]) == []
    assert client.calls == []


def test_simulate_cap_sleep_khi_reset_la_epoch_tuyet_doi():
    """AN TOÀN: nếu X-Ratelimit-Reset là epoch tuyệt đối (khổng lồ), sleep phải bị CAP ở
    MAX_RATE_LIMIT_SLEEP thay vì chờ ~vô tận."""
    client = FakeClient()
    client.queue_post(
        FakeResponse(
            201,
            headers={
                "Location": "/simulations/sim-9",
                "X-Ratelimit-Remaining": "0",
                "X-Ratelimit-Reset": "1720000000",  # epoch tuyệt đối (không phải giây tương đối)
            },
        )
    )
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-11"}))
    client.queue_get(FakeResponse(200, json_data={"is": {"sharpe": 1.0, "checks": []}}))

    sleeps: list[float] = []
    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=sleeps.append, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")

    assert sleeps == [Simulator.MAX_RATE_LIMIT_SLEEP]   # bị cap, không chờ 1.7 tỉ giây
    assert result.status == "passed"


# ----------------------- auto_tag: gắn tag alpha ngay sau sim (yêu cầu 2026-07-18) --------
# Người dùng cần lọc alpha do tool sinh trên web WQ Brain (tab Alphas, filter theo tag).


def _queue_sim_ok(client, alpha_id="alpha-9"):
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-1"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": alpha_id}))
    client.queue_get(FakeResponse(200, json_data={"is": {"sharpe": 1.0, "checks": []}}))


def test_auto_tag_patch_tags_sau_khi_sim_xong():
    client = FakeClient()
    _queue_sim_ok(client)
    client.queue_patch(FakeResponse(200))
    sim = Simulator(client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None,
                    time_func=lambda: 0.0, auto_tag="wqtool")
    result = sim.simulate("rank(close)")

    assert result.alpha_id == "alpha-9"
    patches = [c for c in client.calls if c[0] == "PATCH"]
    assert len(patches) == 1
    assert patches[0][1] == "/alphas/alpha-9"
    assert patches[0][2]["json"] == {"tags": ["wqtool"]}


def test_auto_tag_none_khong_patch():
    client = FakeClient()
    _queue_sim_ok(client)
    sim = Simulator(client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None,
                    time_func=lambda: 0.0)
    sim.simulate("rank(close)")
    assert [c for c in client.calls if c[0] == "PATCH"] == []


def test_auto_tag_patch_loi_khong_lam_hong_ket_qua_sim():
    class _PatchNoClient(FakeClient):
        def patch(self, path, **kwargs):
            self.calls.append(("PATCH", path, kwargs))
            raise RuntimeError("mạng đứt")

    client = _PatchNoClient()
    _queue_sim_ok(client)
    sim = Simulator(client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None,
                    time_func=lambda: 0.0, auto_tag="wqtool")
    result = sim.simulate("rank(close)")
    assert result.status == "passed"  # PATCH lỗi chỉ log warning, không phá sim
