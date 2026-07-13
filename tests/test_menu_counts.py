"""Menu hiện số fields/operators ngay sau đăng nhập để người dùng quyết định tải lại.
Kiểm `_menu_counts`: đếm fields theo đúng scope (region/universe/delay) + tổng operators."""

from __future__ import annotations

import main
from src.storage.db import init_db, make_engine, make_session_factory
from src.storage.models import DataFieldModel, OperatorModel


def _seed(url: str):
    sf = make_session_factory(init_db(make_engine(url)))
    s = sf()
    s.add(DataFieldModel(id="close", region="USA", universe="TOP3000", delay=1, type="MATRIX"))
    s.add(DataFieldModel(id="open", region="USA", universe="TOP3000", delay=1, type="MATRIX"))
    s.add(DataFieldModel(id="vwap_eur", region="EUR", universe="TOP3000", delay=1, type="MATRIX"))
    s.add(OperatorModel(name="rank", category="Cross Sectional", definition="rank(x)", arity=1))
    s.commit()
    s.close()
    return sf


def test_counts_respect_scope(tmp_path):
    sf = _seed(f"sqlite:///{tmp_path / 'wq.db'}")
    state = main._MenuState()  # scope mặc định USA/TOP3000/delay=1
    state.session_factory = sf
    n_fields, n_ops = main._menu_counts(state)
    assert n_fields == 2  # chỉ đếm scope USA/TOP3000/delay=1 (bỏ EUR)
    assert n_ops == 1


def test_counts_zero_before_db_open():
    state = main._MenuState()
    assert main._menu_counts(state) == (0, 0)
    assert state.logged_in is False


# ------------------------------------------------------- mục 6: xem & nộp
def _seed_candidate(sf, *, sharpe=1.5, hypothesis=None):
    from src.storage.models import AlphaModel, SimulationModel

    session = sf()
    session.add(AlphaModel(id="a1", expression="rank(close)", source="ga", hypothesis=hypothesis))
    session.add(SimulationModel(
        id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=sharpe, fitness=1.2, score=0.9, status="passed",
    ))
    session.commit()
    session.close()


def test_menu_view_submit_khong_co_candidate_thi_khong_goi_submit(monkeypatch):
    from tests.fakes import FakeClient

    sf = make_session_factory(init_db(make_engine("sqlite:///:memory:")))
    state = main._MenuState()
    state.session_factory = sf
    state.client = FakeClient()
    monkeypatch.setattr("builtins.input", lambda _: "")

    main._menu_view_submit(state)

    assert state.client.calls == []


def test_menu_view_submit_nguoi_dung_tu_choi_thi_khong_nop(monkeypatch):
    from tests.fakes import FakeClient, FakeResponse

    sf = make_session_factory(init_db(make_engine("sqlite:///:memory:")))
    _seed_candidate(sf)
    state = main._MenuState()
    state.session_factory = sf
    state.client = FakeClient()
    state.client.queue_get(FakeResponse(200, json_data={"max": 0.1}))  # dry-run preview: check corr
    monkeypatch.setattr("builtins.input", lambda _: "")  # Enter -> từ chối

    main._menu_view_submit(state)

    assert not any(c[0] == "POST" for c in state.client.calls)


# ------------------------------------------------------- mục 5: Auto SIM tự tìm MarketData
def _seed_one_field(sf):
    s = sf()
    s.add(DataFieldModel(id="close", region="USA", universe="TOP3000", delay=1, type="MATRIX"))
    s.commit()
    s.close()


def test_menu_auto_sim_tu_dong_tim_market_data_khong_hoi_input(monkeypatch):
    """Mục 5 không được hỏi gì cả (chỉ cần LLM cấu hình sẵn) — tự tìm MarketData như mục 4."""
    sf = make_session_factory(init_db(make_engine("sqlite:///:memory:")))
    _seed_one_field(sf)
    state = main._MenuState()
    state.session_factory = sf

    def _boom(*_a, **_k):
        raise AssertionError("mục 5 không được gọi input()")

    monkeypatch.setattr("builtins.input", _boom)
    monkeypatch.setattr(main, "_find_market_data_dir", lambda: "data/market_yf")
    called = {}
    monkeypatch.setattr(
        main, "_run_closed_loop_session",
        lambda session_factory, client, region, universe, delay, market_data_dir, **_kw:
            called.setdefault("market_data_dir", market_data_dir),
    )

    main._menu_auto_sim(state)

    assert called["market_data_dir"] == "data/market_yf"


def test_menu_auto_sim_khong_thay_market_data_thi_bao_loi_khong_chay(monkeypatch):
    sf = make_session_factory(init_db(make_engine("sqlite:///:memory:")))
    _seed_one_field(sf)
    state = main._MenuState()
    state.session_factory = sf

    monkeypatch.setattr("builtins.input", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("mục 5 không được gọi input()")))
    monkeypatch.setattr(main, "_find_market_data_dir", lambda: None)
    called = {"ran": False}
    monkeypatch.setattr(
        main, "_run_closed_loop_session",
        lambda *a, **k: called.__setitem__("ran", True),
    )

    main._menu_auto_sim(state)

    assert called["ran"] is False


def test_menu_view_submit_xac_nhan_yes_thi_nop_that(monkeypatch):
    from tests.fakes import FakeClient, FakeResponse

    sf = make_session_factory(init_db(make_engine("sqlite:///:memory:")))
    # sharpe=1.6 (Bug 2: >= SUBMIT_MIN_SHARPE 1.58, khác mặc định 1.5 của helper — dưới
    # ngưỡng nộp thật sẽ bị select_candidates() loại thẳng, không có gì để nộp).
    _seed_candidate(sf, sharpe=1.6)  # không có hypothesis -> không kích hoạt gắn tag Power Pool (đã test riêng)
    state = main._MenuState()
    state.session_factory = sf
    state.client = FakeClient()
    state.client.queue_get(FakeResponse(200, json_data={"max": 0.1}))  # dry-run preview: is_acceptable
    state.client.queue_get(FakeResponse(200, json_data={"max": 0.1}))  # nộp thật: is_acceptable lại
    state.client.queue_get(FakeResponse(200, json_data={"max": 0.1}))  # nộp thật: submit() tự check corr lần nữa trước POST
    state.client.queue_post(FakeResponse(201))  # submit thật
    # Bug 1: sau POST, submit() poll GET /alphas/{id}/submit rồi xác nhận GET /alphas/{id}.
    state.client.queue_get(FakeResponse(200, json_data={"is": {"checks": []}}, text="non-empty"))
    state.client.queue_get(FakeResponse(200, json_data={"dateSubmitted": "2026-07-14T00:00:00Z"}))
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    main._menu_view_submit(state)

    assert sum(1 for c in state.client.calls if c[0] == "POST") == 1
