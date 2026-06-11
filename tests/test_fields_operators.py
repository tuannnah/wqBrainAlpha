"""Test FieldRepository (phân trang + cache) và OperatorRepository (arity)."""

from __future__ import annotations

from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository, _count_arity
from src.storage.db import init_db, make_session_factory
from tests.fakes import FakeClient, FakeResponse


def _engine():
    from sqlalchemy import create_engine

    return create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})


def test_fetch_all_phan_trang_va_cache():
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)

    client = FakeClient()
    # Trang 1 đầy (2 field), trang 2 rỗng → dừng.
    client.queue_get(
        FakeResponse(
            200,
            json_data={
                "count": 2,
                "results": [
                    {"id": "close", "type": "MATRIX", "dataset": {"id": "pv"}},
                    {"id": "open", "type": "MATRIX", "dataset": {"id": "pv"}},
                ],
            },
        )
    )

    repo = FieldRepository(client, session_factory)
    fields = repo.fetch_all("USA", "TOP3000", 1, page_size=2)

    assert {f.id for f in fields} == {"close", "open"}
    assert fields[0].dataset_id == "pv"
    # Đã cache vào DB.
    assert len(repo.load_cached()) == 2


def test_fetch_all_gui_instrument_type():
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)
    client = FakeClient()
    client.queue_get(
        FakeResponse(200, json_data={"count": 1, "results": [{"id": "close", "type": "MATRIX"}]})
    )
    repo = FieldRepository(client, session_factory)
    repo.fetch_all("USA", "TOP3000", 1, page_size=1)
    _, path, kwargs = client.calls[0]
    assert path == "/data-fields"
    assert kwargs["params"]["instrumentType"] == "EQUITY"


def test_ensure_dung_cache_khong_goi_api():
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)

    seed = FakeClient()
    seed.queue_get(
        FakeResponse(200, json_data={"count": 1, "results": [{"id": "close", "type": "MATRIX"}]})
    )
    FieldRepository(seed, session_factory).fetch_all("USA", "TOP3000", 1, page_size=1)

    empty_client = FakeClient()  # không có response: gọi API sẽ lỗi
    repo = FieldRepository(empty_client, session_factory)
    fields, fetched = repo.ensure("USA", "TOP3000", 1)
    assert fetched is False
    assert len(fields) == 1
    assert empty_client.calls == []


def test_ensure_force_tai_moi():
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)
    seed = FakeClient()
    seed.queue_get(
        FakeResponse(200, json_data={"count": 1, "results": [{"id": "close", "type": "MATRIX"}]})
    )
    FieldRepository(seed, session_factory).fetch_all("USA", "TOP3000", 1, page_size=1)

    refetch = FakeClient()
    refetch.queue_get(
        FakeResponse(200, json_data={"count": 2, "results": [{"id": "close"}, {"id": "open"}]})
    )
    repo = FieldRepository(refetch, session_factory)
    _, fetched = repo.ensure("USA", "TOP3000", 1, force=True)
    assert fetched is True
    assert len(refetch.calls) == 1


def test_fetch_all_loi_http_raise():
    from src.data.fields import FieldFetchError

    engine = init_db(_engine())
    session_factory = make_session_factory(engine)
    client = FakeClient()
    client.queue_get(FakeResponse(400, text="bad request"))
    repo = FieldRepository(client, session_factory)
    try:
        repo.fetch_all("USA", "TOP3000", 1)
        assert False, "phải raise"
    except FieldFetchError:
        pass


def test_reload_ghi_de_khong_nhan_doi():
    engine = init_db(_engine())
    sf = make_session_factory(engine)

    seed = FakeClient()
    seed.queue_get(
        FakeResponse(200, json_data={"count": 2, "results": [{"id": "close"}, {"id": "open"}]})
    )
    FieldRepository(seed, sf).fetch_all("USA", "TOP3000", 1, page_size=10)

    refetch = FakeClient()  # lần reload trả ít field hơn
    refetch.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "close"}]}))
    repo = FieldRepository(refetch, sf)
    repo.get_fields("USA", "TOP3000", 1, force_reload=True)
    # Ghi đè: chỉ còn 1 field cho scope đó, không phải 3.
    assert repo.cached_count("USA", "TOP3000", 1) == 1


def test_ttl_het_han_trigger_fetch():
    from datetime import timedelta

    from src.storage.models import FetchStateModel

    engine = init_db(_engine())
    sf = make_session_factory(engine)
    seed = FakeClient()
    seed.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "close"}]}))
    FieldRepository(seed, sf).fetch_all("USA", "TOP3000", 1, page_size=10)

    # Đẩy fetched_at lùi quá TTL (mặc định 30 ngày).
    session = sf()
    state = session.get(FetchStateModel, "data_fields:USA:TOP3000:1")
    state.fetched_at = state.fetched_at - timedelta(days=40)
    session.merge(state)
    session.commit()
    session.close()

    refetch = FakeClient()
    refetch.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "open"}]}))
    repo = FieldRepository(refetch, sf)
    repo.get_fields("USA", "TOP3000", 1)  # phải fetch lại vì cache hết hạn
    assert len(refetch.calls) == 1


def test_scope_khac_nhau_khong_de_len_nhau():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    c1 = FakeClient()
    c1.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "close"}]}))
    FieldRepository(c1, sf).fetch_all("USA", "TOP3000", 1, page_size=10)
    c2 = FakeClient()
    c2.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "close"}]}))
    FieldRepository(c2, sf).fetch_all("EUR", "TOP1000", 0, page_size=10)
    # Cùng id 'close' nhưng khác scope -> 2 dòng riêng (khóa kép).
    repo = FieldRepository(None, sf)
    assert repo.cached_count("USA", "TOP3000", 1) == 1
    assert repo.cached_count("EUR", "TOP1000", 0) == 1
    assert repo.cached_count() == 2


def test_get_state_va_all_states():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    c = FakeClient()
    c.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "close"}]}))
    FieldRepository(c, sf).fetch_all("USA", "TOP3000", 1, page_size=10)
    repo = FieldRepository(None, sf)
    state = repo.get_state("USA", "TOP3000", 1)
    assert state.status == "complete"
    assert state.total_count == 1
    assert len(repo.all_states()) == 1


def test_count_arity():
    assert _count_arity("ts_mean(x, d)") == 2
    assert _count_arity("rank(x)") == 1
    assert _count_arity("close") == 0


def test_fetch_operators_parse_arity():
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)

    client = FakeClient()
    client.queue_get(
        FakeResponse(
            200,
            json_data={
                "results": [
                    {"name": "ts_mean", "definition": "ts_mean(x, d)", "category": "Time Series"},
                    {"name": "rank", "definition": "rank(x)", "category": "Cross Sectional"},
                ]
            },
        )
    )

    repo = OperatorRepository(client, session_factory)
    operators = repo.fetch_all()
    by_name = {o.name: o for o in operators}
    assert by_name["ts_mean"].arity == 2
    assert by_name["rank"].arity == 1
    assert len(repo.load_cached()) == 2
