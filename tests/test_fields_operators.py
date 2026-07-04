"""Test FieldRepository (phân trang + cache) và OperatorRepository (arity)."""

from __future__ import annotations

from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository, _count_arity, count_positional_arity
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


def test_fetch_vuot_cua_so_10000_chuyen_sang_tung_dataset():
    """API báo count=15000 nhưng chỉ trả về được ít field rồi results rỗng
    (giới hạn cửa sổ tìm kiếm của WQ Brain) -> phải tự tải riêng từng dataset."""
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)
    client = FakeClient()

    # Fetch chung (không lọc dataset): trang 1 có 1 field, trang 2 rỗng dù count=15000.
    client.queue_get(
        FakeResponse(200, json_data={"count": 15000, "results": [{"id": "a", "dataset": {"id": "pv1"}}]})
    )
    client.queue_get(FakeResponse(200, json_data={"count": 15000, "results": []}))

    # Danh sách dataset đúng scope.
    client.queue_get(
        FakeResponse(200, json_data={"count": 2, "results": [{"id": "pv1"}, {"id": "fnd6"}]})
    )

    # Tải riêng từng dataset.
    client.queue_get(
        FakeResponse(200, json_data={"count": 1, "results": [{"id": "a", "dataset": {"id": "pv1"}}]})
    )
    client.queue_get(
        FakeResponse(200, json_data={"count": 1, "results": [{"id": "b", "dataset": {"id": "fnd6"}}]})
    )

    repo = FieldRepository(client, session_factory)
    fields = repo.fetch_all("USA", "TOP3000", 1, page_size=1)

    assert {f.id for f in fields} == {"a", "b"}
    assert repo.cached_count() == 2

    # Có gọi /data-sets với đúng scope để lấy danh sách dataset.
    dataset_call = next(c for c in client.calls if c[1] == "/data-sets")
    assert dataset_call[2]["params"]["region"] == "USA"
    assert dataset_call[2]["params"]["universe"] == "TOP3000"
    assert dataset_call[2]["params"]["delay"] == 1

    # Các call fetch theo dataset phải mang đúng dataset.id.
    field_calls = [c for c in client.calls if c[1] == "/data-fields"]
    dataset_ids_used = {c[2]["params"].get("dataset.id") for c in field_calls if c[2]["params"].get("dataset.id")}
    assert dataset_ids_used == {"pv1", "fnd6"}


def test_fetch_dung_nguong_cua_so_van_nghi_bi_cat_du_count_noi_doi():
    """Gặp thật trên WQ Brain: count cũng bị chặn = đúng SEARCH_WINDOW_CAP (không phải tổng
    thật) -> so sánh len(fields) < declared_total sẽ KHÔNG bắt được (bằng nhau). Phải tự
    nghi ngờ khi số field nhận được chạm đúng ngưỡng, bất kể count nói gì."""
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)
    client = FakeClient()

    # Broad query: đúng 3 field, count CŨNG báo 3 (nói dối giống hệt số nhận được).
    client.queue_get(
        FakeResponse(200, json_data={
            "count": 3,
            "results": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        })
    )
    client.queue_get(
        FakeResponse(200, json_data={"count": 2, "results": [{"id": "pv1"}, {"id": "fnd6"}]})
    )
    client.queue_get(
        FakeResponse(200, json_data={"count": 1, "results": [{"id": "a", "dataset": {"id": "pv1"}}]})
    )
    client.queue_get(
        FakeResponse(200, json_data={
            "count": 3,
            "results": [
                {"id": "b", "dataset": {"id": "fnd6"}},
                {"id": "c", "dataset": {"id": "fnd6"}},
                {"id": "d", "dataset": {"id": "fnd6"}},
            ],
        })
    )

    repo = FieldRepository(client, session_factory)
    repo.SEARCH_WINDOW_CAP = 3  # hạ ngưỡng để test không cần giả lập thật 10000 dòng
    fields = repo.fetch_all("USA", "TOP3000", 1, page_size=3)

    assert {f.id for f in fields} == {"a", "b", "c", "d"}
    assert any(c[1] == "/data-sets" for c in client.calls)


def test_fetch_khong_vuot_cua_so_thi_khong_goi_data_sets():
    """Trường hợp bình thường (không bị cắt) thì không cần fallback -> không gọi /data-sets."""
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)
    client = FakeClient()
    client.queue_get(
        FakeResponse(200, json_data={"count": 2, "results": [{"id": "close"}, {"id": "open"}]})
    )
    client.queue_get(FakeResponse(200, json_data={"count": 2, "results": []}))

    repo = FieldRepository(client, session_factory)
    fields = repo.fetch_all("USA", "TOP3000", 1, page_size=2)

    assert {f.id for f in fields} == {"close", "open"}
    assert all(c[1] != "/data-sets" for c in client.calls)


def test_fetch_dataset_van_vuot_cua_so_khong_crash():
    """Một dataset đơn lẻ vẫn bị cắt (hiếm) -> không crash, trả về phần đã lấy được."""
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)
    client = FakeClient()
    client.queue_get(
        FakeResponse(200, json_data={"count": 20000, "results": [{"id": "a", "dataset": {"id": "huge"}}]})
    )
    client.queue_get(FakeResponse(200, json_data={"count": 20000, "results": []}))
    client.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "huge"}]}))
    # Fetch theo dataset "huge" cũng bị cắt tương tự.
    client.queue_get(
        FakeResponse(200, json_data={"count": 20000, "results": [{"id": "a", "dataset": {"id": "huge"}}]})
    )
    client.queue_get(FakeResponse(200, json_data={"count": 20000, "results": []}))

    repo = FieldRepository(client, session_factory)
    fields = repo.fetch_all("USA", "TOP3000", 1, page_size=1)

    assert {f.id for f in fields} == {"a"}


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


def test_load_cached_loc_theo_scope():
    """T6.4: load_cached(scope) chỉ trả fields của đúng (region, universe, delay)."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    c1 = FakeClient()
    c1.queue_get(FakeResponse(200, json_data={"count": 2, "results": [{"id": "close"}, {"id": "open"}]}))
    FieldRepository(c1, sf).fetch_all("USA", "TOP3000", 1, page_size=10)
    c2 = FakeClient()
    c2.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "vwap"}]}))
    FieldRepository(c2, sf).fetch_all("EUR", "TOP1000", 0, page_size=10)

    repo = FieldRepository(None, sf)
    usa = {f.id for f in repo.load_cached(region="USA", universe="TOP3000", delay=1)}
    assert usa == {"close", "open"}  # không lẫn 'vwap' của EUR
    eur = {f.id for f in repo.load_cached(region="EUR", universe="TOP1000", delay=0)}
    assert eur == {"vwap"}
    # Không truyền scope -> tất cả (tương thích ngược).
    assert len(repo.load_cached()) == 3


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


def test_count_positional_arity_loai_named_param():
    """Tham số có '=' (named-only) không tính positional -> chặn được named-param."""
    # positional thuần
    assert count_positional_arity("ts_zscore(x, d)") == 2
    assert count_positional_arity("trade_when(x, y, z)") == 3
    assert count_positional_arity("group_neutralize(x, group)") == 2
    # named-only: chỉ x là positional
    assert count_positional_arity("winsorize(x, std=4)") == 1
    assert count_positional_arity("rank(x, rate=2)") == 1
    assert count_positional_arity("ts_decay_linear(x, d, dense = false)") == 2
    assert count_positional_arity("close") == 0


def test_count_positional_arity_bucket_da_chu_ky_va_phay_trong_ngoac_kep():
    """bucket có nhiều chữ ký + giá trị chứa dấu phẩy trong ngoặc kép -> vẫn ra 1."""
    definition = (
        "bucket(rank(x), range=“0, 1, 0.1”, skipBoth=False, NaNGroup=False)\r\n"
        "or\r\nbucket(rank(x), buckets = “2,5,6,7,10”, skipBoth=False, NaNGroup=False)"
    )
    assert count_positional_arity(definition) == 1


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


def test_field_fetch_error_mang_status_code():
    from src.data.fields import FieldFetchError

    err = FieldFetchError("không có quyền", status_code=403)
    assert err.status_code == 403
    assert FieldFetchError("x").status_code is None


def test_fetch_403_raise_kem_status_code():
    from src.data.fields import FieldFetchError

    engine = init_db(_engine())
    sf = make_session_factory(engine)
    client = FakeClient()
    client.queue_get(FakeResponse(403, text="forbidden"))
    repo = FieldRepository(client, sf)
    try:
        repo.fetch_all("USA", "TOP3000", 1)
        assert False, "phải raise"
    except FieldFetchError as exc:
        assert exc.status_code == 403


def test_mark_no_access_ghi_trang_thai():
    from src.storage.models import FetchStateModel

    engine = init_db(_engine())
    sf = make_session_factory(engine)
    repo = FieldRepository(None, sf)
    repo.mark_no_access("EUR", "TOP400", 0)
    state = repo.get_state("EUR", "TOP400", 0)
    assert state is not None
    assert state.status == "no_access"
