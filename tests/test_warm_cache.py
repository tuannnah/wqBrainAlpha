"""Test warm_cache: resume, probe no_access, gom lỗi."""

from __future__ import annotations

from sqlalchemy import create_engine

from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository
from src.data.warm_cache import WarmCacheReport, warm_cache
from src.storage.db import init_db, make_session_factory
from tests.fakes import FakeClient, FakeResponse


def _engine():
    return create_engine(
        "sqlite:///:memory:", future=True, connect_args={"check_same_thread": False}
    )


def _seed_operators(sf):
    oc = FakeClient()
    oc.queue_get(
        FakeResponse(200, json_data={"results": [{"name": "rank", "definition": "rank(x)"}]})
    )
    OperatorRepository(oc, sf).fetch_all()


def _noop_sleep(_s):
    pass


def test_warm_cache_fetch_moi():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)

    fc = FakeClient()
    fc.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "close"}]}))
    field_repo = FieldRepository(fc, sf)
    op_repo = OperatorRepository(FakeClient(), sf)  # operators đã cache -> không gọi API

    report = warm_cache(
        field_repo, op_repo, [("USA", "TOP3000", 1)], sleep_func=_noop_sleep
    )
    assert isinstance(report, WarmCacheReport)
    assert report.fetched == 1
    assert report.skipped == 0
    assert report.operators == 1


def test_warm_cache_resume_bo_qua_scope_da_complete():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)
    # Seed scope đã complete.
    seed = FakeClient()
    seed.queue_get(FakeResponse(200, json_data={"count": 1, "results": [{"id": "close"}]}))
    FieldRepository(seed, sf).fetch_all("USA", "TOP3000", 1, page_size=10)

    # Client rỗng: nếu gọi API sẽ IndexError -> chứng minh không gọi.
    field_repo = FieldRepository(FakeClient(), sf)
    op_repo = OperatorRepository(FakeClient(), sf)
    report = warm_cache(
        field_repo, op_repo, [("USA", "TOP3000", 1)], sleep_func=_noop_sleep
    )
    assert report.skipped == 1
    assert report.fetched == 0


def test_warm_cache_empty_danh_dau_no_access():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)

    fc = FakeClient()
    fc.queue_get(FakeResponse(200, json_data={"count": 0, "results": []}))
    field_repo = FieldRepository(fc, sf)
    op_repo = OperatorRepository(FakeClient(), sf)
    report = warm_cache(
        field_repo, op_repo, [("ASI", "MINVOL1M", 1)], sleep_func=_noop_sleep
    )
    assert report.no_access == 1
    assert report.fetched == 0
    assert field_repo.get_state("ASI", "MINVOL1M", 1).status == "no_access"


def test_warm_cache_403_danh_dau_no_access():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)

    fc = FakeClient()
    fc.queue_get(FakeResponse(403, text="forbidden"))
    field_repo = FieldRepository(fc, sf)
    op_repo = OperatorRepository(FakeClient(), sf)
    report = warm_cache(
        field_repo, op_repo, [("CHN", "TOP2000U", 0)], sleep_func=_noop_sleep
    )
    assert report.no_access == 1
    assert field_repo.get_state("CHN", "TOP2000U", 0).status == "no_access"


def test_warm_cache_resume_bo_qua_no_access():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)
    field_repo = FieldRepository(FakeClient(), sf)  # client rỗng
    field_repo.mark_no_access("EUR", "TOP400", 0)
    op_repo = OperatorRepository(FakeClient(), sf)

    report = warm_cache(
        field_repo, op_repo, [("EUR", "TOP400", 0)], sleep_func=_noop_sleep
    )
    assert report.no_access == 1
    assert report.fetched == 0


def test_warm_cache_loi_http_khac_gom_vao_errors():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_operators(sf)
    fc = FakeClient()
    fc.queue_get(FakeResponse(500, text="server error"))
    field_repo = FieldRepository(fc, sf)
    op_repo = OperatorRepository(FakeClient(), sf)
    report = warm_cache(
        field_repo, op_repo, [("USA", "TOP3000", 1)], sleep_func=_noop_sleep
    )
    assert len(report.errors) == 1
    assert report.errors[0][0] == ("USA", "TOP3000", 1)
    assert report.no_access == 0
