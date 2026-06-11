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
