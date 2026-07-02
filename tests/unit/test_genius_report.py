"""Test báo cáo BRAIN Genius (sub-project G) — 4 metric tie-break tính được LOCAL từ alpha đã
nộp. KHÔNG phải gate, chỉ để tham khảo."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from src.scoring.genius_report import (
    average_distinct_fields_per_alpha,
    average_distinct_operators_per_alpha,
    total_distinct_fields,
    total_distinct_operators,
)
from src.storage.db import init_db, make_session_factory
from src.storage.models import AlphaModel, SimulationModel, SubmissionModel


def _engine():
    return create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})


def _seed_submitted(session_factory):
    session = session_factory()
    try:
        session.add(AlphaModel(id="a1", expression="rank(add(close, open))", source="ga"))
        session.add(SimulationModel(
            id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000", status="passed",
        ))
        session.add(SubmissionModel(id="sub1", alpha_id="WQ1", status="submitted"))

        session.add(AlphaModel(id="a2", expression="ts_delta(close, 5)", source="ga"))
        session.add(SimulationModel(
            id="s2", alpha_id="a2", wq_alpha_id="WQ2", region="USA", universe="TOP3000", status="passed",
        ))
        session.add(SubmissionModel(id="sub2", alpha_id="WQ2", status="submitted"))

        # chưa nộp -> KHÔNG được tính vào report
        session.add(AlphaModel(id="a3", expression="rank(rank(rank(close)))", source="ga"))
        session.add(SimulationModel(
            id="s3", alpha_id="a3", wq_alpha_id="WQ3", region="USA", universe="TOP3000", status="passed",
        ))
        session.commit()
    finally:
        session.close()


def test_average_distinct_operators_per_alpha():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_submitted(sf)
    # a1: {rank, add}=2 operator; a2: {ts_delta}=1 operator -> avg (2+1)/2 = 1.5
    assert average_distinct_operators_per_alpha(sf) == pytest.approx(1.5)


def test_average_distinct_fields_per_alpha():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_submitted(sf)
    # a1: {close, open}=2 field; a2: {close}=1 field -> avg (2+1)/2 = 1.5
    assert average_distinct_fields_per_alpha(sf) == pytest.approx(1.5)


def test_total_distinct_operators():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_submitted(sf)
    # union {rank, add} | {ts_delta} = {rank, add, ts_delta} = 3
    assert total_distinct_operators(sf) == 3


def test_total_distinct_fields():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed_submitted(sf)
    # union {close, open} | {close} = {close, open} = 2
    assert total_distinct_fields(sf) == 2


def test_none_khi_chua_co_alpha_nao_nop():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    assert average_distinct_operators_per_alpha(sf) is None
    assert average_distinct_fields_per_alpha(sf) is None
    assert total_distinct_operators(sf) == 0
    assert total_distinct_fields(sf) == 0


def test_ts_backfill_group_backfill_khong_tinh_vao_operator():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    session.add(AlphaModel(id="a1", expression="rank(ts_backfill(group_backfill(close, sector), 5))", source="ga"))
    session.add(SimulationModel(
        id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000", status="passed",
    ))
    session.add(SubmissionModel(id="sub1", alpha_id="WQ1", status="submitted"))
    session.commit()
    session.close()
    assert average_distinct_operators_per_alpha(sf) == pytest.approx(1.0)  # chỉ 'rank'
