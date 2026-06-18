"""Test migrate_all copy đúng & idempotent giữa hai engine."""

from __future__ import annotations

from sqlalchemy import create_engine

from src.simulation.simulator import SimulationResult
from src.storage.db import init_db, make_session_factory
from src.storage.migrate import migrate_all
from src.storage.models import AlphaModel, OperatorModel, SimulationModel
from src.storage.repository import AlphaRepository


def _engine():
    return create_engine(
        "sqlite:///:memory:", future=True, connect_args={"check_same_thread": False}
    )


def _seed_source(engine):
    sf = make_session_factory(engine)
    repo = AlphaRepository(sf)
    aid = repo.save_alpha("rank(close)", source="llm", description="seed")
    repo.save_simulation(
        SimulationResult(
            expression="rank(close)", alpha_id=aid, status="passed",
            sharpe=1.5, fitness=1.0, turnover=0.2, raw={"is": {}},
        ),
        region="USA", universe="TOP3000", alpha_id=aid,
    )
    s = sf()
    try:
        s.merge(OperatorModel(name="rank", definition="rank(x)", arity=1))
        s.commit()
    finally:
        s.close()


def test_migrate_all_copy_dung_so_rows():
    src = init_db(_engine())
    _seed_source(src)
    dst = _engine()  # chưa init: migrate_all tự init schema đích

    counts = migrate_all(src, dst)

    assert counts["alphas"] == 1
    assert counts["simulations"] == 1
    assert counts["operators"] == 1

    dsf = make_session_factory(dst)
    s = dsf()
    try:
        assert s.query(AlphaModel).count() == 1
        assert s.query(SimulationModel).count() == 1
        assert s.query(OperatorModel).count() == 1
    finally:
        s.close()


def test_migrate_all_idempotent():
    src = init_db(_engine())
    _seed_source(src)
    dst = _engine()

    migrate_all(src, dst)
    migrate_all(src, dst)  # chạy lần hai không nhân đôi

    dsf = make_session_factory(dst)
    s = dsf()
    try:
        assert s.query(AlphaModel).count() == 1
        assert s.query(SimulationModel).count() == 1
    finally:
        s.close()


def test_migrate_all_bo_qua_bang_thieu_o_nguon():
    # Nguồn chỉ có schema, không có bảng tùy biến nào ngoài bộ models -> vẫn chạy.
    src = init_db(_engine())
    dst = _engine()
    counts = migrate_all(src, dst)
    assert counts["alphas"] == 0  # rỗng nhưng không lỗi
