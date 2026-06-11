"""Test lưu simulation vào DB."""

from __future__ import annotations

from sqlalchemy import create_engine

from src.simulation.simulator import SimulationResult
from src.storage.db import init_db, make_session_factory
from src.storage.models import AlphaModel, SimulationModel
from src.storage.repository import AlphaRepository


def _engine():
    return create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})


def test_save_simulation_persists_alpha_va_metrics():
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)

    result = SimulationResult(
        expression="rank(close)",
        alpha_id="a1",
        status="passed",
        sharpe=1.7,
        fitness=1.2,
        turnover=0.3,
        raw={"is": {}},
    )
    repo = AlphaRepository(session_factory)
    sim_id = repo.save_simulation(result, region="USA", universe="TOP3000")

    session = session_factory()
    try:
        sim = session.get(SimulationModel, sim_id)
        assert sim.sharpe == 1.7
        assert sim.status == "passed"
        assert session.query(AlphaModel).count() == 1
    finally:
        session.close()
