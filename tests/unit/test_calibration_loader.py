"""Test load_brain_records: đọc AlphaModel+SimulationModel(+SubmissionModel) -> BrainRecord.

DB tạm in-memory (sqlite:///:memory:) tạo trong test, KHÔNG đụng wq_alpha_*.db thật."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.calibration.loader import BrainRecord, load_brain_records
from src.storage.models import AlphaModel, Base, SimulationModel, SubmissionModel


def _make_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_loads_passed_simulation_into_brain_record():
    sf = _make_session_factory()
    s = sf()
    s.add(AlphaModel(id="a1", expression="rank(close)", source="manual"))
    s.add(SimulationModel(
        id="s1", alpha_id="a1", region="USA", universe="TOP3000",
        sharpe=1.5, fitness=1.2, turnover=0.3, status="passed",
    ))
    s.commit()
    s.close()

    records = load_brain_records(sf)
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, BrainRecord)
    assert rec.expr_string == "rank(close)"
    assert rec.brain_sharpe == 1.5
    assert rec.brain_fitness == 1.2
    assert rec.brain_turnover == 0.3
    assert rec.brain_self_corr is None  # chưa submit


def test_self_corr_filled_when_submission_exists():
    sf = _make_session_factory()
    s = sf()
    s.add(AlphaModel(id="a1", expression="rank(close)", source="manual"))
    s.add(SimulationModel(
        id="s1", alpha_id="a1", region="USA", universe="TOP3000",
        sharpe=1.5, fitness=1.2, turnover=0.3, status="passed",
    ))
    s.add(SubmissionModel(id="sub1", alpha_id="a1", status="submitted", self_correlation=0.42))
    s.commit()
    s.close()

    records = load_brain_records(sf)
    assert records[0].brain_self_corr == 0.42


def test_excludes_error_status_and_null_sharpe():
    sf = _make_session_factory()
    s = sf()
    s.add(AlphaModel(id="a1", expression="rank(close)", source="manual"))
    s.add(SimulationModel(id="s1", alpha_id="a1", region="USA", universe="TOP3000", status="error"))
    s.add(AlphaModel(id="a2", expression="ts_mean(volume,5)", source="manual"))
    s.add(SimulationModel(id="s2", alpha_id="a2", region="USA", universe="TOP3000", status="passed", sharpe=None))
    s.commit()
    s.close()

    records = load_brain_records(sf)
    assert records == []


def test_takes_latest_simulation_per_alpha():
    sf = _make_session_factory()
    s = sf()
    s.add(AlphaModel(id="a1", expression="rank(close)", source="manual"))
    s.add(SimulationModel(
        id="s_old", alpha_id="a1", region="USA", universe="TOP3000",
        sharpe=1.0, status="passed",
        sim_at=__import__("datetime").datetime(2024, 1, 1),
    ))
    s.add(SimulationModel(
        id="s_new", alpha_id="a1", region="USA", universe="TOP3000",
        sharpe=2.0, status="passed",
        sim_at=__import__("datetime").datetime(2024, 6, 1),
    ))
    s.commit()
    s.close()

    records = load_brain_records(sf)
    assert len(records) == 1
    assert records[0].brain_sharpe == 2.0


def test_limit_caps_number_of_records():
    sf = _make_session_factory()
    s = sf()
    for i in range(5):
        s.add(AlphaModel(id=f"a{i}", expression=f"rank(close_{i})", source="manual"))
        s.add(SimulationModel(
            id=f"s{i}", alpha_id=f"a{i}", region="USA", universe="TOP3000",
            sharpe=float(i), status="passed",
        ))
    s.commit()
    s.close()

    records = load_brain_records(sf, limit=3)
    assert len(records) == 3
