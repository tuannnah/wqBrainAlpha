"""Test lưu simulation vào DB."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from src.simulation.simulator import SimulationResult
from src.storage.db import init_db, make_session_factory
from src.storage.models import AlphaModel, FailureModel, SimulationModel
from src.storage.repository import AlphaRepository, expr_hash


def _engine():
    return create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})


def _passed(expr, sharpe=1.7, fitness=1.2, status="passed"):
    return SimulationResult(
        expression=expr,
        alpha_id="a-" + expr,
        status=status,
        sharpe=sharpe,
        fitness=fitness,
        turnover=0.3,
        drawdown=0.1,
        raw={"is": {}},
    )


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


def test_save_alpha_luu_bo_ba_gia_thuyet():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    repo = AlphaRepository(sf)
    aid = repo.save_alpha(
        "rank(close)",
        source="llm",
        hypothesis={"observation": "x", "economic_rationale": "y"},
        description="mô tả alpha",
    )
    child = repo.save_alpha("rank(rank(close))", source="llm", parent_id=aid)
    session = sf()
    try:
        a = session.get(AlphaModel, aid)
        assert "observation" in (a.hypothesis or "")
        assert a.description == "mô tả alpha"
        c = session.get(AlphaModel, child)
        assert c.parent_id == aid
    finally:
        session.close()


def test_get_cached_simulation_hit_theo_hash():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    repo = AlphaRepository(sf)
    repo.save_simulation(_passed("rank(close)"), region="USA", universe="TOP3000")

    hit = repo.get_cached_simulation("rank(close)")
    assert hit is not None and hit.expr_hash == expr_hash("rank(close)")
    assert repo.get_cached_simulation("rank(open)") is None


def test_get_cached_simulation_bo_qua_status_error():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    repo = AlphaRepository(sf)
    repo.save_simulation(_passed("rank(x)", status="error"), region="USA", universe="TOP3000")
    # Kết quả lỗi không được coi là cache hợp lệ.
    assert repo.get_cached_simulation("rank(x)") is None


def test_record_failure_va_recent_failures():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    repo = AlphaRepository(sf)
    repo.record_failure("bad((", category="syntax", reason="ngoặc lệch", source="llm")
    repo.record_failure("rank(close)", category="low_score", reason="sharpe thấp", source="llm")

    fails = repo.recent_failures(10)
    assert len(fails) == 2
    assert {f.category for f in fails} == {"syntax", "low_score"}


def test_zoo_chi_tra_alpha_pass_sort_theo_score():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    repo = AlphaRepository(sf)
    repo.save_simulation(_passed("rank(a)", sharpe=2.0), region="USA", universe="TOP3000", score=2.0)
    repo.save_simulation(_passed("rank(b)", sharpe=1.0), region="USA", universe="TOP3000", score=1.0)
    repo.save_simulation(_passed("rank(c)", status="failed"), region="USA", universe="TOP3000", score=0.5)

    zoo = repo.zoo(10)
    exprs = [z.expression for z in zoo]
    assert exprs == ["rank(a)", "rank(b)"]  # 'failed' bị loại, sort giảm theo score


def test_migration_them_cot_thieu_cho_db_cu():
    """init_db trên DB thiếu cột mới phải ALTER thêm, không lỗi, không nhân đôi."""
    engine = _engine()
    # Giả lập DB cũ: bảng alphas thiếu hypothesis/description/parent_id.
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE alphas (id VARCHAR PRIMARY KEY, expression TEXT NOT NULL, "
                "source VARCHAR, created_at DATETIME)"
            )
        )
    init_db(engine)  # phải thêm cột thiếu mà không lỗi
    sf = make_session_factory(engine)
    repo = AlphaRepository(sf)
    aid = repo.save_alpha("rank(close)", source="llm", description="ok")
    session = sf()
    try:
        assert session.get(AlphaModel, aid).description == "ok"
    finally:
        session.close()
