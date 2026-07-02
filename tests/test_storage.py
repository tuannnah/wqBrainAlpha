"""Test lưu simulation vào DB."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from src.simulation.simulator import SimulationResult
from src.storage.db import init_db, make_session_factory
from src.storage.models import AlphaModel, FailureModel, SimulationModel
from src.storage.repository import AlphaRepository, InvalidFieldRepository, expr_hash


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


def test_save_simulation_luu_failed_checks():
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)

    result = SimulationResult(
        expression="rank(close)", alpha_id="a1", status="failed",
        sharpe=0.2, failed_checks=["LOW_SHARPE", "LOW_FITNESS"], raw={"is": {}},
    )
    repo = AlphaRepository(session_factory)
    sim_id = repo.save_simulation(result, region="USA", universe="TOP3000")

    session = session_factory()
    try:
        sim = session.get(SimulationModel, sim_id)
        import json
        assert json.loads(sim.failed_checks) == ["LOW_SHARPE", "LOW_FITNESS"]
    finally:
        session.close()


def test_invalid_field_repo_record_va_blacklist():
    """Ghi field chết rồi đọc lại blacklist; ghi trùng không nhân đôi."""
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    repo = InvalidFieldRepository(sf)

    repo.record("mdl77_2gdna_cfroi", region="USA", universe="TOP3000", reason="Invalid data field")
    repo.record("opt6_10dorhv", region="USA", universe="TOP3000")
    repo.record("mdl77_2gdna_cfroi", region="USA", universe="TOP3000")  # trùng -> idempotent

    bl = repo.blacklist()
    assert bl == {"mdl77_2gdna_cfroi", "opt6_10dorhv"}


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


def test_get_cached_simulation_phan_biet_theo_config_key():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    repo = AlphaRepository(sf)
    cfg_a = "USA|TOP3000|delay=1|SUBINDUSTRY|decay=0|truncation=0.08"
    cfg_b = "USA|TOP3000|delay=1|INDUSTRY|decay=6|truncation=0.05"

    repo.save_simulation(
        _passed("rank(close)", sharpe=1.1),
        region="USA",
        universe="TOP3000",
        config_key=cfg_a,
    )

    hit_a = repo.get_cached_simulation("rank(close)", config_key=cfg_a)
    assert hit_a is not None
    assert hit_a.expr_hash == expr_hash("rank(close)", cfg_a)
    assert repo.get_cached_simulation("rank(close)", config_key=cfg_b) is None
    assert repo.get_cached_simulation("rank(close)") is None


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


def test_top_simulated_sap_xep_theo_sharpe_bo_error():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    repo = AlphaRepository(sf)
    repo.save_simulation(_passed("a", sharpe=0.5, fitness=0.1, status="failed"), region="USA", universe="TOP3000")
    repo.save_simulation(_passed("b", sharpe=1.9, fitness=0.8, status="failed"), region="USA", universe="TOP3000")
    repo.save_simulation(_passed("c", sharpe=2.0, status="error"), region="USA", universe="TOP3000")
    top = repo.top_simulated(5)
    exprs = [t[0] for t in top]
    assert exprs == ["b", "a"]          # theo sharpe giảm, loại 'error'
    assert top[0] == ("b", 1.9, 0.8)
