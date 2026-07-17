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


def test_near_miss_exprs_chon_dung_dai_sharpe_dedup_va_sort():
    """near_miss_exprs(min,max): chỉ sim Brain THẬT (BrainSimLinkModel, wq_alpha_id có),
    Sharpe trong [min, max), dedup theo expr_string (giữ sharpe cao nhất), sort giảm dần,
    tôn trọng limit — nguồn cho NearMissVariantSource (log 2026-07-16: near-miss 0.8-0.9
    chưa từng được thử biến thể)."""
    from src.storage.repository import MiniBrainRepository

    engine = init_db(_engine())
    sf = make_session_factory(engine)
    repo = MiniBrainRepository(sf)

    def rec(h, expr, sharpe, wq_id="WQx"):
        repo.record_brain_sim(
            h, expr, wq_alpha_id=wq_id, region="USA", universe="TOP1000",
            sharpe=sharpe, fitness=0.3, turnover=0.4, self_corr=None, status="failed",
        )

    rec("h1", "expr_a", 0.89)
    rec("h2", "expr_a", 0.80)  # trùng expr, sharpe thấp hơn -> giữ 0.89
    rec("h3", "expr_b", 0.75)
    rec("h4", "expr_cao", 1.3)   # ngoài dải
    rec("h5", "expr_thap", 0.3)  # ngoài dải
    rec("h6", "expr_local", 0.85, wq_id=None)  # chưa chạm Brain -> loại

    rows = repo.near_miss_exprs(0.6, 1.0, limit=5)
    assert rows == [("expr_a", 0.89), ("expr_b", 0.75)]
    assert repo.near_miss_exprs(0.6, 1.0, limit=1) == [("expr_a", 0.89)]


def test_dataset_of_fields_tra_map_field_dataset():
    """dataset_of_fields: map {field -> dataset_id} từ bảng data_fields (cross-read có chủ
    đích như submit_ready_alphas) — phục vụ combo cùng-dataset của NearMissVariantSource.
    Field không có trong catalog -> vắng mặt trong map (caller tự quyết, không đoán)."""
    from src.storage.models import DataFieldModel
    from src.storage.repository import MiniBrainRepository

    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    try:
        session.add(DataFieldModel(id="firm_vol_imbalance", region="USA", universe="TOP1000",
                                   delay=1, dataset_id="order_flow_imb"))
        session.add(DataFieldModel(id="snt_social_value", region="USA", universe="TOP1000",
                                   delay=1, dataset_id="sentiment1"))
        session.commit()
    finally:
        session.close()

    repo = MiniBrainRepository(sf)
    m = repo.dataset_of_fields({"firm_vol_imbalance", "snt_social_value", "field_la"})
    assert m == {"firm_vol_imbalance": "order_flow_imb", "snt_social_value": "sentiment1"}
    assert repo.dataset_of_fields(set()) == {}
