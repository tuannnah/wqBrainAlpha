"""Test cầu DB Brain SIM ↔ MiniBrain: model BrainSimLinkModel + method repository
record_brain_sim/load_brain_sims/brain_pnl_pool. SQLite in-memory, không mạng."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.db import init_db
from src.storage.models import BrainSimLinkModel
from src.storage.repository import MiniBrainRepository


@pytest.fixture
def repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    sf = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(sf)


def test_brain_sim_link_table_created_and_insertable(repo) -> None:  # noqa: ANN001
    """Bảng brain_sim_links được init_db tạo; chèn 1 row đọc lại đúng giá trị."""
    s = repo.session_factory()
    try:
        row = BrainSimLinkModel(
            canonical_hash="h1", expr_string="rank(close)", wq_alpha_id="WQ123",
            region="USA", universe="TOP3000", sharpe=1.5, fitness=1.2, turnover=0.3,
            self_corr=0.4, status="passed", raw_json="{}",
        )
        s.add(row)
        s.commit()
        got = s.query(BrainSimLinkModel).filter_by(canonical_hash="h1").one()
        assert got.wq_alpha_id == "WQ123"
        assert got.sharpe == 1.5
        assert got.self_corr == 0.4
        assert got.created_at is not None
    finally:
        s.close()


def test_record_brain_sim_inserts_then_updates_by_key(repo) -> None:  # noqa: ANN001
    """record_brain_sim merge theo (canonical_hash, region, universe): lần 2 cập nhật,
    KHÔNG nhân đôi row."""
    id1 = repo.record_brain_sim(
        "hA", "rank(close)", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=1.0, fitness=0.9, turnover=0.2, self_corr=0.3, status="passed",
    )
    id2 = repo.record_brain_sim(
        "hA", "rank(close)", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=1.8, fitness=1.5, turnover=0.25, self_corr=0.35, status="passed",
    )
    assert id1 == id2  # cùng key -> cùng row
    sims = repo.load_brain_sims()
    assert len(sims) == 1
    assert sims[0].sharpe == 1.8  # đã cập nhật giá trị mới


def test_load_brain_sims_returns_all(repo) -> None:  # noqa: ANN001
    repo.record_brain_sim("h1", "close", wq_alpha_id=None, region="USA", universe="TOP3000",
                          sharpe=1.0, fitness=1.0, turnover=0.1, self_corr=0.1, status="passed")
    repo.record_brain_sim("h2", "open", wq_alpha_id=None, region="USA", universe="TOP3000",
                          sharpe=None, fitness=None, turnover=None, self_corr=None, status="error")
    assert len(repo.load_brain_sims()) == 2


def test_brain_pnl_pool_only_passed_with_self_corr(repo) -> None:  # noqa: ANN001
    """brain_pnl_pool chỉ trả link passed có self_corr != None."""
    repo.record_brain_sim("hp", "close", wq_alpha_id="W", region="USA", universe="TOP3000",
                          sharpe=1.0, fitness=1.0, turnover=0.1, self_corr=0.5, status="passed")
    repo.record_brain_sim("hf", "open", wq_alpha_id="W2", region="USA", universe="TOP3000",
                          sharpe=0.0, fitness=0.0, turnover=0.0, self_corr=None, status="failed")
    pool = repo.brain_pnl_pool()
    assert pool == {"hp": 0.5}


def test_avoided_exprs_returns_failed_expr_strings(repo) -> None:  # noqa: ANN001
    repo.record_brain_sim("hf", "rank(volume)", wq_alpha_id=None, region="USA",
                          universe="TOP3000", sharpe=0.0, fitness=0.0, turnover=0.0,
                          self_corr=None, status="failed")
    repo.record_brain_sim("hp", "rank(close)", wq_alpha_id="W", region="USA",
                          universe="TOP3000", sharpe=1.5, fitness=1.2, turnover=0.2,
                          self_corr=0.3, status="passed")
    assert repo.avoided_exprs() == {"rank(volume)"}  # chỉ cái failed


def test_avoided_hashes_returns_failed_canonical_hashes(repo) -> None:  # noqa: ANN001
    """avoid-list theo canonical_hash (Pha 1.2): dedup cross-session bắt cả biến thể fold
    scale, không chỉ chuỗi y hệt như avoided_exprs."""
    repo.record_brain_sim("hf", "rank(volume)", wq_alpha_id=None, region="USA",
                          universe="TOP3000", sharpe=0.0, fitness=0.0, turnover=0.0,
                          self_corr=None, status="failed")
    repo.record_brain_sim("hp", "rank(close)", wq_alpha_id="W", region="USA",
                          universe="TOP3000", sharpe=1.5, fitness=1.2, turnover=0.2,
                          self_corr=0.3, status="passed")
    assert repo.avoided_hashes() == {"hf"}  # chỉ hash của cái failed


def test_brain_local_sharpe_pairs_matches_by_canonical_hash(repo) -> None:  # noqa: ANN001
    # 1 expression có cả local eval + brain sim -> ghép cặp; 1 chỉ có brain -> bỏ.
    expr_id = repo.upsert_expression("rank(close)", "hX", 2, 3, {"close"})
    from src.backtest.metrics_local import AlphaMetrics
    m = AlphaMetrics(sharpe=0.8, annual_return=0.1, turnover=0.2, max_drawdown=0.05,
                     fitness=1.0, per_year_sharpe={2021: 1.0}, weight_concentration=0.05)
    repo.record_evaluation(expr_id, "{}", "default", m, 0.0, "passed", [], 42)
    repo.record_brain_sim("hX", "rank(close)", wq_alpha_id="W", region="USA",
                          universe="TOP3000", sharpe=1.6, fitness=1.3, turnover=0.2,
                          self_corr=0.3, status="passed")
    repo.record_brain_sim("hNoLocal", "open", wq_alpha_id="W2", region="USA",
                          universe="TOP3000", sharpe=2.0, fitness=2.0, turnover=0.1,
                          self_corr=0.1, status="passed")  # không có local eval -> bỏ
    pairs = repo.brain_local_sharpe_pairs()
    assert pairs == [(0.8, 1.6)]  # chỉ hX ghép được (local 0.8, brain 1.6)
