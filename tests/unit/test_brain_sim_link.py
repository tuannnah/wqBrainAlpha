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
