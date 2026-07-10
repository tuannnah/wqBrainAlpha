"""Test bảng tried_hashes: hash GỐC (pre-tune) đã thử — avoid-list cross-session đúng
không gian hash (Task 6 fix). Tách khỏi BrainSimLinkModel/EvaluationModel — không được lẫn
vào brain_local_sharpe_pairs (giữ calibration sạch)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.db import init_db
from src.storage.models import TriedHashModel
from src.storage.repository import MiniBrainRepository


@pytest.fixture
def repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    sf = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(sf)


def test_tried_hashes_table_created_and_insertable(repo) -> None:  # noqa: ANN001
    """init_db (create_all) tạo bảng tried_hashes tự động — không cần migration thủ công vì
    đây là bảng MỚI (không phải cột thêm vào bảng cũ)."""
    session = repo.session_factory()
    try:
        session.add(TriedHashModel(hash="h1"))
        session.commit()
        got = session.query(TriedHashModel).filter_by(hash="h1").one()
        assert got.hash == "h1"
        assert got.created_at is not None
    finally:
        session.close()


def test_record_avoided_hash_then_read_back(repo) -> None:  # noqa: ANN001
    repo.record_avoided_hash("hash_a")
    repo.record_avoided_hash("hash_b")
    assert repo.avoided_hashes_original() == {"hash_a", "hash_b"}


def test_record_avoided_hash_idempotent(repo) -> None:  # noqa: ANN001
    """Ghi cùng hash 2 lần không lỗi (merge theo PK), không nhân đôi."""
    repo.record_avoided_hash("dup")
    repo.record_avoided_hash("dup")
    session = repo.session_factory()
    try:
        count = session.query(TriedHashModel).filter_by(hash="dup").count()
        assert count == 1
    finally:
        session.close()
    assert repo.avoided_hashes_original() == {"dup"}


def test_avoided_hashes_original_empty_when_none_recorded(repo) -> None:  # noqa: ANN001
    assert repo.avoided_hashes_original() == set()


def test_tried_hashes_isolated_from_calibration_join(repo) -> None:  # noqa: ANN001
    """record_avoided_hash KHÔNG được tạo row BrainSimLinkModel/EvaluationModel giả — join
    calibration (brain_local_sharpe_pairs) phải không bị nhiễm hash gốc pre-tune."""
    repo.record_avoided_hash("some_original_hash")
    assert repo.brain_local_sharpe_pairs() == []
    assert repo.brain_pnl_pool() == {}
    assert repo.load_brain_sims() == []
