"""Test migrate_all copy đủ bảng MiniBrain mới (Expression/Evaluation/PoolPnl/DeadField/
BrainRecord), tôn trọng thứ tự FK, idempotent khi chạy lại."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.db import init_db
from src.storage.migrate import migrate_all
from src.storage.models import (
    BrainRecordModel,
    DeadFieldModel,
    EvaluationModel,
    ExpressionModel,
    PoolPnlModel,
)


def _seeded_source():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    Session = sessionmaker(bind=engine, future=True)
    s = Session()
    expr = ExpressionModel(
        canonical_hash="h1", expr_string="close", depth=1, complexity=1, fields_json="[]",
    )
    s.add(expr)
    s.commit()
    ev = EvaluationModel(
        expression_id=expr.id, config_json="{}", data_window="w", status="passed",
    )
    s.add(ev)
    s.commit()
    s.add(PoolPnlModel(evaluation_id=ev.id, dates_blob=b"\x00", pnl_blob=b"\x00"))
    s.add(DeadFieldModel(name="bad_field", reason="rejected"))
    s.add(BrainRecordModel(expr_string="close", brain_sharpe=1.0, submitted=1))
    s.commit()
    s.close()
    return engine


def test_migrate_all_copies_minibrain_tables():
    src = _seeded_source()
    dst = create_engine("sqlite:///:memory:", future=True)
    counts = migrate_all(src, dst)
    assert counts["expressions"] == 1
    assert counts["evaluations"] == 1
    assert counts["pool_pnl"] == 1
    assert counts["dead_fields_minibrain"] == 1
    assert counts["brain_records"] == 1


def test_migrate_all_is_idempotent_on_minibrain_tables():
    src = _seeded_source()
    dst = create_engine("sqlite:///:memory:", future=True)
    migrate_all(src, dst)
    counts2 = migrate_all(src, dst)  # chạy lại — không lỗi unique/FK
    assert counts2["expressions"] == 1
