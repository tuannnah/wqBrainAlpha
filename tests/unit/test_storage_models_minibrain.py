"""Test models MiniBrain mới: bảng tạo đúng cột, FK, unique constraint (B11 schema)."""

from __future__ import annotations

from sqlalchemy import create_engine, inspect

from src.storage.db import init_db
from src.storage.models import (
    BrainRecordModel,
    DeadFieldModel,
    EvaluationModel,
    ExpressionModel,
    PoolPnlModel,
)


def _fresh_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    return engine


def test_all_minibrain_tables_created():
    engine = _fresh_engine()
    tables = set(inspect(engine).get_table_names())
    assert {"expressions", "evaluations", "pool_pnl", "dead_fields_minibrain",
            "brain_records"} <= tables


def test_expression_table_columns():
    engine = _fresh_engine()
    cols = {c["name"] for c in inspect(engine).get_columns("expressions")}
    assert cols == {"id", "canonical_hash", "expr_string", "depth", "complexity",
                     "fields_json", "created_at"}


def test_evaluation_table_columns_and_fk():
    engine = _fresh_engine()
    cols = {c["name"] for c in inspect(engine).get_columns("evaluations")}
    assert cols == {
        "id", "expression_id", "config_json", "data_window", "sharpe", "annual_return",
        "turnover", "max_drawdown", "fitness", "weight_concentration", "per_year_json",
        "self_corr_max", "status", "fail_reasons", "seed", "created_at",
    }
    fks = inspect(engine).get_foreign_keys("evaluations")
    assert any(fk["referred_table"] == "expressions" for fk in fks)


def test_evaluation_unique_constraint_blocks_duplicate():
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    engine = _fresh_engine()
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    expr = ExpressionModel(
        canonical_hash="h1", expr_string="close", depth=1, complexity=1, fields_json="[]",
    )
    session.add(expr)
    session.commit()
    session.add(EvaluationModel(
        expression_id=expr.id, config_json="{}", data_window="2020..2021", status="passed",
    ))
    session.commit()
    session.add(EvaluationModel(
        expression_id=expr.id, config_json="{}", data_window="2020..2021", status="passed",
    ))
    try:
        session.commit()
        raised = False
    except IntegrityError:
        session.rollback()
        raised = True
    finally:
        session.close()
    assert raised


def test_pool_pnl_fk_to_evaluation():
    engine = _fresh_engine()
    cols = {c["name"] for c in inspect(engine).get_columns("pool_pnl")}
    assert cols == {"evaluation_id", "dates_blob", "pnl_blob"}
    fks = inspect(engine).get_foreign_keys("pool_pnl")
    assert any(fk["referred_table"] == "evaluations" for fk in fks)


def test_dead_field_and_brain_record_columns():
    engine = _fresh_engine()
    dead_cols = {c["name"] for c in inspect(engine).get_columns("dead_fields_minibrain")}
    assert dead_cols == {"name", "reason", "created_at"}
    brain_cols = {c["name"] for c in inspect(engine).get_columns("brain_records")}
    assert brain_cols == {
        "id", "expr_string", "brain_sharpe", "brain_fitness", "brain_turnover",
        "brain_self_corr", "submitted", "created_at",
    }


def test_canonical_hash_is_unique():
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    engine = _fresh_engine()
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    session.add(ExpressionModel(
        canonical_hash="dup", expr_string="close", depth=1, complexity=1, fields_json="[]",
    ))
    session.commit()
    session.add(ExpressionModel(
        canonical_hash="dup", expr_string="open", depth=1, complexity=1, fields_json="[]",
    ))
    try:
        session.commit()
        raised = False
    except IntegrityError:
        session.rollback()
        raised = True
    finally:
        session.close()
    assert raised
