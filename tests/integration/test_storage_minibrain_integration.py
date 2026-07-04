"""Integration: parse (Phase 1 thật) -> visitors thật -> MiniBrainRepository -> ResultCache.
Không hardcode canonical_hash/depth/complexity — tính bằng visitor thật trên AST thật."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.metrics_local import AlphaMetrics
from src.cache.result_cache import ResultCache
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import CanonicalHasher, ComplexityVisitor, DepthVisitor, FieldCollector
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


def _make_repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    session_factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(session_factory)


def test_parse_visit_upsert_cache_roundtrip_with_real_ast() -> None:
    expr_string = "ts_mean(close, 5)"
    node = parse(expr_string)
    depth = node.accept(DepthVisitor())
    fields = node.accept(FieldCollector(default_registry()))
    canonical_hash = node.accept(CanonicalHasher())
    complexity = node.accept(ComplexityVisitor())

    repo = _make_repo()
    expr_id = repo.upsert_expression(expr_string, canonical_hash, depth, complexity, fields)
    assert isinstance(expr_id, int)

    cfg_json = json.dumps({"delay": 1})
    metrics = AlphaMetrics(
        sharpe=1.4, annual_return=0.09, turnover=0.25, max_drawdown=-0.03,
        fitness=1.7, per_year_sharpe={2023: 1.4}, weight_concentration=0.06,
    )
    repo.record_evaluation(
        expr_id, cfg_json, "2023..2024", metrics, self_corr_max=0.2,
        status="passed", fail_reasons=[], seed=11,
    )

    cache = ResultCache(repo)
    hit = cache.get(canonical_hash, cfg_json, "2023..2024")
    assert hit is not None
    assert hit.sharpe == pytest.approx(1.4)
    assert hit.per_year_sharpe == {2023: 1.4}


def test_failed_expression_recorded_with_reasons_not_cached() -> None:
    expr_string = "ts_mean(volume, 999)"  # window lớn -> giả định pass parse, fail gate
    node = parse(expr_string)
    depth = node.accept(DepthVisitor())
    fields = node.accept(FieldCollector(default_registry()))
    canonical_hash = node.accept(CanonicalHasher())
    complexity = node.accept(ComplexityVisitor())

    repo = _make_repo()
    expr_id = repo.upsert_expression(expr_string, canonical_hash, depth, complexity, fields)
    repo.record_evaluation(
        expr_id, json.dumps({"delay": 1}), "2023..2024", metrics=None,
        self_corr_max=None, status="failed_gate",
        fail_reasons=["self_corr 0.91 >= SELF_CORR_MAX 0.70"], seed=None,
    )

    cache = ResultCache(repo)
    assert cache.get(canonical_hash, json.dumps({"delay": 1}), "2023..2024") is None


def test_dedup_real_canonical_hash_for_commutative_expression() -> None:
    """CanonicalHasher (Phase 1) sort commutative args -> 'add(close, volume)' và
    'add(volume, close)' phải cho CÙNG canonical_hash -> upsert_expression dedup."""
    repo = _make_repo()
    hash_a = parse("add(close, volume)").accept(CanonicalHasher())
    hash_b = parse("add(volume, close)").accept(CanonicalHasher())
    assert hash_a == hash_b  # xác nhận tiền đề trước khi test dedup qua repo

    id1 = repo.upsert_expression("add(close, volume)", hash_a, 2, 3, {"close", "volume"})
    id2 = repo.upsert_expression("add(volume, close)", hash_b, 2, 3, {"close", "volume"})
    assert id1 == id2
