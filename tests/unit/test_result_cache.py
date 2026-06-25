# tests/unit/test_result_cache.py
"""Test ResultCache: bọc MiniBrainRepository, hit sau put, miss khi key khác."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.metrics_local import AlphaMetrics
from src.cache.result_cache import ResultCache
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


@pytest.fixture
def cache():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    session_factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    repo = MiniBrainRepository(session_factory)
    return ResultCache(repo)


def _metrics() -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.3, annual_return=0.08, turnover=0.3, max_drawdown=-0.02,
        fitness=1.9, per_year_sharpe={2022: 1.1}, weight_concentration=0.04,
    )


def test_get_returns_none_on_cold_cache(cache):
    assert cache.get("hash_x", "{}", "w1") is None


def test_put_then_get_hits(cache):
    cache.put(
        "hash_x", "ts_mean(close, 5)", depth=2, complexity=3, fields={"close"},
        config_json="{}", data_window="w1", metrics=_metrics(), seed=3,
    )
    hit = cache.get("hash_x", "{}", "w1")
    assert hit is not None
    assert hit.sharpe == pytest.approx(1.3)
    assert hit.per_year_sharpe == {2022: 1.1}


def test_different_config_json_is_a_miss(cache):
    cache.put(
        "hash_x", "close", depth=1, complexity=1, fields={"close"},
        config_json="{}", data_window="w1", metrics=_metrics(), seed=None,
    )
    assert cache.get("hash_x", '{"decay": 5}', "w1") is None


def test_different_data_window_is_a_miss(cache):
    cache.put(
        "hash_x", "close", depth=1, complexity=1, fields={"close"},
        config_json="{}", data_window="w1", metrics=_metrics(), seed=None,
    )
    assert cache.get("hash_x", "{}", "w2") is None
