# tests/unit/test_minibrain_repository.py
"""Test MiniBrainRepository: upsert_expression dedup, record_evaluation (pass&fail),
load_pool/save_pool_pnl round-trip, dead_field, result_cache hit/miss, top_n."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.metrics_local import AlphaMetrics
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


@pytest.fixture
def repo():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    session_factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(session_factory)


def _metrics(sharpe=1.5) -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=sharpe, annual_return=0.1, turnover=0.2, max_drawdown=-0.05,
        fitness=2.0, per_year_sharpe={2021: 1.2, 2022: 1.8}, weight_concentration=0.05,
    )


def _cfg_json() -> str:
    cfg = PortfolioConfig(neutralization=Neutralization.SECTOR)
    return json.dumps({"neutralization": cfg.neutralization.name, "decay": cfg.decay,
                        "truncation": cfg.truncation, "scale_book": cfg.scale_book,
                        "delay": cfg.delay})


def test_upsert_expression_dedups_by_canonical_hash(repo):
    id1 = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    id2 = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    assert id1 == id2


def test_upsert_expression_distinct_hash_creates_new_row(repo):
    id1 = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    id2 = repo.upsert_expression("open", "hash2", depth=1, complexity=1, fields={"open"})
    assert id1 != id2


def test_record_evaluation_passed_stores_full_metrics(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    eval_id = repo.record_evaluation(
        expr_id, _cfg_json(), "2020..2021", _metrics(), self_corr_max=0.1,
        status="passed", fail_reasons=[], seed=42,
    )
    assert isinstance(eval_id, int)


def test_record_evaluation_failed_stores_reasons_without_metrics(repo):
    expr_id = repo.upsert_expression("bad(", "hash_bad", depth=0, complexity=0, fields=set())
    eval_id = repo.record_evaluation(
        expr_id, _cfg_json(), "2020..2021", metrics=None, self_corr_max=None,
        status="invalid", fail_reasons=["parse lỗi: unexpected token"], seed=None,
    )
    assert isinstance(eval_id, int)


def test_record_evaluation_upsert_same_key_updates_not_duplicates(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(1.0), 0.1, "passed", [], 1)
    repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(2.0), 0.1, "passed", [], 1)
    cached = repo.result_cache_get("hash1", _cfg_json(), "w1")
    assert cached is not None
    assert cached.sharpe == pytest.approx(2.0)  # ghi đè, không nhân đôi


def test_save_and_load_pool_pnl_roundtrip(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    eval_id = repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(), 0.1, "passed", [], 1)
    dates = np.array(["2021-01-01", "2021-01-02", "2021-01-03"], dtype="datetime64[D]")
    pnl = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    repo.save_pool_pnl(eval_id, dates, pnl)
    pool = repo.load_pool()
    assert eval_id in pool
    loaded_dates, loaded_pnl = pool[eval_id]
    assert np.array_equal(loaded_dates, dates)
    np.testing.assert_allclose(loaded_pnl, pnl)


def test_load_pool_returns_writeable_array_for_inplace_ops(repo):
    """Phase 6 (max_corr trên pool) cần thao tác in-place (vd demean `arr -= mean`) trên
    mảng pnl trả về từ load_pool; np.frombuffer() trần trả mảng read-only và sẽ raise
    ValueError khi bị trừ in-place — load_pool phải trả bản pnl ghi-được (.copy())."""
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    eval_id = repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(), 0.1, "passed", [], 1)
    dates = np.array(["2021-01-01", "2021-01-02", "2021-01-03"], dtype="datetime64[D]")
    pnl = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    repo.save_pool_pnl(eval_id, dates, pnl)

    pool = repo.load_pool()

    _, loaded_pnl = pool[eval_id]
    assert loaded_pnl.flags.writeable is True
    loaded_pnl -= 1.0  # thao tác in-place kiểu Phase 6 max_corr; không được raise
    np.testing.assert_allclose(loaded_pnl, pnl - 1.0)


def test_load_pool_roundtrips_dates_and_pnl_as_tuple(repo):
    """Khóa hành vi: load_pool trả {evaluation_id: (dates, pnl)} đúng format mà
    PoolCorrelation.__init__ (Task 6.1) mong đợi — kể cả khi 2 alpha trong pool có lịch
    sử dài-ngắn khác nhau (alpha A 3 ngày, alpha B 10 ngày, mốc thời gian khác nhau)."""
    expr_a = repo.upsert_expression("close", "hash_a", depth=1, complexity=1, fields={"close"})
    eval_a = repo.record_evaluation(expr_a, _cfg_json(), "w1", _metrics(), 0.1, "passed", [], 1)
    dates_a = np.array(["2021-01-01", "2021-01-02", "2021-01-03"], dtype="datetime64[D]")
    pnl_a = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    repo.save_pool_pnl(eval_a, dates_a, pnl_a)

    expr_b = repo.upsert_expression("open", "hash_b", depth=1, complexity=1, fields={"open"})
    eval_b = repo.record_evaluation(expr_b, _cfg_json(), "w1", _metrics(), 0.2, "passed", [], 2)
    dates_b = np.array(
        ["2022-06-01", "2022-06-02", "2022-06-03", "2022-06-04", "2022-06-05",
         "2022-06-06", "2022-06-07", "2022-06-08", "2022-06-09", "2022-06-10"],
        dtype="datetime64[D]",
    )
    pnl_b = np.arange(10, dtype=np.float64) * 0.001
    repo.save_pool_pnl(eval_b, dates_b, pnl_b)

    pool = repo.load_pool()

    assert set(pool.keys()) == {eval_a, eval_b}

    loaded_dates_a, loaded_pnl_a = pool[eval_a]
    assert loaded_dates_a.shape == (3,)
    assert loaded_pnl_a.shape == (3,)
    assert np.array_equal(loaded_dates_a, dates_a)
    np.testing.assert_allclose(loaded_pnl_a, pnl_a)

    loaded_dates_b, loaded_pnl_b = pool[eval_b]
    assert loaded_dates_b.shape == (10,)
    assert loaded_pnl_b.shape == (10,)
    assert np.array_equal(loaded_dates_b, dates_b)
    np.testing.assert_allclose(loaded_pnl_b, pnl_b)


def test_dead_field_add_and_check(repo):
    assert repo.is_dead_field("bad_field") is False
    repo.add_dead_field("bad_field", reason="brain rejected")
    assert repo.is_dead_field("bad_field") is True


def test_dead_field_add_is_idempotent(repo):
    repo.add_dead_field("bad_field", reason="r1")
    repo.add_dead_field("bad_field", reason="r2")  # ghi đè, không lỗi PK trùng
    assert repo.is_dead_field("bad_field") is True


def test_result_cache_miss_returns_none(repo):
    assert repo.result_cache_get("never_seen_hash", _cfg_json(), "w1") is None


def test_result_cache_hit_after_passed_evaluation(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(1.7), 0.1, "passed", [], 9)
    cached = repo.result_cache_get("hash1", _cfg_json(), "w1")
    assert cached is not None
    assert cached.sharpe == pytest.approx(1.7)
    assert cached.per_year_sharpe == {2021: 1.2, 2022: 1.8}


def test_result_cache_no_hit_for_failed_evaluation(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    repo.record_evaluation(expr_id, _cfg_json(), "w1", None, None, "invalid", ["x"], None)
    assert repo.result_cache_get("hash1", _cfg_json(), "w1") is None


def test_result_cache_put_then_get(repo):
    m = _metrics(2.5)
    repo.result_cache_put(
        "hash_new", "ts_mean(close, 5)", depth=2, complexity=3, fields={"close"},
        config_json=_cfg_json(), data_window="w1", metrics=m, seed=7,
    )
    cached = repo.result_cache_get("hash_new", _cfg_json(), "w1")
    assert cached is not None
    assert cached.sharpe == pytest.approx(2.5)


def test_top_n_orders_by_sharpe_desc_passed_only(repo):
    id_a = repo.upsert_expression("a", "ha", depth=1, complexity=1, fields=set())
    id_b = repo.upsert_expression("b", "hb", depth=1, complexity=1, fields=set())
    repo.record_evaluation(id_a, _cfg_json(), "w1", _metrics(1.0), 0.1, "passed", [], 1)
    repo.record_evaluation(id_b, _cfg_json(), "w1", _metrics(3.0), 0.1, "passed", [], 1)
    top = repo.top_n(5)
    assert top[0][0] == "b"
    assert top[0][1] == pytest.approx(3.0)
