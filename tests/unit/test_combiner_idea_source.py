"""TDD cho CombinerIdeaSource: gom tín hiệu (batch + DB) và các nhánh passthrough.

Không chạm scorer thật (_score_one_full) — chỉ test logic augment/gom tín hiệu; đường
chấm+lọc combo đã test riêng ở test_combine_stage.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.app.closed_loop_adapters import CombinerIdeaSource
from src.pipeline.shortlist import ShortlistCandidate

DATES = np.arange(50)


@dataclass
class _M:
    fitness: float


class _FakeFallback:
    def __init__(self, batch):
        self._batch = batch

    def next_batch(self):
        return self._batch


class _FakeRepo:
    def __init__(self, db_signals):
        self._db = db_signals

    def good_signals_for_combine(self, limit=50):
        return self._db

    def load_pool(self):
        return {}


def _cand(expr, pnl, fitness):
    return ShortlistCandidate(expr=expr, metrics=_M(fitness), pnl=pnl, dates=DATES.copy())


def _src(batch, db_signals):
    return CombinerIdeaSource(
        fallback=_FakeFallback(batch), data=object(), repo=_FakeRepo(db_signals),
        config=object(), registry=None,
    )


def test_batch_rong_tra_rong_khong_dung_toi_repo():
    src = _src([], [])
    assert src.next_batch() == []


def test_thieu_tin_hieu_tra_batch_nguyen():
    # Batch chỉ có candidate PnL rỗng (kiểu curated) + DB rỗng -> < n_min -> trả batch nguyên.
    empty = np.zeros(0, dtype=np.float64)
    curated = ShortlistCandidate(expr="x", metrics=None, pnl=empty, dates=empty)
    src = _src([curated], [])
    assert src.next_batch() == [curated]


def test_gom_tin_hieu_run_va_db_bo_curated_pnl_rong():
    rng = np.random.default_rng(0)
    valid = _cand("rank(ts_delta(close, 5))", rng.normal(size=50), 1.0)
    empty = np.zeros(0, dtype=np.float64)
    curated = ShortlistCandidate(expr="curated", metrics=None, pnl=empty, dates=empty)
    db = [("rank(ts_delta(close, 20))", DATES.copy(), rng.normal(size=50), 1.2)]
    src = _src([valid, curated], db)

    sigs = src._signals([valid, curated])

    exprs = {s.expr: s.source for s in sigs}
    assert exprs == {"rank(ts_delta(close, 5))": "run", "rank(ts_delta(close, 20))": "db"}
    assert "curated" not in exprs  # candidate PnL rỗng bị loại
