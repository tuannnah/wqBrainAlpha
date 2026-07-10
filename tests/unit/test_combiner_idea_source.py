"""TDD cho CombinerIdeaSource: gom tín hiệu (batch + DB) và các nhánh passthrough.

Phần lớn test dùng data/config giả (KHÔNG chạm scorer thật) — chỉ test logic augment/gom
tín hiệu; đường chấm+lọc combo đã test riêng ở test_combine_stage.py. Riêng nhóm test
"local-usable" (Task 7, sửa bug 0-combo) dùng `small_panel` THẬT để xác nhận candidate
curated (pnl rỗng) được backtest local thật ra tín hiệu con, và dùng monkeypatch
`_score_one_full` để lái combine_stage tất định khi kiểm next_batch sinh combo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.app.closed_loop_adapters import CombinerIdeaSource
from src.backtest.config import PortfolioConfig
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


# ------------------------- Task 7: sửa bug "0 combo" -------------------------
# Root cause đã kiểm chứng: CuratedIdeaSource/AltDataIdeaSource yield core với
# metrics=None, pnl rỗng (chưa backtest) -> `_signals` cũ vứt thẳng, DB rỗng ở phiên mới
# -> < n_min -> next_batch không bao giờ sinh combo. Sửa: candidate pnl rỗng NHƯNG expr
# local-usable (field nằm trong panel) thì backtest NGAY bằng đúng đường _score_one_full
# (parse->eval->portfolio->backtest->metrics, dùng chung với tuner/generate_many) để lấy
# pnl/fitness làm tín hiệu con; alt-data (field ngoài panel) vẫn bị loại như cũ.


def _fake_score_one_full(sub_pnls: dict[str, np.ndarray], dates: np.ndarray):
    """Scorer giả tất định: expr có trong `sub_pnls` -> trả đúng pnl/fitness thấp (0.5,
    pass); expr khác (combo do combine_stage tự dựng) -> fitness cao (5.0, pass) để chắc
    chắn vượt tín hiệu con -> combo được giữ. Chữ ký khớp `_score_one_full(expr, cfg, data,
    pool=None)` để monkeypatch thay thế xuyên suốt cả `_signals` lẫn `_score_fn`."""

    @dataclass
    class _FakeMetrics:
        fitness: float

    @dataclass
    class _FakeVerdict:
        passed: bool

    @dataclass
    class _FakeResult:
        metrics: _FakeMetrics
        verdict: _FakeVerdict
        pnl: np.ndarray
        dates: np.ndarray

    combo_pnl = np.ones(len(dates), dtype=np.float64)

    def fake(expr: str, cfg, data, pool=None):
        if expr in sub_pnls:
            return _FakeResult(_FakeMetrics(0.5), _FakeVerdict(True), sub_pnls[expr], dates)
        return _FakeResult(_FakeMetrics(5.0), _FakeVerdict(True), combo_pnl, dates)

    return fake


def test_curated_local_usable_duoc_backtest_thanh_tin_hieu(small_panel):  # noqa: ANN001
    """(a) Candidate kiểu curated (local-usable, pnl rỗng, metrics=None) PHẢI trở thành
    tín hiệu con qua backtest local thật — panel THẬT (small_panel), không mock."""
    empty_pnl = np.zeros(0, dtype=np.float64)
    empty_dates = np.zeros(0, dtype="datetime64[ns]")
    curated_a = ShortlistCandidate(
        expr="rank(ts_delta(close, 5))", metrics=None, pnl=empty_pnl, dates=empty_dates,
    )
    curated_b = ShortlistCandidate(
        expr="rank(ts_delta(volume, 5))", metrics=None, pnl=empty_pnl, dates=empty_dates,
    )
    src = CombinerIdeaSource(
        fallback=_FakeFallback([curated_a, curated_b]), data=small_panel,
        repo=_FakeRepo([]), config=PortfolioConfig(decay=0, truncation=0.10), registry=None,
    )

    sigs = src._signals([curated_a, curated_b])

    exprs = {s.expr: s for s in sigs}
    assert set(exprs) == {"rank(ts_delta(close, 5))", "rank(ts_delta(volume, 5))"}
    for sig in exprs.values():
        assert sig.source == "run"
        assert sig.pnl.size > 0  # backtest local thật đã chạy, không còn pnl rỗng


def test_alt_data_expr_van_bi_loai_khoi_signals(small_panel):  # noqa: ANN001
    """(b) Expr ngoài panel (alt-data, local_usable=False) vẫn bị loại — không backtest
    được local nên không thể thành tín hiệu con."""
    empty_pnl = np.zeros(0, dtype=np.float64)
    empty_dates = np.zeros(0, dtype="datetime64[ns]")
    alt = ShortlistCandidate(
        expr="rank(ts_delta(anl4_afv4_eps_mean, 5))", metrics=None,
        pnl=empty_pnl, dates=empty_dates,
    )
    src = CombinerIdeaSource(
        fallback=_FakeFallback([alt]), data=small_panel,
        repo=_FakeRepo([]), config=PortfolioConfig(), registry=None,
    )

    assert src._signals([alt]) == []


def test_next_batch_du_n_min_sinh_combo_va_ghi_instrumentation(monkeypatch):
    """(a)+(c): đủ n_min tín hiệu con local-usable -> next_batch sinh >=1 combo appended;
    đồng thời `last_stats` (instrumentation) ghi đúng số đếm run/db/total/n_combos."""
    import src.app.closed_loop_adapters as cla

    rng = np.random.default_rng(3)
    dates = DATES.copy()
    e1, e2 = "rank(ts_delta(close, 5))", "rank(ts_delta(volume, 5))"
    sub_pnls = {e1: rng.normal(size=50), e2: rng.normal(size=50)}
    monkeypatch.setattr(cla, "_score_one_full", _fake_score_one_full(sub_pnls, dates))

    empty_pnl = np.zeros(0, dtype=np.float64)
    empty_dates = np.zeros(0, dtype="datetime64[ns]")
    curated_a = ShortlistCandidate(expr=e1, metrics=None, pnl=empty_pnl, dates=empty_dates)
    curated_b = ShortlistCandidate(expr=e2, metrics=None, pnl=empty_pnl, dates=empty_dates)
    src = cla.CombinerIdeaSource(
        fallback=_FakeFallback([curated_a, curated_b]), data=object(),
        repo=_FakeRepo([]), config=object(), registry=None, tau=0.9,
    )

    out = src.next_batch()

    assert len(out) == 3  # 2 batch gốc + >=1 combo
    combo = out[-1]
    assert combo.metrics is not None
    assert combo.expr not in (e1, e2)
    assert src.last_stats["n_run_signals"] == 2
    assert src.last_stats["n_db_signals"] == 0
    assert src.last_stats["total_signals"] == 2
    assert src.last_stats["skipped"] is False
    assert src.last_stats["n_combos"] == 1


def test_next_batch_skip_van_ghi_instrumentation():
    """(c): nhánh skip (< n_min) vẫn ghi last_stats để chẩn đoán được vì sao 0 combo."""
    empty = np.zeros(0, dtype=np.float64)
    curated = ShortlistCandidate(expr="x", metrics=None, pnl=empty, dates=empty)
    src = _src([curated], [])

    out = src.next_batch()

    assert out == [curated]
    assert src.last_stats["skipped"] is True
    assert src.last_stats["total_signals"] < src.n_min
