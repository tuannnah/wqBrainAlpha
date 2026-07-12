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
        # Fix 1 (Task 2): db_signals nay là [(expr, sharpe_brain)] — khớp
        # `repo.brain_proven_signals(min_sharpe)` thật (KHÔNG còn (expr, dates, pnl, fitness)
        # của `good_signals_for_combine` cũ, đã bị xoá vì fitness LOCAL không đáng tin, xem
        # `logs/diag_combiner_20260712.md`).
        self._db = db_signals

    def brain_proven_signals(self, min_sharpe=0.8, limit=50):
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


def test_gom_tin_hieu_run_va_db_backtest_local_score_la_sharpe_brain(small_panel):  # noqa: ANN001
    """Fix 1 (Task 2): nguồn db nay qua `repo.brain_proven_signals` — CombinerIdeaSource tự
    backtest local từng expr (qua `_score_one_full`) để lấy PnL, và SCORE gán cho SubSignal
    là sharpe BRAIN thật (từ DB), KHÔNG PHẢI fitness đo được từ backtest local đó."""
    rng = np.random.default_rng(0)
    valid = _cand("rank(ts_delta(close, 5))", rng.normal(size=50), 1.0)
    empty_pnl = np.zeros(0, dtype=np.float64)
    empty_dates = np.zeros(0, dtype="datetime64[ns]")
    # field ngoài panel (như test alt-data khác) -> local_usable=False -> loại, không backtest.
    curated = ShortlistCandidate(
        expr="rank(ts_delta(anl4_afv4_eps_mean, 5))", metrics=None,
        pnl=empty_pnl, dates=empty_dates,
    )
    db_expr = "rank(ts_delta(volume, 5))"
    db_sharpe = 1.9  # sharpe Brain giả — cố tình khác xa fitness local thật để phân biệt nguồn.
    src = CombinerIdeaSource(
        fallback=_FakeFallback([valid, curated]), data=small_panel,
        repo=_FakeRepo([(db_expr, db_sharpe)]),
        config=PortfolioConfig(decay=0, truncation=0.10), registry=None,
    )

    sigs = src._signals([valid, curated])

    by_expr = {s.expr: s for s in sigs}
    assert set(by_expr) == {"rank(ts_delta(close, 5))", db_expr}
    assert by_expr["rank(ts_delta(close, 5))"].source == "run"
    assert by_expr["rank(ts_delta(close, 5))"].score == 1.0  # candidate.metrics.fitness, không đổi
    assert by_expr[db_expr].source == "db"
    assert by_expr[db_expr].score == db_sharpe  # sharpe Brain -- KHÔNG PHẢI fitness local backtest
    assert by_expr[db_expr].pnl.size > 0  # backtest local thật đã chạy để lấy PnL
    assert "rank(ts_delta(anl4_afv4_eps_mean, 5))" not in by_expr  # curated pnl rỗng bị loại


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
        sharpe: float  # Fix 4: điểm-nộp cần cả sharpe — gán = fitness để giữ nguyên ý định
        # gốc của fake này (combo "mạnh hơn" -> điểm-nộp cao hơn, không đổi hành vi test cũ).

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
            return _FakeResult(_FakeMetrics(0.5, 0.5), _FakeVerdict(True), sub_pnls[expr], dates)
        return _FakeResult(_FakeMetrics(5.0, 5.0), _FakeVerdict(True), combo_pnl, dates)

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


def test_next_batch_khong_con_dung_repo_load_pool(monkeypatch):
    """Fix 2 (Task 2): gate pool không còn là `repo.load_pool()` (1321+ eval LOCAL bão hòa —
    đo được giết oan combo self-corr 0.70-0.86 trong khi Brain thật đo 0.40-0.46, xem
    `logs/diag_combiner_20260712.md`) — next_batch phải dùng score_fn_factory (pool = tín
    hiệu Brain-proven NGOÀI combo) thay vì gọi `load_pool`."""
    import src.app.closed_loop_adapters as cla

    rng = np.random.default_rng(4)
    dates = DATES.copy()
    e1, e2 = "rank(ts_delta(close, 5))", "rank(ts_delta(volume, 5))"
    sub_pnls = {e1: rng.normal(size=50), e2: rng.normal(size=50)}
    monkeypatch.setattr(cla, "_score_one_full", _fake_score_one_full(sub_pnls, dates))

    empty_pnl = np.zeros(0, dtype=np.float64)
    empty_dates = np.zeros(0, dtype="datetime64[ns]")
    curated_a = ShortlistCandidate(expr=e1, metrics=None, pnl=empty_pnl, dates=empty_dates)
    curated_b = ShortlistCandidate(expr=e2, metrics=None, pnl=empty_pnl, dates=empty_dates)

    class _RepoNoLoadPool(_FakeRepo):
        def load_pool(self):
            raise AssertionError("next_batch KHÔNG được gọi repo.load_pool() nữa (Fix 2)")

    src = cla.CombinerIdeaSource(
        fallback=_FakeFallback([curated_a, curated_b]), data=object(),
        repo=_RepoNoLoadPool([]), config=object(), registry=None, tau=0.9,
    )

    out = src.next_batch()  # không raise AssertionError -> load_pool() thật sự không bị gọi

    assert len(out) == 3  # 2 batch gốc + 1 combo — vẫn hoạt động bình thường


def test_next_batch_gop_drop_stats_vao_last_stats_va_log(monkeypatch):  # noqa: ANN001
    """Fix 4 (Task 2): next_batch gộp drop_stats (depth/gate/not_better/greedy_empty) từ
    combine_stage vào last_stats VÀ log dòng "Combiner drop: ..." — chẩn đoán được TẠI SAO
    0 combo mà không cần bắt log (đúng như comment gốc của `last_stats` yêu cầu)."""
    import src.app.closed_loop_adapters as cla

    rng = np.random.default_rng(6)
    dates = DATES.copy()
    e1, e2 = "rank(ts_delta(close, 5))", "rank(ts_delta(volume, 5))"
    sub_pnls = {e1: rng.normal(size=50), e2: rng.normal(size=50)}

    @dataclass
    class _M:
        fitness: float
        sharpe: float

    @dataclass
    class _V:
        passed: bool

    @dataclass
    class _R:
        metrics: _M
        verdict: _V
        pnl: np.ndarray
        dates: np.ndarray

    def fake(expr, cfg, data, pool=None):
        if expr in sub_pnls:
            return _R(_M(0.5, 0.5), _V(True), sub_pnls[expr], dates)
        # combo -> điểm-nộp THẤP hơn component -> "not_better", KHÔNG được giữ.
        return _R(_M(0.1, 0.1), _V(True), np.ones(len(dates)), dates)

    monkeypatch.setattr(cla, "_score_one_full", fake)

    empty_pnl = np.zeros(0, dtype=np.float64)
    empty_dates = np.zeros(0, dtype="datetime64[ns]")
    curated_a = ShortlistCandidate(expr=e1, metrics=None, pnl=empty_pnl, dates=empty_dates)
    curated_b = ShortlistCandidate(expr=e2, metrics=None, pnl=empty_pnl, dates=empty_dates)
    src = cla.CombinerIdeaSource(
        fallback=_FakeFallback([curated_a, curated_b]), data=object(),
        repo=_FakeRepo([]), config=object(), registry=None, tau=0.9,
    )

    # Bắt log loguru bằng sink riêng (KHÔNG đụng sink stderr WARNING toàn phiên ở conftest)
    # để xác nhận dòng "Combiner drop: ..." thật sự được emit, không chỉ suy từ last_stats.
    logs: list[str] = []
    sink_id = cla.logger.add(logs.append, level="INFO")
    try:
        out = src.next_batch()
    finally:
        cla.logger.remove(sink_id)

    assert len(out) == 2  # combo bị loại (not_better) -> chỉ batch gốc
    assert src.last_stats["n_combos"] == 0
    assert src.last_stats["depth"] == 0
    assert src.last_stats["gate"] == 0
    assert src.last_stats["not_better"] == 1
    assert src.last_stats["greedy_empty"] == 0
    assert any("Combiner drop" in m for m in logs)


# ------------------- Review fix: cache PnL nguồn db, không backtest lặp -------------------
# next_batch() chạy trong vòng while của closed-loop; không cache thì MỖI batch backtest lại
# TOÀN BỘ expr Brain-proven (~20s/expr) cho đúng những PnL series KHÔNG ĐỔI (panel local bất
# biến trong phiên), và danh sách chỉ tăng khi DB tích luỹ. Cache expr -> (pnl, dates) sống
# theo đời CombinerIdeaSource; kết quả ÂM (không local-usable/backtest lỗi) cũng cache để
# không thử lại vô ích.


def _dem_backtest_db(monkeypatch, cla, dem: dict[str, int], loi: set[str] | None = None):
    """Monkeypatch `_score_one_full` module-level: đếm số lần backtest theo expr; expr trong
    `loi` -> raise (mô phỏng backtest hỏng, _local_backtest phải trả None + cache âm)."""
    rng = np.random.default_rng(7)

    @dataclass
    class _M:
        fitness: float
        sharpe: float

    @dataclass
    class _V:
        passed: bool

    @dataclass
    class _R:
        metrics: _M
        verdict: _V
        pnl: np.ndarray
        dates: np.ndarray

    def fake(expr, cfg, data, pool=None):
        dem[expr] = dem.get(expr, 0) + 1
        if loi and expr in loi:
            raise ValueError("backtest hỏng (giả lập)")
        return _R(_M(0.5, 0.5), _V(True), rng.normal(size=50), DATES.copy())

    monkeypatch.setattr(cla, "_score_one_full", fake)


def test_cache_khong_backtest_lai_expr_db_o_lan_hai(monkeypatch):
    """(a) Gọi next_batch() 2 lần, repo trả CÙNG danh sách -> expr db chỉ backtest 1 lần
    (lần 2 lấy từ cache, 0 backtest thêm)."""
    import src.app.closed_loop_adapters as cla

    dem: dict[str, int] = {}
    _dem_backtest_db(monkeypatch, cla, dem)

    run_cand = _cand("rank(ts_delta(close, 5))", np.random.default_rng(8).normal(size=50), 1.0)
    db_expr = "rank(ts_delta(volume, 5))"
    # n_min=3 > tổng tín hiệu (2) -> combine_stage không chạy, chỉ đo đường _signals.
    src = cla.CombinerIdeaSource(
        fallback=_FakeFallback([run_cand]), data=object(),
        repo=_FakeRepo([(db_expr, 1.2)]), config=object(), registry=None, n_min=3,
    )

    src.next_batch()
    assert dem.get(db_expr, 0) == 1  # lần 1: backtest đúng 1 lần

    src.next_batch()
    assert dem.get(db_expr, 0) == 1  # lần 2: 0 backtest thêm — lấy từ cache

    # Tín hiệu db vẫn xuất hiện đầy đủ ở lần 2 (cache trả pnl, không phải bỏ qua expr).
    assert src.last_stats["n_db_signals"] == 1


def test_cache_chi_backtest_expr_moi_xuat_hien(monkeypatch):
    """(b) Lần 2 repo trả thêm expr MỚI -> chỉ backtest đúng expr mới; expr lỗi cache ÂM,
    không thử lại ở lần sau."""
    import src.app.closed_loop_adapters as cla

    dem: dict[str, int] = {}
    e_cu, e_moi, e_loi = (
        "rank(ts_delta(volume, 5))", "rank(ts_delta(close, 20))", "rank(ts_delta(open, 5))",
    )
    _dem_backtest_db(monkeypatch, cla, dem, loi={e_loi})

    run_cand = _cand("rank(ts_delta(close, 5))", np.random.default_rng(9).normal(size=50), 1.0)

    class _RepoDoiDanhSach(_FakeRepo):
        """Lần 1 trả [e_cu, e_loi]; từ lần 2 thêm e_moi."""

        def __init__(self):
            self.n_goi = 0

        def brain_proven_signals(self, min_sharpe=0.8, limit=50):
            self.n_goi += 1
            base = [(e_cu, 1.2), (e_loi, 1.1)]
            return base if self.n_goi == 1 else base + [(e_moi, 1.0)]

        def load_pool(self):
            return {}

    src = cla.CombinerIdeaSource(
        fallback=_FakeFallback([run_cand]), data=object(),
        repo=_RepoDoiDanhSach(), config=object(), registry=None, n_min=99,
    )

    src.next_batch()
    assert dem == {e_cu: 1, e_loi: 1}  # lần 1: backtest cả hai (e_loi raise -> cache âm)

    src.next_batch()
    # Lần 2: CHỈ backtest e_moi; e_cu lấy cache, e_loi cache âm không thử lại.
    assert dem == {e_cu: 1, e_loi: 1, e_moi: 1}
    assert src.last_stats["n_db_signals"] == 2  # e_cu (cache) + e_moi; e_loi bị loại


def test_db_limit_truyen_xuong_brain_proven_signals():
    """(c) `db_limit` của CombinerIdeaSource truyền xuống `brain_proven_signals(limit=...)`."""
    nhan: list[int] = []

    class _RepoGhiLimit(_FakeRepo):
        def brain_proven_signals(self, min_sharpe=0.8, limit=50):
            nhan.append(limit)
            return []

    src = CombinerIdeaSource(
        fallback=_FakeFallback([_cand("x", np.zeros(50), 1.0)]), data=object(),
        repo=_RepoGhiLimit([]), config=object(), registry=None, db_limit=7,
    )
    src.next_batch()
    assert nhan == [7]


def test_next_batch_skip_van_ghi_instrumentation():
    """(c): nhánh skip (< n_min) vẫn ghi last_stats để chẩn đoán được vì sao 0 combo."""
    empty = np.zeros(0, dtype=np.float64)
    curated = ShortlistCandidate(expr="x", metrics=None, pnl=empty, dates=empty)
    src = _src([curated], [])

    out = src.next_batch()

    assert out == [curated]
    assert src.last_stats["skipped"] is True
    assert src.last_stats["total_signals"] < src.n_min
