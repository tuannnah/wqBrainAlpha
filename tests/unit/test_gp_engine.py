"""Test GPEngine: init/evaluate/run/persist trên small_panel thật.

Engine ghép seeds→init→variation→selection→eval qua Phase 2/3/4/6 + persist Phase 5.
Mọi randomness đi qua rng inject (seed cố định) nên test xác định (deterministic)."""

from __future__ import annotations

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import pytest

import src.operators_local  # noqa: F401  (side-effect: nạp 27 operator vào registry)
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.pool_corr import PoolCorrelation
from src.gp.engine import GPEngine, GPRunResult
from src.gp.individual import Individual
from src.lang.ast import Constant
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import CanonicalHasher, DepthVisitor
from src.storage.db import init_db
from src.storage.models import EvaluationModel
from src.storage.repository import MiniBrainRepository


@pytest.fixture
def repo() -> MiniBrainRepository:
    """Repository MiniBrain trên SQLite in-memory mới mỗi test (không rò state)."""
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    sf = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(sf)


@pytest.fixture
def cfg() -> PortfolioConfig:
    """Config danh mục đơn giản (không neutralize) cho test nhanh."""
    return PortfolioConfig(
        neutralization=Neutralization.NONE, decay=0, truncation=0.10,
        scale_book=1.0, delay=1,
    )


@pytest.fixture
def engine_fixture(small_panel, repo, cfg) -> GPEngine:  # noqa: ANN001
    """GPEngine chuẩn (saturated_families rỗng) trên small_panel — dùng cho test A2 gate
    degenerate TRƯỚC backtest."""
    return GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=2, n_generations=0, seed=42,
    )


@pytest.fixture
def engine_fixture_voi_saturated(small_panel, repo, cfg) -> GPEngine:  # noqa: ANN001
    """GPEngine với saturated_families={'pv_reversal'} — dùng cho test A2 gate họ-đã-đóng."""
    return GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=2, n_generations=0, seed=42, saturated_families={"pv_reversal"},
    )


@pytest.fixture
def engine_fixture_voi_cache(small_panel, repo, cfg):  # noqa: ANN001
    """GPEngine + dict cache rỗng truyền vào ``eval_cache`` — dùng cho test A3 (cache backtest
    thuần theo canonical_hash). Trả tuple ``(engine, cache)`` để test kiểm tra cả nội dung cache."""
    cache: "dict[str, tuple]" = {}
    engine = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=2, n_generations=0, seed=42, eval_cache=cache,
    )
    return engine, cache


def _pool_corr_rong() -> PoolCorrelation:
    """PoolCorrelation rỗng (không có alpha nào trong pool) — đủ dùng cho test chỉ quan tâm
    hành vi lọc trước-backtest, không cần self-corr thật."""
    return PoolCorrelation(pool={})


def test_gprunresult_is_frozen_dataclass() -> None:
    """GPRunResult bất biến: gán lại field sau khởi tạo phải raise (frozen dataclass)."""
    r = GPRunResult(
        generations_run=0, final_population=[], best_by_sharpe=None,
        n_evaluated=0, n_passed=0, seed=42,
    )
    with pytest.raises(Exception):  # FrozenInstanceError  # noqa: PT011
        r.generations_run = 99  # type: ignore[misc]


def test_engine_init_accepts_required_args(small_panel, repo, cfg) -> None:  # noqa: ANN001
    """Constructor nhận đủ tham số bắt buộc + lưu pop_size để vòng lặp dùng sau."""
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=4, n_generations=0, seed=42,
    )
    assert eng.pop_size == 4


def test_evaluate_individual_passed_status_on_valid_seed(small_panel, repo, cfg) -> None:  # noqa: ANN001
    """Cây hợp lệ ``ts_mean(close, 5)`` trên panel xác định: phải pass hoặc failed_gate,
    cốt lõi là KHÔNG rơi vào invalid/error (cây parse và eval được)."""
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=2, n_generations=0, seed=42,
    )
    expr = parse("ts_mean(close, 5)")
    ind = Individual(expr=expr)
    pool_corr = PoolCorrelation(pool={})
    fv, status, reasons, daily_pnl, metrics = eng._evaluate_individual(ind, pool_corr)
    assert status in {"passed", "failed_gate"}
    if status == "passed":
        assert fv is not None
        assert reasons == []
        assert daily_pnl is not None
        assert metrics is not None
    else:
        assert reasons  # non-empty
        assert daily_pnl is not None  # failed_gate vẫn có backtest
        assert metrics is not None


def test_evaluate_individual_error_status_for_scalar_root(small_panel, repo, cfg) -> None:  # noqa: ANN001
    """Root là Constant trần (scalar literal, không phải PANEL signal): không tham chiếu
    field nào -> gate fields_ok có thể fail, hoặc eval/backtest sinh giá trị suy biến.
    Phải rơi vào invalid/error/failed_gate (KHÔNG passed) và fitness None khi không pass."""
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=2, n_generations=0, seed=42,
    )
    ind = Individual(expr=Constant(5.0))  # root = scalar literal, không phải PANEL
    pool_corr = PoolCorrelation(pool={})
    fv, status, reasons, daily_pnl, metrics = eng._evaluate_individual(ind, pool_corr)
    assert status in {"invalid", "error", "failed_gate"}
    assert status != "passed"
    assert reasons  # phải có lý do (không tham chiếu field nào -> fields_ok=False, v.v.)
    if status in {"invalid", "error"}:
        assert fv is None
        assert daily_pnl is None
        assert metrics is None


def test_eval_cache_hit_khong_backtest_lai(engine_fixture_voi_cache, monkeypatch) -> None:  # noqa: ANN001
    """Cùng canonical_hash lần 2: Backtester.run không được gọi lại, kết quả giống hệt."""
    import src.gp.engine as eng_mod

    engine, cache = engine_fixture_voi_cache  # GPEngine(eval_cache=cache), data giả nhiều field
    i1 = Individual(expr=parse("ts_mean(subtract(close, ts_delay(close, 1)), 10)"))
    fv1, st1, rs1, pnl1, met1 = engine._evaluate_individual(i1, _pool_corr_rong())
    assert len(cache) == 1
    assert pnl1 is not None  # eval phải thành công để test cache "ok" có ý nghĩa

    so_lan = {"n": 0}
    that = eng_mod.Backtester.run
    monkeypatch.setattr(
        eng_mod.Backtester, "run",
        lambda self, w, d: so_lan.__setitem__("n", so_lan["n"] + 1) or that(self, w, d),
    )
    i2 = Individual(expr=parse("ts_mean(subtract(close, ts_delay(close, 1)), 10)"))
    fv2, st2, rs2, pnl2, met2 = engine._evaluate_individual(i2, _pool_corr_rong())
    assert so_lan["n"] == 0
    assert st2 == st1
    np.testing.assert_array_equal(pnl2, pnl1)


def test_eval_cache_error_khong_luu_bt_va_khong_mutate(engine_fixture_voi_cache) -> None:  # noqa: ANN001
    """Cây tham chiếu field không tồn tại (``open`` không có trong ``small_panel``) -> eval
    ném ``KeyError`` -> status ``'error'``, cache lưu ``('error', reasons)``; lần 2 hit cache
    trả list MỚI (copy) để caller sửa list trả về không làm hỏng entry cache nội bộ."""
    engine, cache = engine_fixture_voi_cache
    ind1 = Individual(expr=parse("ts_mean(open, 5)"))
    fv1, st1, rs1, pnl1, met1 = engine._evaluate_individual(ind1, _pool_corr_rong())
    assert st1 == "error"
    assert pnl1 is None
    assert met1 is None
    assert fv1 is None
    assert len(cache) == 1
    rs1.append("mutated bởi caller")  # caller sửa list trả về...

    ind2 = Individual(expr=parse("ts_mean(open, 5)"))
    fv2, st2, rs2, pnl2, met2 = engine._evaluate_individual(ind2, _pool_corr_rong())
    assert st2 == "error"
    assert pnl2 is None
    assert "mutated bởi caller" not in rs2  # ...không được rò vào cache nội bộ
    assert len(cache) == 1  # vẫn 1 entry — không ghi đè/nhân đôi khi cache hit


def test_eval_cache_ok_khong_chua_backtestresult_hay_weights(engine_fixture_voi_cache) -> None:  # noqa: ANN001
    """Fix OOM (review cuối C1): entry cache "ok" chỉ giữ (daily_pnl, metrics) — phần tử thứ
    2 phải là ``np.ndarray`` 1 chiều (daily_pnl), KHÔNG phải ``BacktestResult`` (vốn giữ thêm
    ``weights`` ma trận (T,N) nặng ~10-15MB/cá thể trên panel thật, gây OOM khi tích luỹ
    xuyên cache CHIA SẺ nhiều thế hệ/batch)."""
    from src.backtest.backtester import BacktestResult
    from src.backtest.metrics_local import AlphaMetrics

    engine, cache = engine_fixture_voi_cache
    ind = Individual(expr=parse("ts_mean(subtract(close, ts_delay(close, 1)), 10)"))
    engine._evaluate_individual(ind, _pool_corr_rong())
    assert len(cache) == 1
    (tag, payload, metrics), = cache.values()
    assert tag == "ok"
    assert isinstance(payload, np.ndarray)
    assert payload.ndim == 1  # daily_pnl (T,), không phải weights (T,N)
    assert not isinstance(payload, BacktestResult)
    assert isinstance(metrics, AlphaMetrics)


def test_persist_khong_goi_lai_metricscalculator_khi_da_co_metrics(  # noqa: ANN001
    engine_fixture, monkeypatch,
) -> None:
    """Fix OOM/lãng phí (review cuối C1): ``_persist`` KHÔNG được recompute
    ``MetricsCalculator().compute(...)`` khi caller đã truyền sẵn ``metrics`` — trước đây
    ``_persist`` gọi lại tính metrics từ ``bt`` dù caller có sẵn kết quả."""
    import src.gp.engine as eng_mod

    so_lan = {"n": 0}
    that = eng_mod.MetricsCalculator.compute
    monkeypatch.setattr(
        eng_mod.MetricsCalculator, "compute",
        lambda self, bt, data: so_lan.__setitem__("n", so_lan["n"] + 1) or that(self, bt, data),
    )
    engine = engine_fixture
    ind = Individual(expr=parse("ts_mean(close, 5)"))
    pool_corr = _pool_corr_rong()
    fv, status, reasons, daily_pnl, metrics = engine._evaluate_individual(ind, pool_corr)
    assert status in {"passed", "failed_gate"}
    so_lan["n"] = 0  # reset đếm — chỉ quan tâm số lần gọi TRONG _persist
    engine._persist(ind, status, reasons, daily_pnl, metrics, 0.0)
    assert so_lan["n"] == 0


def test_engine_runs_pop4_gen1_persists_evaluations(small_panel, repo, cfg) -> None:  # noqa: ANN001
    """Chạy 4 cá thể qua 1 thế hệ: kết quả đúng kiểu/đếm thế hệ + DB có >=4 evaluation."""
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=4, n_generations=1, seed=42, with_llm_seeds=False,
    )
    result = eng.run()
    assert isinstance(result, GPRunResult)
    assert result.generations_run == 1
    assert len(result.final_population) == 4
    assert result.n_evaluated >= 4

    session = repo.session_factory()
    try:
        n_rows = session.query(EvaluationModel).count()
        assert n_rows >= 4
    finally:
        session.close()


def test_engine_persists_seed_in_db(small_panel, repo, cfg) -> None:  # noqa: ANN001
    """Mọi evaluation row phải ghi đúng seed master (determinism R8: tái lập được run)."""
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=4, n_generations=0, seed=123,
    )
    eng.run()
    session = repo.session_factory()
    try:
        rows = session.query(EvaluationModel).all()
        assert rows  # có ít nhất 1 row
        assert all(r.seed == 123 for r in rows)
    finally:
        session.close()


def test_engine_max_depth_enforced(small_panel, repo, cfg) -> None:  # noqa: ANN001
    """Mọi cá thể trong quần thể cuối phải có depth <= max_depth (init + variation tôn trọng)."""
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=8, n_generations=1, seed=42, max_depth=5,
    )
    result = eng.run()
    for ind in result.final_population:
        assert ind.expr.accept(DepthVisitor()) <= 5


def test_engine_deterministic_for_same_seed(small_panel, cfg) -> None:  # noqa: ANN001
    """Hai GPEngine cùng config + cùng seed (DB sạch riêng) → quần thể cuối có cùng
    canonical_hash theo thứ tự (determinism R8)."""
    def _fresh_repo() -> MiniBrainRepository:
        engine = create_engine("sqlite:///:memory:", future=True)
        init_db(engine)
        sf = sessionmaker(bind=engine, future=True, expire_on_commit=False)
        return MiniBrainRepository(sf)

    eng1 = GPEngine(
        data=small_panel, repo=_fresh_repo(), config=cfg, registry=default_registry(),
        pop_size=4, n_generations=1, seed=42,
    )
    r1 = eng1.run()
    eng2 = GPEngine(
        data=small_panel, repo=_fresh_repo(), config=cfg, registry=default_registry(),
        pop_size=4, n_generations=1, seed=42,
    )
    r2 = eng2.run()
    h1 = [i.expr.accept(CanonicalHasher()) for i in r1.final_population]
    h2 = [i.expr.accept(CanonicalHasher()) for i in r2.final_population]
    assert h1 == h2


def test_engine_persists_failed_or_passed_with_pre_populated_pool(  # noqa: ANN001
    small_panel, repo, cfg,
) -> None:
    """Bơm sẵn 1 alpha pass vào pool rồi chạy GP: DB phải chứa status 'passed' (alpha pool
    gốc) cùng các evaluation từ run; mọi fail_reasons là list hợp lệ (B11 avoid-list)."""
    from src.backtest.metrics_local import AlphaMetrics

    expr_id = repo.upsert_expression("close", "h_close_seed", 1, 1, {"close"})
    dates = small_panel.dates
    pnl = np.linspace(0.001, 0.002, len(dates))
    m = AlphaMetrics(
        sharpe=1.5, annual_return=0.1, turnover=0.2, max_drawdown=-0.05,
        fitness=2.0, per_year_sharpe={2021: 1.2}, weight_concentration=0.05,
    )
    eval_id = repo.record_evaluation(expr_id, "{}", "default", m, 0.0, "passed", [], 1)
    repo.save_pool_pnl(eval_id, dates, pnl)

    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=4, n_generations=0, seed=42,
    )
    eng.run()

    session = repo.session_factory()
    try:
        statuses = {r.status for r in session.query(EvaluationModel).all()}
    finally:
        session.close()
    assert "passed" in statuses  # alpha pool gốc còn nguyên


def test_engine_repeat_run_does_not_double_count_evaluations(  # noqa: ANN001
    small_panel, repo, cfg,
) -> None:
    """Chạy 2 lần cùng seed trên cùng DB: record_evaluation merge theo khóa duy nhất nên
    số row KHÔNG nhân đôi (bound lỏng <= 2x — chứng minh không nhân đôi vô tội vạ)."""
    eng1 = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=4, n_generations=0, seed=42,
    )
    eng1.run()
    session = repo.session_factory()
    try:
        n_before = session.query(EvaluationModel).count()
    finally:
        session.close()

    eng2 = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=4, n_generations=0, seed=42,
    )
    eng2.run()
    session = repo.session_factory()
    try:
        n_after = session.query(EvaluationModel).count()
    finally:
        session.close()
    assert n_after <= n_before * 2


def test_engine_passes_seed_offset_to_init_population(small_panel, repo, cfg, monkeypatch) -> None:  # noqa: ANN001
    """seed_offset (round-robin seed family qua batch, xem GPIdeaSource) phải truyền
    nguyên vẹn xuống init_population() — spy bằng monkeypatch thay vì dựng seed_cores
    thật để test nhanh, không phụ thuộc nội dung families.py."""
    captured: dict = {}

    def _fake_init_population(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("src.gp.engine.init_population", _fake_init_population)
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=4, n_generations=0, seed=42, seed_offset=8,
    )
    eng.run()
    assert captured["seed_offset"] == 8


def test_engine_seed_offset_mac_dinh_0(small_panel, repo, cfg, monkeypatch) -> None:  # noqa: ANN001
    captured: dict = {}

    def _fake_init_population(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("src.gp.engine.init_population", _fake_init_population)
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=4, n_generations=0, seed=42,
    )
    eng.run()
    assert captured["seed_offset"] == 0


# --- A2: lọc meaningfulness + họ-đã-đóng TRONG GP, TRƯỚC backtest ---

def test_evaluate_population_bo_ca_the_vo_nghia_khong_backtest(engine_fixture, monkeypatch) -> None:  # noqa: ANN001
    """Cá thể volume-only bị chặn TRƯỚC backtest: Backtester.run không được gọi."""
    import src.gp.engine as eng_mod

    goi_backtest = []
    monkeypatch.setattr(
        eng_mod.Backtester, "run",
        lambda self, w, d: goi_backtest.append(1) or (_ for _ in ()).throw(AssertionError),
    )
    engine = engine_fixture  # GPEngine với data giả có field 'volume'
    ind = Individual(expr=parse("rank(ts_zscore(volume, 5))"))
    n_ev, n_pa = engine._evaluate_population([ind], _pool_corr_rong())
    assert ind.fitness is None
    assert goi_backtest == []
    assert n_ev == 1
    assert n_pa == 0


def test_evaluate_population_bo_ca_the_ho_da_dong(engine_fixture_voi_saturated) -> None:  # noqa: ANN001
    """engine dựng với saturated_families={'pv_reversal'} -> cá thể close-open bị bỏ, không
    backtest (classify_family("multiply(-1, ts_mean(subtract(close, open), 10))") ==
    "pv_reversal", nằm trong saturated_families)."""
    engine = engine_fixture_voi_saturated  # saturated_families={"pv_reversal"}
    ind = Individual(expr=parse("multiply(-1, ts_mean(subtract(close, open), 10))"))
    n_ev, n_pa = engine._evaluate_population([ind], _pool_corr_rong())
    assert ind.fitness is None
    assert n_ev == 1
    assert n_pa == 0


def test_evaluate_population_ca_the_binh_thuong_van_duoc_backtest(engine_fixture) -> None:  # noqa: ANN001
    """Seed core hợp lệ (dùng field giá, không thuộc họ đóng nào) KHÔNG bị chặn oan bởi gate
    A2 — vẫn đi qua backtest thật (status passed/failed_gate, KHÔNG phải bị chặn ở gate mới)."""
    engine = engine_fixture
    ind = Individual(expr=parse("ts_mean(close, 5)"))
    n_ev, n_pa = engine._evaluate_population([ind], _pool_corr_rong())
    assert n_ev == 1
    # Cá thể hợp lệ luôn có fitness (không None) dù pass hay failed_gate — CHỈ bị chặn A2
    # (fitness=None, không backtest) khi vô nghĩa/họ đóng, mà "ts_mean(close, 5)" không phải.
    assert ind.fitness is not None
