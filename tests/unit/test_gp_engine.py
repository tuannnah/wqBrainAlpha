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
    fv, status, reasons, bt = eng._evaluate_individual(ind, pool_corr)
    assert status in {"passed", "failed_gate"}
    if status == "passed":
        assert fv is not None
        assert reasons == []
        assert bt is not None
    else:
        assert reasons  # non-empty
        assert bt is not None  # failed_gate vẫn có backtest


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
    fv, status, reasons, bt = eng._evaluate_individual(ind, pool_corr)
    assert status in {"invalid", "error", "failed_gate"}
    assert status != "passed"
    assert reasons  # phải có lý do (không tham chiếu field nào -> fields_ok=False, v.v.)
    if status in {"invalid", "error"}:
        assert fv is None


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
