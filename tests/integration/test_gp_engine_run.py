"""Integration GPEngine: end-to-end seed→eval→variation→select→persist trên DB +
small_panel thật (không mock). Xác minh DB có rows ExpressionModel + EvaluationModel,
engine ổn định (không quá nhiều status 'error')."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.operators_local  # noqa: F401  (side-effect: nạp 27 operator vào registry)
from src.backtest.config import Neutralization, PortfolioConfig
from src.gp.engine import GPEngine, GPRunResult
from src.lang.registry import default_registry
from src.storage.db import init_db
from src.storage.models import EvaluationModel, ExpressionModel
from src.storage.repository import MiniBrainRepository


@pytest.fixture
def repo() -> MiniBrainRepository:
    """Repository MiniBrain trên SQLite in-memory mới mỗi test (không rò state)."""
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    sf = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(sf)


def test_gp_engine_run_end_to_end_small_panel(small_panel, repo) -> None:  # noqa: ANN001
    """Chạy thật pop_size=8, n_generations=2: kết quả đúng kiểu + DB có expression/
    evaluation; tỷ lệ status 'error' không vượt nửa (engine sống sót dữ liệu thật)."""
    cfg = PortfolioConfig(
        neutralization=Neutralization.NONE, decay=0, truncation=0.10,
        scale_book=1.0, delay=1,
    )
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=8, n_generations=2, seed=42,
    )
    result = eng.run()
    assert isinstance(result, GPRunResult)
    assert result.generations_run == 2
    assert result.n_evaluated > 0
    assert len(result.final_population) > 0

    session = repo.session_factory()
    try:
        n_expr = session.query(ExpressionModel).count()
        n_eval = session.query(EvaluationModel).count()
        n_error = session.query(EvaluationModel).filter_by(status="error").count()
    finally:
        session.close()

    assert n_expr > 0
    assert n_eval > 0
    # Bao dung: error <= 50% (small_panel có thể có edge case ts ops).
    assert n_error <= n_eval // 2


def test_gp_engine_run_zero_gen_evaluates_initial_population(  # noqa: ANN001
    small_panel, repo,
) -> None:
    """n_generations=0: chỉ đánh giá quần thể khởi tạo (seed cores + ramped) và persist;
    không biến đổi. final_population = pop_size, mọi cá thể đã có fitness hoặc đã persist."""
    cfg = PortfolioConfig(
        neutralization=Neutralization.NONE, decay=0, truncation=0.10,
        scale_book=1.0, delay=1,
    )
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=6, n_generations=0, seed=7,
    )
    result = eng.run()
    assert result.generations_run == 0
    assert len(result.final_population) == 6
    assert result.n_evaluated == 6

    session = repo.session_factory()
    try:
        n_eval = session.query(EvaluationModel).count()
    finally:
        session.close()
    assert n_eval >= 1
