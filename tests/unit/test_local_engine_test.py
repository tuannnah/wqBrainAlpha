"""Test mục 4 menu 'Test engine': GP local -> LLM refine thật -> re-score local, không sim
Brain. Dùng FakeDeepSeek (không gọi mạng) + small_panel (panel giả nhỏ, xem conftest.py)."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.operators_local  # noqa: F401
from src.app.local_engine_test import run_local_engine_test
from src.backtest.config import Neutralization, PortfolioConfig
from src.lang.registry import default_registry
from src.simulation.pre_filter import PreFilter
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository
from tests.fakes import FakeDeepSeek, FakeSymbolRepo


@pytest.fixture
def repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    return MiniBrainRepository(sessionmaker(bind=engine, future=True, expire_on_commit=False))


@pytest.fixture
def cfg() -> PortfolioConfig:
    return PortfolioConfig(
        neutralization=Neutralization.NONE, decay=0, truncation=0.10, scale_book=1.0, delay=1,
    )


def _common_kwargs(small_panel, repo, cfg, deepseek):
    prefilter = PreFilter(known_operators={"rank"}, known_fields={"close", "volume"})
    return dict(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(), deepseek=deepseek,
        field_repo=FakeSymbolRepo(["close", "volume"]), operator_repo=FakeSymbolRepo(["rank"]),
        prefilter=prefilter, pop_size=8, n_generations=1,
    )


def test_happy_path_sinh_refine_rescore(small_panel, repo, cfg) -> None:  # noqa: ANN001
    ds = FakeDeepSeek([
        json.dumps({"description": "mô tả cải tiến"}),
        json.dumps({"expression": "rank(close)"}),
    ])
    result = run_local_engine_test(**_common_kwargs(small_panel, repo, cfg, ds))

    assert result.ok, result.error
    assert result.idea_expr  # GP sinh được ứng viên
    assert result.llm_ok is True
    assert result.refined_expr == "rank(close)"
    assert result.sharpe_after is not None
    assert result.passed is not None


def test_llm_refine_khong_sinh_duoc_bieu_thuc_bao_loi_ro(small_panel, repo, cfg) -> None:  # noqa: ANN001
    ds = FakeDeepSeek(
        [json.dumps({"description": "d"})] + [json.dumps({"expression": "bad_op(x)"})] * 5,
    )
    result = run_local_engine_test(**_common_kwargs(small_panel, repo, cfg, ds))

    assert not result.ok
    assert result.llm_ok is False
    assert "LLM" in result.error


def test_gp_khong_sinh_duoc_ung_vien_bao_loi_ro(small_panel, repo, cfg, monkeypatch) -> None:  # noqa: ANN001
    from src.gp.engine import GPRunResult

    class _EmptyEngine:
        def __init__(self, *a, **k) -> None:
            pass

        def run(self):
            return GPRunResult(
                generations_run=0, final_population=[], best_by_sharpe=None,
                n_evaluated=0, n_passed=0, seed=42,
            )

    monkeypatch.setattr("src.app.closed_loop_adapters.GPEngine", _EmptyEngine)
    ds = FakeDeepSeek([])
    result = run_local_engine_test(**_common_kwargs(small_panel, repo, cfg, ds))

    assert not result.ok
    assert "GP" in result.error
    assert ds.calls == []  # không tốn lượt LLM khi GP đã rỗng


def test_loi_bat_ngo_khong_lam_crash_ma_gom_vao_error(small_panel, repo, cfg, monkeypatch) -> None:  # noqa: ANN001
    def _boom(*a, **k):
        raise RuntimeError("panel hỏng")

    monkeypatch.setattr("src.app.local_engine_test.score_one", _boom)
    ds = FakeDeepSeek([
        json.dumps({"description": "mô tả cải tiến"}),
        json.dumps({"expression": "rank(close)"}),
    ])
    result = run_local_engine_test(**_common_kwargs(small_panel, repo, cfg, ds))

    assert not result.ok
    assert "RuntimeError" in result.error
    assert "panel hỏng" in result.error
