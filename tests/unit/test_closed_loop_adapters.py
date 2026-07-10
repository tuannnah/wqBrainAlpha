"""Test adapter vòng kín: RefinementLoopRefiner map LoopResult->IdeaOutcome; GPIdeaSource bọc
generate_many với seed tăng dần. RefinementLoopRefiner test bằng fake loop (không AI/sim thật);
GPIdeaSource test trên small_panel + DB in-memory."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.operators_local  # noqa: F401
from src.app.closed_loop_adapters import GPIdeaSource, RefinementLoopRefiner
from src.backtest.config import Neutralization, PortfolioConfig
from src.lang.registry import default_registry
from src.pipeline.closed_loop import IdeaOutcome
from src.pipeline.shortlist import ShortlistCandidate
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


class _FakeLoopResult:
    def __init__(self) -> None:
        self.best_candidate = type("C", (), {"expression": "rank(close)"})()
        self.best_passed = True
        self.best_alpha_id = "WQ42"
        self.best_metrics = {"sharpe": 1.6, "fitness": 1.3, "turnover": 0.2}
        self.best_self_corr = 0.35
        self.sims_used = 3
        self.stop_reason = "patience"


class _FakeLoop:
    def __init__(self) -> None:
        self.seeds: list[str] = []

    def run_from_seed(self, expression: str, on_progress=None) -> _FakeLoopResult:
        self.seeds.append(expression)
        return _FakeLoopResult()


def _cand(expr: str) -> ShortlistCandidate:
    from src.backtest.metrics_local import AlphaMetrics
    m = AlphaMetrics(sharpe=1.0, annual_return=0.1, turnover=0.3, max_drawdown=0.05,
                     fitness=1.0, per_year_sharpe={2021: 1.0}, weight_concentration=0.05)
    d = (np.datetime64("2021-01-01") + np.arange(5)).astype("datetime64[D]")
    return ShortlistCandidate(expr=expr, metrics=m, pnl=np.ones(5), dates=d)


def test_refiner_maps_loopresult_to_ideaoutcome() -> None:
    refiner = RefinementLoopRefiner(_FakeLoop())
    outcome = refiner.refine_and_sim(_cand("rank(close)"))
    assert isinstance(outcome, IdeaOutcome)
    assert outcome.passed is True
    assert outcome.wq_alpha_id == "WQ42"
    assert outcome.sharpe == 1.6
    assert outcome.fitness == 1.3
    assert outcome.turnover == 0.2
    assert outcome.self_corr == 0.35
    assert outcome.sims_used == 3
    assert outcome.stop_reason == "patience"
    assert outcome.canonical_hash  # tính được từ expr (parse+CanonicalHasher), không rỗng


def test_refiner_seeds_loop_with_candidate_expr() -> None:
    loop = _FakeLoop()
    RefinementLoopRefiner(loop).refine_and_sim(_cand("ts_mean(close, 5)"))
    assert loop.seeds == ["ts_mean(close, 5)"]  # seed loop bằng đúng expr candidate


@pytest.fixture
def repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    return MiniBrainRepository(sessionmaker(bind=engine, future=True, expire_on_commit=False))


def test_gp_idea_source_yields_candidates_and_advances_seed(small_panel, repo) -> None:  # noqa: ANN001
    from unittest.mock import patch
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    src = GPIdeaSource(small_panel, repo, cfg, default_registry(),
                       pop_size=6, n_generations=0, base_seed=42, top_k=5, max_corr=0.99,
                       max_empty_retries=1)
    seeds_seen: list[int] = []

    class _StubEngine:
        def __init__(self, *a, seed: int, **k) -> None:
            seeds_seen.append(seed)
        def run(self):
            from src.gp.engine import GPRunResult
            return GPRunResult(generations_run=0, final_population=[], best_by_sharpe=None,
                               n_evaluated=0, n_passed=0, seed=42)

    with patch("src.app.closed_loop_adapters.GPEngine", _StubEngine):
        b1 = src.next_batch()
        b2 = src.next_batch()
    assert seeds_seen == [42, 43]  # seed tăng dần mỗi batch
    assert isinstance(b1, list) and isinstance(b2, list)


def test_gp_idea_source_seed_offset_tang_theo_pop_size(small_panel, repo) -> None:  # noqa: ANN001
    """Moi batch phai dung 1 lo seed khac nhau (round-robin) - offset tang dung pop_size,
    khop cach seed=base_seed+batch da co san."""
    from unittest.mock import patch
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    src = GPIdeaSource(small_panel, repo, cfg, default_registry(),
                       pop_size=6, n_generations=0, base_seed=42, top_k=5, max_corr=0.99,
                       max_empty_retries=1)
    offsets_seen: list[int] = []

    class _StubEngine:
        def __init__(self, *a, seed_offset: int, **k) -> None:
            offsets_seen.append(seed_offset)
        def run(self):
            from src.gp.engine import GPRunResult
            return GPRunResult(generations_run=0, final_population=[], best_by_sharpe=None,
                               n_evaluated=0, n_passed=0, seed=42)

    with patch("src.app.closed_loop_adapters.GPEngine", _StubEngine):
        src.next_batch()
        src.next_batch()
        src.next_batch()
    assert offsets_seen == [0, 6, 12]  # tang dung pop_size (6) moi batch


def test_next_batch_thu_nhieu_seed_khi_batch_rong(small_panel, repo) -> None:  # noqa: ANN001
    """Một seed cho 0 ứng viên -> KHÔNG trả rỗng ngay, mà thử seed kế cho tới khi có ý
    tưởng (tránh no_more_ideas oan vì 1 seed xui)."""
    from unittest.mock import patch
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    src = GPIdeaSource(small_panel, repo, cfg, default_registry(),
                       pop_size=6, n_generations=0, base_seed=42, top_k=5, max_corr=0.99,
                       max_empty_retries=5)
    calls = {"n": 0}

    def _fake_generate_many(**_kw):
        calls["n"] += 1
        return [] if calls["n"] < 3 else [_cand("rank(close)")]

    class _StubEngine:
        def __init__(self, *a, **k) -> None: ...

    with patch("src.app.closed_loop_adapters.GPEngine", _StubEngine), \
         patch("src.app.closed_loop_adapters.generate_many", _fake_generate_many):
        batch = src.next_batch()
    assert calls["n"] == 3  # thử 3 seed mới có ý tưởng
    assert len(batch) == 1
    assert src._batch == 3  # đã tiêu 3 lô seed


def test_next_batch_tra_rong_khi_moi_seed_deu_can(small_panel, repo) -> None:  # noqa: ANN001
    """Mọi seed đều rỗng (thật sự cạn) -> trả [] sau đúng max_empty_retries lần thử,
    không lặp vô hạn."""
    from unittest.mock import patch
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    src = GPIdeaSource(small_panel, repo, cfg, default_registry(),
                       pop_size=6, n_generations=0, base_seed=42, top_k=5, max_corr=0.99,
                       max_empty_retries=4)
    calls = {"n": 0}

    def _always_empty(**_kw):
        calls["n"] += 1
        return []

    class _StubEngine:
        def __init__(self, *a, **k) -> None: ...

    with patch("src.app.closed_loop_adapters.GPEngine", _StubEngine), \
         patch("src.app.closed_loop_adapters.generate_many", _always_empty):
        batch = src.next_batch()
    assert batch == []
    assert calls["n"] == 4  # đúng max_empty_retries, không hơn


def test_refiner_raises_quota_exhausted_on_auth_expired() -> None:
    from src.pipeline.closed_loop import QuotaExhausted
    from src.simulation.simulator import AuthExpiredError

    class _AuthDeadLoop:
        def run_from_seed(self, expression: str, on_progress: object = None) -> object:
            raise AuthExpiredError("session het han / quota")

    with pytest.raises(QuotaExhausted):
        RefinementLoopRefiner(_AuthDeadLoop()).refine_and_sim(_cand("rank(close)"))


def test_refiner_raises_quota_exhausted_on_quota_exceeded() -> None:
    """QuotaExceededError (hết quota ngày thật, KHÁC lỗi auth) cũng phải ánh xạ sang
    QuotaExhausted để ClosedLoop dừng gọn — không chỉ AuthExpiredError."""
    from src.pipeline.closed_loop import QuotaExhausted
    from src.simulation.simulator import QuotaExceededError

    class _QuotaDeadLoop:
        def run_from_seed(self, expression: str, on_progress: object = None) -> object:
            raise QuotaExceededError("het quota simulation ngay")

    with pytest.raises(QuotaExhausted):
        RefinementLoopRefiner(_QuotaDeadLoop()).refine_and_sim(_cand("rank(close)"))


def test_build_closed_loop_wires_components(small_panel, repo) -> None:  # noqa: ANN001
    from src.app.closed_loop_adapters import build_closed_loop
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.lang.registry import default_registry
    from src.pipeline.closed_loop import ClosedLoop

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)

    class _NoopLoop:
        def run_from_seed(self, expression, on_progress=None):
            # trả LoopResult-like tối thiểu: không pass, không sim
            return type("R", (), {"best_candidate": None, "best_passed": False,
                                  "best_alpha_id": None, "best_metrics": {},
                                  "best_self_corr": None, "sims_used": 0,
                                  "stop_reason": "no_seed"})()

    loop = build_closed_loop(data=small_panel, repo=repo, config=cfg,
                             registry=default_registry(), loop=_NoopLoop(),
                             pop_size=6, n_generations=0, top_k=3, max_ideas=2)
    assert isinstance(loop, ClosedLoop)
    report = loop.run()  # chạy với GP thật (pop nhỏ) + _NoopLoop refiner -> không crash
    assert report.ideas_tried >= 0
    assert report.stop_reason in {"no_more_ideas", "quota"}


def test_build_closed_loop_include_alt_data(small_panel, repo) -> None:  # noqa: ANN001
    """include_alt_data=True -> batch ĐẦU của idea_source là các core alt-data (đi thẳng Brain)."""
    from src.app.closed_loop_adapters import build_closed_loop
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.generation.alt_data_seeds import ALT_DATA_CORES
    from src.lang.registry import default_registry

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)

    class _NoopLoop:
        def run_from_seed(self, expression, on_progress=None):
            return type("R", (), {"best_candidate": None})()

    loop = build_closed_loop(data=small_panel, repo=repo, config=cfg,
                             registry=default_registry(), loop=_NoopLoop(),
                             pop_size=6, n_generations=0, top_k=3, max_ideas=2,
                             curated_seeds=False, include_alt_data=True,
                             include_fundamental=False)
    batch = loop.idea_source.next_batch()
    assert [c.expr for c in batch] == list(ALT_DATA_CORES)


def test_build_closed_loop_fundamental_mac_dinh(small_panel, repo) -> None:  # noqa: ANN001
    """Pha 2.1: fundamental cores có mặt trong batch đầu khi include_fundamental (mặc định True).
    Field fundamental ngoài panel local -> refiner sim thẳng Brain (như alt-data)."""
    from src.app.closed_loop_adapters import build_closed_loop
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.generation.fundamental_seeds import FUNDAMENTAL_CORES
    from src.lang.registry import default_registry

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)

    class _NoopLoop:
        def run_from_seed(self, expression, on_progress=None):
            return type("R", (), {"best_candidate": None})()

    loop = build_closed_loop(data=small_panel, repo=repo, config=cfg,
                             registry=default_registry(), loop=_NoopLoop(),
                             pop_size=6, n_generations=0, top_k=3, max_ideas=2,
                             curated_seeds=False)  # KHÔNG truyền include_fundamental
    exprs = [c.expr for c in loop.idea_source.next_batch()]
    for core in FUNDAMENTAL_CORES:
        assert core in exprs


def test_build_closed_loop_alt_data_bat_mac_dinh(small_panel, repo) -> None:  # noqa: ANN001
    """Pha 2.1: alt-data BẬT mặc định (đòn bẩy yield #1) — không cần truyền include_alt_data.
    AltDataIdeaSource bọc ngoài nên các core alt-data nằm trong batch đầu (combiner có thể
    nối thêm combo phía sau, nên dùng superset thay vì bằng tuyệt đối)."""
    from src.app.closed_loop_adapters import build_closed_loop
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.generation.alt_data_seeds import ALT_DATA_CORES
    from src.lang.registry import default_registry

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)

    class _NoopLoop:
        def run_from_seed(self, expression, on_progress=None):
            return type("R", (), {"best_candidate": None})()

    loop = build_closed_loop(data=small_panel, repo=repo, config=cfg,
                             registry=default_registry(), loop=_NoopLoop(),
                             pop_size=6, n_generations=0, top_k=3, max_ideas=2,
                             curated_seeds=False)  # KHÔNG truyền include_alt_data
    exprs = [c.expr for c in loop.idea_source.next_batch()]
    for core in ALT_DATA_CORES:
        assert core in exprs
