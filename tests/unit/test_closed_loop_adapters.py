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
                             include_fundamental=False, include_hypothesis=False)
    batch = loop.idea_source.next_batch()
    assert [c.expr for c in batch] == list(ALT_DATA_CORES)


def test_build_closed_loop_noi_on_family_closed_toi_generator(small_panel, repo) -> None:  # noqa: ANN001
    """Fix gap Pha 2.3: khi truyền idea_generator, build_closed_loop phải nối on_family_closed
    -> generator.set_saturated_families (để họ bão hoà tiêm vào prompt LLM lượt sau)."""
    from src.app.closed_loop_adapters import build_closed_loop
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.lang.registry import default_registry

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)

    class _Gen:
        def __init__(self):
            self.received = None
        def set_saturated_families(self, fams):
            self.received = set(fams)

    class _NoopLoop:
        def run_from_seed(self, expression, on_progress=None):
            return type("R", (), {"best_candidate": None})()

    gen = _Gen()
    loop = build_closed_loop(data=small_panel, repo=repo, config=cfg,
                             registry=default_registry(), loop=_NoopLoop(),
                             pop_size=6, n_generations=0, top_k=3, max_ideas=1,
                             idea_generator=gen)
    assert loop.on_family_closed is not None
    loop.on_family_closed({"pv_reversal"})
    assert gen.received == {"pv_reversal"}


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


def test_gp_idea_source_loc_ho_da_dong(small_panel, repo) -> None:  # noqa: ANN001
    """Task 4: set_saturated_families lọc candidate thuộc họ đã đóng khỏi next_batch — trước
    fix, GPIdeaSource không có setter này nên tín hiệu đóng họ không bao giờ chặn được sinh."""
    from unittest.mock import patch

    from src.reporting.diagnostics import classify_family

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    src = GPIdeaSource(small_panel, repo, cfg, default_registry(),
                       pop_size=6, n_generations=0, base_seed=42, top_k=5, max_corr=0.99,
                       max_empty_retries=1)

    def _fake_generate_many(**_kw):
        return [
            _cand("multiply(-1, ts_mean(subtract(close, vwap), 10))"),  # pv_reversal
            _cand("ts_rank(assets, 10)"),  # fundamental
        ]

    class _StubEngine:
        def __init__(self, *a, **k) -> None: ...

    src.set_saturated_families({"pv_reversal"})
    with patch("src.app.closed_loop_adapters.GPEngine", _StubEngine), \
         patch("src.app.closed_loop_adapters.generate_many", _fake_generate_many):
        batch = src.next_batch()
    assert batch and all(classify_family(c.expr) != "pv_reversal" for c in batch)


def test_curated_idea_source_uy_quyen_saturated_xuong_fallback(small_panel, repo) -> None:  # noqa: ANN001
    """Task 4: wrapper CuratedIdeaSource phải ủy quyền set_saturated_families xuống fallback
    (GPIdeaSource) để CẢ CHUỖI học tín hiệu đóng họ, không chỉ dừng ở wrapper ngoài cùng."""
    from src.app.closed_loop_adapters import CuratedIdeaSource

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    gp = GPIdeaSource(small_panel, repo, cfg, default_registry(), pop_size=6, n_generations=0)
    outer = CuratedIdeaSource(fallback=gp)

    outer.set_saturated_families({"pv_reversal"})

    assert gp._saturated == {"pv_reversal"}


def test_chuoi_wrapper_lan_toa_loc_ho_dong_toi_gp(small_panel, repo) -> None:  # noqa: ANN001
    """Task 4, verify (a): set_saturated_families gọi ở wrapper NGOÀI CÙNG (CuratedIdeaSource)
    phải lan toả xuống GPIdeaSource — cores curated (đều pv_reversal) bị lọc sạch nên rơi
    xuống GP fallback (đã stub trả 1 candidate fundamental), kết quả không còn pv_reversal."""
    from unittest.mock import patch

    from src.app.closed_loop_adapters import CuratedIdeaSource
    from src.reporting.diagnostics import classify_family

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    gp = GPIdeaSource(small_panel, repo, cfg, default_registry(),
                       pop_size=6, n_generations=0, base_seed=42, top_k=5, max_corr=0.99,
                       max_empty_retries=1)
    outer = CuratedIdeaSource(fallback=gp)

    def _fake_generate_many(**_kw):
        return [_cand("ts_rank(assets, 10)")]  # fundamental, không pv

    class _StubEngine:
        def __init__(self, *a, **k) -> None: ...

    outer.set_saturated_families({"pv_reversal"})
    with patch("src.app.closed_loop_adapters.GPEngine", _StubEngine), \
         patch("src.app.closed_loop_adapters.generate_many", _fake_generate_many):
        batch = outer.next_batch()
    assert batch and all(classify_family(c.expr) != "pv_reversal" for c in batch)


def test_build_closed_loop_on_family_closed_noi_toi_idea_source(small_panel, repo) -> None:  # noqa: ANN001
    """Task 4, verify (b): build_closed_loop phải nối on_family_closed -> idea_source (chuỗi
    generator THẬT), không chỉ tới idea_generator (LLM re-seed riêng, mặc định None). Gọi
    on_family_closed phải lọc được batch tiếp theo sinh từ idea_source."""
    from unittest.mock import patch

    from src.app.closed_loop_adapters import build_closed_loop
    from src.reporting.diagnostics import classify_family

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)

    class _NoopLoop:
        def run_from_seed(self, expression, on_progress=None):
            return type("R", (), {"best_candidate": None})()

    # Tắt hết wrapper khác -> idea_source CHÍNH LÀ GPIdeaSource, kiểm tra trực diện dây nối.
    loop = build_closed_loop(data=small_panel, repo=repo, config=cfg,
                             registry=default_registry(), loop=_NoopLoop(),
                             pop_size=6, n_generations=0, top_k=3, max_ideas=2,
                             curated_seeds=False, include_alt_data=False,
                             include_fundamental=False, include_combiner=False,
                             include_hypothesis=False)
    assert loop.on_family_closed is not None

    loop.on_family_closed({"pv_reversal"})
    assert loop.idea_source._saturated == {"pv_reversal"}

    def _fake_generate_many(**_kw):
        return [
            _cand("multiply(-1, ts_mean(subtract(close, vwap), 10))"),  # pv_reversal
            _cand("ts_rank(assets, 10)"),  # fundamental
        ]

    class _StubEngine:
        def __init__(self, *a, **k) -> None: ...

    with patch("src.app.closed_loop_adapters.GPEngine", _StubEngine), \
         patch("src.app.closed_loop_adapters.generate_many", _fake_generate_many):
        batch = loop.idea_source.next_batch()
    assert batch and all(classify_family(c.expr) != "pv_reversal" for c in batch)


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


def test_build_closed_loop_loc_core_da_that_bai_qua_avoided_hashes(small_panel, repo) -> None:  # noqa: ANN001
    """Finding #2 (Important) — khe hash-space: `build_closed_loop` trước đây chỉ nạp
    `avoided_hashes_original()` (TriedHashModel, hash GỐC pre-tune) để lọc core tại NGUỒN
    (AltDataIdeaSource/CuratedIdeaSource), trong khi `ClosedLoop.run` chặn refine trùng bằng
    UNION CẢ BA tập: avoided_hashes_original() ∪ avoided_hashes() (BrainSimLinkModel
    status='failed', post-tune) ∪ dedup(avoided_exprs()). Core đã Brain-sim-fail ở phiên
    TRƯỚC (chỉ có mặt trong avoided_hashes(), KHÔNG có trong avoided_hashes_original() vì
    record_avoided_hash chưa từng gọi cho nó) phải bị lọc NGAY TẠI NGUỒN — không được lọt vào
    batch alt-data rồi đốt quota thật thêm lần nữa (dù cuối cùng vẫn bị `seen` ở run() chặn,
    chuyện đó xảy ra SAU KHI đã sim thật, không còn dấu vết trong batch)."""
    from src.app.closed_loop_adapters import build_closed_loop
    from src.generation.alt_data_seeds import ALT_DATA_CORES
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.lang.registry import default_registry
    from src.lang.parser import parse
    from src.lang.visitors import CanonicalHasher

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)

    class _NoopLoop:
        def run_from_seed(self, expression, on_progress=None):
            return type("R", (), {"best_candidate": None})()

    # Core alt-data ĐẦU TIÊN đã Brain-sim status='failed' ở phiên trước — chỉ ghi vào
    # BrainSimLinkModel (avoided_hashes()), KHÔNG ghi TriedHashModel (avoided_hashes_original()
    # rỗng) — đúng khe hở review chỉ ra.
    core_da_fail = ALT_DATA_CORES[0]
    canonical_hash = CanonicalHasher(default_registry()).visit(parse(core_da_fail))
    repo.record_brain_sim(
        canonical_hash, core_da_fail, wq_alpha_id="wq-old", region="USA", universe="TOP3000",
        sharpe=0.1, fitness=0.1, turnover=0.5, self_corr=None, status="failed",
    )
    assert canonical_hash in repo.avoided_hashes()
    assert canonical_hash not in repo.avoided_hashes_original()

    loop = build_closed_loop(data=small_panel, repo=repo, config=cfg,
                             registry=default_registry(), loop=_NoopLoop(),
                             pop_size=6, n_generations=0, top_k=3, max_ideas=2,
                             curated_seeds=False)
    exprs = [c.expr for c in loop.idea_source.next_batch()]
    assert core_da_fail not in exprs
    # Core ALT_DATA khác (chưa từng sim) vẫn được phục vụ bình thường.
    for core in ALT_DATA_CORES[1:]:
        assert core in exprs


# --- Field-validity guard (RC1/RC2 fix idea-generator): core field bịa/chưa cache bị lọc
# TRƯỚC khi chạm Brain sim, thay vì đốt quota rồi WQ từ chối (cardinal rule #1). ------------


def test_alt_data_idea_source_loc_core_field_khong_co_trong_catalog() -> None:
    """known_fields KHÔNG chứa field của 1 core -> core đó bị lọc, core còn lại (field đủ)
    vẫn được phục vụ."""
    from src.app.closed_loop_adapters import AltDataIdeaSource
    from src.lang.registry import default_registry

    cores = (
        "rank(close)",                       # field "close" -> CÓ trong catalog
        "rank(field_bia_khong_ton_tai)",      # field bịa -> KHÔNG có trong catalog
    )
    src = AltDataIdeaSource(
        fallback=None, cores=cores,
        known_fields=frozenset({"close", "volume"}), registry=default_registry(),
    )
    served = [c.expr for c in src.next_batch()]
    assert served == ["rank(close)"]
    assert "rank(field_bia_khong_ton_tai)" not in served


def test_alt_data_idea_source_known_fields_none_khong_loc() -> None:
    """known_fields=None (mặc định, catalog chưa load) -> KHÔNG lọc gì — hành vi cũ y nguyên
    (tương thích ngược, mọi test hiện có không truyền known_fields đều phải chạy y như trước)."""
    from src.app.closed_loop_adapters import AltDataIdeaSource

    cores = ("rank(close)", "rank(field_bia_khong_ton_tai)")
    src = AltDataIdeaSource(fallback=None, cores=cores)
    served = [c.expr for c in src.next_batch()]
    assert served == list(cores)


def test_build_closed_loop_known_fields_loc_core_thieu_field(small_panel, repo) -> None:  # noqa: ANN001
    """build_closed_loop(known_fields=...) phải thread xuống AltDataIdeaSource: core hypothesis
    tham chiếu field không nằm trong catalog cache KHÔNG BAO GIỜ được idea_source phục vụ."""
    from src.app.closed_loop_adapters import build_closed_loop
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.generation.hypothesis_seeds import HYPOTHESIS_CORES
    from src.lang.registry import default_registry

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)

    class _NoopLoop:
        def run_from_seed(self, expression, on_progress=None):
            return type("R", (), {"best_candidate": None})()

    # Catalog THẬT chỉ có field fundamental đã verify live (khớp value_quality) — mọi field
    # analyst4/short-interest suy đoán (chưa verify) phải bị lọc.
    known = frozenset({"operating_income", "assets", "sales_growth", "close", "volume"})
    loop = build_closed_loop(data=small_panel, repo=repo, config=cfg,
                             registry=default_registry(), loop=_NoopLoop(),
                             pop_size=6, n_generations=0, top_k=3, max_ideas=2,
                             curated_seeds=False, include_alt_data=False,
                             include_fundamental=False, include_combiner=False,
                             known_fields=known)
    exprs = [c.expr for c in loop.idea_source.next_batch()]
    # Core value_quality (chỉ dùng field đã verify live) PHẢI có mặt.
    value_quality_core = [e for e in HYPOTHESIS_CORES if "sales_growth" in e][0]
    assert value_quality_core in exprs
    # Core dùng field chưa verify (anl4_*, days_to_cover, shares_short) PHẢI bị lọc hết.
    for core in HYPOTHESIS_CORES:
        if core != value_quality_core:
            assert core not in exprs


# --- Task 5: mini-sweep cho đường sim-thẳng alt-data (flip dấu + biến thể decay) ---
# Mỗi hypothesis alt-data trước đây chỉ được sim ĐÚNG 1 LẦN rồi vứt (bằng chứng: seed social
# từng SAI DẤU, Sharpe -0.48 -> đáng lẽ +0.48 nếu flip; analyst revision 1-shot 0.64 rồi bỏ).
# `_sim_direct` nay sim thêm CÓ KỶ LUẬT (ngân sách `alt_sweep_budget`, mặc định 2) rồi chọn
# outcome có điểm-nộp cao nhất trong toàn bộ các lần đã sim.


def _sweep_cand(expr: str) -> ShortlistCandidate:
    """Candidate alt-data (field ngoài panel local) — metrics=None/pnl rỗng như
    AltDataIdeaSource thật yield (không tự backtest local)."""
    return ShortlistCandidate(
        expr=expr, metrics=None, pnl=np.zeros(0),
        dates=np.zeros(0, dtype="datetime64[ns]"),
    )


class _SweepRepo:
    def __init__(self) -> None:
        # (expr, source, description) — description để phân biệt bản ghi THẮNG (finalize)
        # với bản ghi attempt THUA của sweep.
        self.saved: list[tuple[str, str | None, str | None]] = []
        self.sim_saved: int = 0

    def save_alpha(self, expr, **k):
        self.saved.append((expr, k.get("source"), k.get("description")))
        return f"alpha-{len(self.saved)}"

    def save_simulation(self, *a, **k):
        self.sim_saved += 1
        return None


class _SweepPVData:
    """Panel local chỉ có price/volume — field alt-data (snt_social_value) nằm NGOÀI đây nên
    `_is_alt_data` nhận diện đúng, đi thẳng `_sim_direct`."""

    def field_names(self):
        return {"close", "open", "vwap", "volume", "high", "low", "returns"}


def _sweep_boom_tune(*a, **k):
    raise AssertionError("nhánh alt-data KHÔNG được gọi tune local")


class _SeqSimulator:
    """Simulator giả trả kết quả THEO THỨ TỰ lần gọi — dùng để dựng kịch bản mini-sweep
    (sim #1 core, sim #2 flip/decay...). Lần gọi vượt quá danh sách kết quả -> lặp lại phần tử
    cuối (an toàn nếu implementation gọi nhiều hơn dự kiến, test vẫn assert đúng `calls`)."""

    def __init__(self, results: list) -> None:
        self._results = list(results)
        self.calls = 0
        self.seen: list[tuple[str, dict]] = []

    def simulate(self, expr, settings=None):
        self.calls += 1
        self.seen.append((expr, settings))
        idx = min(self.calls - 1, len(self._results) - 1)
        return self._results[idx]


_SWEEP_EXPR = "ts_mean(snt_social_value, 5)"
_SWEEP_EXPR_FLIPPED = "multiply(-1, ts_mean(snt_social_value, 5))"


def _sweep_result(expr, *, status, sharpe, fitness=1.0, alpha_id="wq-x"):
    from src.simulation.simulator import SimulationResult

    return SimulationResult(
        expression=expr, alpha_id=alpha_id, status=status, sharpe=sharpe,
        fitness=fitness, turnover=0.2, drawdown=0.1, raw={},
    )


def _sweep_refiner(sim, *, sim_config=None, local_decay=4, alt_sweep_budget=2):
    from src.app.closed_loop_adapters import LocalTunerRefiner
    from src.backtest.config import PortfolioConfig
    from src.simulation.config import SimConfig

    return LocalTunerRefiner(
        simulator=sim, repo=_SweepRepo(), data=_SweepPVData(),
        local_config=PortfolioConfig(decay=local_decay, truncation=0.08),
        sim_config=sim_config or SimConfig.default(),
        tune_fn=_sweep_boom_tune,  # nếu bị gọi -> test đỏ (chứng minh alt-data không tune local)
        alt_sweep_budget=alt_sweep_budget,
    )


def test_flip_sign_boc_neu_da_multiply_am_1() -> None:
    """AST (không xử lý chuỗi): expr gốc dạng multiply(-1, X) -> flip BÓC thành X."""
    from src.app.closed_loop_adapters import _flip_sign

    assert _flip_sign(_SWEEP_EXPR_FLIPPED) == _SWEEP_EXPR


def test_flip_sign_boc_neu_chua_co_dau_am() -> None:
    """Expr gốc chưa có multiply(-1, ...) -> flip BỌC bằng multiply(-1, <expr>)."""
    from src.app.closed_loop_adapters import _flip_sign

    assert _flip_sign(_SWEEP_EXPR) == _SWEEP_EXPR_FLIPPED


def test_sim_direct_sharpe_am_sau_flip_dau_va_sim_lai() -> None:
    """Kịch bản (a): sharpe core -0.9 (<= -ALT_SWEEP_MIN_ABS_SHARPE) -> đúng 2 lần simulate,
    lần 2 với expr đã flip dấu, outcome cuối lấy kết quả TỐT HƠN (flip), sims_used == 2."""
    r1 = _sweep_result(_SWEEP_EXPR, status="failed", sharpe=-0.9, fitness=0.3, alpha_id="wq-1")
    r2 = _sweep_result(_SWEEP_EXPR_FLIPPED, status="passed", sharpe=1.6, fitness=1.2, alpha_id="wq-2")
    sim = _SeqSimulator([r1, r2])
    refiner = _sweep_refiner(sim)

    out = refiner.refine_and_sim(_sweep_cand(_SWEEP_EXPR))

    assert sim.calls == 2
    assert sim.seen[1][0] == _SWEEP_EXPR_FLIPPED
    assert out.sims_used == 2
    assert out.expr == _SWEEP_EXPR_FLIPPED
    assert out.sharpe == 1.6
    assert out.passed is True
    # Finding reviewer (Important): sim THẬT không thắng (sim #1, sharpe -0.9) đã đốt quota +
    # tạo alpha thật trên platform -> PHẢI có bản ghi local (save_alpha + save_simulation),
    # không được vứt không dấu vết (mất dữ liệu calibration/audit; alpha mồ côi trên Brain).
    repo = refiner.repo
    assert len(repo.saved) == 2
    assert repo.sim_saved == 2
    saved_exprs = {expr for expr, _src, _desc in repo.saved}
    assert saved_exprs == {_SWEEP_EXPR, _SWEEP_EXPR_FLIPPED}
    assert all(src == "alt_data" for _e, src, _d in repo.saved)
    # Bản ghi attempt THUA phải phân biệt được với bản ghi thắng qua description.
    desc_by_expr = {expr: desc for expr, _src, desc in repo.saved}
    assert desc_by_expr[_SWEEP_EXPR] == "alt-data sweep attempt (thua)"
    assert desc_by_expr[_SWEEP_EXPR_FLIPPED] == "alt-data direct"
    # Hợp đồng outcome KHÔNG đổi: refine_and_sim vẫn trả đúng 1 IdeaOutcome (best).
    assert isinstance(out, IdeaOutcome)


def test_sim_direct_sharpe_yeu_duoi_nguong_khong_sweep() -> None:
    """Kịch bản (b): sharpe core 0.2 (|sharpe| < ALT_SWEEP_MIN_ABS_SHARPE) -> KHÔNG đủ tín hiệu
    để biết nên flip hay đổi decay -> đúng 1 sim, không sweep."""
    r1 = _sweep_result(_SWEEP_EXPR, status="failed", sharpe=0.2, fitness=0.4, alpha_id="wq-1")
    sim = _SeqSimulator([r1])
    refiner = _sweep_refiner(sim)

    out = refiner.refine_and_sim(_sweep_cand(_SWEEP_EXPR))

    assert sim.calls == 1
    assert out.sims_used == 1
    assert out.sharpe == 0.2


def test_sim_direct_budget_0_khong_sweep_du_sharpe_am() -> None:
    """Kịch bản (c): alt_sweep_budget=0 -> đúng 1 sim dù sharpe -0.9 đủ điều kiện flip (ngân
    sách 0 = không còn lượt sweep nào)."""
    r1 = _sweep_result(_SWEEP_EXPR, status="failed", sharpe=-0.9, fitness=0.3, alpha_id="wq-1")
    sim = _SeqSimulator([r1])
    refiner = _sweep_refiner(sim, alt_sweep_budget=0)

    out = refiner.refine_and_sim(_sweep_cand(_SWEEP_EXPR))

    assert sim.calls == 1
    assert out.sims_used == 1
    assert out.sharpe == -0.9


def test_sim_direct_sharpe_duong_yeu_thu_decay_khac_0_thanh_4() -> None:
    """Kịch bản decay (0.5 <= sharpe, chưa pass): sim #1 dùng decay mặc định 0 (SimConfig.
    default()) -> sim #2 lại BEST-SO-FAR với decay đổi sang 4 ('ngược lại -> 4'), expr KHÔNG
    đổi (chỉ config đổi). sim #2 pass -> dừng sweep, outcome lấy kết quả tốt hơn."""
    r1 = _sweep_result(_SWEEP_EXPR, status="failed", sharpe=0.6, fitness=0.5, alpha_id="wq-1")
    r2 = _sweep_result(_SWEEP_EXPR, status="passed", sharpe=1.6, fitness=1.2, alpha_id="wq-2")
    sim = _SeqSimulator([r1, r2])
    refiner = _sweep_refiner(sim)  # SimConfig.default() -> decay=0

    out = refiner.refine_and_sim(_sweep_cand(_SWEEP_EXPR))

    assert sim.calls == 2
    assert sim.seen[0][1]["decay"] == 0
    assert sim.seen[1][1]["decay"] == 4
    assert sim.seen[1][0] == _SWEEP_EXPR  # expr không đổi, chỉ decay đổi
    assert out.sims_used == 2
    assert out.sharpe == 1.6


def test_sim_direct_sharpe_duong_yeu_thu_decay_khac_4_thanh_8() -> None:
    """Kịch bản decay hướng ngược lại: sim_config gốc đã decay=4 -> sweep đổi sang 8."""
    from src.simulation.config import SimConfig

    r1 = _sweep_result(_SWEEP_EXPR, status="failed", sharpe=0.7, fitness=0.5, alpha_id="wq-1")
    r2 = _sweep_result(_SWEEP_EXPR, status="passed", sharpe=1.5, fitness=1.2, alpha_id="wq-2")
    sim = _SeqSimulator([r1, r2])
    refiner = _sweep_refiner(sim, sim_config=SimConfig.default().with_overrides(decay=4))

    out = refiner.refine_and_sim(_sweep_cand(_SWEEP_EXPR))

    assert sim.seen[0][1]["decay"] == 4
    assert sim.seen[1][1]["decay"] == 8
    assert out.sims_used == 2


def test_sim_direct_toggle_decay_khong_sim_trung_cau_hinh_da_thu() -> None:
    """Finding #1 (CRITICAL): decay khởi điểm 4 (mặc định production), budget=2, sharpe dương
    yếu (>= ngưỡng) nhưng CHƯA pass ở cả 2 lần đầu -> toggle decay 4->8->4 khiến sim #3 TRÙNG
    y hệt sim #1 (cùng expr, cùng decay=4) — đốt quota + tạo alpha trùng vô ích. Sau khi nhớ
    tập (expr, cfg.key()) đã sim trong vòng sweep, biến thể trùng phải bị chặn -> DỪNG ở sim #2,
    KHÔNG có sim #3."""
    from src.simulation.config import SimConfig

    r1 = _sweep_result(_SWEEP_EXPR, status="failed", sharpe=0.6, fitness=0.5, alpha_id="wq-1")
    r2 = _sweep_result(_SWEEP_EXPR, status="failed", sharpe=0.7, fitness=0.55, alpha_id="wq-2")
    # Nếu bug còn sống, sim #3 sẽ được gọi lại với đúng cấu hình sim #1 (decay=4) -> để lộ
    # bug, "gài mìn" bằng cách nếu bị gọi lần 3 thì trả kết quả rất khác (sharpe cao ngất) mà
    # test assert KHÔNG được nhìn thấy.
    r3_neu_bi_goi_trung = _sweep_result(
        _SWEEP_EXPR, status="passed", sharpe=9.9, fitness=9.9, alpha_id="wq-3-KHONG-DUOC-GOI",
    )
    sim = _SeqSimulator([r1, r2, r3_neu_bi_goi_trung])
    refiner = _sweep_refiner(sim, sim_config=SimConfig.default().with_overrides(decay=4))

    out = refiner.refine_and_sim(_sweep_cand(_SWEEP_EXPR))

    assert sim.calls == 2, "sim #3 trùng hệt sim #1 (expr+decay=4) lẽ ra phải bị chặn"
    assert out.sims_used == 2
    assert out.sharpe == 0.7  # best trong 2 attempt thật, KHÔNG phải 9.9 (sim trùng ảo)
    # Không attempt nào lặp cấu hình: (expr, decay) của 2 lần sim phải khác nhau.
    seen_configs = {(e, s["decay"]) for e, s in sim.seen}
    assert len(seen_configs) == sim.calls


def test_sim_direct_khong_sweep_khi_sim_dau_da_pass() -> None:
    """sim #1 đã 'passed' -> dừng ngay (rule 4), không đốt thêm sim dù còn ngân sách."""
    r1 = _sweep_result(_SWEEP_EXPR, status="passed", sharpe=1.8, fitness=1.3, alpha_id="wq-1")
    sim = _SeqSimulator([r1])
    refiner = _sweep_refiner(sim)

    out = refiner.refine_and_sim(_sweep_cand(_SWEEP_EXPR))

    assert sim.calls == 1
    assert out.sims_used == 1
    assert out.passed is True


def test_sim_direct_quota_het_giua_luc_sweep_nem_quota_exhausted() -> None:
    """Sweep phải dừng gọn khi sim thêm (không phải sim #1) gặp QuotaExceededError -> map
    sang QuotaExhausted như code hiện tại (không nuốt lỗi rồi coi như 'sim thường')."""
    from src.pipeline.closed_loop import QuotaExhausted
    from src.simulation.simulator import QuotaExceededError

    class _BoomOnSecondCall:
        def __init__(self) -> None:
            self.calls = 0

        def simulate(self, expr, settings=None):
            self.calls += 1
            if self.calls == 1:
                return _sweep_result(_SWEEP_EXPR, status="failed", sharpe=-0.9, fitness=0.3)
            raise QuotaExceededError("hết quota ngày")

    refiner = _sweep_refiner(_BoomOnSecondCall())

    with pytest.raises(QuotaExhausted):
        refiner.refine_and_sim(_sweep_cand(_SWEEP_EXPR))


def test_sim_direct_quota_het_giua_sweep_van_luu_attempt_da_sim_that() -> None:
    """Finding #5 (Important): sim #1 (sim THẬT, đã tạo alpha thật trên Brain) không được
    finalize làm 'best' vì sim #2 (mini-sweep) ném QuotaExceededError giữa chừng -> toàn bộ
    hàm ném QuotaExhausted TRƯỚC khi chạm tới đoạn persist attempts (vốn chỉ chạy SAU vòng
    while) -> attempt #1 mất cả save_alpha/save_simulation (alpha mồ côi trên Brain, mất dấu
    vết audit/calibration). Sau fix: attempts đã sim THẬT (sim #1) phải được persist qua
    `_persist_sweep_attempt_thua` TRƯỚC khi re-raise QuotaExhausted."""
    from src.pipeline.closed_loop import QuotaExhausted
    from src.simulation.simulator import QuotaExceededError

    class _BoomOnSecondCall:
        def __init__(self) -> None:
            self.calls = 0

        def simulate(self, expr, settings=None):
            self.calls += 1
            if self.calls == 1:
                return _sweep_result(_SWEEP_EXPR, status="failed", sharpe=-0.9, fitness=0.3,
                                     alpha_id="wq-1")
            raise QuotaExceededError("hết quota ngày")

    refiner = _sweep_refiner(_BoomOnSecondCall())

    with pytest.raises(QuotaExhausted):
        refiner.refine_and_sim(_sweep_cand(_SWEEP_EXPR))

    repo = refiner.repo
    # sim #1 ĐÃ sim thật (alpha_id="wq-1") -> phải có bản ghi local dù QuotaExhausted đã raise.
    assert len(repo.saved) == 1
    assert repo.sim_saved == 1
    saved_expr, saved_source, saved_desc = repo.saved[0]
    assert saved_expr == _SWEEP_EXPR
    assert saved_source == "alt_data"
    assert saved_desc == "alt-data sweep attempt (thua)"


# --- Task 6: build_closed_loop nối simulator/sim_config của LocalTunerRefiner xuống
# AltDataIdeaSource -> batch core alt-data đầu tiên sim CẢ NHÓM 1 lần qua simulate_many, rồi
# refine_and_sim đọc lại cache thay vì sim lần 2. ------------------------------------------------


def test_build_closed_loop_noi_simulate_many_xuong_alt_data_source(small_panel, repo) -> None:  # noqa: ANN001
    """refiner=LocalTunerRefiner có simulator hỗ trợ simulate_many -> build_closed_loop tự gắn
    presim_cache dùng chung; next_batch() đầu tiên gọi simulate_many ĐÚNG 1 lần cho toàn bộ
    core alt-data (thay vì mỗi core đợi refine_and_sim tự sim tuần tự)."""
    from src.app.closed_loop_adapters import LocalTunerRefiner, build_closed_loop
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.generation.alt_data_seeds import ALT_DATA_CORES
    from src.lang.registry import default_registry
    from src.simulation.config import SimConfig
    from src.simulation.simulator import SimulationResult

    class _MultiSim:
        def __init__(self) -> None:
            self.multi_calls: list[list] = []
            self.single_calls = 0

        def simulate(self, expr, settings=None):
            self.single_calls += 1
            return SimulationResult(expression=expr, alpha_id="wq-single", status="passed", sharpe=1.0)

        def simulate_many(self, jobs):
            self.multi_calls.append(list(jobs))
            return [
                SimulationResult(expression=e, alpha_id=f"wq-{i}", status="passed", sharpe=1.2)
                for i, (e, _s) in enumerate(jobs)
            ]

    class _AlphaRepo:
        """Fake vai trò AlphaRepository (save_alpha/save_simulation) — khác `repo` fixture
        (MiniBrainRepository) build_closed_loop dùng cho GPIdeaSource/ClosedLoop, đúng như
        wiring thật (main.py: loop.repo là AlphaRepository, repo truyền ClosedLoop khác)."""

        def save_alpha(self, *a, **k):
            return "a1"

        def save_simulation(self, *a, **k):
            return None

    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    sim = _MultiSim()
    refiner = LocalTunerRefiner(
        simulator=sim, repo=_AlphaRepo(), data=small_panel,
        local_config=cfg, sim_config=SimConfig.default(),
    )

    class _NoopLoop:
        def run_from_seed(self, expression, on_progress=None):
            return type("R", (), {"best_candidate": None})()

    loop = build_closed_loop(data=small_panel, repo=repo, config=cfg,
                             registry=default_registry(), loop=_NoopLoop(),
                             pop_size=6, n_generations=0, top_k=3, max_ideas=2,
                             curated_seeds=False, include_fundamental=False,
                             include_hypothesis=False, include_combiner=False,
                             refiner=refiner)

    batch = loop.idea_source.next_batch()
    assert len(sim.multi_calls) == 1
    # Finding #1: max_ideas=2 -> presim_cap=2, chỉ 2 job vào batch multi-sim (KHÔNG sim cả 6
    # core rồi để ClosedLoop vứt 4 kết quả vượt trần — lãng phí quota lặp qua các phiên).
    assert len(sim.multi_calls[0]) == 2
    assert sim.single_calls == 0  # chưa refine_and_sim gì — chỉ next_batch() đã sim xong batch
    # Core vượt trần vẫn được yield làm candidate (đi đường sim đơn nếu tới lượt, không mất).
    assert len(batch) == len(ALT_DATA_CORES)

    # refine_and_sim CORE đầu tiên phải đọc lại cache, KHÔNG sim đơn lần 2.
    outcome = refiner.refine_and_sim(batch[0])
    assert sim.single_calls == 0
    assert outcome.wq_alpha_id == "wq-0"
