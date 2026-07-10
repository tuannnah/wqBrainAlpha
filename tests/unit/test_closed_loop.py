"""Test orchestrator ClosedLoop bằng fake (không mạng/AI/sim). Kiểm luồng: lấy ý tưởng →
refine+sim mỗi cái → record_brain_sim → tránh trùng → dừng khi hết quota / cạn ý tưởng."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.metrics_local import AlphaMetrics
from src.pipeline.closed_loop import (
    ClosedLoop,  # noqa: F401  — dùng ở test Task 2+
    ClosedLoopReport,
    IdeaOutcome,
    QuotaExhausted,
)
from src.pipeline.shortlist import ShortlistCandidate
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


def _cand(expr: str) -> ShortlistCandidate:
    m = AlphaMetrics(sharpe=1.0, annual_return=0.1, turnover=0.3, max_drawdown=0.05,
                     fitness=1.0, per_year_sharpe={2021: 1.0}, weight_concentration=0.05)
    dates = (np.datetime64("2021-01-01") + np.arange(5)).astype("datetime64[D]")
    return ShortlistCandidate(expr=expr, metrics=m, pnl=np.ones(5), dates=dates)


@pytest.fixture
def repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    sf = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(sf)


def test_idea_outcome_and_report_are_frozen() -> None:
    o = IdeaOutcome(expr="close", canonical_hash="h", passed=True, wq_alpha_id="W",
                    sharpe=1.0, fitness=1.0, turnover=0.2, self_corr=0.3, sims_used=1,
                    stop_reason="passed")
    with pytest.raises(Exception):  # FrozenInstanceError  # noqa: PT011
        o.passed = False  # type: ignore[misc]
    r = ClosedLoopReport(ideas_tried=0, sims_used=0, n_passed=0, n_abandoned=0,
                         stop_reason="no_more_ideas")
    with pytest.raises(Exception):  # noqa: PT011
        r.sims_used = 9  # type: ignore[misc]


def test_quota_exhausted_is_exception() -> None:
    assert issubclass(QuotaExhausted, Exception)


class _FakeIdeaSource:
    """Trả các batch cố định rồi cạn ([] -> ClosedLoop dừng)."""

    def __init__(self, batches: list[list[ShortlistCandidate]]) -> None:
        self._batches = list(batches)

    def next_batch(self) -> list[ShortlistCandidate]:
        return self._batches.pop(0) if self._batches else []


class _FakeRefiner:
    """Trả IdeaOutcome theo map expr->outcome; expr không có map -> failed mặc định.
    Nếu expr nằm trong `quota_on` -> ném QuotaExhausted (giả lập Brain hết quota)."""

    def __init__(self, outcomes: dict[str, IdeaOutcome], quota_on: set[str] | None = None) -> None:
        self._outcomes = outcomes
        self._quota_on = quota_on or set()
        self.calls: list[str] = []

    def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome:
        self.calls.append(candidate.expr)
        if candidate.expr in self._quota_on:
            raise QuotaExhausted("het quota")
        return self._outcomes.get(
            candidate.expr,
            IdeaOutcome(expr=candidate.expr, canonical_hash="h_" + candidate.expr,
                        passed=False, wq_alpha_id=None, sharpe=None, fitness=None,
                        turnover=None, self_corr=None, sims_used=1, stop_reason="patience"),
        )


def _passed(expr: str) -> IdeaOutcome:
    return IdeaOutcome(expr=expr, canonical_hash="h_" + expr, passed=True,
                       wq_alpha_id="WQ_" + expr, sharpe=1.5, fitness=1.2, turnover=0.2,
                       self_corr=0.3, sims_used=2, stop_reason="passed")


def test_run_persists_each_outcome_and_counts(repo) -> None:  # noqa: ANN001
    src = _FakeIdeaSource([[_cand("close"), _cand("open")]])
    refiner = _FakeRefiner({"close": _passed("close")})  # open -> failed mặc định
    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo)
    report = loop.run()
    assert isinstance(report, ClosedLoopReport)
    assert report.ideas_tried == 2
    assert report.n_passed == 1
    assert report.n_abandoned == 1
    assert report.sims_used == 3  # 2 (close passed) + 1 (open failed)
    assert report.stop_reason == "no_more_ideas"
    sims = repo.load_brain_sims()
    assert len(sims) == 2
    assert {s.status for s in sims} == {"passed", "failed"}


def test_run_stops_on_quota_exhausted(repo) -> None:  # noqa: ANN001
    src = _FakeIdeaSource([[_cand("a"), _cand("b"), _cand("c")]])
    refiner = _FakeRefiner({"a": _passed("a")}, quota_on={"b"})  # b -> hết quota
    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo)
    report = loop.run()
    assert report.stop_reason == "quota"
    assert report.ideas_tried == 1   # chỉ 'a' xong; 'b' ném quota trước khi tính
    assert refiner.calls == ["a", "b"]  # 'c' không bao giờ được gọi
    assert len(repo.load_brain_sims()) == 1  # chỉ 'a' kịp ghi


def test_run_skips_duplicate_expr_within_session(repo) -> None:  # noqa: ANN001
    src = _FakeIdeaSource([[_cand("dup"), _cand("dup")]])
    refiner = _FakeRefiner({"dup": _passed("dup")})
    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo)
    report = loop.run()
    assert refiner.calls == ["dup"]  # lần 2 bị bỏ qua
    assert report.ideas_tried == 1


def test_run_stops_on_empty_batch(repo) -> None:  # noqa: ANN001
    src = _FakeIdeaSource([])  # cạn ngay
    refiner = _FakeRefiner({})
    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo)
    report = loop.run()
    assert report.ideas_tried == 0
    assert report.stop_reason == "no_more_ideas"


def test_run_respects_max_ideas(repo) -> None:  # noqa: ANN001
    # idea_source vô hạn (mỗi batch 1 ý tưởng mới) -> max_ideas chặn.
    class _Infinite:
        def __init__(self) -> None:
            self.i = 0

        def next_batch(self) -> list[ShortlistCandidate]:
            self.i += 1
            return [_cand(f"x{self.i}")]

    loop = ClosedLoop(idea_source=_Infinite(), refiner=_FakeRefiner({}), repo=repo,
                      max_ideas=3)
    report = loop.run()
    assert report.ideas_tried == 3
    assert report.stop_reason == "no_more_ideas"


def test_calibration_tracker_computes_rho_at_interval(repo) -> None:  # noqa: ANN001
    from src.pipeline.closed_loop import CalibrationTracker
    # seed 3 cặp (local, brain) tương quan dương hoàn hảo -> rho=1.0
    from src.backtest.metrics_local import AlphaMetrics
    for i, (ls, bs) in enumerate([(0.5, 1.0), (1.0, 2.0), (1.5, 3.0)]):
        eid = repo.upsert_expression(f"e{i}", f"h{i}", 1, 1, {"close"})
        m = AlphaMetrics(sharpe=ls, annual_return=0.1, turnover=0.2, max_drawdown=0.05,
                         fitness=1.0, per_year_sharpe={2021: 1.0}, weight_concentration=0.05)
        repo.record_evaluation(eid, "{}", "default", m, 0.0, "passed", [], 1)
        repo.record_brain_sim(f"h{i}", f"e{i}", wq_alpha_id=None, region="USA",
                              universe="TOP3000", sharpe=bs, fitness=1.0, turnover=0.2,
                              self_corr=0.1, status="passed")
    tr = CalibrationTracker(repo, every=2, rho_bar=0.5)
    assert tr.maybe_calibrate(1) is None      # chưa tới mốc (1 < 2)
    rho = tr.maybe_calibrate(2)               # tới mốc bội số 2
    assert rho is not None
    assert rho == pytest.approx(1.0, abs=1e-9)
    assert tr.last_rho == pytest.approx(1.0, abs=1e-9)


def test_closed_loop_skips_avoided_exprs_from_db(repo) -> None:  # noqa: ANN001
    # pre-seed 1 expr failed trên Brain -> ClosedLoop phải bỏ qua, không refine lại.
    repo.record_brain_sim("hbad", "bad_expr", wq_alpha_id=None, region="USA",
                          universe="TOP3000", sharpe=0.0, fitness=0.0, turnover=0.0,
                          self_corr=None, status="failed")
    src = _FakeIdeaSource([[_cand("bad_expr"), _cand("good_expr")]])
    refiner = _FakeRefiner({"good_expr": _passed("good_expr")})
    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo)
    report = loop.run()
    assert refiner.calls == ["good_expr"]   # bad_expr bị avoid-list bỏ qua
    assert report.ideas_tried == 1


def test_closed_loop_report_includes_rho_when_tracker_set(repo) -> None:  # noqa: ANN001
    from src.pipeline.closed_loop import CalibrationTracker
    src = _FakeIdeaSource([[_cand("close")]])
    refiner = _FakeRefiner({"close": _passed("close")})
    tracker = CalibrationTracker(repo, every=1, rho_bar=0.5)
    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo,
                      calibration_tracker=tracker)
    report = loop.run()
    # 1 cặp (close) -> spearman < 2 cặp -> NaN -> last_rho có thể NaN; report.rho_sharpe gán
    # từ tracker.last_rho (không crash). Cốt lõi: field tồn tại + vòng chạy xong.
    assert hasattr(report, "rho_sharpe")
    assert report.ideas_tried == 1


def test_run_logs_progress(repo) -> None:  # noqa: ANN001
    """ClosedLoop.run phát log tiến trình (loguru -> hiện màn hình khi chạy mục 5): batch,
    mỗi ý tưởng, kết quả sim, và tổng running. Trước đây run() im lặng hoàn toàn."""
    from loguru import logger

    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        src = _FakeIdeaSource([[_cand("close"), _cand("open")]])
        refiner = _FakeRefiner({"close": _passed("close")})  # open -> failed mặc định
        ClosedLoop(idea_source=src, refiner=refiner, repo=repo).run()
    finally:
        logger.remove(sink_id)

    joined = "\n".join(msgs)
    assert "Batch:" in joined and "ứng viên" in joined          # log batch
    assert "Ý tưởng #1" in joined and "close" in joined         # log mỗi ý tưởng + expr
    assert "PASSED" in joined                                    # kết quả pass của 'close'
    assert "ý tưởng /" in joined and "sim /" in joined          # dòng tổng running


def test_run_logs_local_gate_block_when_zero_sims(repo) -> None:  # noqa: ANN001
    """Outcome sims_used=0 (bị gate local chặn trước sim) được log rõ, không lẫn với sim thật."""
    from loguru import logger

    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        blocked = IdeaOutcome(expr="blk", canonical_hash="h_blk", passed=False,
                              wq_alpha_id=None, sharpe=None, fitness=None, turnover=None,
                              self_corr=None, sims_used=0, stop_reason="local_gate")
        src = _FakeIdeaSource([[_cand("blk")]])
        ClosedLoop(idea_source=src, refiner=_FakeRefiner({"blk": blocked}), repo=repo).run()
    finally:
        logger.remove(sink_id)
    assert "gate local chặn" in "\n".join(msgs)


def test_closed_loop_goi_alpha_logger_moi_y_tuong():
    """ClosedLoop gọi alpha_logger.log cho mỗi ý tưởng có outcome (index tăng dần)."""
    from src.pipeline.closed_loop import ClosedLoop, IdeaOutcome
    from src.pipeline.shortlist import ShortlistCandidate
    import numpy as np

    cand = ShortlistCandidate(
        expr="rank(close)", metrics=None, pnl=np.zeros(2),
        dates=np.arange("2020-01-01", "2020-01-03", dtype="datetime64[D]"),
    )

    class _Src:
        def __init__(self):
            self.done = False
        def next_batch(self):
            if self.done:
                return []
            self.done = True
            return [cand]

    class _Ref:
        def refine_and_sim(self, c):
            return IdeaOutcome(
                expr=c.expr, canonical_hash="h", passed=True, wq_alpha_id="wq",
                sharpe=1.5, fitness=1.1, turnover=0.3, self_corr=0.2, sims_used=1,
                stop_reason="ok",
            )

    class _Repo:
        def avoided_exprs(self): return set()
        def record_brain_sim(self, **kw): return None

    logged = []

    class _Logger:
        def log(self, index, outcome):
            logged.append((index, outcome.expr))

    cl = ClosedLoop(_Src(), _Ref(), _Repo(), max_ideas=1, alpha_logger=_Logger())
    cl.run()
    assert logged == [(1, "rank(close)")]


def test_family_budget_dong_ho_khi_can_ma_khong_pass(repo) -> None:  # noqa: ANN001
    """Pha 2.2: family budget — họ sinh >= max_per_family ứng viên mà 0 pass -> ĐÓNG họ,
    candidate cùng họ sau đó bị bỏ (chuyển ngân sách sang họ khác)."""
    src = _FakeIdeaSource([[_cand("pv_a"), _cand("pv_b"), _cand("pv_c"), _cand("mom_a")]])
    refiner = _FakeRefiner({})  # tất cả fail mặc định

    def family_fn(expr: str) -> str:
        return "pv" if expr.startswith("pv") else "mom"

    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo,
                      family_fn=family_fn, max_per_family=2)
    loop.run()
    # pv: chỉ 2 đầu được refine (pv_c bị đóng họ); mom_a vẫn refine (họ khác còn ngân sách)
    assert refiner.calls == ["pv_a", "pv_b", "mom_a"]


def test_family_closed_goi_callback(repo) -> None:  # noqa: ANN001
    """Pha 2.3: khi đóng họ, on_family_closed nhận set họ bão hoà (để nối LLM prompt)."""
    src = _FakeIdeaSource([[_cand("pv_a"), _cand("pv_b")]])
    refiner = _FakeRefiner({})
    seen_sets: list[set] = []

    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo,
                      family_fn=lambda e: "pv", max_per_family=2,
                      on_family_closed=lambda fams: seen_sets.append(set(fams)))
    loop.run()
    assert seen_sets and "pv" in seen_sets[-1]


def test_family_budget_khong_dong_khi_co_pass(repo) -> None:  # noqa: ANN001
    """Họ có ít nhất 1 pass thì KHÔNG đóng dù vượt max_per_family (họ còn tiềm năng)."""
    src = _FakeIdeaSource([[_cand("pv_a"), _cand("pv_b"), _cand("pv_c")]])
    refiner = _FakeRefiner({"pv_a": _passed("pv_a")})  # pv_a pass

    def family_fn(expr: str) -> str:
        return "pv"

    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo,
                      family_fn=family_fn, max_per_family=2)
    loop.run()
    assert refiner.calls == ["pv_a", "pv_b", "pv_c"]  # không đóng vì đã có pass


def test_dedup_theo_canonical_key_chan_bien_the_scale(repo) -> None:  # noqa: ANN001
    """Pha 1.2: dedup dùng dedup_key_fn (canonical) TRƯỚC refine -> multiply(2,X) và
    multiply(4,X) coi là trùng, chỉ refine 1 lần (không tốn 2 backtest)."""
    src = _FakeIdeaSource([[_cand("multiply(2, close)"), _cand("multiply(4, close)")]])
    refiner = _FakeRefiner({})

    def dedup_key(expr: str) -> str:
        # Giả canonical: strip hệ số multiply dương -> cùng key.
        import re
        return re.sub(r"multiply\(\d+(\.\d+)?,\s*", "", expr).rstrip(")")

    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo, dedup_key_fn=dedup_key)
    loop.run()
    assert len(refiner.calls) == 1  # biến thể scale thứ 2 bị chặn ở dedup


def test_dedup_nap_avoided_hashes_cross_session(repo) -> None:  # noqa: ANN001
    """Pha 1.2: avoid-list cross-session dùng hash. Expr có key nằm trong avoided_hashes ->
    chặn ngay, không refine."""
    src = _FakeIdeaSource([[_cand("multiply(2, close)")]])
    refiner = _FakeRefiner({})

    class _RepoAvoid:
        def avoided_exprs(self):
            return set()
        def avoided_hashes(self):
            return {"close"}  # key đã fail phiên trước
        def record_brain_sim(self, **kw):
            return None

    def dedup_key(expr: str) -> str:
        import re
        return re.sub(r"multiply\(\d+(\.\d+)?,\s*", "", expr).rstrip(")")

    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=_RepoAvoid(), dedup_key_fn=dedup_key)
    loop.run()
    assert refiner.calls == []  # bị chặn bởi avoid-list hash cross-session


def test_gen_ms_duoc_dien_vao_outcome(repo) -> None:  # noqa: ANN001
    """Fix gap Pha 0: ClosedLoop đo thời gian next_batch (GP generation) và điền gen_ms vào
    outcome (refiner không biết chi phí sinh batch). Trước đây gen_ms luôn None -> cột 'gen'
    trong funnel luôn '—'."""
    src = _FakeIdeaSource([[_cand("close"), _cand("open")]])
    refiner = _FakeRefiner({})
    logged = []

    class _Logger:
        def log(self, index, outcome):
            logged.append(outcome)

    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo, alpha_logger=_Logger())
    loop.run()
    assert logged
    for o in logged:
        assert o.gen_ms is not None and o.gen_ms >= 0.0


def test_closed_loop_feed_session_summary(repo) -> None:  # noqa: ANN001
    """ClosedLoop record mỗi outcome vào session_summary + đếm dup bị chặn (Pha 0)."""
    from src.reporting.session_summary import SessionSummary

    src = _FakeIdeaSource([[_cand("close"), _cand("open"), _cand("close")]])  # "close" lặp
    refiner = _FakeRefiner({"close": _passed("close")})
    summary = SessionSummary()
    loop = ClosedLoop(idea_source=src, refiner=refiner, repo=repo, session_summary=summary)
    loop.run()
    d = summary.as_dict()
    assert d["total"] == 2          # close + open (close lần 2 bị chặn, không refine)
    assert d["dup_blocked"] == 1    # close lần 2 tính dup
    assert d["passed"] == 1
