# Closed-Loop Phase 2 — Orchestrator `ClosedLoop` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (khuyến nghị) hoặc superpowers:executing-plans. Steps dùng checkbox (`- [ ]`). Phase 2 của
> feature "Vòng kín AI + MiniBrain" (spec
> `docs/superpowers/specs/2026-06-26-ai-minibrain-closed-loop-design.md`). Phase 1 (cầu DB
> `BrainSimLinkModel` + `record_brain_sim`/`load_brain_sims`/`brain_pnl_pool`) ĐÃ xong trên
> nhánh `closed-loop-integration`. Phase 3 (feedback) + Phase 4 (menu/wiring RefinementLoop
> thật) + Phase 5 (prompt) sau.

**Goal:** Viết orchestrator `ClosedLoop` thuần logic, network-agnostic: lặp lấy ý tưởng (GP→
short-list) → với mỗi ý tưởng gọi refiner-protocol (`refine_and_sim`, bọc RefinementLoop
thật ở phase sau) → ghi kết quả SIM qua `record_brain_sim` (cầu DB Phase 1) → tránh trùng →
dừng gọn khi hết quota Brain hoặc cạn ý tưởng. Test hoàn toàn bằng fake (không mạng/AI/sim).

**Architecture:** `src/pipeline/closed_loop.py` chứa `ClosedLoop` + `IdeaOutcome` +
`ClosedLoopReport` + `QuotaExhausted` + 2 Protocol structural (`_GeneratesIdeas`,
`_RefinesIdea`). KHÔNG import `src.llm`/`src.gp`/`src.simulation` — mọi dependency injected
qua Protocol/tham số (giống `generate_many` Phase 8). Việc DỰNG idea source thật (GPEngine+
build_shortlist) và refiner thật (bọc RefinementLoop) nằm ở `main.py`/adapter Phase 4.

**Tech Stack:** Python 3.12, dataclasses, typing.Protocol, pytest.

## Global Constraints

- Python 3.12; cú pháp hiện đại (`@dataclass(frozen=True, slots=True)`, `X | None`,
  `Protocol`).
- Full type hints; `mypy --strict --follow-imports=silent src/pipeline/closed_loop.py`
  clean; `ruff` clean; không unused import.
- **Dependency rule B1:** `src/pipeline/closed_loop.py` KHÔNG import `src.llm`/`src.gp`/
  `src.simulation`. Chỉ import `src.pipeline.shortlist` (ShortlistCandidate), `src.storage`
  (kiểu cho type hint repo — dùng Protocol nếu tránh được import cứng), stdlib.
- Determinism: không có randomness trong orchestrator (thứ tự ý tưởng do idea source quyết).
- **Tiếng Việt giữ dấu đúng chính tả** trong mọi docstring/comment mới.
- TDD: test trước (đỏ) → code tối thiểu (xanh) → commit. Mỗi task = 1 commit.
- KHÔNG đốt sim/AI thật trong test — chỉ fake.

## Pre-condition (chữ ký thật đã xác minh)

```python
# src/pipeline/shortlist.py (Phase 8)
@dataclass(frozen=True, slots=True)
class ShortlistCandidate:
    expr: str
    metrics: AlphaMetrics
    pnl: npt.NDArray[np.float64]
    dates: Dates

# src/storage/repository.py (Phase 1 — đã có)
class MiniBrainRepository:
    def record_brain_sim(self, canonical_hash, expr_string, *, wq_alpha_id, region,
        universe, sharpe, fitness, turnover, self_corr, status, raw_json=None) -> int: ...
    def load_brain_sims(self) -> list[BrainSimLinkModel]: ...
    def brain_pnl_pool(self) -> dict[str, float]: ...

# RefinementLoop (sẽ bọc ở Phase 4, KHÔNG dùng trực tiếp ở Phase 2):
#   run(direction: str) -> LoopResult(best_candidate, best_vector, sims_used, stop_reason, ...)
#   stop_reason ∈ {'abandon','budget','patience','no_seed'}; Brain hết quota -> Simulator ném lỗi.
```

## File Structure

- **Create** `src/pipeline/closed_loop.py` (~150 dòng): types + `ClosedLoop`.
- **Create** `tests/unit/test_closed_loop.py` (~180 dòng): test orchestration bằng fake.

---

### Task 1: Types — `IdeaOutcome`, `ClosedLoopReport`, `QuotaExhausted`, Protocols

**Files:**
- Create: `src/pipeline/closed_loop.py`
- Test: `tests/unit/test_closed_loop.py`

**Interfaces:**
- Consumes: `ShortlistCandidate` (src.pipeline.shortlist).
- Produces:
  ```python
  class QuotaExhausted(Exception): ...   # refiner ném khi Brain hết quota -> ClosedLoop dừng gọn

  @dataclass(frozen=True, slots=True)
  class IdeaOutcome:
      """Kết quả refine+sim một ý tưởng (do refiner-protocol trả về)."""
      expr: str
      canonical_hash: str
      passed: bool                 # alpha qua gate Brain (đáng giữ)
      wq_alpha_id: str | None
      sharpe: float | None
      fitness: float | None
      turnover: float | None
      self_corr: float | None
      sims_used: int               # số sim ý tưởng này tiêu
      stop_reason: str             # 'abandon'|'budget'|'patience'|'no_seed'|'passed'

  @dataclass(frozen=True, slots=True)
  class ClosedLoopReport:
      ideas_tried: int
      sims_used: int
      n_passed: int                # số ý tưởng passed Brain
      n_abandoned: int             # bỏ sau patience/abandon/no_seed
      stop_reason: str             # 'quota' | 'no_more_ideas' | 'interrupted'

  class _GeneratesIdeas(Protocol):
      def next_batch(self) -> list[ShortlistCandidate]: ...   # [] khi cạn ý tưởng

  class _RefinesIdea(Protocol):
      def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome: ...
      # ném QuotaExhausted khi Brain hết quota.
  ```

- [ ] **Step 1: Viết test đỏ (phần types) `tests/unit/test_closed_loop.py`**

```python
"""Test orchestrator ClosedLoop bằng fake (không mạng/AI/sim). Kiểm luồng: lấy ý tưởng →
refine+sim mỗi cái → record_brain_sim → tránh trùng → dừng khi hết quota / cạn ý tưởng."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.metrics_local import AlphaMetrics
from src.pipeline.closed_loop import (
    ClosedLoop,
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
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop.py -q
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.pipeline.closed_loop'`.

- [ ] **Step 3: Tạo `src/pipeline/closed_loop.py` (types + skeleton ClosedLoop)**

```python
"""ClosedLoop — orchestrator vòng kín AI + MiniBrain (thuần logic, network-agnostic).

Lặp: lấy ý tưởng (GP→short-list, qua `idea_source`) → với mỗi ý tưởng gọi `refiner.
refine_and_sim` (bọc RefinementLoop thật ở phase wiring) → ghi kết quả SIM qua
`repo.record_brain_sim` (cầu DB Phase 1) → tránh trùng → dừng gọn khi Brain hết quota
(`QuotaExhausted`) hoặc cạn ý tưởng.

Dependency rule B1: KHÔNG import `src.llm`/`src.gp`/`src.simulation` — mọi dependency
injected qua Protocol structural; việc dựng cụ thể nằm ở `main.py`/adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.pipeline.shortlist import ShortlistCandidate


class QuotaExhausted(Exception):
    """Refiner ném khi Brain hết quota SIM — ClosedLoop dừng vòng gọn, persist mọi thứ."""


@dataclass(frozen=True, slots=True)
class IdeaOutcome:
    """Kết quả refine+sim một ý tưởng (refiner-protocol trả về)."""

    expr: str
    canonical_hash: str
    passed: bool
    wq_alpha_id: str | None
    sharpe: float | None
    fitness: float | None
    turnover: float | None
    self_corr: float | None
    sims_used: int
    stop_reason: str


@dataclass(frozen=True, slots=True)
class ClosedLoopReport:
    """Thống kê một lần chạy ClosedLoop + lý do dừng."""

    ideas_tried: int
    sims_used: int
    n_passed: int
    n_abandoned: int
    stop_reason: str


class _GeneratesIdeas(Protocol):
    def next_batch(self) -> list[ShortlistCandidate]: ...


class _RefinesIdea(Protocol):
    def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome: ...
```

- [ ] **Step 4: Chạy test — PASS (2 test)**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop.py -q
```
Expected: 2 PASS.

- [ ] **Step 5: ruff + mypy + kiểm dấu + commit**

```bash
venv/Scripts/python.exe -m ruff check src/pipeline/closed_loop.py tests/unit/test_closed_loop.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/pipeline/closed_loop.py
git add src/pipeline/closed_loop.py tests/unit/test_closed_loop.py
git commit -m "feat(pipeline): types ClosedLoop - IdeaOutcome/ClosedLoopReport/QuotaExhausted/Protocols"
```
Expected: ruff + mypy sạch.

---

### Task 2: `ClosedLoop.run()` — vòng lặp lấy ý tưởng → refine+sim → persist → dừng

**Files:**
- Modify: `src/pipeline/closed_loop.py` (thêm `class ClosedLoop`)
- Test: `tests/unit/test_closed_loop.py` (thêm test)

**Interfaces:**
- Consumes: `IdeaOutcome`, `QuotaExhausted`, `_GeneratesIdeas`, `_RefinesIdea`,
  `ShortlistCandidate` (Task 1), `MiniBrainRepository.record_brain_sim` (Phase 1).
- Produces:
  ```python
  class ClosedLoop:
      def __init__(self, idea_source: _GeneratesIdeas, refiner: _RefinesIdea,
                   repo: MiniBrainRepository, *, region: str = "USA",
                   universe: str = "TOP3000", max_ideas: int | None = None) -> None: ...
      def run(self) -> ClosedLoopReport: ...
  ```

**Hành vi `run()` (đặt vào docstring):**
1. Lặp: `batch = idea_source.next_batch()`; `batch == []` → dừng (`stop_reason='no_more_ideas'`).
2. Mỗi `cand` trong batch: bỏ qua nếu `cand.expr` đã thấy (tránh refine trùng trong phiên).
3. Gọi `refiner.refine_and_sim(cand)`; bắt `QuotaExhausted` → dừng (`stop_reason='quota'`),
   trả report với số liệu tới thời điểm đó.
4. Ghi outcome qua `repo.record_brain_sim(canonical_hash=outcome.canonical_hash,
   expr_string=outcome.expr, wq_alpha_id=..., region, universe, sharpe/fitness/turnover/
   self_corr, status='passed' nếu outcome.passed else 'failed')`.
5. Đếm: `ideas_tried += 1`; `sims_used += outcome.sims_used`; `n_passed/n_abandoned`.
6. `max_ideas` (None = không trần) — trần an toàn tùy chọn để test/chạy ngắn; đạt → dừng
   (`stop_reason='no_more_ideas'`).

- [ ] **Step 1: Viết test đỏ (thêm vào `tests/unit/test_closed_loop.py`)**

```python
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
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop.py -q
```
Expected: FAIL `AttributeError: 'ClosedLoop'`/`ImportError` (chưa có class ClosedLoop).

- [ ] **Step 3: Thêm `class ClosedLoop` vào `src/pipeline/closed_loop.py`**

Thêm import `MiniBrainRepository` (cho type hint) vào đầu file:
```python
from src.storage.repository import MiniBrainRepository
```
Thêm class (sau các Protocol):
```python
class ClosedLoop:
    """Vòng kín: lấy ý tưởng → refine+sim mỗi cái → persist kết quả SIM → dừng khi hết
    quota / cạn ý tưởng. Thuần điều phối; mọi dependency injected."""

    def __init__(
        self,
        idea_source: _GeneratesIdeas,
        refiner: _RefinesIdea,
        repo: MiniBrainRepository,
        *,
        region: str = "USA",
        universe: str = "TOP3000",
        max_ideas: int | None = None,
    ) -> None:
        self.idea_source = idea_source
        self.refiner = refiner
        self.repo = repo
        self.region = region
        self.universe = universe
        self.max_ideas = max_ideas

    def run(self) -> ClosedLoopReport:
        """Lặp: next_batch → mỗi ý tưởng refine_and_sim → record_brain_sim → đếm. Dừng khi
        batch rỗng (cạn ý tưởng), đạt max_ideas, hoặc refiner ném QuotaExhausted (hết quota
        Brain). Bỏ qua expr đã thấy trong phiên (tránh refine trùng)."""
        ideas_tried = 0
        sims_used = 0
        n_passed = 0
        n_abandoned = 0
        seen: set[str] = set()

        while True:
            batch = self.idea_source.next_batch()
            if not batch:
                return ClosedLoopReport(ideas_tried, sims_used, n_passed, n_abandoned,
                                        "no_more_ideas")
            for cand in batch:
                if self.max_ideas is not None and ideas_tried >= self.max_ideas:
                    return ClosedLoopReport(ideas_tried, sims_used, n_passed, n_abandoned,
                                            "no_more_ideas")
                if cand.expr in seen:
                    continue
                seen.add(cand.expr)
                try:
                    outcome = self.refiner.refine_and_sim(cand)
                except QuotaExhausted:
                    return ClosedLoopReport(ideas_tried, sims_used, n_passed, n_abandoned,
                                            "quota")
                self.repo.record_brain_sim(
                    canonical_hash=outcome.canonical_hash, expr_string=outcome.expr,
                    wq_alpha_id=outcome.wq_alpha_id, region=self.region,
                    universe=self.universe, sharpe=outcome.sharpe, fitness=outcome.fitness,
                    turnover=outcome.turnover, self_corr=outcome.self_corr,
                    status="passed" if outcome.passed else "failed",
                )
                ideas_tried += 1
                sims_used += outcome.sims_used
                if outcome.passed:
                    n_passed += 1
                else:
                    n_abandoned += 1
```

- [ ] **Step 4: Chạy test — PASS (7 test tổng)**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop.py -q
```
Expected: 7 PASS.

- [ ] **Step 5: ruff + mypy + kiểm dấu tiếng Việt**

```bash
venv/Scripts/python.exe -m ruff check src/pipeline/closed_loop.py tests/unit/test_closed_loop.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/pipeline/closed_loop.py
```
Expected: ruff sạch. mypy sạch (file mới — nếu than `MiniBrainRepository` import kéo lỗi
baseline từ src.storage, dùng `--follow-imports=silent` đã chặn; lỗi chỉ-trong file mới phải
sạch).

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/closed_loop.py tests/unit/test_closed_loop.py
git commit -m "feat(pipeline): ClosedLoop.run - orchestrator vong kin (idea->refine+sim->persist->dung)"
```

---

## Self-review

**Spec coverage (Phase 2 scope):**
- [x] Orchestrator nối GP→shortlist (idea_source) → refine+sim (refiner protocol) →
  persist (record_brain_sim) — Task 1+2.
- [x] Dừng gọn khi Brain hết quota (`QuotaExhausted`) — Task 2 test `test_run_stops_on_quota`.
- [x] Tránh trùng (avoid refine lặp trong phiên) — Task 2 test `skips_duplicate`.
- [x] Network-agnostic, test bằng fake hoàn toàn — mọi test dùng `_FakeIdeaSource`/
  `_FakeRefiner`.
- [x] Dependency rule B1 (không import src.llm/src.gp/src.simulation) — chỉ import shortlist
  + storage.
- [~] Feedback 4 cơ chế — KHÔNG thuộc Phase 2 (Phase 3); Phase 2 chỉ persist outcome thô.
- [~] Wiring RefinementLoop thật thành `_RefinesIdea` — KHÔNG thuộc Phase 2 (Phase 4 adapter).

**Placeholder scan:** ✅ Mọi step có code/lệnh cụ thể.

**Type consistency:**
- `IdeaOutcome(expr, canonical_hash, passed, wq_alpha_id, sharpe, fitness, turnover,
  self_corr, sims_used, stop_reason)` — nhất quán Task 1 def, Task 2 test fakes, `run()`
  consume.
- `record_brain_sim(canonical_hash=, expr_string=, wq_alpha_id=, region=, universe=,
  sharpe=, fitness=, turnover=, self_corr=, status=)` — khớp chữ ký Phase 1 (đã merge).
- `ClosedLoop.__init__(idea_source, refiner, repo, *, region, universe, max_ideas)` +
  `run() -> ClosedLoopReport` — khớp Task 2 Interfaces + mọi test.
- `_GeneratesIdeas.next_batch() -> list[ShortlistCandidate]`, `_RefinesIdea.
  refine_and_sim(candidate) -> IdeaOutcome` — khớp fakes.

**Risks / gotchas:**
1. `seen` theo `expr` (chuỗi) — đủ cho v1; avoid-list bền (theo canonical_hash, qua DB) là
   Phase 3.
2. `record_brain_sim` mở/đóng session mỗi ý tưởng — chấp nhận (không hot path; sim Brain
   chậm hơn nhiều). Tối ưu sau nếu cần.
3. `max_ideas=None` mặc định = chạy đến hết quota/ý tưởng (đúng yêu cầu "không trần"); test
   dùng `max_ideas` để chặn idea_source vô hạn.
