# Closed-Loop Phase 3 — Feedback DB-driven (avoid-list + calibrate ρ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development hoặc
> superpowers:executing-plans. Steps dùng checkbox (`- [ ]`). Phase 3 của feature "Vòng kín
> AI + MiniBrain" (spec `docs/superpowers/specs/2026-06-26-ai-minibrain-closed-loop-design.md`).
> Phase 1 (cầu DB `BrainSimLinkModel`) + Phase 2 (`ClosedLoop`) ĐÃ xong trên nhánh
> `closed-loop-integration`.

**Goal:** Nối 2 feedback THUẦN DỮ LIỆU (không cần AI/Brain sống) vào vòng kín: (b)
**avoid-list bền** — bỏ qua expression đã fail SIM Brain ở phiên trước; (c) **tự tái
calibrate ρ** — sau mỗi N sim tính lại Spearman ρ (local sharpe vs Brain sharpe) từ dữ liệu
đã lưu, cảnh báo nếu ρ tụt. Hai feedback còn lại (pool tầng-2 Brain qua `get_alpha_pnl`/
`check_correlation`; AI học từ SIM + dead-field-từ-Brain) phụ thuộc AI/Brain sống → để
**Phase 4 (adapter)** nơi dựng RefinementLoop/Simulator/Brain API thật.

**Architecture:** Thêm 2 query DB-driven vào `MiniBrainRepository` (`avoided_exprs`,
`brain_local_sharpe_pairs`) + class `CalibrationTracker` (`src/pipeline/closed_loop.py`,
dùng `spearman` của `src.calibration.stats`) + wire avoid-list & tracker vào `ClosedLoop.run`
(backward-compatible: tham số mặc định no-op nên test Phase 2 vẫn xanh).

**Tech Stack:** Python 3.12, SQLAlchemy, numpy, `src.calibration.stats.spearman`, pytest.

## Global Constraints

- Python 3.12; full type hints; `mypy --strict --follow-imports=silent` clean trên code mới
  (`closed_loop.py`); `ruff` clean; không unused import.
- **Dependency rule B1:** `closed_loop.py` được phép import `src.calibration.stats` (stdlib-ish,
  thuần hàm) + `src.storage.repository`; KHÔNG import `src.llm`/`src.gp`/`src.simulation`.
- Backward-compatible: KHÔNG phá test Phase 2 (`tests/unit/test_closed_loop.py` hiện có 7 test
  xanh). Tham số mới của `ClosedLoop` có default no-op.
- **Tiếng Việt giữ dấu đúng chính tả** trong docstring/comment mới.
- TDD: test trước (đỏ) → code (xanh) → commit. Mỗi task = 1 commit.

## Pre-condition (chữ ký thật đã xác minh)

```python
# src/storage/models.py
class BrainSimLinkModel(Base):  # canonical_hash, expr_string, sharpe, fitness, turnover,
                                # self_corr, status ('passed'|'failed'|'error'), region, universe
class ExpressionModel(Base):    # id, canonical_hash (unique)
class EvaluationModel(Base):    # expression_id (FK), sharpe (Float|None), status

# src/storage/repository.py — MiniBrainRepository (session pattern: s=self.session_factory();
#   try: ...; finally: s.close()). Đã có record_brain_sim/load_brain_sims/brain_pnl_pool.

# src/calibration/stats.py
def spearman(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    # pairwise-complete (loại cặp NaN); trả NaN nếu < 2 cặp hợp lệ hoặc std=0.

# src/pipeline/closed_loop.py (Phase 2)
@dataclass(frozen=True, slots=True)
class ClosedLoopReport:  # ideas_tried, sims_used, n_passed, n_abandoned, stop_reason
class ClosedLoop:
    def __init__(self, idea_source, refiner, repo, *, region="USA", universe="TOP3000",
                 max_ideas=None) -> None: ...
    def run(self) -> ClosedLoopReport: ...  # vòng: next_batch → refine_and_sim → record_brain_sim
```

## File Structure

- **Modify** `src/storage/repository.py` (~40 dòng): `avoided_exprs`, `brain_local_sharpe_pairs`.
- **Modify** `src/pipeline/closed_loop.py` (~55 dòng): `CalibrationTracker` + thêm field
  `rho_sharpe` vào `ClosedLoopReport` + wire vào `ClosedLoop`.
- **Modify** `tests/unit/test_brain_sim_link.py` (~50 dòng): test 2 query.
- **Modify** `tests/unit/test_closed_loop.py` (~70 dòng): test tracker + wire.

---

### Task 1: Repo queries `avoided_exprs` + `brain_local_sharpe_pairs`

**Files:**
- Modify: `src/storage/repository.py`
- Test: `tests/unit/test_brain_sim_link.py`

**Interfaces:**
- Consumes: `BrainSimLinkModel`, `ExpressionModel`, `EvaluationModel`.
- Produces (thêm vào `MiniBrainRepository`):
  ```python
  def avoided_exprs(self) -> set[str]: ...
  # {expr_string} của brain_sim_links status='failed' (đã fail SIM Brain -> tránh refine lại).

  def brain_local_sharpe_pairs(self) -> list[tuple[float, float]]: ...
  # [(local_sharpe, brain_sharpe)] cho expression có CẢ brain_sim_links.sharpe != None VÀ
  # một EvaluationModel.sharpe != None (match qua canonical_hash). Cho calibrate ρ.
  ```

- [ ] **Step 1: Viết test đỏ (thêm vào `tests/unit/test_brain_sim_link.py`)**

```python
def test_avoided_exprs_returns_failed_expr_strings(repo) -> None:  # noqa: ANN001
    repo.record_brain_sim("hf", "rank(volume)", wq_alpha_id=None, region="USA",
                          universe="TOP3000", sharpe=0.0, fitness=0.0, turnover=0.0,
                          self_corr=None, status="failed")
    repo.record_brain_sim("hp", "rank(close)", wq_alpha_id="W", region="USA",
                          universe="TOP3000", sharpe=1.5, fitness=1.2, turnover=0.2,
                          self_corr=0.3, status="passed")
    assert repo.avoided_exprs() == {"rank(volume)"}  # chỉ cái failed


def test_brain_local_sharpe_pairs_matches_by_canonical_hash(repo) -> None:  # noqa: ANN001
    # 1 expression có cả local eval + brain sim -> ghép cặp; 1 chỉ có brain -> bỏ.
    expr_id = repo.upsert_expression("rank(close)", "hX", 2, 3, {"close"})
    from src.backtest.metrics_local import AlphaMetrics
    m = AlphaMetrics(sharpe=0.8, annual_return=0.1, turnover=0.2, max_drawdown=0.05,
                     fitness=1.0, per_year_sharpe={2021: 1.0}, weight_concentration=0.05)
    repo.record_evaluation(expr_id, "{}", "default", m, 0.0, "passed", [], 42)
    repo.record_brain_sim("hX", "rank(close)", wq_alpha_id="W", region="USA",
                          universe="TOP3000", sharpe=1.6, fitness=1.3, turnover=0.2,
                          self_corr=0.3, status="passed")
    repo.record_brain_sim("hNoLocal", "open", wq_alpha_id="W2", region="USA",
                          universe="TOP3000", sharpe=2.0, fitness=2.0, turnover=0.1,
                          self_corr=0.1, status="passed")  # không có local eval -> bỏ
    pairs = repo.brain_local_sharpe_pairs()
    assert pairs == [(0.8, 1.6)]  # chỉ hX ghép được (local 0.8, brain 1.6)
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_brain_sim_link.py -q
```
Expected: FAIL `AttributeError: ... has no attribute 'avoided_exprs'`.

- [ ] **Step 3: Thêm 2 method vào `MiniBrainRepository` (`src/storage/repository.py`)**

Đảm bảo `ExpressionModel`, `EvaluationModel`, `BrainSimLinkModel` đã trong khối import models
(BrainSimLinkModel thêm ở Phase 1; ExpressionModel/EvaluationModel đã có). Thêm:

```python
    def avoided_exprs(self) -> set[str]:
        """Trả {expr_string} của các link Brain SIM status='failed' — vòng kín bỏ qua, tránh
        refine lại ý tưởng đã hỏng trên Brain (avoid-list bền B11)."""
        session = self.session_factory()
        try:
            rows = (
                session.query(BrainSimLinkModel.expr_string)
                .filter(BrainSimLinkModel.status == "failed")
                .all()
            )
            return {r[0] for r in rows}
        finally:
            session.close()

    def brain_local_sharpe_pairs(self) -> list[tuple[float, float]]:
        """Trả [(local_sharpe, brain_sharpe)] cho expression có CẢ local evaluation lẫn Brain
        sim (match theo canonical_hash), cả hai sharpe != None. Phục vụ calibrate ρ Spearman
        (local vs Brain). Mỗi canonical_hash lấy 1 local sharpe (eval đầu tiên có sharpe)."""
        session = self.session_factory()
        try:
            pairs: list[tuple[float, float]] = []
            links = (
                session.query(BrainSimLinkModel)
                .filter(BrainSimLinkModel.sharpe.isnot(None))
                .all()
            )
            for link in links:
                expr = (
                    session.query(ExpressionModel)
                    .filter_by(canonical_hash=link.canonical_hash)
                    .first()
                )
                if expr is None:
                    continue
                ev = (
                    session.query(EvaluationModel)
                    .filter(EvaluationModel.expression_id == expr.id)
                    .filter(EvaluationModel.sharpe.isnot(None))
                    .order_by(EvaluationModel.id)
                    .first()
                )
                if ev is None:
                    continue
                pairs.append((float(ev.sharpe), float(link.sharpe)))
            return pairs
        finally:
            session.close()
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_brain_sim_link.py -q
```
Expected: tất cả PASS (6 cũ + 2 mới).

- [ ] **Step 5: ruff + mypy + kiểm dấu + commit**

```bash
venv/Scripts/python.exe -m ruff check src/storage/repository.py tests/unit/test_brain_sim_link.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/storage/repository.py
git add src/storage/repository.py tests/unit/test_brain_sim_link.py
git commit -m "feat(storage): avoided_exprs + brain_local_sharpe_pairs cho feedback vong kin"
```
mypy: nếu `float(ev.sharpe)`/`link.sharpe` báo no-any (ORM động) — chấp nhận pattern
`# type: ignore` như method khác; nhưng `float(...)` thường đã ép kiểu nên sạch.

---

### Task 2: `CalibrationTracker` + wire avoid-list & ρ vào `ClosedLoop`

**Files:**
- Modify: `src/pipeline/closed_loop.py`
- Test: `tests/unit/test_closed_loop.py`

**Interfaces:**
- Consumes: `spearman` (src.calibration.stats), `MiniBrainRepository.brain_local_sharpe_pairs`/
  `avoided_exprs` (Task 1), `ClosedLoop`/`ClosedLoopReport` (Phase 2).
- Produces:
  ```python
  class CalibrationTracker:
      def __init__(self, repo: MiniBrainRepository, *, every: int = 10,
                   rho_bar: float = 0.5) -> None: ...
      def maybe_calibrate(self, sims_total: int) -> float | None: ...
      # Khi sims_total vượt mốc bội số `every` kể từ lần trước -> tính spearman trên
      # brain_local_sharpe_pairs(); lưu self.last_rho; trả ρ (hoặc None nếu chưa tới mốc /
      # < 2 cặp). ρ < rho_bar -> log cảnh báo.
  # ClosedLoopReport: THÊM field `rho_sharpe: float | None = None`.
  # ClosedLoop.__init__: THÊM tham số `calibration_tracker: CalibrationTracker | None = None`.
  # ClosedLoop.run: nạp avoid-list từ repo.avoided_exprs() lúc bắt đầu (gộp vào `seen`);
  #   sau mỗi record_brain_sim, gọi tracker.maybe_calibrate(sims_used) nếu có; report kèm
  #   rho_sharpe = tracker.last_rho.
  ```

- [ ] **Step 1: Viết test đỏ (thêm vào `tests/unit/test_closed_loop.py`)**

```python
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
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop.py -q
```
Expected: FAIL (`ImportError: CalibrationTracker` / `ClosedLoop` chưa nhận `calibration_tracker`).

- [ ] **Step 3: Sửa `src/pipeline/closed_loop.py`**

Thêm import đầu file:
```python
import logging

import numpy as np

from src.calibration.stats import spearman

logger = logging.getLogger(__name__)
```
Thêm field vào `ClosedLoopReport` (cuối, có default — backward-compatible):
```python
    rho_sharpe: float | None = None
```
Thêm class `CalibrationTracker` (trước `class ClosedLoop`):
```python
class CalibrationTracker:
    """Theo dõi độ tin ranking local: sau mỗi `every` sim, tính lại Spearman ρ giữa local
    sharpe và Brain sharpe (trên các expression đã có cả hai). ρ < `rho_bar` -> cảnh báo
    (ranking local có thể không còn đáng tin -> nên điều tra data/operator fidelity)."""

    def __init__(
        self, repo: MiniBrainRepository, *, every: int = 10, rho_bar: float = 0.5,
    ) -> None:
        self.repo = repo
        self.every = every
        self.rho_bar = rho_bar
        self.last_rho: float | None = None
        self._last_mark = 0

    def maybe_calibrate(self, sims_total: int) -> float | None:
        """Tính ρ nếu `sims_total` đã qua mốc bội số `every` kể từ lần trước; ngược lại None.
        ρ tính qua `spearman` trên `brain_local_sharpe_pairs()` (NaN nếu < 2 cặp)."""
        if sims_total < self._last_mark + self.every:
            return None
        self._last_mark = sims_total - (sims_total % self.every)
        pairs = self.repo.brain_local_sharpe_pairs()
        if len(pairs) < 2:
            self.last_rho = None
            return None
        local = np.array([p[0] for p in pairs], dtype=np.float64)
        brain = np.array([p[1] for p in pairs], dtype=np.float64)
        rho = spearman(local, brain)
        self.last_rho = rho
        if not np.isnan(rho) and rho < self.rho_bar:
            logger.warning("Calibration ρ=%.3f < bar %.2f — ranking local kém tin", rho, self.rho_bar)
        return rho
```
Sửa `ClosedLoop.__init__` thêm tham số:
```python
        calibration_tracker: CalibrationTracker | None = None,
```
và `self.calibration_tracker = calibration_tracker` trong thân.

Sửa `ClosedLoop.run`:
- Ngay sau `seen: set[str] = set()`, thêm: `seen |= self.repo.avoided_exprs()` (nạp avoid-list
  bền lúc bắt đầu — expr đã fail Brain bị bỏ qua).
- Sau khối `record_brain_sim(...)` + tăng `sims_used`, thêm:
  ```python
                if self.calibration_tracker is not None:
                    self.calibration_tracker.maybe_calibrate(sims_used)
  ```
- Mọi điểm `return ClosedLoopReport(...)` thêm tham số cuối:
  `rho_sharpe=self.calibration_tracker.last_rho if self.calibration_tracker else None`.

(Để DRY, có thể gom tạo report vào 1 helper `_report(self, ...)` — tùy chọn; nếu giữ inline
thì cả 3 điểm return phải thêm `rho_sharpe=...` nhất quán.)

- [ ] **Step 4: Chạy test — PASS (Phase 2: 7 cũ + 3 mới = 10)**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop.py -q
```
Expected: 10 PASS. Xác nhận 7 test Phase 2 KHÔNG vỡ (avoid-list rỗng + tracker None = no-op).

- [ ] **Step 5: ruff + mypy + kiểm dấu tiếng Việt**

```bash
venv/Scripts/python.exe -m ruff check src/pipeline/closed_loop.py tests/unit/test_closed_loop.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/pipeline/closed_loop.py
```
Expected: sạch.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/closed_loop.py tests/unit/test_closed_loop.py
git commit -m "feat(pipeline): CalibrationTracker + wire avoid-list & rho vao ClosedLoop"
```

---

## Self-review

**Spec coverage (Phase 3 scope đã chốt):**
- [x] (b) avoid-list bền — `avoided_exprs` (Task 1) + wire `seen |= avoided_exprs()` (Task 2).
- [x] (c) tự tái calibrate ρ — `brain_local_sharpe_pairs` (Task 1) + `CalibrationTracker`
  (Task 2) + `ClosedLoopReport.rho_sharpe`.
- [~] (a) pool tầng-2 Brain (`get_alpha_pnl`/`check_correlation`) — DEFER Phase 4 (cần Brain API sống).
- [~] (b-phần dead-field-từ-Brain) + (d) AI học từ SIM — DEFER Phase 4/5 (cần adapter AI/sim sống).
- [x] Backward-compatible Phase 2 (tham số default no-op) — Task 2 Step 4 verify 7 test cũ.

**Placeholder scan:** ✅ Mọi step có code/lệnh cụ thể.

**Type consistency:**
- `avoided_exprs() -> set[str]`, `brain_local_sharpe_pairs() -> list[tuple[float, float]]` —
  khớp Task 1 def, test, Task 2 consume.
- `CalibrationTracker(repo, *, every, rho_bar)`, `.maybe_calibrate(sims_total) -> float|None`,
  `.last_rho` — khớp Task 2 def + test.
- `ClosedLoopReport` thêm `rho_sharpe: float | None = None` (default) — không phá construction
  5-field ở test Phase 2.
- `ClosedLoop.__init__(..., calibration_tracker=None)` — backward-compatible.

**Risks / gotchas:**
1. avoid-list theo expr_string (không canonical_hash) — đủ cho v1 (serializer cho chuỗi ổn
   định); canonical-hash-based là refinement sau.
2. `maybe_calibrate` mốc theo bội số `every`: dùng `_last_mark` để không tính lại nhiều lần
   trong cùng mốc; test `every=2` kiểm.
3. `spearman` trả NaN khi < 2 cặp — `CalibrationTracker` trả None ở nhánh đó, không gán NaN
   gây nhiễu; nhưng nếu ≥2 cặp mà std=0 thì last_rho=NaN (chấp nhận; report mang NaN, không crash).
4. `seen |= avoided_exprs()` nạp 1 lần lúc bắt đầu run — alpha fail TRONG phiên vẫn được
   `seen.add` xử lý riêng; reload giữa phiên là tối ưu sau.
