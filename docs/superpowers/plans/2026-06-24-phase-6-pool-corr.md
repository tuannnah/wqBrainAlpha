# Phase 6 — Pool Correlation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (khuyến nghị) hoặc superpowers:executing-plans để thực thi từng task. Steps dùng checkbox
> (`- [ ]`) để theo dõi; chạy tuần tự — Task 6.1 (`pool_corr.py`, độc lập) → Task 6.2
> (`gates.py`, phụ thuộc 6.1) → Task 6.3 (`repository.py` + điểm tích hợp ghi pool, phụ thuộc
> 6.1) → Task 6.4 (integration end-to-end) → Task 6.5 (review + merge + push, luôn cuối).

**Goal:** Biến self-correlation từ tham số `self_corr: float` do caller tự truyền tay
(Phase 4, `GateEvaluator.evaluate`) thành một **giá trị tính thật** từ pool các alpha đã
pass, lưu trong DB — đây là tính năng đòn bẩy cao nhất của MiniBrain (B9 master design:
"the single highest-leverage thing MiniBrain can do that Brain can't do for free"). Cụ
thể: `PoolCorrelation` đọc `dict[id, pnl-vector(+dates)]` từ pool, tính `max|Pearson ρ|`
của một candidate PnL so với từng alpha trong pool (align trên ngày giao nhau), trả
`(max|ρ|, worst_alpha_id)`; wire kết quả này vào `GateEvaluator`/đường gọi gate thay cho
giá trị `self_corr` truyền tay; và khi một alpha **pass** gate, ghi PnL của nó vào pool để
các candidate sau so sánh.

**Architecture:** `src/backtest/pool_corr.py` mới, cùng tầng với `metrics_local.py`/
`gates.py` (Phase 4) — **không** import `src/gp`, `src/storage`, `src/llm` (dependency rule
B1/master plan: `backtest` không phụ thuộc `storage`). `PoolCorrelation` nhận pool đã
**vật chất hóa trong RAM** (`dict[int, tuple[Dates, Panel-1D]]`) qua constructor — việc đọc
pool từ DB là trách nhiệm của `src/storage/repository.py` (`load_pool`), gọi ở tầng
`pipeline`/`loop` (ngoài phạm vi `backtest`), rồi truyền kết quả vào `PoolCorrelation`.
Việc **ghi** PnL khi pass (Task 6.3) chạm `src/storage/repository.py` (tầng `storage`, được
phép phụ thuộc `backtest` một chiều — đối xứng với pattern Phase 4 Task 4.4
`scoring.filter.evaluate_local` phụ thuộc `backtest.gates`) và điểm gọi trong loop/pipeline.

**Tech Stack:** Python 3.12, numpy, pytest, ruff, mypy --strict. Không thêm dependency mới.

## Global Constraints

- Python 3.12; cú pháp hiện đại (`match`, `X | None`, `type` alias, `@dataclass(frozen=True, slots=True)`, `Protocol`).
- Full type hints; `mypy --strict` clean; `ruff` clean; không unused import.
- **No look-ahead:** time-series ops chỉ đọc rows ≤ t; thiếu lịch sử → NaN.
- **No survivorship:** universe mask per-day; out-of-universe = NaN (không phải 0).
- **Delay-1:** `pnl_t = nansum(weights_{t-1} * returns_t)`.
- **Stage separation:** expression = signal core; neut/decay/trunc/scale/delay ở `PortfolioConfig`.
- **Thresholds chỉ ở `config/thresholds.py`** — không hardcode gate number ở call site.
- **Determinism:** randomness qua seed inject; ghi seed vào DB.
- **WQ operator fidelity:** tra skill `worldquant-brain` trước khi viết FASTEXPR/operator.
- **TDD:** test trước, đỏ → code tối thiểu → xanh → commit. Mỗi phase = 1 nhánh git → merge → push.
- **Per-phase ritual:** Design → Implement → Explain → Review (test+ruff+mypy) → Gate → Journal (`PROGRESS.md`).

## Pre-condition (đọc trước khi bắt đầu) — quan trọng: Phase 5 plan CHƯA tồn tại

Tại thời điểm viết plan này, kiểm tra repo thật cho thấy:

```bash
venv/Scripts/python.exe -c "from src.backtest.gates import GateEvaluator, GateVerdict; from src.backtest.metrics_local import AlphaMetrics; print('phase4 ok')"
venv/Scripts/python.exe -c "from src.storage.repository import AlphaRepository; print('repo ok')"
```

- `src/backtest/gates.py` + `metrics_local.py` (Phase 4) là **plan đã viết step-by-step**
  (`docs/superpowers/plans/2026-06-24-phase-4-metrics-gates.md`) — giả định đã merge vào
  `main` khi Phase 6 bắt đầu. Chữ ký dùng trong plan này (`GateEvaluator.evaluate(self, m,
  self_corr, depth, fields_ok) -> GateVerdict`) lấy nguyên văn từ đó.
- `src/storage/repository.py` **đã tồn tại trong repo thật** nhưng **chưa có** các API mà
  master plan Phase 5 mô tả (`upsert_expression`, `record_evaluation`, `load_pool`,
  `add_dead_field`, `result_cache_get/put`, `top_n`) — file hiện tại chỉ có
  `AlphaRepository` (save_alpha/save_simulation/get_cached_simulation/record_failure/
  recent_failures/top_simulated/zoo) cho luồng sim-Brain cũ, và `InvalidFieldRepository`.
  **Không có file plan Phase 5 step-by-step** (`docs/superpowers/plans/2026-06-24-phase-5-database.md`
  không tồn tại) — chỉ có dòng tóm tắt trong master plan (B11 schema +
  `load_pool()`/`save_pool_pnl` mô tả ở mức gợi ý trong yêu cầu nhiệm vụ này).

**Quyết định (ghi rõ vì không có đặc tả Phase 5 chi tiết để khớp 100%):**

1. **Chữ ký `load_pool()` giả định cho Phase 6**, khớp B11 schema
   (`pool_pnl(evaluation_id PK, dates_blob, pnl_blob)`) và B9 design
   (`PoolCorrelation.__init__(pool_pnls: dict[int, npt.NDArray[np.float64]])`):

   ```python
   def load_pool(self) -> dict[int, tuple[Dates, Panel1D]]: ...
   #   Panel1D = npt.NDArray[np.float64]  shape (T,)
   #   key = evaluation_id (PK của bảng pool_pnl theo B11, KHÔNG phải alpha_id —
   #         một expression có thể có nhiều evaluation dưới config/window khác nhau,
   #         mỗi evaluation pass riêng một entry pool)
   ```

   Lý do trả `tuple[Dates, Panel1D]` (kèm dates) thay vì chỉ `Panel1D`: B11 lưu cả
   `dates_blob` và `pnl_blob` trong cùng dòng `pool_pnl` — không có dates thì không align
   được khi các alpha trong pool có lịch sử dài khác nhau (alpha cũ có nhiều ngày hơn alpha
   mới). Nếu Phase 5 thật triển khai `load_pool()` trả `dict[int, Panel1D]` (không kèm
   dates, ngụ ý "mọi pool entry cùng chung một trục ngày cố định"), `PoolCorrelation` ở
   Task 6.1 vẫn tương thích được — xem "Lối thoát nếu Phase 5 khác giả định" cuối Task 6.1.

2. **Chữ ký `save_pool_pnl`** (đối xứng `load_pool`, theo B11):

   ```python
   def save_pool_pnl(self, evaluation_id: int, dates: Dates, pnl: Panel1D) -> None: ...
   ```

3. Nếu khi thực thi Task 6.3, `src/storage/repository.py` thật (sau khi Phase 5 thật chạy)
   có chữ ký khác — DỪNG, đọc lại implementation thật, sửa Task 6.3 cho khớp (không tự ý vá
   Phase 5). Task 6.1/6.2 (`pool_corr.py`, `gates.py`) **không phụ thuộc trực tiếp** vào
   `repository.py` nên không bị ảnh hưởng nếu chữ ký DB lệch — đây là lý do thiết kế
   `PoolCorrelation` nhận `dict` đã vật chất hóa thay vì tự đọc DB (giữ đúng dependency rule
   B1: `backtest` không import `storage`).

```bash
venv/Scripts/python.exe -c "from config.thresholds import SELF_CORR_MAX; print(SELF_CORR_MAX)"
```
Expected: `0.7`. Nếu lỗi, Phase 0 chưa merge — DỪNG, báo cáo block.

---

### Task 6.1: `PoolCorrelation` (`src/backtest/pool_corr.py`)

**Files:**
- Create: `src/backtest/pool_corr.py`
- Test: `tests/unit/test_pool_corr.py`

**Interfaces:**
- Consumes: `Dates`, `Panel` (Phase 0, `src/local_types.py` — dùng `Panel` cho PnL 1-D dù
  tên alias gốc là cho `(T,N)`; ở đây dùng `npt.NDArray[np.float64]` shape `(T,)` trực tiếp,
  không ép kiểu `Panel` 2-D cho rõ ràng — xem ghi chú type alias trong code mẫu dưới).
- Produces:

  ```python
  # src/backtest/pool_corr.py
  class PoolCorrelation:
      def __init__(
          self, pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]]
      ) -> None: ...

      def max_corr(
          self, candidate_pnl: npt.NDArray[np.float64], dates: Dates
      ) -> tuple[float, int | None]: ...
  ```

  - `pool` rỗng (`{}`) → `max_corr` luôn trả `(0.0, None)` bất kể candidate.
  - Với mỗi `(pool_id, (pool_dates, pool_pnl))` trong pool: align `dates` (candidate) và
    `pool_dates` trên **giao ngày chung** (`np.intersect1d` hoặc tương đương); nếu giao nhau
    có **< 2 điểm hữu hạn** (sau khi loại NaN ở cả hai phía) → coi alpha đó là "không so
    sánh được", **bỏ qua** khỏi việc tính max (không tính ρ=0 giả, không làm worst_id sai
    lệch) — ghi rõ docstring lý do (Pearson cần ≥2 điểm để có phương sai xác định).
  - ρ = Pearson correlation trên các điểm align được (`np.corrcoef` hoặc tính tay
    `cov/std`); nếu std của một trong hai vector bằng 0 trên đoạn giao → bỏ qua alpha đó
    (cùng lý do — Pearson không xác định).
  - Trả `(max(|ρ_i|), id của alpha cho |ρ| lớn nhất)`; nếu **mọi** alpha trong pool đều bị
    bỏ qua (không đủ overlap/variance) → trả `(0.0, None)` (coi như không có thông tin, an
    toàn theo hướng không chặn oan).
  - **Không** dùng `pandas` — toàn bộ bằng numpy (tech stack Phase 6 chỉ thêm numpy/pytest,
    đã có sẵn từ Phase 0).

- [ ] **Step 1: Tạo nhánh từ main sạch**

```bash
git checkout main && git pull --ff-only 2>/dev/null; git checkout -b phase-6-pool-corr
git status
```
Expected: "On branch phase-6-pool-corr", working tree clean.

- [ ] **Step 2: Viết test đỏ — pair PnL có ρ biết trước + align dates lệch + pool rỗng**

```python
# tests/unit/test_pool_corr.py
"""Test PoolCorrelation.max_corr: Pearson |rho| align trên dates chung, pool rỗng ->
(0.0, None), bỏ qua alpha không đủ overlap/variance để so sánh."""

from __future__ import annotations

import numpy as np

from src.backtest.pool_corr import PoolCorrelation


def _dates(start: str, n: int) -> np.ndarray:
    return (np.datetime64(start) + np.arange(n)).astype("datetime64[D]")


def test_empty_pool_returns_zero_and_none():
    pc = PoolCorrelation(pool={})
    candidate = np.array([0.01, -0.02, 0.03])
    rho, worst_id = pc.max_corr(candidate, _dates("2021-01-01", 3))
    assert rho == 0.0
    assert worst_id is None


def test_identical_series_gives_rho_one():
    pnl = np.array([0.01, -0.02, 0.03, 0.01, -0.01])
    dates = _dates("2021-01-01", 5)
    pc = PoolCorrelation(pool={1: (dates, pnl.copy())})
    rho, worst_id = pc.max_corr(pnl.copy(), dates)
    assert np.isclose(rho, 1.0, atol=1e-9)
    assert worst_id == 1


def test_sign_flipped_series_gives_rho_minus_one_abs_one():
    pnl = np.array([0.01, -0.02, 0.03, 0.01, -0.01])
    dates = _dates("2021-01-01", 5)
    pc = PoolCorrelation(pool={1: (dates, pnl.copy())})
    rho, worst_id = pc.max_corr(-pnl.copy(), dates)
    # Pearson(x, -x) = -1 -> |rho| = 1
    assert np.isclose(rho, 1.0, atol=1e-9)
    assert worst_id == 1


def test_independent_series_gives_low_rho():
    rng = np.random.default_rng(42)
    pool_pnl = rng.normal(size=2000)
    candidate_pnl = rng.normal(size=2000)  # độc lập (seed khác draw)
    dates = _dates("2021-01-01", 2000)
    pc = PoolCorrelation(pool={1: (dates, pool_pnl)})
    rho, worst_id = pc.max_corr(candidate_pnl, dates)
    assert rho < 0.10  # độc lập -> rho gần 0, ngưỡng lỏng tránh flaky
    assert worst_id == 1


def test_picks_worst_alpha_id_as_max_abs_rho_across_pool():
    dates = _dates("2021-01-01", 5)
    base = np.array([0.01, -0.02, 0.03, 0.01, -0.01])
    rng = np.random.default_rng(7)
    pool = {
        1: (dates, rng.normal(size=5)),       # độc lập, |rho| thấp
        2: (dates, base.copy()),              # giống hệt candidate -> |rho|=1
        3: (dates, rng.normal(size=5)),       # độc lập, |rho| thấp
    }
    pc = PoolCorrelation(pool=pool)
    rho, worst_id = pc.max_corr(base.copy(), dates)
    assert np.isclose(rho, 1.0, atol=1e-9)
    assert worst_id == 2


def test_partial_date_overlap_aligns_on_intersection_only():
    # Pool alpha có lịch sử dài hơn candidate; chỉ 3 ngày cuối trùng nhau.
    pool_dates = _dates("2021-01-01", 6)
    pool_pnl = np.array([100.0, 100.0, 100.0, 0.01, -0.02, 0.03])  # 3 đầu là nhiễu lớn
    candidate_dates = _dates("2021-01-04", 3)  # trùng 3 ngày cuối của pool
    candidate_pnl = np.array([0.01, -0.02, 0.03])  # giống hệt phần overlap

    pc = PoolCorrelation(pool={9: (pool_dates, pool_pnl)})
    rho, worst_id = pc.max_corr(candidate_pnl, candidate_dates)
    assert np.isclose(rho, 1.0, atol=1e-9)
    assert worst_id == 9


def test_no_date_overlap_is_skipped_not_zero_forced():
    pool_dates = _dates("2020-01-01", 3)
    pool_pnl = np.array([0.01, -0.02, 0.03])
    candidate_dates = _dates("2025-01-01", 3)  # không trùng ngày nào
    candidate_pnl = np.array([0.05, 0.05, 0.05])

    pc = PoolCorrelation(pool={5: (pool_dates, pool_pnl)})
    rho, worst_id = pc.max_corr(candidate_pnl, candidate_dates)
    assert rho == 0.0
    assert worst_id is None  # không đủ overlap -> bỏ qua, không phải "so sánh ra 0"


def test_zero_variance_pool_series_is_skipped():
    dates = _dates("2021-01-01", 5)
    flat_pnl = np.full(5, 0.02)  # std = 0 -> Pearson không xác định
    candidate_pnl = np.array([0.01, -0.02, 0.03, 0.01, -0.01])

    pc = PoolCorrelation(pool={3: (dates, flat_pnl)})
    rho, worst_id = pc.max_corr(candidate_pnl, dates)
    assert rho == 0.0
    assert worst_id is None
```

- [ ] **Step 3: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_pool_corr.py -v
```
Expected: FAIL `ModuleNotFoundError: No module named 'src.backtest.pool_corr'`.

- [ ] **Step 4: Tạo `src/backtest/pool_corr.py`**

```python
# src/backtest/pool_corr.py
"""PoolCorrelation — self-correlation cục bộ, tính năng đòn bẩy cao nhất của MiniBrain
(B9 master design): max |Pearson rho| của PnL candidate so với từng alpha đã PASS trong
pool, align trên ngày giao nhau. Đây là PROXY LOCAL, miễn phí quota; checker thật của Brain
là authoritative trước khi submit thật — không thay thế, chỉ lọc trước.

Pool được truyền vào dưới dạng dict đã vật chất hóa trong RAM (đọc từ DB ở tầng
storage/pipeline, KHÔNG ở đây) — pool_corr.py không import src.storage để giữ dependency
rule (lang/operators_local/engine/backtest không phụ thuộc storage/gp/llm).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.local_types import Dates

_MIN_OVERLAP_POINTS = 2  # Pearson cần >=2 điểm hữu hạn để có phương sai xác định


class PoolCorrelation:
    """Max |Pearson rho| của candidate PnL so với từng alpha trong pool."""

    def __init__(
        self, pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]]
    ) -> None:
        self._pool = pool

    def max_corr(
        self, candidate_pnl: npt.NDArray[np.float64], dates: Dates
    ) -> tuple[float, int | None]:
        if not self._pool:
            return 0.0, None

        best_abs_rho = 0.0
        best_id: int | None = None

        for pool_id, (pool_dates, pool_pnl) in self._pool.items():
            rho = self._pairwise_rho(candidate_pnl, dates, pool_pnl, pool_dates)
            if rho is None:
                continue
            abs_rho = abs(rho)
            if best_id is None or abs_rho > best_abs_rho:
                best_abs_rho = abs_rho
                best_id = pool_id

        if best_id is None:
            return 0.0, None
        return best_abs_rho, best_id

    def _pairwise_rho(
        self,
        candidate_pnl: npt.NDArray[np.float64],
        candidate_dates: Dates,
        pool_pnl: npt.NDArray[np.float64],
        pool_dates: Dates,
    ) -> float | None:
        common = np.intersect1d(candidate_dates, pool_dates)
        if common.size < _MIN_OVERLAP_POINTS:
            return None

        cand_idx = np.searchsorted(candidate_dates, common)
        pool_idx = np.searchsorted(pool_dates, common)
        cand_aligned = candidate_pnl[cand_idx]
        pool_aligned = pool_pnl[pool_idx]

        finite = np.isfinite(cand_aligned) & np.isfinite(pool_aligned)
        if finite.sum() < _MIN_OVERLAP_POINTS:
            return None
        cand_aligned = cand_aligned[finite]
        pool_aligned = pool_aligned[finite]

        if cand_aligned.std(ddof=0) == 0.0 or pool_aligned.std(ddof=0) == 0.0:
            return None

        rho = float(np.corrcoef(cand_aligned, pool_aligned)[0, 1])
        if not np.isfinite(rho):
            return None
        return rho
```

- [ ] **Step 5: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_pool_corr.py -v
```
Expected: PASS (8 test).

- [ ] **Step 6: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/backtest/pool_corr.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/backtest/pool_corr.py
```
Expected: cả hai sạch.

- [ ] **Step 7: Commit**

```bash
git add src/backtest/pool_corr.py tests/unit/test_pool_corr.py
git commit -m "feat(backtest): PoolCorrelation max|rho| align dates chung, bo qua alpha khong du overlap/variance"
```

> **Lối thoát nếu Phase 5 thật khác giả định (`load_pool()` trả `dict[int, Panel1D]` không
> kèm dates, nghĩa là pool dùng chung một trục ngày cố định):** `PoolCorrelation` ở trên vẫn
> hoạt động đúng nếu tầng gọi (`pipeline`/loop, Task 6.3) tự cặp `dates` chung đó vào mỗi
> entry trước khi truyền vào constructor — không cần sửa `pool_corr.py`. Chỉ Task 6.3 (điểm
> đọc/viết DB) cần sửa cho khớp chữ ký `load_pool` thật.

---

### Task 6.2: Wire `PoolCorrelation` vào `GateEvaluator` (`src/backtest/gates.py`)

**Files:**
- Modify: `src/backtest/gates.py` (Phase 4 — đọc toàn văn trước khi sửa; **không đổi** chữ
  ký `GateEvaluator.evaluate(self, m, self_corr, depth, fields_ok) -> GateVerdict` hiện có —
  nhiều caller Phase 4 (`scoring/filter.evaluate_local`, test Phase 4) đã phụ thuộc chữ ký
  này; Task 6.2 **thêm** một entrypoint mới thay vì sửa chữ ký cũ).
- Test: `tests/unit/test_gates_pool_corr.py`

**Interfaces:**
- Consumes: `PoolCorrelation` (Task 6.1, `src/backtest/pool_corr.py`), `GateEvaluator`
  (Phase 4, `src/backtest/gates.py`).
- Produces: thêm method mới trên `GateEvaluator`:

  ```python
  # THÊM vào class GateEvaluator trong src/backtest/gates.py — KHÔNG sửa evaluate() cũ
  def evaluate_with_pool(
      self,
      m: AlphaMetrics,
      candidate_pnl: npt.NDArray[np.float64],
      candidate_dates: Dates,
      pool_corr: PoolCorrelation,
      depth: int,
      fields_ok: bool,
  ) -> GateVerdict:
      """Như evaluate(), nhưng self_corr được TÍNH THẬT từ pool_corr.max_corr(...) thay vì
      nhận tay. Đây là entrypoint Phase 6+ nên dùng; evaluate() cũ vẫn giữ cho test/caller
      đã truyền self_corr tính sẵn (ví dụ calibration harness Phase 4.5 dùng self_corr từ
      brain_record, không qua pool local)."""
      self_corr, _worst_id = pool_corr.max_corr(candidate_pnl, candidate_dates)
      return self.evaluate(m, self_corr=self_corr, depth=depth, fields_ok=fields_ok)
  ```

  Lý do **thêm method mới thay vì sửa `evaluate()`**: `evaluate()` (Phase 4) nhận
  `self_corr: float` đã tính — đúng tách bạch "GateEvaluator đánh giá AlphaMetrics đã tính
  sẵn, không tự backtest" (docstring Phase 4 gốc). `PoolCorrelation` cần `candidate_pnl` +
  `dates` (không phải `AlphaMetrics`), nên ép vào `evaluate()` sẽ phá chữ ký đã có test phụ
  thuộc. `evaluate_with_pool` là lớp mỏng gọi `max_corr` rồi delegate — giữ nguyên logic hard
  gate/soft score của `evaluate()` (không trùng lặp).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_gates_pool_corr.py
"""Test GateEvaluator.evaluate_with_pool: self_corr tinh THAT tu PoolCorrelation.max_corr,
khong phai gia tri truyen tay; hard gate SELF_CORR_MAX van ap dung dung nhu evaluate()."""

from __future__ import annotations

import numpy as np

from config.thresholds import SELF_CORR_MAX
from src.backtest.gates import GateEvaluator
from src.backtest.metrics_local import AlphaMetrics
from src.backtest.pool_corr import PoolCorrelation


def _dates(start: str, n: int) -> np.ndarray:
    return (np.datetime64(start) + np.arange(n)).astype("datetime64[D]")


def _good_metrics() -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.5, annual_return=0.20, turnover=0.30, max_drawdown=0.10,
        fitness=2.0, per_year_sharpe={2021: 1.2}, weight_concentration=0.05,
    )


def test_evaluate_with_pool_passes_when_pool_empty():
    verdict = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=np.array([0.01, -0.01, 0.02]),
        candidate_dates=_dates("2021-01-01", 3), pool_corr=PoolCorrelation(pool={}),
        depth=3, fields_ok=True,
    )
    assert verdict.passed is True
    assert verdict.hard_failures == []


def test_evaluate_with_pool_hard_fails_when_identical_to_pool_alpha():
    dates = _dates("2021-01-01", 10)
    pnl = np.linspace(0.01, 0.10, 10)
    pool_corr = PoolCorrelation(pool={1: (dates, pnl.copy())})
    verdict = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=pnl.copy(), candidate_dates=dates,
        pool_corr=pool_corr, depth=3, fields_ok=True,
    )
    assert verdict.passed is False
    assert any("self_corr" in f for f in verdict.hard_failures)


def test_evaluate_with_pool_uses_same_threshold_as_evaluate():
    dates = _dates("2021-01-01", 10)
    pnl = np.linspace(0.01, 0.10, 10)
    pool_corr = PoolCorrelation(pool={1: (dates, pnl.copy())})
    rho, _ = pool_corr.max_corr(pnl.copy(), dates)
    assert rho >= SELF_CORR_MAX  # identical series -> rho=1.0 >= 0.70

    verdict_pool = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=pnl.copy(), candidate_dates=dates,
        pool_corr=pool_corr, depth=3, fields_ok=True,
    )
    verdict_manual = GateEvaluator().evaluate(
        _good_metrics(), self_corr=rho, depth=3, fields_ok=True
    )
    assert verdict_pool.hard_failures == verdict_manual.hard_failures
    assert verdict_pool.soft_scores == verdict_manual.soft_scores


def test_evaluate_unchanged_signature_still_works():
    # evaluate() Phase 4 KHONG bi pha vo boi Task 6.2
    verdict = GateEvaluator().evaluate(_good_metrics(), self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.passed is True
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gates_pool_corr.py -v
```
Expected: FAIL `AttributeError: 'GateEvaluator' object has no attribute 'evaluate_with_pool'`.

- [ ] **Step 3: Thêm `evaluate_with_pool` vào `src/backtest/gates.py`**

```python
# THÊM vào đầu file src/backtest/gates.py, trong khối import hiện có:
import numpy as np
import numpy.typing as npt

from src.local_types import Dates
from src.backtest.pool_corr import PoolCorrelation

# THÊM method vào class GateEvaluator, SAU method evaluate() hiện có (không xóa/sửa evaluate):

    def evaluate_with_pool(
        self,
        m: AlphaMetrics,
        candidate_pnl: npt.NDArray[np.float64],
        candidate_dates: Dates,
        pool_corr: PoolCorrelation,
        depth: int,
        fields_ok: bool,
    ) -> GateVerdict:
        """Như evaluate(), nhưng self_corr tính thật từ pool_corr.max_corr(...) (Phase 6,
        B9 master design) thay vì nhận tay — entrypoint nên dùng cho mọi candidate mới."""
        self_corr, _worst_id = pool_corr.max_corr(candidate_pnl, candidate_dates)
        return self.evaluate(m, self_corr=self_corr, depth=depth, fields_ok=fields_ok)
```

- [ ] **Step 4: Chạy test — PASS, rồi chạy lại toàn bộ test Phase 4 để xác nhận không phá `evaluate()` cũ**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_gates_pool_corr.py tests/unit/test_gates.py -v
```
Expected: PASS toàn bộ (4 + 13 test).

- [ ] **Step 5: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/backtest/gates.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/backtest/gates.py
```
Expected: cả hai sạch.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/gates.py tests/unit/test_gates_pool_corr.py
git commit -m "feat(backtest): GateEvaluator.evaluate_with_pool tinh self_corr that tu PoolCorrelation"
```

---

### Task 6.3: Lưu PnL khi pass (`src/storage/repository.py` + điểm tích hợp loop)

**Files:**
- Modify: `src/storage/repository.py` (đọc toàn văn trước khi sửa — file hiện có
  `AlphaRepository`/`InvalidFieldRepository` cho luồng sim-Brain cũ; **không sửa/xoá** các
  method hiện có, chỉ THÊM method mới).
- Modify: `src/storage/models.py` (thêm `PoolPnlModel` nếu Phase 5 thật chưa tạo bảng này —
  xem điều kiện kiểm tra ở Step 1).
- Test: `tests/unit/test_repository_pool_pnl.py`

**Interfaces:**
- Consumes: `Dates`, `npt.NDArray[np.float64]` (PnL 1-D).
- Produces:

  ```python
  # THÊM vào class AlphaRepository trong src/storage/repository.py
  def save_pool_pnl(
      self, evaluation_id: int, dates: Dates, pnl: npt.NDArray[np.float64]
  ) -> None: ...

  def load_pool(self) -> dict[int, tuple[Dates, npt.NDArray[np.float64]]]: ...
  ```

  `dates`/`pnl` lưu dưới dạng blob (theo B11 `pool_pnl(evaluation_id PK, dates_blob,
  pnl_blob)`): `dates.astype("datetime64[D]").tobytes()` + `pnl.astype(np.float64).tobytes()`,
  đọc lại bằng `np.frombuffer(blob, dtype=...)`. `evaluation_id` là PK — gọi `save_pool_pnl`
  hai lần cùng `evaluation_id` thì **upsert** (ghi đè), không nhân bản dòng (idempotent —
  cùng pattern `session.merge` đã dùng ở `InvalidFieldRepository.record`).

> **Điểm tích hợp gọi `save_pool_pnl` khi alpha pass — GIẢ ĐỊNH, ghi rõ vì chưa có
> loop/pipeline thật chạm Phase 6.** Tại thời điểm viết plan này, `src/llm/loop.py`
> (`RefinementLoop`, nhắc tới ở Phase 3 Task 3.5 "gỡ đường cũ D9" và Phase 7 Task 7.8) là nơi
> tự nhiên để gọi gate rồi lưu pool, nhưng **chưa tồn tại đường nối Phase 3→4→6 cụ thể** ở
> thời điểm lập plan (Phase 3/4 cũng mới là plan, chưa code thật). Vì vậy Task 6.3 **không**
> tự ý sửa `src/llm/loop.py` (sẽ lấn phạm vi/đoán sai cấu trúc loop thật khi nó được code).
> Thay vào đó: (a) implement `save_pool_pnl`/`load_pool` đầy đủ + test (đây là phạm vi B9
> đúng nghĩa của Phase 6), và (b) viết **một hàm tiện ích nhỏ** ở
> `src/backtest/pool_corr.py`-adjacent KHÔNG — thay vào `src/storage/repository.py` luôn, vì
> "ghi khi pass" về bản chất là một quyết định của tầng gọi (biết `GateVerdict.passed` +
> `evaluation_id` + PnL), nên expose `save_pool_pnl` là đủ; **điểm gọi thật** (`if
> verdict.passed: repo.save_pool_pnl(...)`) sẽ nằm trong `src/llm/loop.py` hoặc
> `src/pipeline/runner.py` (Phase 8) khi các file đó được implement — Task 6.3 ghi rõ đoạn
> code mẫu dưới đây dưới dạng **docstring hướng dẫn tích hợp** trong `save_pool_pnl`, không
> tạo file loop giả. Nếu khi thực thi Task 6.3, `src/llm/loop.py` **đã** tồn tại với một hàm
> rõ ràng đang gọi `GateEvaluator` (kiểm tra bằng
> `grep -rn "GateEvaluator\|evaluate_with_pool" src/llm/loop.py`), thì **mở rộng thêm** lệnh
> gọi `save_pool_pnl` ngay sau khi `verdict.passed` đúng, trong cùng commit này, và ghi log
> trong báo cáo review Task 6.5.

- [ ] **Step 1: Kiểm tra `PoolPnlModel` đã tồn tại trong `src/storage/models.py` chưa**

```bash
venv/Scripts/python.exe -c "from src.storage.models import PoolPnlModel; print('exists')"
```
- Nếu in `exists` (Phase 5 thật đã tạo bảng) → **bỏ qua Step 2** (không tạo lại model),
  dùng `PoolPnlModel` thật, đọc cột thật của nó (có thể khác tên cột giả định dưới) và sửa
  Step 3/4 cho khớp tên cột thật — KHÔNG đổi tên cột đã có.
- Nếu `ImportError`/`AttributeError` (Phase 5 thật chưa làm bảng này, đúng như xác nhận ở
  Pre-condition) → tiếp tục Step 2, tự tạo `PoolPnlModel` tối giản theo B11.

- [ ] **Step 2: Viết test đỏ**

```python
# tests/unit/test_repository_pool_pnl.py
"""Test AlphaRepository.save_pool_pnl + load_pool: roundtrip dates+pnl qua blob, upsert
idempotent theo evaluation_id, load_pool tra ve dict khop dinh dang PoolCorrelation can."""

from __future__ import annotations

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.models import Base
from src.storage.repository import AlphaRepository


def _make_repo():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return AlphaRepository(session_factory)


def test_save_then_load_pool_roundtrips_dates_and_pnl():
    repo = _make_repo()
    dates = (np.datetime64("2021-01-01") + np.arange(5)).astype("datetime64[D]")
    pnl = np.array([0.01, -0.02, 0.03, 0.0, -0.01])

    repo.save_pool_pnl(evaluation_id=1, dates=dates, pnl=pnl)
    pool = repo.load_pool()

    assert set(pool.keys()) == {1}
    loaded_dates, loaded_pnl = pool[1]
    assert np.array_equal(loaded_dates, dates)
    assert np.allclose(loaded_pnl, pnl)


def test_load_pool_empty_db_returns_empty_dict():
    repo = _make_repo()
    assert repo.load_pool() == {}


def test_save_pool_pnl_is_idempotent_upsert_by_evaluation_id():
    repo = _make_repo()
    dates1 = (np.datetime64("2021-01-01") + np.arange(3)).astype("datetime64[D]")
    pnl1 = np.array([0.01, 0.02, 0.03])
    repo.save_pool_pnl(evaluation_id=1, dates=dates1, pnl=pnl1)

    dates2 = (np.datetime64("2022-01-01") + np.arange(4)).astype("datetime64[D]")
    pnl2 = np.array([0.04, 0.05, 0.06, 0.07])
    repo.save_pool_pnl(evaluation_id=1, dates=dates2, pnl=pnl2)  # ghi de cung id

    pool = repo.load_pool()
    assert len(pool) == 1  # khong nhan ban dong
    loaded_dates, loaded_pnl = pool[1]
    assert np.array_equal(loaded_dates, dates2)
    assert np.allclose(loaded_pnl, pnl2)


def test_load_pool_handles_multiple_alphas_with_different_lengths():
    repo = _make_repo()
    dates_a = (np.datetime64("2021-01-01") + np.arange(3)).astype("datetime64[D]")
    dates_b = (np.datetime64("2021-06-01") + np.arange(10)).astype("datetime64[D]")
    repo.save_pool_pnl(evaluation_id=10, dates=dates_a, pnl=np.array([0.1, 0.2, 0.3]))
    repo.save_pool_pnl(evaluation_id=20, dates=dates_b, pnl=np.arange(10, dtype=np.float64))

    pool = repo.load_pool()
    assert set(pool.keys()) == {10, 20}
    assert pool[10][1].shape == (3,)
    assert pool[20][1].shape == (10,)
```

- [ ] **Step 3: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_repository_pool_pnl.py -v
```
Expected: FAIL `AttributeError: 'AlphaRepository' object has no attribute 'save_pool_pnl'`
(hoặc `ImportError: PoolPnlModel` nếu Step 1 xác nhận model chưa tồn tại — implement Step 4
trước rồi chạy lại).

- [ ] **Step 4: (Chỉ nếu Step 1 báo model chưa tồn tại) Thêm `PoolPnlModel` vào `src/storage/models.py`**

```python
# THÊM vào src/storage/models.py — KHÔNG sửa các model hiện có
class PoolPnlModel(Base):
    """PnL daily của alpha đã PASS gate -> pool self-correlation local (B9/B11).
    evaluation_id là PK logic (1 entry pool / evaluation pass); dates_blob/pnl_blob là
    bytes packed datetime64[D]/float64 theo thứ tự thời gian tăng dần."""

    __tablename__ = "pool_pnl"

    evaluation_id = Column(Integer, primary_key=True)
    dates_blob = Column(LargeBinary, nullable=False)
    pnl_blob = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=_utcnow)
```

Thêm `LargeBinary` vào import `sqlalchemy` đầu file (`from sqlalchemy import (...,
LargeBinary, ...)`).

- [ ] **Step 5: Thêm `save_pool_pnl` + `load_pool` vào `class AlphaRepository` trong `src/storage/repository.py`**

```python
# THÊM vào đầu file (khối import):
import numpy as np
import numpy.typing as npt

from src.local_types import Dates
from src.storage.models import PoolPnlModel  # cùng nhóm import models hiện có

# THÊM method vào class AlphaRepository, sau zoo():

    def save_pool_pnl(
        self, evaluation_id: int, dates: Dates, pnl: npt.NDArray[np.float64]
    ) -> None:
        """Ghi PnL daily của một evaluation đã PASS gate vào pool self-corr local (B9).
        Idempotent theo evaluation_id (upsert qua session.merge — gọi lại với cùng id sẽ
        GHI ĐÈ, không nhân bản). Điểm gọi: tầng loop/pipeline ngay sau khi
        GateEvaluator.evaluate_with_pool(...).passed is True (xem Task 6.3 trong
        docs/superpowers/plans/2026-06-24-phase-6-pool-corr.md để biết điểm tích hợp thật
        khi src/llm/loop.py hoặc src/pipeline/runner.py tồn tại)."""
        dates_blob = dates.astype("datetime64[D]").tobytes()
        pnl_blob = pnl.astype(np.float64).tobytes()
        session = self.session_factory()
        try:
            session.merge(
                PoolPnlModel(
                    evaluation_id=evaluation_id, dates_blob=dates_blob, pnl_blob=pnl_blob
                )
            )
            session.commit()
        finally:
            session.close()

    def load_pool(self) -> dict[int, tuple[Dates, npt.NDArray[np.float64]]]:
        """Đọc toàn bộ pool self-corr local thành dict[evaluation_id -> (dates, pnl)] —
        định dạng PoolCorrelation.__init__ (src/backtest/pool_corr.py, Task 6.1) cần trực
        tiếp, không xử lý thêm ở caller."""
        session = self.session_factory()
        try:
            rows = session.query(PoolPnlModel).all()
            pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] = {}
            for row in rows:
                dates = np.frombuffer(row.dates_blob, dtype="datetime64[D]")
                pnl = np.frombuffer(row.pnl_blob, dtype=np.float64)
                pool[row.evaluation_id] = (dates, pnl)
            return pool
        finally:
            session.close()
```

- [ ] **Step 6: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_repository_pool_pnl.py -v
```
Expected: PASS (4 test).

- [ ] **Step 7: Chạy lại toàn bộ test repository cũ để xác nhận không phá luồng sim-Brain**

```bash
venv/Scripts/python.exe -m pytest tests/unit -k "repository or repo" -v
```
Expected: tất cả PASS (test repository cũ + test mới Task 6.3).

- [ ] **Step 8: ruff + mypy**

```bash
venv/Scripts/python.exe -m ruff check src/storage/repository.py src/storage/models.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/storage/repository.py src/storage/models.py
```
Expected: cả hai sạch.

- [ ] **Step 9: Commit**

```bash
git add src/storage/repository.py src/storage/models.py tests/unit/test_repository_pool_pnl.py
git commit -m "feat(storage): save_pool_pnl/load_pool cho pool self-corr local (B11 pool_pnl)"
```

---

### Task 6.4: Integration — pool đa-alpha → gate end-to-end qua DB thật

**Files:**
- Create: `tests/integration/test_pool_corr_gate.py`

**Interfaces:**
- Consumes: `AlphaRepository.save_pool_pnl/load_pool` (Task 6.3), `PoolCorrelation` (Task
  6.1), `GateEvaluator.evaluate_with_pool` (Task 6.2), `AlphaMetrics` (Phase 4).
- Produces: không có module mới — 1 test integration chứng minh round-trip DB→
  `PoolCorrelation`→`GateEvaluator` thật, không mock.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/integration/test_pool_corr_gate.py
"""Integration Phase 6: ghi pool qua AlphaRepository (DB thật, sqlite tmp) -> load_pool ->
PoolCorrelation -> GateEvaluator.evaluate_with_pool, end-to-end khong mock."""

from __future__ import annotations

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.gates import GateEvaluator
from src.backtest.metrics_local import AlphaMetrics
from src.backtest.pool_corr import PoolCorrelation
from src.storage.models import Base
from src.storage.repository import AlphaRepository


def _good_metrics() -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.5, annual_return=0.20, turnover=0.30, max_drawdown=0.10,
        fitness=2.0, per_year_sharpe={2021: 1.2}, weight_concentration=0.05,
    )


def test_pool_grows_and_blocks_correlated_candidate_end_to_end(tmp_path):
    db_path = tmp_path / "pool_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    repo = AlphaRepository(sessionmaker(bind=engine))

    dates = (np.datetime64("2021-01-01") + np.arange(20)).astype("datetime64[D]")
    rng = np.random.default_rng(123)
    alpha_a_pnl = rng.normal(size=20)

    # Pool rỗng -> candidate đầu tiên luôn pass gate self_corr (self_corr=0.0)
    pool = PoolCorrelation(pool=repo.load_pool())
    verdict_first = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=alpha_a_pnl, candidate_dates=dates,
        pool_corr=pool, depth=3, fields_ok=True,
    )
    assert verdict_first.passed is True

    # Alpha A pass -> ghi vào pool thật
    repo.save_pool_pnl(evaluation_id=1, dates=dates, pnl=alpha_a_pnl)

    # Candidate B giống hệt A -> phải bị hard-fail self_corr khi load lại pool từ DB
    pool_after = PoolCorrelation(pool=repo.load_pool())
    verdict_second = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=alpha_a_pnl.copy(), candidate_dates=dates,
        pool_corr=pool_after, depth=3, fields_ok=True,
    )
    assert verdict_second.passed is False
    assert any("self_corr" in f for f in verdict_second.hard_failures)

    # Candidate C độc lập -> vẫn pass
    alpha_c_pnl = rng.normal(size=20)
    verdict_third = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=alpha_c_pnl, candidate_dates=dates,
        pool_corr=pool_after, depth=3, fields_ok=True,
    )
    assert verdict_third.passed is True
```

- [ ] **Step 2: Chạy test — kỳ vọng PASS ngay nếu Task 6.1–6.3 đúng (chạy để xác nhận, không giả định)**

```bash
venv/Scripts/python.exe -m pytest tests/integration/test_pool_corr_gate.py -v
```
Expected: PASS (1 test). Nếu FAIL, quay lại Task tương ứng (6.1 nếu lỗi `max_corr`, 6.2 nếu
lỗi `evaluate_with_pool`, 6.3 nếu lỗi DB round-trip) — sửa tại đó, không patch test để né lỗi.

- [ ] **Step 3: Chạy toàn bộ test Phase 6 + Phase 4 cùng lúc để xác nhận không có regression**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_pool_corr.py tests/unit/test_gates_pool_corr.py tests/unit/test_repository_pool_pnl.py tests/unit/test_gates.py tests/integration/test_pool_corr_gate.py -v
```
Expected: PASS toàn bộ.

- [ ] **Step 4: ruff + mypy toàn bộ thay đổi Phase 6**

```bash
venv/Scripts/python.exe -m ruff check src/backtest/pool_corr.py src/backtest/gates.py src/storage/repository.py src/storage/models.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/backtest/pool_corr.py src/backtest/gates.py src/storage/repository.py src/storage/models.py
```
Expected: cả hai sạch.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_pool_corr_gate.py
git commit -m "test(integration): pool corr end-to-end DB->PoolCorrelation->GateEvaluator"
```

---

### Task 6.5: Review + merge + push

**Files:** không tạo file mới — review toàn bộ thay đổi nhánh `phase-6-pool-corr`.

- [ ] **Step 1: Chạy full test suite + lint + type-check toàn repo**

```bash
venv/Scripts/python.exe -m pytest -v
venv/Scripts/python.exe -m ruff check .
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src config
```
Expected: tất cả PASS/sạch. Nếu có test cũ (Phase 0–5) đỏ vì lý do KHÔNG liên quan Phase 6
(ví dụ Phase 5 thật chưa tồn tại đúng như Pre-condition đã ghi), liệt kê rõ trong báo cáo
review — không sửa code Phase khác để "làm xanh giả".

- [ ] **Step 2: Diff review tự đọc lại — đối chiếu Global Constraints**

Tự kiểm tra (không cần script, đọc lại diff bằng `git diff main...HEAD`):
- [ ] Không hardcode `0.70`/`SELF_CORR_MAX` ở call site nào ngoài `config/thresholds.py`
  (cả `pool_corr.py` và `gates.py` Task 6.2 chỉ truyền `self_corr` đã tính, không so sánh số
  trực tiếp — so sánh nằm trong `evaluate()` Phase 4 cũ).
- [ ] Align trên dates **giao nhau** (không forward-fill, không giả định cùng độ dài).
- [ ] `backtest/pool_corr.py` không import `src.storage`/`src.gp`/`src.llm` (dependency rule
  B1) — xác nhận: `grep -n "^import\|^from" src/backtest/pool_corr.py`.
- [ ] Mọi giá trị float trả ra là `|ρ|` (giá trị tuyệt đối), không phải ρ có dấu — đối chiếu
  test `test_sign_flipped_series_gives_rho_minus_one_abs_one`.
- [ ] `tiếng Việt giữ dấu` trong toàn bộ docstring/comment mới thêm.

- [ ] **Step 3: Cập nhật `PROGRESS.md`/journal (nếu repo có file này) — ghi giả định Phase 5 chưa hoàn thiện đã xử lý ở Task 6.3**

- [ ] **Step 4: Merge vào main**

```bash
git checkout main
git merge --no-ff phase-6-pool-corr -m "merge: Phase 6 - Pool correlation (PoolCorrelation + wire gate + save/load pool)"
git push origin main
```

---

## Self-review (đối chiếu Global Constraints + yêu cầu nhiệm vụ)

- **Align trên dates chung:** Task 6.1 dùng `np.intersect1d` trên cả hai trục ngày, không
  giả định cùng độ dài/cùng offset — test `test_partial_date_overlap_aligns_on_intersection_only`
  và `test_no_date_overlap_is_skipped_not_zero_forced` phủ cả overlap-một-phần và
  không-overlap.
- **|Pearson ρ|:** `max_corr` trả `abs(rho)`; test `test_sign_flipped_series_gives_rho_minus_one_abs_one`
  xác nhận ρ=-1 cho ra |ρ|=1, không bị coi là "không tương quan".
- **Hard gate 0.70 từ `config/thresholds.py`:** Task 6.2 không hardcode số — `evaluate_with_pool`
  delegate sang `evaluate()` Phase 4 (nơi duy nhất so sánh với `SELF_CORR_MAX`).
- **PoolCorrelation là proxy local, Brain authoritative:** ghi rõ trong docstring
  `pool_corr.py` (Task 6.1) và trong Goal — không có chỗ nào plan này tuyên bố
  `PoolCorrelation` thay thế checker thật của Brain trước khi submit.
- **Chữ ký `load_pool` khớp Phase 5:** Phase 5 **chưa có file plan/implementation thật**
  tại thời điểm viết plan này (đã xác minh bằng cách đọc trực tiếp `src/storage/repository.py`
  và `src/storage/models.py` — không có `ExpressionModel`/`EvaluationModel`/`PoolPnlModel`,
  không có `load_pool`). Plan này **tự quyết định** chữ ký
  `load_pool() -> dict[int, tuple[Dates, Panel1D]]` khớp B11 schema (`pool_pnl` table có cả
  `dates_blob` và `pnl_blob`) và B9 design (key theo id, value là PnL), và ghi rõ điều kiện
  kiểm tra (Task 6.3 Step 1) để Task tự phát hiện nếu Phase 5 thật triển khai khác, tránh
  nhân bản logic nếu `PoolPnlModel`/`load_pool` đã có sẵn lúc thực thi.
- **Điểm tích hợp ghi PnL khi pass:** Task 6.3 ghi rõ đây là giả định (loop/pipeline thật
  chưa tồn tại) — implement đầy đủ `save_pool_pnl`/`load_pool` (phạm vi B9 đúng nghĩa của
  Phase 6) nhưng **không** tự viết `src/llm/loop.py` để tránh đoán sai cấu trúc của Phase
  chưa code; có hướng dẫn rõ cách nối khi loop tồn tại.
- **Không placeholder, TDD đỏ→xanh→commit từng task:** mỗi Task có step viết test trước,
  chạy FAIL với lý do cụ thể (`ModuleNotFoundError`/`AttributeError`), code thật tối thiểu
  đủ pass, chạy lại PASS, ruff+mypy, rồi mới commit.
- **Concern còn mở (không tự giải quyết, để báo cáo người dùng):** Phase 0–5 hiện chỉ là
  *plan* (trừ `config/thresholds.py` và `src/local_types.py` đã có code thật, và
  `src/storage/{repository,models}.py` có code thật nhưng cho luồng sim-Brain cũ, chưa có
  phần Phase 5 MiniBrain). Khi các phase đó được thực thi thật, cần re-xác nhận Pre-condition
  của Phase 6 (đặc biệt `GateEvaluator.evaluate`, `AlphaMetrics`, `load_pool` thật) trước khi
  chạy Task 6.1–6.4 — nếu chữ ký lệch, sửa tại Task tương ứng (đã ghi rõ "lối thoát" ở cuối
  Task 6.1 và điều kiện kiểm tra ở Task 6.3 Step 1).
