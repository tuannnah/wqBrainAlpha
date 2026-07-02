# Closed-Loop Phase 4A — `RefinementLoop.run_from_seed` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development hoặc
> superpowers:executing-plans. Steps dùng checkbox. Phase 4A của feature "Vòng kín AI +
> MiniBrain" (spec `docs/superpowers/specs/2026-06-26-ai-minibrain-closed-loop-design.md`).
> Phase 1-3 ĐÃ xong trên nhánh `closed-loop-integration`. 4A là KEYSTONE cho 4B (adapter)/4C
> (menu). REQUIRED SUB-SKILL khi refactor: superpowers:test-driven-development.

**Goal:** Thêm `RefinementLoop.run_from_seed(expression)` — chạy y hệt `run()` nhưng hạt
giống là MỘT công thức FASTEXPR cho sẵn (core từ GPEngine), KHÔNG qua hypothesis_gen/
translator. Cho phép vòng kín "GP trục, AI tăng cường": GP sinh core → AI refine ≤ patience
lần + sim. Refactor tách lõi refine thành `_refine_loop` dùng chung bởi `run()` và
`run_from_seed`; `run()` giữ hành vi y nguyên (5 file test loop hiện có phải xanh).

**Architecture:** Tách phần thân vòng refine của `run()` (từ sau khi chọn seed tới hết) thành
method `_refine_loop(self, best_cand, best_ev, research_direction, current_config, history,
emit) -> LoopResult`. `run()` giữ phần seed_candidates rồi gọi `_refine_loop`. Thêm
`run_from_seed` dựng seed từ công thức rồi gọi cùng `_refine_loop`.

**Tech Stack:** Python 3.12, pytest. Sửa `src/llm/loop.py` (file có 5 test hiện hành).

## Global Constraints

- Python 3.12; full type hints trên method mới; `ruff` clean; không unused import.
- **REGRESSION CỨNG:** 5 file test loop hiện có PHẢI xanh sau refactor —
  `tests/test_loop.py`, `tests/test_loop_referee.py`, `tests/test_loop_reseed.py`,
  `tests/test_loop_seed.py`, `tests/unit/test_loop_local_gate.py`. `run()` đổi hành vi =
  refactor SAI.
- **Tiếng Việt giữ dấu đúng chính tả** trong docstring/comment mới.
- TDD: test `run_from_seed` đỏ trước → refactor + impl → xanh → chạy regression → commit.
- mypy: `src/llm/loop.py` có thể có debt baseline; KHÔNG làm phát sinh lỗi MỚI từ method mới.

## Pre-condition (chữ ký thật đã xác minh)

```python
# src/llm/loop.py
class RefinementLoop:
    def run(self, research_direction: str, on_progress=None) -> LoopResult: ...
    def seed_candidates(self, research_direction: str) -> list: ...   # -> [AlphaCandidate]
    def _evaluate(self, candidate, parent_id: str | None, config=None) -> _Eval | None: ...
    # _Eval: vector (ScoreVector, .total), metrics(dict), alpha_id(str|None), passed(bool),
    #        effective_total(float), pool_corr(float|None), regime_blocked(bool)
    # run() thân vòng refine: sau khi chọn best seed (best_cand/best_ev), vòng
    #   `while sims_used < max_simulations and patience < no_improve_patience:` gồm
    #   referee/tune_config/reseed/refine + tính stop_reason + return LoopResult.
@dataclass
class LoopResult:  # best_candidate, best_vector, history, zoo_added, failures, sims_used, stop_reason

# src/llm/translator.py
@dataclass
class AlphaCandidate:
    hypothesis: Hypothesis
    description: str
    expression: str
    parent_id: str | None = None

# src/llm/hypothesis.py
@dataclass
class Hypothesis:  # observation/background/economic_rationale/implementation_spec đều default ""
    # Hypothesis() dựng được rỗng hợp lệ.
# refiner.refine(candidate, metrics, weak) dùng candidate.expression + candidate.description
# (hypothesis chỉ truyền tiếp) -> seed GP với Hypothesis() rỗng + description=expression OK.
```

## File Structure

- **Modify** `src/llm/loop.py` (~30 dòng net): tách `_refine_loop` + thêm `run_from_seed`.
- **Create** `tests/unit/test_loop_run_from_seed.py` (~70 dòng): test `run_from_seed` bằng
  fake simulator/refiner (theo pattern test_loop hiện có).

---

### Task 1: Refactor `_refine_loop` + thêm `run_from_seed` (+ test + regression)

**Files:**
- Modify: `src/llm/loop.py`
- Test: `tests/unit/test_loop_run_from_seed.py`

**Interfaces:**
- Consumes: `_evaluate`, `seed_candidates`, `LoopResult`, `LoopProgress`, `AlphaCandidate`
  (translator), `Hypothesis` (hypothesis).
- Produces:
  ```python
  def _refine_loop(self, best_cand, best_ev, research_direction: str, current_config,
                   history: list, emit) -> LoopResult: ...
  def run_from_seed(self, expression: str, on_progress=None) -> LoopResult: ...
  ```

- [ ] **Step 0: Đọc toàn văn `RefinementLoop.run` (src/llm/loop.py)** để xác định ranh giới
  trích: phần "seed" (đầu run, qua seed_candidates + chọn best seed + history seed) GIỮ trong
  `run()`; phần "vòng refine" (từ `patience = 0` tới hết, gồm tính stop_reason + return
  LoopResult) CHUYỂN vào `_refine_loop`. Đọc cả pattern fake trong `tests/test_loop_seed.py`
  để viết test mới đúng kiểu (fake hypothesis_gen/translator/refiner/simulator/repo).

- [ ] **Step 1: Viết test đỏ `tests/unit/test_loop_run_from_seed.py`**

```python
"""Test RefinementLoop.run_from_seed: hạt giống là công thức cho sẵn (core GP), KHÔNG qua
hypothesis_gen/translator. Tái dùng pattern fake của tests/test_loop_seed.py."""

from __future__ import annotations

# NOTE-AT-EXEC: import đúng các fake/helper mà tests/test_loop_seed.py dùng để dựng
# RefinementLoop (fake simulator trả SimulationResult pass, fake repo, prefilter thật/giả,
# score_vector_fn/hard_filter_fn mặc định). Đọc test_loop_seed.py ở Step 0 rồi tái dùng.

from src.llm.loop import LoopResult, RefinementLoop


def _make_loop(...):  # dựng RefinementLoop với fakes — sao theo test_loop_seed.py
    ...


def test_run_from_seed_uses_given_expression_as_seed() -> None:
    """run_from_seed('rank(close)') đánh giá đúng công thức đó làm seed (KHÔNG gọi
    hypothesis_gen/seed_candidates), trả LoopResult với best_candidate.expression bắt nguồn
    từ seed."""
    loop = _make_loop(...)
    result = loop.run_from_seed("rank(close)")
    assert isinstance(result, LoopResult)
    assert result.best_candidate is not None
    # seed eval được -> best_candidate.expression là 'rank(close)' (hoặc biến thể refine của nó)
    assert result.sims_used >= 1


def test_run_from_seed_unparseable_seed_returns_no_seed() -> None:
    """Seed bị prefilter loại / eval None -> LoopResult stop_reason='no_seed', best=None."""
    loop = _make_loop(...)  # prefilter loại mọi thứ, hoặc simulator khiến _evaluate None
    result = loop.run_from_seed("khong_hop_le(")
    assert result.best_candidate is None
    assert result.stop_reason == "no_seed"
```

> Ở Step 0 đọc `tests/test_loop_seed.py` để biết CHÍNH XÁC cách dựng RefinementLoop với fake
> (constructor cần hypothesis_gen/translator/refiner/simulator/prefilter/repo/region/...).
> Điền `_make_loop` + 2 assertion cho khớp fake thật. KHÔNG đoán API fake — sao chép pattern.

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_loop_run_from_seed.py -q
```
Expected: FAIL `AttributeError: 'RefinementLoop' object has no attribute 'run_from_seed'`.

- [ ] **Step 3: Refactor — tách `_refine_loop` khỏi `run()`**

Trong `src/llm/loop.py`, tách phần thân vòng refine của `run()` (từ `patience = 0` tới hết
hàm — gồm khối while, tính `stop_reason`, `emit("done", ...)`, `return LoopResult(...)`) vào
method mới, GIỮ NGUYÊN từng dòng logic (chỉ đổi chỗ + nhận tham số qua signature):

```python
    def _refine_loop(
        self, best_cand, best_ev, research_direction: str, current_config,
        history: list, emit,
    ) -> LoopResult:
        """Lõi vòng refine dùng chung bởi run() và run_from_seed: từ (best_cand, best_ev) đã
        có, lặp refine/tune/reseed tới patience/budget/abandon rồi trả LoopResult. Tách ra để
        seed có thể đến từ hypothesis (run) hoặc từ công thức cho sẵn (run_from_seed)."""
        # <<< DÁN NGUYÊN phần thân vòng refine của run() cũ vào đây (patience=0 ... return). >>>
```

Sửa `run()` để sau khi chọn best seed + append history seed + emit("seed", ...), KẾT bằng:
```python
        return self._refine_loop(best_cand, best_ev, research_direction, current_config,
                                 history, emit)
```
(Bỏ phần thân vòng đã chuyển đi khỏi `run()`.)

- [ ] **Step 4: Thêm `run_from_seed`**

```python
    def run_from_seed(self, expression: str, on_progress=None) -> LoopResult:
        """Như run() nhưng hạt giống là MỘT công thức FASTEXPR cho sẵn (vd core từ GPEngine),
        KHÔNG qua hypothesis_gen/translator. Phục vụ vòng kín 'GP trục, AI tăng cường'."""
        from src.llm.hypothesis import Hypothesis
        from src.llm.translator import AlphaCandidate

        self.sims_used = 0
        self.zoo_added = 0
        history: list = []

        def emit(phase, best_total, detail=""):
            if on_progress:
                on_progress(LoopProgress(self.sims_used, best_total, phase, detail))

        current_config = self.sim_config
        seed = AlphaCandidate(hypothesis=Hypothesis(), description=expression,
                              expression=expression)
        emit("seed", 0.0, expression)
        best_ev = self._evaluate(seed, parent_id=None, config=current_config)
        if best_ev is None:
            return LoopResult(None, None, history, self.zoo_added,
                              self.repo.recent_failures(50), self.sims_used,
                              stop_reason="no_seed")
        history.append(
            {"step": 0, "action": "seed", "dimension": "-", "total": best_ev.vector.total,
             "expression": expression, "accepted": True}
        )
        emit("seed", best_ev.vector.total, expression)
        return self._refine_loop(seed, best_ev, expression, current_config, history, emit)
```

- [ ] **Step 5: Chạy test mới — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_loop_run_from_seed.py -q
```
Expected: 2 PASS.

- [ ] **Step 6: REGRESSION — 5 file test loop hiện có phải xanh**

```bash
venv/Scripts/python.exe -m pytest tests/test_loop.py tests/test_loop_referee.py tests/test_loop_reseed.py tests/test_loop_seed.py tests/unit/test_loop_local_gate.py -q
```
Expected: TẤT CẢ PASS (run() hành vi y nguyên). Nếu đỏ → refactor làm lệch logic; sửa cho
khớp run() cũ TỪNG DÒNG. Đây là gate cứng.

- [ ] **Step 7: ruff + kiểm dấu tiếng Việt + commit**

```bash
venv/Scripts/python.exe -m ruff check src/llm/loop.py tests/unit/test_loop_run_from_seed.py
git add src/llm/loop.py tests/unit/test_loop_run_from_seed.py
git commit -m "feat(llm): RefinementLoop.run_from_seed - seed bang cong thuc GP (tach _refine_loop)"
```

---

## Self-review

**Spec coverage (4A scope):**
- [x] `run_from_seed(expression)` seed bằng công thức cho sẵn — Task 1 Step 4.
- [x] Tái dùng lõi refine (patience/abandon/sim) qua `_refine_loop` — Task 1 Step 3.
- [x] `run()` giữ hành vi (regression 5 file test) — Task 1 Step 6 (gate cứng).
- [~] Adapter `_RefinesIdea`/menu — KHÔNG thuộc 4A (4B/4C).

**Placeholder scan:** Có chủ ý ở `_make_loop(...)`/`...` trong test (Step 1): phụ thuộc pattern
fake CHÍNH XÁC của `tests/test_loop_seed.py` — Step 0 bắt buộc đọc + sao chép, KHÔNG đoán. Mọi
phần khác (run_from_seed, _refine_loop signature, run() rewire) có code cụ thể.

**Type consistency:**
- `_refine_loop(self, best_cand, best_ev, research_direction, current_config, history, emit)
  -> LoopResult` — dùng bởi cả `run()` và `run_from_seed`.
- `run_from_seed(self, expression: str, on_progress=None) -> LoopResult` — khớp test.
- `AlphaCandidate(hypothesis=Hypothesis(), description=expression, expression=expression)` —
  khớp chữ ký translator.AlphaCandidate (hypothesis: Hypothesis).

**Risks / gotchas:**
1. Trích `_refine_loop` phải GIỮ NGUYÊN logic run() (đặc biệt khởi tạo `patience/stuck/step/
   abandoned/reseed_on` ở đầu lõi). Regression 5 file test là lưới an toàn.
2. `run_from_seed` không gọi referee.judge khác run() — lõi `_refine_loop` xử referee y nhau;
   research_direction truyền = expression (referee.judge nhận chuỗi này — chấp nhận).
3. Test fake phải khớp constructor RefinementLoop thật (nhiều tham số) — Step 0 đọc
   test_loop_seed.py để sao, tránh sai chữ ký fake.
