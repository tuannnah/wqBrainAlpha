# Lệnh toàn trình `auto` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm một lệnh `auto` chạy toàn trình (đăng nhập → cache fields/operators → tìm/mô phỏng/cải thiện alpha → log đầy đủ, KHÔNG nộp), điều phối bởi một orchestrator thuần `AutoPipeline` test được không cần mạng.

**Architecture:** `AutoPipeline` (src/pipeline/auto.py) là orchestrator thuần nhận 3 callback (prepare, propose_directions, run_direction) + cấu hình dừng (K-pass / trần sim / số hướng), lo vòng lặp và thu thập kết quả. Lệnh `auto` trong main.py bọc client + engine thật (AI/GA) thành 3 callback đó. `start()` đổi thành gọi thẳng `auto`, bỏ menu wizard.

**Tech Stack:** Python 3.12, dataclasses, typer (CLI), rich (in bảng/console), loguru (log file), pytest. Engine tái dùng: `RefinementLoop`, `GeneticOptimizer`, `FieldRepository.ensure`, `OperatorRepository.ensure`, `_make_research_loop`, `_make_llm_generator`.

---

## File Structure

- **Create** `src/pipeline/__init__.py` — package rỗng.
- **Create** `src/pipeline/auto.py` — dataclasses (`PassedAlpha`, `DirectionOutcome`, `PrepareInfo`, `AutoEvent`, `AutoResult`) + class `AutoPipeline`.
- **Create** `tests/test_auto_pipeline.py` — test orchestrator bằng fake callback.
- **Modify** `main.py` — thêm lệnh `auto`, các hàm bọc callback (`_auto_prepare`, `_auto_propose_directions`, `_auto_run_direction_*`), đổi `start()` gọi `auto`, xóa các hàm `_wizard_*` + `_WizardState` + `_ask`.

---

## Task 1: Khung dataclasses + AutoPipeline dừng khi "hết hướng"

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/auto.py`
- Test: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Tạo package file rỗng**

Tạo `src/pipeline/__init__.py` với nội dung rỗng (một dòng comment):

```python
"""Orchestrator toàn trình cho WQ Auto-Alpha."""
```

- [ ] **Step 2: Viết test "hết hướng" (RED)**

Tạo `tests/test_auto_pipeline.py`:

```python
"""Test AutoPipeline bằng fake callback — không gọi mạng."""

from __future__ import annotations

from src.pipeline.auto import (
    AutoPipeline,
    DirectionOutcome,
    PassedAlpha,
    PrepareInfo,
)


def _pa(expr: str, direction: str = "") -> PassedAlpha:
    return PassedAlpha(expression=expr, sharpe=1.5, fitness=1.1, direction=direction)


def test_dung_khi_het_huong():
    calls = {"run": 0}

    def prepare() -> PrepareInfo:
        return PrepareInfo(fields=10, operators=5)

    def propose(n: int) -> list[str]:
        return ["h1", "h2"]

    def run_direction(direction: str) -> DirectionOutcome:
        calls["run"] += 1
        return DirectionOutcome(passed=[], sims_used=1)

    pipe = AutoPipeline(
        prepare=prepare,
        propose_directions=propose,
        run_direction=run_direction,
        target_passes=99,
        max_total_sims=999,
        max_directions=5,
    )
    result = pipe.run()

    assert calls["run"] == 2           # chạy đúng 2 hướng được đề xuất
    assert result.directions_run == 2
    assert result.total_sims == 2
    assert result.stop_reason == "hết_hướng"
    assert result.passed_alphas == []
```

- [ ] **Step 3: Chạy test để xác nhận FAIL**

Run: `venv/Scripts/python.exe -m pytest tests/test_auto_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'AutoPipeline'` (module chưa tồn tại).

- [ ] **Step 4: Viết tối thiểu để PASS**

Tạo `src/pipeline/auto.py`:

```python
"""Orchestrator toàn trình: điều phối thuần, không biết httpx/CLI.

Nhận 3 callback (prepare, propose_directions, run_direction) + cấu hình dừng.
Lo vòng lặp + điều kiện dừng (K-pass / trần sim / hết hướng) + thu thập kết quả.
Test được bằng fake callback, không gọi mạng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PassedAlpha:
    expression: str
    sharpe: float | None
    fitness: float | None
    direction: str  # hướng nguồn (rỗng nếu GA)


@dataclass
class DirectionOutcome:
    passed: list[PassedAlpha]
    sims_used: int


@dataclass
class PrepareInfo:
    fields: int
    operators: int


@dataclass
class AutoEvent:
    kind: str       # prepare | directions | direction_start | direction_done | stop
    message: str
    data: dict = field(default_factory=dict)


@dataclass
class AutoResult:
    passed_alphas: list[PassedAlpha]
    directions_run: int
    total_sims: int
    stop_reason: str


@dataclass
class AutoPipeline:
    prepare: Callable[[], PrepareInfo]
    propose_directions: Callable[[int], list[str]]
    run_direction: Callable[[str], DirectionOutcome]
    target_passes: int = 3
    max_total_sims: int = 60
    max_directions: int = 5
    on_event: Callable[[AutoEvent], None] | None = None

    def _emit(self, kind: str, message: str, **data) -> None:
        if self.on_event is not None:
            self.on_event(AutoEvent(kind=kind, message=message, data=data))

    def run(self) -> AutoResult:
        passed: list[PassedAlpha] = []
        total_sims = 0
        directions_run = 0
        stop_reason = "hết_hướng"

        self.prepare()
        directions = self.propose_directions(self.max_directions)

        for direction in directions:
            outcome = self.run_direction(direction)
            passed.extend(outcome.passed)
            total_sims += outcome.sims_used
            directions_run += 1

        return AutoResult(
            passed_alphas=passed,
            directions_run=directions_run,
            total_sims=total_sims,
            stop_reason=stop_reason,
        )
```

- [ ] **Step 5: Chạy test để xác nhận PASS**

Run: `venv/Scripts/python.exe -m pytest tests/test_auto_pipeline.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/__init__.py src/pipeline/auto.py tests/test_auto_pipeline.py
git commit -m "feat(auto): khung AutoPipeline + dừng khi hết hướng (T1)"
```

---

## Task 2: Dừng khi đủ K alpha pass (kiểm ở ĐẦU vòng)

**Files:**
- Modify: `src/pipeline/auto.py` (hàm `run`)
- Test: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Viết test "đủ K pass" + "kiểm ở đầu vòng" (RED)**

Thêm vào `tests/test_auto_pipeline.py`:

```python
def test_dung_khi_du_k_pass():
    calls = {"run": 0}

    def run_direction(direction: str) -> DirectionOutcome:
        calls["run"] += 1
        return DirectionOutcome(passed=[_pa(f"e{calls['run']}a"), _pa(f"e{calls['run']}b")], sims_used=3)

    pipe = AutoPipeline(
        prepare=lambda: PrepareInfo(10, 5),
        propose_directions=lambda n: ["h1", "h2", "h3", "h4", "h5"],
        run_direction=run_direction,
        target_passes=3,
        max_total_sims=999,
        max_directions=5,
    )
    result = pipe.run()

    assert calls["run"] == 2           # mỗi hướng 2 pass; sau hướng 2 đã có 4 >= 3 -> dừng
    assert len(result.passed_alphas) == 4
    assert result.stop_reason == "đủ_K_pass"
    assert result.directions_run == 2


def test_kiem_dieu_kien_dung_o_dau_vong():
    calls = {"run": 0}

    def run_direction(direction: str) -> DirectionOutcome:
        calls["run"] += 1
        return DirectionOutcome(passed=[_pa("only")], sims_used=1)

    pipe = AutoPipeline(
        prepare=lambda: PrepareInfo(10, 5),
        propose_directions=lambda n: ["h1", "h2", "h3"],
        run_direction=run_direction,
        target_passes=1,
        max_total_sims=999,
        max_directions=3,
    )
    result = pipe.run()

    assert calls["run"] == 1           # hướng đầu đủ 1 pass -> hướng 2 KHÔNG được gọi
    assert result.stop_reason == "đủ_K_pass"
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `venv/Scripts/python.exe -m pytest tests/test_auto_pipeline.py -v`
Expected: FAIL — `test_dung_khi_du_k_pass` và `test_kiem_dieu_kien_dung_o_dau_vong` fail vì `stop_reason` luôn là `"hết_hướng"` và `run` được gọi đủ số hướng.

- [ ] **Step 3: Thêm kiểm tra K-pass ở đầu vòng**

Sửa vòng `for` trong `AutoPipeline.run` (src/pipeline/auto.py):

```python
        for direction in directions:
            if len(passed) >= self.target_passes:
                stop_reason = "đủ_K_pass"
                break
            outcome = self.run_direction(direction)
            passed.extend(outcome.passed)
            total_sims += outcome.sims_used
            directions_run += 1

        if len(passed) >= self.target_passes:
            stop_reason = "đủ_K_pass"
```

Lưu ý: kiểm thêm một lần sau vòng để bắt trường hợp hướng cuối cùng vừa đủ K (vòng kết thúc tự nhiên nhưng lý do thực là đủ K, không phải hết hướng).

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `venv/Scripts/python.exe -m pytest tests/test_auto_pipeline.py -v`
Expected: PASS (cả test cũ Task 1 vẫn xanh).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/auto.py tests/test_auto_pipeline.py
git commit -m "feat(auto): dừng khi đủ K alpha pass, kiểm ở đầu vòng (T2)"
```

---

## Task 3: Dừng khi chạm trần sim

**Files:**
- Modify: `src/pipeline/auto.py` (hàm `run`)
- Test: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Viết test "chạm trần sim" (RED)**

Thêm vào `tests/test_auto_pipeline.py`:

```python
def test_dung_khi_cham_tran_sim():
    calls = {"run": 0}

    def run_direction(direction: str) -> DirectionOutcome:
        calls["run"] += 1
        return DirectionOutcome(passed=[], sims_used=25)

    pipe = AutoPipeline(
        prepare=lambda: PrepareInfo(10, 5),
        propose_directions=lambda n: ["h1", "h2", "h3", "h4", "h5"],
        run_direction=run_direction,
        target_passes=99,
        max_total_sims=60,
        max_directions=5,
    )
    result = pipe.run()

    # Hướng 1 (25) + hướng 2 (50): chưa chạm; đầu vòng 3 tổng=50<60 vẫn chạy -> 75.
    # Đầu vòng 4: 75 >= 60 -> dừng. Vậy chạy 3 hướng.
    assert calls["run"] == 3
    assert result.total_sims == 75
    assert result.stop_reason == "chạm_trần_sim"
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `venv/Scripts/python.exe -m pytest tests/test_auto_pipeline.py::test_dung_khi_cham_tran_sim -v`
Expected: FAIL — `stop_reason` là `"hết_hướng"` (chưa có nhánh trần sim).

- [ ] **Step 3: Thêm kiểm tra trần sim ở đầu vòng**

Sửa đầu vòng `for` trong `AutoPipeline.run` để kiểm cả trần sim (đặt SAU kiểm K-pass):

```python
        for direction in directions:
            if len(passed) >= self.target_passes:
                stop_reason = "đủ_K_pass"
                break
            if total_sims >= self.max_total_sims:
                stop_reason = "chạm_trần_sim"
                break
            outcome = self.run_direction(direction)
            passed.extend(outcome.passed)
            total_sims += outcome.sims_used
            directions_run += 1

        if len(passed) >= self.target_passes:
            stop_reason = "đủ_K_pass"
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `venv/Scripts/python.exe -m pytest tests/test_auto_pipeline.py -v`
Expected: PASS (mọi test trước vẫn xanh).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/auto.py tests/test_auto_pipeline.py
git commit -m "feat(auto): dừng khi chạm trần sim (T3)"
```

---

## Task 4: Phát sự kiện đầy đủ + prepare lỗi dừng sạch

**Files:**
- Modify: `src/pipeline/auto.py` (hàm `run`)
- Test: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Viết test sự kiện + prepare lỗi (RED)**

Thêm vào `tests/test_auto_pipeline.py`:

```python
import pytest


def test_phat_du_su_kien():
    events = []

    pipe = AutoPipeline(
        prepare=lambda: PrepareInfo(10, 5),
        propose_directions=lambda n: ["h1", "h2"],
        run_direction=lambda d: DirectionOutcome(passed=[], sims_used=1),
        target_passes=99,
        max_total_sims=999,
        max_directions=5,
        on_event=lambda ev: events.append(ev),
    )
    pipe.run()

    kinds = [e.kind for e in events]
    assert kinds == [
        "prepare",
        "directions",
        "direction_start",
        "direction_done",
        "direction_start",
        "direction_done",
        "stop",
    ]
    # sự kiện stop mang lý do dừng
    assert events[-1].data.get("stop_reason") == "hết_hướng"


def test_prepare_loi_thi_dung_sach():
    calls = {"run": 0, "propose": 0}

    def prepare() -> PrepareInfo:
        raise RuntimeError("login hỏng")

    def run_direction(d):
        calls["run"] += 1
        return DirectionOutcome(passed=[], sims_used=1)

    def propose(n):
        calls["propose"] += 1
        return ["h1"]

    pipe = AutoPipeline(
        prepare=prepare,
        propose_directions=propose,
        run_direction=run_direction,
    )

    with pytest.raises(RuntimeError, match="login hỏng"):
        pipe.run()

    assert calls["run"] == 0        # chưa chạy hướng nào
    assert calls["propose"] == 0    # cũng chưa sinh hướng
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `venv/Scripts/python.exe -m pytest tests/test_auto_pipeline.py::test_phat_du_su_kien -v`
Expected: FAIL — chưa có sự kiện nào được phát (events rỗng).

- [ ] **Step 3: Thêm phát sự kiện vào `run`**

Cập nhật `AutoPipeline.run` (src/pipeline/auto.py) thành bản đầy đủ:

```python
    def run(self) -> AutoResult:
        passed: list[PassedAlpha] = []
        total_sims = 0
        directions_run = 0
        stop_reason = "hết_hướng"

        info = self.prepare()
        self._emit(
            "prepare",
            f"✓ đăng nhập | fields={info.fields} | operators={info.operators}",
            fields=info.fields,
            operators=info.operators,
        )

        directions = self.propose_directions(self.max_directions)
        self._emit(
            "directions",
            f"Sẽ thử {len(directions)} hướng",
            directions=list(directions),
        )

        total = len(directions)
        for i, direction in enumerate(directions, start=1):
            if len(passed) >= self.target_passes:
                stop_reason = "đủ_K_pass"
                break
            if total_sims >= self.max_total_sims:
                stop_reason = "chạm_trần_sim"
                break

            self._emit(
                "direction_start",
                f"[Hướng {i}/{total}] {direction!r}",
                index=i,
                total=total,
                direction=direction,
            )
            outcome = self.run_direction(direction)
            passed.extend(outcome.passed)
            total_sims += outcome.sims_used
            directions_run += 1
            self._emit(
                "direction_done",
                f"+{len(outcome.passed)} alpha đạt | sim lượt={outcome.sims_used} "
                f"| tổng pass={len(passed)}/{self.target_passes} | tổng sim={total_sims}",
                index=i,
                added=len(outcome.passed),
                sims_used=outcome.sims_used,
                total_passed=len(passed),
                total_sims=total_sims,
            )

        if len(passed) >= self.target_passes:
            stop_reason = "đủ_K_pass"

        self._emit(
            "stop",
            f"Dừng: {stop_reason} — pass={len(passed)}, sim={total_sims}, hướng đã chạy={directions_run}",
            stop_reason=stop_reason,
            total_passed=len(passed),
            total_sims=total_sims,
            directions_run=directions_run,
        )

        return AutoResult(
            passed_alphas=passed,
            directions_run=directions_run,
            total_sims=total_sims,
            stop_reason=stop_reason,
        )
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `venv/Scripts/python.exe -m pytest tests/test_auto_pipeline.py -v`
Expected: PASS — tất cả 6 test orchestrator xanh. (prepare lỗi nổi ra trước khi phát "directions" vì `self.prepare()` ném trước.)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/auto.py tests/test_auto_pipeline.py
git commit -m "feat(auto): phát sự kiện đầy đủ + prepare lỗi dừng sạch (T4)"
```

---

## Task 5: Bọc engine GA thành callback + test map alpha pass

**Files:**
- Modify: `main.py` (thêm `_auto_run_direction_ga`, helper map)
- Test: `tests/test_auto_pipeline.py`

Phần này test logic **map kết quả engine GA → PassedAlpha** mà không gọi mạng. Ta tách logic map thành một hàm thuần trong `src/pipeline/auto.py` để test độc lập, rồi `main.py` dùng lại.

- [ ] **Step 1: Viết test hàm map GA (RED)**

Thêm vào `tests/test_auto_pipeline.py`:

```python
from src.pipeline.auto import passed_from_ga


class _FakeSimResult:
    """Giả lập result của simulator cho hard_filter + score."""
    def __init__(self, sharpe, fitness, turnover, drawdown, status="passed"):
        self._m = {"sharpe": sharpe, "fitness": fitness, "turnover": turnover,
                   "returns": 0.1, "drawdown": drawdown, "margin": 0.002}
        self.status = status

    def metrics(self):
        return dict(self._m)


def test_passed_from_ga_loc_alpha_dat_nguong():
    # alpha tốt (đạt ngưỡng filter mặc định) + alpha tệ (trượt)
    good_expr = "rank(close)"
    bad_expr = "rank(open)"
    results = {
        good_expr: _FakeSimResult(sharpe=1.8, fitness=1.3, turnover=0.25, drawdown=0.08),
        bad_expr: _FakeSimResult(sharpe=0.2, fitness=0.1, turnover=0.9, drawdown=0.5),
    }

    passed = passed_from_ga([good_expr, bad_expr], results)

    assert [p.expression for p in passed] == [good_expr]
    assert passed[0].direction == ""           # GA không có hướng
    assert passed[0].sharpe == 1.8
    assert passed[0].fitness == 1.3
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `venv/Scripts/python.exe -m pytest tests/test_auto_pipeline.py::test_passed_from_ga_loc_alpha_dat_nguong -v`
Expected: FAIL — `ImportError: cannot import name 'passed_from_ga'`.

- [ ] **Step 3: Thêm `passed_from_ga` vào auto.py**

Thêm vào cuối `src/pipeline/auto.py`:

```python
def passed_from_ga(expressions, results) -> list[PassedAlpha]:
    """Lọc các biểu thức GA đạt ngưỡng hard-filter -> PassedAlpha (direction='').

    expressions: danh sách expr ứng viên (theo thứ tự tốt→kém).
    results: dict expr -> sim result (có .metrics() và .status).
    """
    from src.scoring.filter import passes as hard_filter
    from src.scoring.metrics import normalize

    out: list[PassedAlpha] = []
    for expr in expressions:
        result = results.get(expr)
        if result is None:
            continue
        ok, _ = hard_filter(result)
        if result.status == "passed" and ok:
            m = normalize(result)
            out.append(
                PassedAlpha(expression=expr, sharpe=m["sharpe"], fitness=m["fitness"], direction="")
            )
    return out
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `venv/Scripts/python.exe -m pytest tests/test_auto_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/auto.py tests/test_auto_pipeline.py
git commit -m "feat(auto): passed_from_ga lọc alpha GA đạt ngưỡng (T5)"
```

---

## Task 6: Lệnh `auto` trong main.py (đấu nối engine thật)

**Files:**
- Modify: `main.py` (import, thêm lệnh `auto` + 3 hàm bọc callback)

Phần này không unit-test (phụ thuộc LLM/WQ thật); tin tưởng test orchestrator + engine sẵn có. Kiểm chứng bằng `--help` chạy được.

- [ ] **Step 1: Thêm import AutoPipeline ở đầu main.py**

Thêm vào khối import (sau dòng `from src.storage.repository import AlphaRepository`, khoảng main.py:26):

```python
from src.pipeline.auto import (
    AutoEvent,
    AutoPipeline,
    DirectionOutcome,
    PassedAlpha,
    PrepareInfo,
    passed_from_ga,
)
```

- [ ] **Step 2: Thêm lệnh `auto` + hàm bọc (trước `def start()`, khoảng main.py:1041)**

```python
def _auto_prepare(client_box: dict, session_factory, region, universe, delay) -> PrepareInfo:
    """Đăng nhập + ensure fields/operators (cache nếu có). Trả PrepareInfo."""
    client = _make_client()
    client.authenticate()
    client_box["client"] = client

    field_repo = FieldRepository(client, session_factory)
    fields, _ = field_repo.ensure(region, universe, delay)

    op_repo = OperatorRepository(client, session_factory)
    operators, _ = op_repo.ensure()

    return PrepareInfo(fields=len(fields), operators=len(operators))


def _auto_run_direction_ai(client_box, session_factory, region, universe, delay, per_direction_box):
    """Trả callback run_direction cho engine AI."""
    def run(direction: str) -> DirectionOutcome:
        loop, _deepseek = _make_research_loop(
            session_factory, client_box["client"], region, universe, delay,
            max_sims=per_direction_box["per_direction"], patience=3,
        )
        result = loop.run(direction)
        passed: list[PassedAlpha] = []
        cand = result.best_candidate
        if cand is not None and result.zoo_added > 0 and result.best_vector is not None:
            d = result.best_vector.dimensions()
            passed.append(
                PassedAlpha(
                    expression=cand.expression,
                    sharpe=d.get("sharpe"),
                    fitness=d.get("fitness"),
                    direction=direction,
                )
            )
        return DirectionOutcome(passed=passed, sims_used=result.sims_used)
    return run


def _auto_run_direction_ga(client_box, session_factory, region, universe, delay, per_direction_box):
    """Trả callback run_direction cho engine GA."""
    import random

    from src.generation.ast_utils import to_expression
    from src.generation.template import TemplateGenerator
    from src.optimization.evolution import GeneticOptimizer
    from src.simulation.pre_filter import PreFilter

    def run(direction: str) -> DirectionOutcome:
        fields, operators = _cached_symbols(session_factory)
        pf = PreFilter(known_operators=operators or None, known_fields=set(fields))
        tgen = TemplateGenerator(fields, pf, rng=random.Random())
        sim = Simulator(client_box["client"])

        def seed_factory():
            exprs = tgen.generate(1)
            return GeneticOptimizer.expr_to_node(exprs[0] if exprs else f"rank({fields[0]})")

        results: dict = {}
        original_simulate = sim.simulate

        def simulate_capture(expr, **kwargs):
            res = original_simulate(expr, **kwargs)
            results[expr] = res
            return res

        sim.simulate = simulate_capture
        opt = GeneticOptimizer(
            simulator=sim, prefilter=pf, seed_factory=seed_factory, fields=fields,
            population_size=30, generations=10,
            max_simulations=per_direction_box["per_direction"],
        )
        best_nodes = opt.run()
        best_exprs = [to_expression(n) for n in best_nodes]
        passed = passed_from_ga(best_exprs, results)
        return DirectionOutcome(passed=passed, sims_used=opt.simulations_used)
    return run


@app.command()
def auto(
    engine: str = typer.Option("ai", help="ai | ga"),
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    target_passes: int = typer.Option(3, "--target", help="Dừng khi đủ K alpha đạt ngưỡng"),
    max_sims: int = typer.Option(60, "--max-sims", help="Trần cứng tổng số simulation"),
    max_directions: int = typer.Option(5, "--directions", help="Số hướng nghiên cứu tối đa (engine ai)"),
) -> None:
    """Chạy toàn trình: login → cache → tìm/mô phỏng/cải thiện → log. KHÔNG nộp."""
    _setup_logging()
    engine = engine.lower().strip()
    if engine not in {"ai", "ga"}:
        console.print("[red]--engine chỉ nhận 'ai' hoặc 'ga'.[/red]")
        raise typer.Exit(code=1)

    engine_box = init_db(make_engine())
    session_factory = make_session_factory(engine_box)
    client_box: dict = {}
    per_direction_box = {"per_direction": max_sims}

    def prepare() -> PrepareInfo:
        return _auto_prepare(client_box, session_factory, region, universe, delay)

    def propose(n: int) -> list[str]:
        if engine == "ga":
            return [""]
        from src.simulation.pre_filter import PreFilter
        gen = _make_llm_generator(session_factory, PreFilter())
        return gen.generate_ideas(n)

    run_builder = _auto_run_direction_ai if engine == "ai" else _auto_run_direction_ga
    run_direction_raw = run_builder(
        client_box, session_factory, region, universe, delay, per_direction_box
    )

    state = {"sims_used": 0, "dirs_total": 1}

    def run_direction(direction: str) -> DirectionOutcome:
        # Chia trần sim: phần còn lại / số hướng còn lại (hướng đầu không ăn hết).
        remaining = max_sims - state["sims_used"]
        dirs_left = max(1, state["dirs_total"] - 0)
        per_direction_box["per_direction"] = max(1, remaining // dirs_left)
        outcome = run_direction_raw(direction)
        state["sims_used"] += outcome.sims_used
        state["dirs_total"] = max(1, state["dirs_total"] - 1)
        return outcome

    def on_event(ev: AutoEvent) -> None:
        logger.info("[auto:{}] {} | {}", ev.kind, ev.message, ev.data)
        style = {"stop": "bold green", "prepare": "cyan"}.get(ev.kind, "")
        console.print(f"[{style}]{ev.message}[/{style}]" if style else ev.message)

    pipe = AutoPipeline(
        prepare=prepare,
        propose_directions=propose,
        run_direction=run_direction,
        target_passes=target_passes,
        max_total_sims=max_sims,
        max_directions=max_directions if engine == "ai" else 1,
        on_event=on_event,
    )

    # Đặt dirs_total đúng sau khi biết số hướng (cập nhật qua sự kiện "directions").
    def on_event_with_total(ev: AutoEvent) -> None:
        if ev.kind == "directions":
            state["dirs_total"] = max(1, len(ev.data.get("directions", [])))
        on_event(ev)

    pipe.on_event = on_event_with_total
    result = pipe.run()

    table = Table(title=f"Alpha đạt ngưỡng ({len(result.passed_alphas)}) — engine={engine}, dừng: {result.stop_reason}")
    table.add_column("Expression", overflow="fold")
    table.add_column("Sharpe", justify="right")
    table.add_column("Fitness", justify="right")
    table.add_column("Hướng nguồn", overflow="fold")
    for p in result.passed_alphas:
        table.add_row(
            p.expression,
            f"{p.sharpe:.3f}" if p.sharpe is not None else "—",
            f"{p.fitness:.3f}" if p.fitness is not None else "—",
            p.direction or "—",
        )
    console.print(table)
    console.print(
        "[dim]Đã lưu DB — xem bằng lệnh 'top'. CHƯA nộp; nộp bằng 'submit' khi muốn.[/dim]"
    )
```

- [ ] **Step 3: Kiểm chứng lệnh import + help chạy được**

Run: `venv/Scripts/python.exe -m main auto --help`
Expected: in ra help của lệnh `auto` với các option `--engine`, `--target`, `--max-sims`, `--directions`. Không lỗi import.

> Nếu chạy `python -m main` không được, thử `venv/Scripts/python.exe main.py auto --help`.

- [ ] **Step 4: Chạy lại toàn bộ test để chắc không vỡ**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: tất cả test xanh (194 cũ + 7 mới).

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(auto): lệnh auto đấu nối engine AI/GA + log + bảng kết quả (T6)"
```

---

## Task 7: `start()` gọi thẳng `auto`, xóa menu wizard

**Files:**
- Modify: `main.py` (viết lại `start()`, xóa `_WizardState`, `_ask`, tất cả `_wizard_*`)

- [ ] **Step 1: Viết lại `start()` gọi `auto`**

Thay toàn bộ thân hàm `start()` (main.py:1041-1093) bằng:

```python
@app.command()
def start(
    engine: str = typer.Option("ai", help="ai | ga"),
) -> None:
    """Chạy toàn trình với mặc định (alias của 'auto'). KHÔNG nộp."""
    auto(engine=engine)
```

- [ ] **Step 2: Xóa code wizard không còn dùng**

Xóa các định nghĩa sau khỏi main.py (không còn ai gọi sau khi `start` đổi):
- `class _WizardState` (và toàn bộ method của nó)
- `def _ask(...)`
- `def _wizard_login`, `_wizard_fields`, `_wizard_operators`, `_wizard_simulate`, `_wizard_generate`, `_wizard_run_ga`, `_wizard_research`, `_wizard_list_fields`, `_wizard_scope`, `_wizard_menu`

Giữ nguyên: các `@app.command` khác (login, probe-fields, fetch-fields, cache-status, fetch-operators, list-fields, simulate, sweep-config, generate, run-ga, research, llm-generate, llm-ideas, top, originality, submit) và các helper `_make_*`, `_cached_symbols`, `_run_research_with_progress`, `_render_research_result`.

- [ ] **Step 3: Kiểm chứng CLI còn chạy + không còn tham chiếu wizard**

Run: `venv/Scripts/python.exe -m main --help`
Expected: liệt kê các lệnh gồm `auto`, `start`, `login`, ... Không lỗi `NameError`.

Run: `grep -rn "_wizard\|_WizardState\|_ask(" main.py` (hoặc dùng Grep tool)
Expected: không còn kết quả nào (đã xóa sạch).

- [ ] **Step 4: Chạy toàn bộ test**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: tất cả test xanh.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(auto): start() gọi thẳng auto, bỏ menu wizard (T7)"
```

---

## Self-Review

**Spec coverage:**
- 1 lệnh toàn trình → Task 6 (`auto`) + Task 7 (`start` alias).
- Chọn engine ai/ga → Task 6 (`--engine`, `propose`/`run_builder`).
- Engine AI brainstorm nhiều hướng → Task 6 (`generate_ideas(n)`).
- Dừng K-pass HOẶC trần sim → Task 2, Task 3.
- Không nộp → Task 6 (không gọi `submit`; có dòng nhắc).
- Bỏ menu, giữ lệnh phụ → Task 7.
- Log đầy đủ → Task 4 (sự kiện) + Task 6 (`on_event` in console + logger + bảng).
- Chia trần sim theo hướng → Task 6 (`per_direction_box`, `remaining // dirs_left`).

**Placeholder scan:** không có TBD/TODO; mọi step có code/lệnh cụ thể.

**Type consistency:** `PassedAlpha(expression, sharpe, fitness, direction)`, `DirectionOutcome(passed, sims_used)`, `PrepareInfo(fields, operators)`, `AutoResult(passed_alphas, directions_run, total_sims, stop_reason)`, `AutoEvent(kind, message, data)` — dùng nhất quán giữa các task. `passed_from_ga(expressions, results)` khớp Task 5 và Task 6. `stop_reason` ∈ {"đủ_K_pass","chạm_trần_sim","hết_hướng"} nhất quán.

**Lưu ý kỹ thuật cần để ý khi thực thi:**
- `best_vector.dimensions()` trả dict có khóa `sharpe`/`fitness` (đã thấy trong `_render_research_result` main.py:510). Xác nhận lại khi code Task 6.
- AI engine hiện chỉ trả 1 `best_candidate`/hướng → tối đa 1 pass/hướng. Đủ để đạt K khi chạy nhiều hướng. Không mở rộng (YAGNI).
