# Căn chỉnh config tăng chất lượng alpha — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đưa closed-loop về một bộ config thống nhất (SUBINDUSTRY/decay=4/truncation=0.08) cho cả local gate lẫn Brain sim, để bộ lọc rẻ đánh giá khớp cách WQ chấm và bật decay theo docs → tăng Sharpe/fitness.

**Architecture:** Tách một helper thuần `_closed_loop_configs()` dựng cặp (PortfolioConfig local, SimConfig Brain) từ cùng bộ tham số; `_run_closed_loop_session` dùng helper và truyền `sim_config` vào `_make_research_loop`; đổi default sang bộ mới ở cả `_run_closed_loop_session` và `closed_loop_cmd`.

**Tech Stack:** Python, typer, pytest.

## Global Constraints

- TDD: viết test trước, chạy đỏ, implement, chạy xanh, commit.
- Giao tiếp/commit tiếng Việt; mỗi task 1 commit.
- Chạy test với biến môi trường `WQ_NO_FILE_LOG=1` để không ghi log production.
- `PortfolioConfig.neutralization` là enum `Neutralization`; `SimConfig.neutralization` là `str` (đã normalize hoa). Cùng biểu diễn giá trị "SUBINDUSTRY".
- Bộ config thống nhất: `neutralization="SUBINDUSTRY", decay=4, truncation=0.08`.

---

### Task 1: Helper `_closed_loop_configs` dựng cặp config khớp nhau

**Files:**
- Modify: `main.py` (thêm helper ngay trước `_run_closed_loop_session`, khoảng dòng 611)
- Test: `tests/test_closed_loop_seed.py` (đã tồn tại, thêm test vào cuối)

**Interfaces:**
- Produces: `_closed_loop_configs(neutralization: str, decay: int, truncation: float, delay: int, region: str, universe: str) -> tuple[PortfolioConfig, SimConfig]` — trả `(cfg, sim_config)` dùng chung neutralization/decay/truncation.

- [ ] **Step 1: Viết test đỏ**

Thêm vào cuối `tests/test_closed_loop_seed.py`:

```python
def test_closed_loop_configs_khop_local_va_brain() -> None:
    """cfg (local gate) và sim_config (Brain sim) phải dùng CHUNG một bộ
    neutralization/decay/truncation — tránh mismatch làm local gate lọc sai."""
    from src.backtest.config import Neutralization

    cfg, sim = main._closed_loop_configs("SUBINDUSTRY", 4, 0.08, 1, "USA", "TOP3000")
    # local gate (PortfolioConfig, neutralization là enum)
    assert cfg.neutralization == Neutralization.SUBINDUSTRY
    assert cfg.decay == 4
    assert cfg.truncation == 0.08
    # Brain sim (SimConfig, neutralization là str đã normalize)
    assert sim.neutralization == "SUBINDUSTRY"
    assert sim.decay == 4
    assert sim.truncation == 0.08
    assert sim.region == "USA" and sim.universe == "TOP3000" and sim.delay == 1
```

- [ ] **Step 2: Chạy test đỏ**

Run: `WQ_NO_FILE_LOG=1 venv/Scripts/python.exe -m pytest tests/test_closed_loop_seed.py::test_closed_loop_configs_khop_local_va_brain -q`
Expected: FAIL — `AttributeError: module 'main' has no attribute '_closed_loop_configs'`.

- [ ] **Step 3: Implement helper**

Thêm vào `main.py` ngay TRƯỚC `def _run_closed_loop_session(` (khoảng dòng 611):

```python
def _closed_loop_configs(
    neutralization: str, decay: int, truncation: float, delay: int,
    region: str, universe: str,
):
    """Dựng cặp (PortfolioConfig local gate, SimConfig Brain sim) DÙNG CHUNG một bộ
    neutralization/decay/truncation — để bộ lọc local đánh giá alpha khớp cách WQ Brain
    chấm điểm (tránh mismatch)."""
    from src.simulation.config import SimConfig

    cfg = _portfolio_config_from_opts(neutralization, decay, truncation, delay)
    sim_config = SimConfig.default(region=region, universe=universe, delay=delay).with_overrides(
        neutralization=neutralization, decay=decay, truncation=truncation,
    )
    return cfg, sim_config
```

- [ ] **Step 4: Chạy test xanh**

Run: `WQ_NO_FILE_LOG=1 venv/Scripts/python.exe -m pytest tests/test_closed_loop_seed.py::test_closed_loop_configs_khop_local_va_brain -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_closed_loop_seed.py
git commit -m "feat(closed-loop): helper _closed_loop_configs dung cap config khop local+Brain"
```

---

### Task 2: Wire helper + truyền sim_config + đổi default sang SUBINDUSTRY/4/0.08

**Files:**
- Modify: `main.py` — `_run_closed_loop_session` (signature dòng ~616 + thân dòng ~637-641); `closed_loop_cmd` options (dòng ~669-671)
- Test: `tests/test_closed_loop_seed.py`

**Interfaces:**
- Consumes: `_closed_loop_configs(...)` từ Task 1; `_make_research_loop(..., sim_config=...)` (param `sim_config` đã tồn tại, default None).

- [ ] **Step 1: Viết test đỏ (default mới)**

Thêm vào cuối `tests/test_closed_loop_seed.py`:

```python
def test_closed_loop_defaults_la_bo_config_thong_nhat() -> None:
    """Default closed-loop phải là bộ thống nhất SUBINDUSTRY/4/0.08 (đổi mặc định engine)."""
    import inspect

    sig = inspect.signature(main._run_closed_loop_session)
    assert sig.parameters["neutralization"].default == "SUBINDUSTRY"
    assert sig.parameters["decay"].default == 4
    assert sig.parameters["truncation"].default == 0.08
```

- [ ] **Step 2: Chạy test đỏ**

Run: `WQ_NO_FILE_LOG=1 venv/Scripts/python.exe -m pytest tests/test_closed_loop_seed.py::test_closed_loop_defaults_la_bo_config_thong_nhat -q`
Expected: FAIL — default hiện là `"NONE"`, `0`, `0.10`.

- [ ] **Step 3: Đổi default signature `_run_closed_loop_session`**

Trong `main.py`, sửa dòng trong signature `_run_closed_loop_session`:

Từ:
```python
    neutralization: str = "NONE", decay: int = 0, truncation: float = 0.10,
```
Thành:
```python
    neutralization: str = "SUBINDUSTRY", decay: int = 4, truncation: float = 0.08,
```

- [ ] **Step 4: Dùng helper + truyền sim_config trong thân**

Trong `main.py`, `_run_closed_loop_session`, thay đoạn:

```python
    cfg = _portfolio_config_from_opts(neutralization, decay, truncation, delay)
    loop, _deepseek = _make_research_loop(
        session_factory, client, region, universe, delay,
        max_sims=10**9, patience=patience, marathon=True,
    )
```
Thành:
```python
    cfg, sim_config = _closed_loop_configs(
        neutralization, decay, truncation, delay, region, universe,
    )
    loop, _deepseek = _make_research_loop(
        session_factory, client, region, universe, delay,
        max_sims=10**9, patience=patience, marathon=True, sim_config=sim_config,
    )
```

- [ ] **Step 5: Chạy test xanh + regression seed cũ**

Run: `WQ_NO_FILE_LOG=1 venv/Scripts/python.exe -m pytest tests/test_closed_loop_seed.py -q`
Expected: PASS toàn bộ (gồm test cũ `_resolve_base_seed`).

- [ ] **Step 6: Đổi default `closed_loop_cmd` options**

Trong `main.py`, `closed_loop_cmd`, sửa 3 dòng option:

Từ:
```python
    neutralization: str = typer.Option("NONE"),
    decay: int = typer.Option(0),
    truncation: float = typer.Option(0.10),
```
Thành:
```python
    neutralization: str = typer.Option("SUBINDUSTRY"),
    decay: int = typer.Option(4),
    truncation: float = typer.Option(0.08),
```

- [ ] **Step 7: Smoke CLI + full test suite**

Run: `WQ_NO_FILE_LOG=1 venv/Scripts/python.exe main.py closed-loop --help`
Expected: hiện help, không lỗi import.

Run: `WQ_NO_FILE_LOG=1 venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`
Expected: chỉ `test_db_postgres` fail (pre-existing ModuleNotFound), còn lại PASS.

- [ ] **Step 8: Commit**

```bash
git add main.py tests/test_closed_loop_seed.py
git commit -m "feat(closed-loop): default SUBINDUSTRY/decay=4/truncation=0.08 + Brain sim khop local gate"
```

---

## Self-Review

- **Spec coverage:** Thay đổi default 3 nơi (closed_loop_cmd, _run_closed_loop_session params, thân truyền sim_config) → Task 2 phủ. Helper khớp config → Task 1. Testing khớp + default → cả 2 task. ✅
- **Placeholder scan:** không có TBD/TODO; mọi step có code/lệnh cụ thể. ✅
- **Type consistency:** `_closed_loop_configs` trả `(PortfolioConfig, SimConfig)`; test dùng `Neutralization.SUBINDUSTRY` (enum) cho cfg và `"SUBINDUSTRY"` (str) cho sim — khớp thực tế 2 kiểu. `_make_research_loop` param `sim_config` đã tồn tại (default None). ✅
