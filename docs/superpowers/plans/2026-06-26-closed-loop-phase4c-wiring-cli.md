# Closed-Loop Phase 4C — Wiring + CLI/menu (chạy thật) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development hoặc
> superpowers:executing-plans. Phase 4C của feature "Vòng kín AI + MiniBrain" (spec
> `docs/superpowers/specs/2026-06-26-ai-minibrain-closed-loop-design.md`). Phase 1-3 + 4A +
> 4B ĐÃ xong trên nhánh `closed-loop-integration`.

**Goal:** Ráp vòng kín chạy được: (1) `QuotaExhausted` best-effort trong
`RefinementLoopRefiner` (bắt `AuthExpiredError` → dừng gọn); (2) factory `build_closed_loop`
composes adapters + `ClosedLoop` + `CalibrationTracker`; (3) CLI `closed-loop` + mục menu
trong `main.py start` (đăng nhập + panel + `.env` AI backend → chạy `ClosedLoop.run`). Sau
4C tool chạy end-to-end; tinh chỉnh phát hiện quota chính xác sau LẦN CHẠY THẬT đầu tiên.

**Architecture:** Sửa `src/app/closed_loop_adapters.py` (QuotaExhausted + factory). Thêm CLI
`closed-loop` command + menu option vào `main.py` (composition root: đăng nhập qua
`_make_client`, dựng RefinementLoop qua `_make_research_loop(marathon=True)`, set
`loop.market_data`, panel qua `ParquetSource`, repo `MiniBrainRepository`). Smoke-test bằng
CliRunner; chạy thật do người dùng (cần login + `.env`).

**Tech Stack:** Python 3.12, Typer/CliRunner, pytest.

## Global Constraints

- Python 3.12; full type hints trên code mới; `ruff` clean (main.py có debt E402/F841 tiền-tồn
  — KHÔNG thêm lỗi MỚI ở code mới); không unused import.
- `src/pipeline/closed_loop.py` KHÔNG đổi. `src/app/` được phép import gp/llm/simulation.
- **Tiếng Việt giữ dấu đúng chính tả** trong docstring/help/comment mới.
- TDD cho phần test được (Task 1 QuotaExhausted, Task 2 factory, Task 3 smoke CLI). Chạy thật
  = việc của người dùng (login + quota) — KHÔNG mock Brain thật trong test.

## Pre-condition (chữ ký thật đã xác minh)

```python
# src/simulation/simulator.py: class AuthExpiredError(RuntimeError)  (raise khi 401/session chết)
# src/app/closed_loop_adapters.py (4B): RefinementLoopRefiner(loop).refine_and_sim(candidate)->IdeaOutcome
#   GPIdeaSource(data, repo, config, registry, *, pop_size, n_generations, base_seed, top_k, max_corr)
# src/pipeline/closed_loop.py: ClosedLoop(idea_source, refiner, repo, *, region, universe,
#   max_ideas=None, calibration_tracker=None).run()->ClosedLoopReport; QuotaExhausted; CalibrationTracker(repo,*,every,rho_bar)
# main.py: _make_client()->WQBrainClient (đọc .env); client.authenticate();
#   _make_research_loop(session_factory, client, region, universe, delay, max_sims, patience,
#       ..., marathon=False) -> (loop, deepseek); loop.market_data (set được sau dựng);
#   loop.max_simulations (int). init_db/make_engine/make_session_factory; app/console/_setup_logging;
#   ParquetSource(dir).load(start,end,universe); default_registry(); PortfolioConfig/Neutralization.
# _portfolio_config_from_opts(neutralization, decay, truncation, delay) -> PortfolioConfig (Phase 8).
```

## File Structure

- **Modify** `src/app/closed_loop_adapters.py` (~25 dòng): QuotaExhausted trong refiner + `build_closed_loop`.
- **Modify** `tests/unit/test_closed_loop_adapters.py` (~30 dòng): test QuotaExhausted + factory.
- **Modify** `main.py` (~70 dòng): CLI `closed-loop` + menu option 7 + helper.
- **Modify** `tests/unit/test_cli_score_one_generate.py` (~20 dòng) HOẶC test mới: smoke `closed-loop`.

---

### Task 1: `QuotaExhausted` best-effort trong `RefinementLoopRefiner`

**Files:**
- Modify: `src/app/closed_loop_adapters.py`
- Test: `tests/unit/test_closed_loop_adapters.py`

**Interfaces:**
- Consumes: `AuthExpiredError` (src.simulation.simulator), `QuotaExhausted` (src.pipeline.closed_loop).
- Produces: `RefinementLoopRefiner.refine_and_sim` bọc `run_from_seed` trong try/except
  `AuthExpiredError` → `raise QuotaExhausted(...) from exc`.

- [ ] **Step 1: Viết test đỏ (thêm vào `tests/unit/test_closed_loop_adapters.py`)**

```python
def test_refiner_raises_quota_exhausted_on_auth_expired() -> None:
    from src.pipeline.closed_loop import QuotaExhausted
    from src.simulation.simulator import AuthExpiredError

    class _AuthDeadLoop:
        def run_from_seed(self, expression, on_progress=None):
            raise AuthExpiredError("session het han / quota")

    with pytest.raises(QuotaExhausted):
        RefinementLoopRefiner(_AuthDeadLoop()).refine_and_sim(_cand("rank(close)"))
```

- [ ] **Step 2: Chạy test — FAIL** (`AuthExpiredError` nổi lên, không phải QuotaExhausted)

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop_adapters.py::test_refiner_raises_quota_exhausted_on_auth_expired -q
```

- [ ] **Step 3: Sửa `RefinementLoopRefiner.refine_and_sim`**

Thêm import đầu file: `from src.pipeline.closed_loop import IdeaOutcome, QuotaExhausted` (gộp
QuotaExhausted vào dòng import closed_loop đã có). Thêm import `AuthExpiredError`:
```python
from src.simulation.simulator import AuthExpiredError
```
Bọc lời gọi `run_from_seed`:
```python
    def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome:
        try:
            result = self.loop.run_from_seed(candidate.expr)  # type: ignore[attr-defined]
        except AuthExpiredError as exc:
            # Best-effort: session chết / hết quota Brain -> báo ClosedLoop dừng gọn.
            # (Tinh chỉnh nhận diện quota-ngày chính xác sau lần chạy thật đầu tiên.)
            raise QuotaExhausted(str(exc)) from exc
        # ... phần map LoopResult -> IdeaOutcome giữ nguyên ...
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop_adapters.py -q
```
Expected: PASS (các test 4B cũ + test mới).

- [ ] **Step 5: ruff + mypy + commit**

```bash
venv/Scripts/python.exe -m ruff check src/app/closed_loop_adapters.py tests/unit/test_closed_loop_adapters.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/app/closed_loop_adapters.py
git add src/app/closed_loop_adapters.py tests/unit/test_closed_loop_adapters.py
git commit -m "feat(app): RefinementLoopRefiner nem QuotaExhausted khi AuthExpired (best-effort quota)"
```

---

### Task 2: Factory `build_closed_loop`

**Files:**
- Modify: `src/app/closed_loop_adapters.py`
- Test: `tests/unit/test_closed_loop_adapters.py`

**Interfaces:**
- Produces:
  ```python
  def build_closed_loop(
      *, data, repo, config, registry, loop,
      region: str = "USA", universe: str = "TOP3000",
      pop_size: int = 30, n_generations: int = 3, base_seed: int = 42,
      top_k: int = 10, max_corr: float = 0.70,
      calibrate_every: int = 10, rho_bar: float = 0.5, max_ideas: int | None = None,
  ) -> ClosedLoop:
      """Ráp GPIdeaSource + RefinementLoopRefiner + CalibrationTracker + ClosedLoop."""
  ```

- [ ] **Step 1: Viết test đỏ (thêm vào test file)**

```python
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
```

- [ ] **Step 2: Chạy test — FAIL** (`ImportError: build_closed_loop`)

- [ ] **Step 3: Thêm `build_closed_loop` vào `src/app/closed_loop_adapters.py`**

```python
def build_closed_loop(
    *, data: object, repo: object, config: object, registry: object, loop: object,
    region: str = "USA", universe: str = "TOP3000",
    pop_size: int = 30, n_generations: int = 3, base_seed: int = 42,
    top_k: int = 10, max_corr: float = 0.70,
    calibrate_every: int = 10, rho_bar: float = 0.5, max_ideas: int | None = None,
) -> "ClosedLoop":
    """Ráp vòng kín: GPIdeaSource (sinh ý tưởng) + RefinementLoopRefiner (AI refine+sim qua
    `loop`) + CalibrationTracker (ρ) + ClosedLoop. `loop` là RefinementLoop đã dựng (đăng nhập
    + Simulator thật) do composition root (main.py) truyền vào."""
    from src.pipeline.closed_loop import CalibrationTracker, ClosedLoop

    idea_source = GPIdeaSource(
        data, repo, config, registry, pop_size=pop_size, n_generations=n_generations,
        base_seed=base_seed, top_k=top_k, max_corr=max_corr,
    )
    refiner = RefinementLoopRefiner(loop)
    tracker = CalibrationTracker(repo, every=calibrate_every, rho_bar=rho_bar)  # type: ignore[arg-type]
    return ClosedLoop(
        idea_source=idea_source, refiner=refiner, repo=repo,  # type: ignore[arg-type]
        region=region, universe=universe, max_ideas=max_ideas,
        calibration_tracker=tracker,
    )
```
Thêm `from src.pipeline.closed_loop import ClosedLoop` vào TYPE_CHECKING nếu cần cho annotation
(hoặc để chuỗi `"ClosedLoop"` + import trong thân — như trên).

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop_adapters.py -q
```

- [ ] **Step 5: ruff + mypy + commit**

```bash
venv/Scripts/python.exe -m ruff check src/app/closed_loop_adapters.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/app/closed_loop_adapters.py
git add src/app/closed_loop_adapters.py tests/unit/test_closed_loop_adapters.py
git commit -m "feat(app): build_closed_loop factory rap GPIdeaSource+Refiner+Tracker+ClosedLoop"
```

---

### Task 3: CLI `closed-loop` command + mục menu

**Files:**
- Modify: `main.py`
- Test: `tests/unit/test_cli_closed_loop.py` (mới)

**Interfaces:**
- Consumes: `build_closed_loop` (Task 2), `_make_client`/`_make_research_loop`/`init_db`/
  `make_engine`/`make_session_factory`/`ParquetSource`/`default_registry`/
  `_portfolio_config_from_opts`/`MiniBrainRepository`/`app`/`console`/`_setup_logging`.
- Produces: `@app.command("closed-loop")` + mục 7 trong `_print_menu`/`start`.

- [ ] **Step 1: Viết smoke test đỏ `tests/unit/test_cli_closed_loop.py`**

```python
"""Smoke test CLI closed-loop: KHÔNG login/sim Brain thật (chỉ kiểm parse + lỗi input rõ)."""

from __future__ import annotations

from typer.testing import CliRunner

from main import app

runner = CliRunner()


def test_closed_loop_missing_market_data_dir_fails_clearly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["closed-loop", "--market-data-dir", str(tmp_path / "nope")],
    )
    assert result.exit_code == 1


def test_closed_loop_help_lists_options() -> None:
    result = runner.invoke(app, ["closed-loop", "--help"])
    assert result.exit_code == 0
    assert "market-data-dir" in result.stdout
    assert "patience" in result.stdout
```

- [ ] **Step 2: Chạy test — FAIL** (`No command "closed-loop"`)

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_cli_closed_loop.py -q
```

- [ ] **Step 3: Thêm lệnh `closed-loop` vào `main.py`**

Đặt gần lệnh `generate`/`score-one`. Kiểm `--market-data-dir` TRƯỚC khi login (để smoke test
missing-dir fail nhanh, không cần credentials):
```python
@app.command("closed-loop")
def closed_loop_cmd(
    market_data_dir: str = typer.Option(..., help="Thư mục parquet MarketData (gate local)"),
    region: str = typer.Option("USA"),
    universe: str = typer.Option("TOP3000"),
    delay: int = typer.Option(1),
    patience: int = typer.Option(5, help="Bỏ ý tưởng sau N lần refine không cải thiện"),
    pop_size: int = typer.Option(30, help="Kích thước quần thể GP mỗi batch ý tưởng"),
    n_generations: int = typer.Option(3),
    top_k: int = typer.Option(10, help="Số ý tưởng/batch sau decorrelate"),
    max_corr: float = typer.Option(0.70),
    max_ideas: int = typer.Option(0, help="0 = không trần (chạy đến hết quota)"),
    neutralization: str = typer.Option("NONE"),
    decay: int = typer.Option(0),
    truncation: float = typer.Option(0.10),
) -> None:
    """Vòng kín AI + MiniBrain: GP sinh ý tưởng → AI refine ≤patience + gate local → SIM Brain
    → lưu DB + feedback → lặp đến khi hết quota (Ctrl+C để dừng tay). Cần đăng nhập + .env AI."""
    _setup_logging()
    from pathlib import Path

    import src.operators_local  # noqa: F401
    from src.app.closed_loop_adapters import build_closed_loop
    from src.data.adapters.parquet_source import ParquetSource
    from src.lang.registry import default_registry
    from src.pipeline.closed_loop import QuotaExhausted
    from src.storage.repository import MiniBrainRepository

    if not Path(market_data_dir).is_dir():
        console.print(f"[red]Không thấy thư mục MarketData: {market_data_dir}[/red]")
        raise typer.Exit(code=1)

    client = _make_client()
    client.authenticate()

    engine_db = init_db(make_engine())
    session_factory = make_session_factory(engine_db)
    repo = MiniBrainRepository(session_factory)

    try:
        data = ParquetSource(market_data_dir).load("1900-01-01", "2999-12-31", universe)
    except (FileNotFoundError, AssertionError, OSError) as exc:
        console.print(f"[red]Không load được MarketData: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    cfg = _portfolio_config_from_opts(neutralization, decay, truncation, delay)
    loop, _deepseek = _make_research_loop(
        session_factory, client, region, universe, delay,
        max_sims=10**9, patience=patience, marathon=True,
    )
    loop.market_data = data          # bật local gate trước sim
    loop.local_gate_cfg = cfg
    loop.max_simulations = 10**9     # không trần local; dừng theo quota Brain (QuotaExhausted)

    cl = build_closed_loop(
        data=data, repo=repo, config=cfg, registry=default_registry(), loop=loop,
        region=region, universe=universe, pop_size=pop_size, n_generations=n_generations,
        top_k=top_k, max_corr=max_corr, max_ideas=(max_ideas or None),
    )
    console.print("[cyan]Bắt đầu vòng kín (Ctrl+C để dừng)…[/cyan]")
    try:
        report = cl.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Đã dừng tay (Ctrl+C). Kết quả đã lưu DB.[/yellow]")
        return
    console.print(
        f"[green]Vòng kín xong[/green] ({report.stop_reason}): ý tưởng={report.ideas_tried} "
        f"sim={report.sims_used} pass={report.n_passed} bỏ={report.n_abandoned} "
        f"ρ={report.rho_sharpe}"
    )
```

- [ ] **Step 4: Thêm mục menu 7 trong `_print_menu` + `start`**

Trong `_print_menu` (sau dòng " 6) Marathon..."):
```python
    console.print(" 7) Vòng kín AI+MiniBrain (GP→refine→SIM→feedback)")
```
Trong `start`, thêm nhánh (yêu cầu đã đăng nhập như mục 2-6):
```python
            elif choice == "7":
                _menu_closed_loop(state)
```
Thêm helper `_menu_closed_loop(state)` hỏi `--market-data-dir` qua input rồi gọi cùng logic
(gọn: hỏi thư mục panel, dùng state.client/region/universe/delay, dựng như lệnh trên). Nếu
gọn hơn: helper in hướng dẫn "dùng lệnh `python main.py closed-loop --market-data-dir ...`"
và trả — nhưng ƯU TIÊN chạy thật: hỏi đường dẫn panel, gọi build_closed_loop với
state.session_factory/state.client. Giữ nhất quán pattern `_menu_research`.

- [ ] **Step 5: Smoke test — PASS + `--help`**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_cli_closed_loop.py -q
venv/Scripts/python.exe main.py closed-loop --help
```
Expected: 2 PASS; `--help` in options không crash.

- [ ] **Step 6: ruff (chỉ lỗi mới) + kiểm dấu tiếng Việt + commit**

```bash
venv/Scripts/python.exe -m ruff check main.py
git add main.py tests/unit/test_cli_closed_loop.py
git commit -m "feat(cli): lenh closed-loop + muc menu 7 - chay vong kin AI+MiniBrain"
```

---

## Self-review

**Spec coverage (4C scope):**
- [x] QuotaExhausted best-effort (AuthExpired) — Task 1.
- [x] Factory ráp vòng kín — Task 2.
- [x] CLI `closed-loop` + mục menu 7 + đăng nhập + panel + .env AI (qua _make_research_loop) — Task 3.
- [~] Phát hiện quota-ngày CHÍNH XÁC — best-effort; chốt sau lần chạy thật (ghi rõ trong report).
- [~] Feedback (a) Brain pool tầng-2 + (d) AI prompt novelty — Phase 5 (sau khi loop chạy thật).

**Placeholder scan:** `_menu_closed_loop` (Task 3 Step 4) mô tả hành vi + tham chiếu pattern
`_menu_research` — implementer viết theo pattern menu thật (đọc `_menu_research`); mọi phần
khác có code cụ thể. Chạy thật cần login (không test tự động được — smoke chỉ kiểm parse/lỗi).

**Type consistency:**
- `build_closed_loop(*, data, repo, config, registry, loop, region, universe, pop_size,
  n_generations, base_seed, top_k, max_corr, calibrate_every, rho_bar, max_ideas) -> ClosedLoop`
  — khớp Task 2 def + lời gọi CLI Task 3.
- `RefinementLoopRefiner.refine_and_sim` ném `QuotaExhausted` (Task 1) — ClosedLoop.run bắt
  (Phase 2) → dừng 'quota'.
- `_make_research_loop(...marathon=True) -> (loop, deepseek)`; `loop.market_data`/
  `loop.max_simulations` set được — khớp loop.py Phase pre-condition.

**Risks / gotchas:**
1. **Quota chính xác:** AuthExpiredError là proxy (session chết) — có thể KHÔNG trùng hết-quota-ngày.
   Chạy thật lần đầu để bắt đúng response quota rồi tinh chỉnh (đã hẹn với user).
2. `max_simulations=10**9`: bỏ trần local để dừng theo quota Brain; RRateLimiter vẫn ghì concurrency.
3. `_menu_closed_loop` chạy blocking đến hết quota — Ctrl+C bắt ở lệnh CLI; trong menu nên
   bọc tương tự (`_menu_research` có pattern hủy).
4. Chạy thật đốt quota + gọi AI thật — smoke test KHÔNG chạm; người dùng chạy có ý thức.
