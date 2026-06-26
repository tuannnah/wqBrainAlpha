# Phase 8 — Short-list + CLI Design

> Spec brainstorm 2026-06-26. Cập nhật thiết kế Phase 8 cho khớp thực tế codebase SAU khi
> Phase 7.7-7.9 (GPEngine + adapter + CLI `generate`) đã merge. Thay thế các giả định lỗi
> thời trong plan gốc `docs/superpowers/plans/2026-06-24-phase-8-cli.md` (viết trước khi có
> GPEngine thật). Plan thực thi sẽ được dựng từ spec này.

## Goal

Dựng tầng orchestration cuối của MiniBrain: chấm một expression (`score_one`) và sinh nhiều
ứng viên rồi rút **short-list xếp hạng + decorrelate pool-aware** (`generate_many`), cộng
2 lệnh CLI (`score-one`, nâng cấp `generate`) để chạy toàn bộ pipeline local từ PowerShell
**không tốn quota sim Brain**.

## Quyết định đã chốt (brainstorm)

1. **Nguồn short-list:** re-score qua `score_one` (một nguồn `AlphaMetrics` duy nhất cho cả
   pipeline), KHÔNG đọc lại metrics đã persist từ DB. `generate_many` drive GP rồi re-score.
2. **Lệnh `generate`:** nâng cấp tại chỗ (lệnh đã tồn tại ở Phase 7.8) để dùng
   `generate_many` + in short-list — KHÔNG tạo lệnh trùng tên.
3. **`score-one`:** nạp pool PnL từ DB (gate self-correlation có nghĩa thật), KHÔNG persist
   kết quả (kiểm tra ad-hoc).
4. **Config CLI:** expose flags chính `--neutralization --decay --truncation --delay` (có
   default hợp lý) cho cả `score-one` và `generate` — đúng tinh thần stage separation.

## Components

### `src/pipeline/__init__.py`
Package marker (rỗng), tạo nếu chưa có.

### `src/pipeline/shortlist.py` (thuần, không I/O)
```python
@dataclass(frozen=True, slots=True)
class ShortlistCandidate:
    expr: str
    metrics: AlphaMetrics
    pnl: npt.NDArray[np.float64]
    dates: Dates

def build_shortlist(
    candidates: list[ShortlistCandidate],
    top_k: int,
    max_corr: float,
    pool_corr: PoolCorrelation | None = None,
) -> list[ShortlistCandidate]: ...
```
Rank theo `metrics.fitness` giảm dần → quét tuần tự: giữ candidate nếu `|ρ|` PnL `< max_corr`
với **mọi candidate đã giữ** VÀ với **pool** (qua `pool_corr.max_corr`, nếu có). Dừng khi đủ
`top_k` hoặc hết. Không đột biến input. Helper `_pairwise_abs_rho` tính Pearson `|ρ|` trên
giao ngày chung; trả `None` (không bịa `ρ=0`) khi thiếu điểm/phương sai bằng 0 — đúng logic
`PoolCorrelation._pairwise_rho` Phase 6. Đây là hiện thực B9 (PnL self-corr là nguyên nhân
reject hàng đầu, không phải AST-hash dedup).

### `src/pipeline/runner.py` (network-agnostic, dependency injected)
```python
def score_one(
    expr: str, cfg: PortfolioConfig, data: MarketData,
    pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] | None = None,
) -> tuple[AlphaMetrics, GateVerdict]: ...

def generate_many(
    gp_engine: _RunsGP, cfg: PortfolioConfig, data: MarketData,
    top_k: int, max_corr: float,
    pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] | None = None,
) -> list[ShortlistCandidate]: ...
```
- `score_one`: parse→eval→portfolio→backtest→metrics→pool-corr→gate. Thuần local, tất định
  với cùng `(expr, cfg, data, pool)`. Lỗi parse/eval → `_EMPTY_METRICS` (tất cả 0.0/{}) +
  `GateVerdict(passed=False, hard_failures=[lý do])`. Có pool → `evaluate_with_pool`; không
  pool → `evaluate(self_corr=0.0)`.
- **Helper nội bộ `_score_one_full(expr, cfg, data, pool) -> _ScoreResult`** trả thêm
  `(pnl, dates)` cạnh `(metrics, verdict)`; `score_one` là wrapper mỏng trên nó. Dùng chung
  bởi `generate_many` để **tránh backtest 2 lần** (plan gốc tự nhận là gotcha — backtest lại
  lần 2 sau `score_one`). Đây là cải tiến có chủ ý so với plan gốc.
- `generate_many`: `result = gp_engine.run()` → `result.final_population`; mỗi `Individual`
  có `fitness is not None`, serialize AST→string (`Serializer`), `_score_one_full`, giữ cái
  `verdict.passed`, gom `ShortlistCandidate(expr, metrics, pnl, dates)`, rồi
  `build_shortlist(top_k, max_corr, pool_corr)`. Cá thể `fitness is None` (chưa eval trong
  GP) bị bỏ qua, không lỗi.
- `_RunsGP` = `Protocol` structural có `run() -> GPRunResultLike` (`.final_population` là
  list các `.expr`/`.fitness`) — để test bằng fake, KHÔNG import cứng `src.gp.engine`.

### `main.py` (lớp mỏng + `rich`)
- **`score-one <expr> --market-data-dir [--db-url] --neutralization --decay --truncation
  --delay`**: `ParquetSource(dir).load(...)` → nạp pool từ DB qua `MiniBrainRepository.
  load_pool()` (nếu `--db-url`) → `score_one` → in `AlphaMetrics` + `GateVerdict`. Thiếu/rỗng
  data-dir → `typer.Exit(code=1)` thông báo rõ (không trả kết quả giả). Expression không
  parse được → exit 0, in `verdict.passed=False` + lý do (CLI không crash).
- **`generate` (nâng cấp)**: thêm `--top-k --max-corr --seed` + config flags. Build
  `GPEngine(data, repo, cfg, registry, pop_size=count, n_generations=..., seed=...)` →
  `generate_many` → in short-list dạng `Table`. GP vẫn persist mọi outcome qua `run()`.
- **`calibrate`:** ĐÃ nối data thật (`main.py:1423` dùng `make_local_scorer` +
  `CalibrationHarness` trên `ParquetSource`). Phase 8 KHÔNG tạo lệnh mới — chỉ ghi chú
  no-op trong PROGRESS.md.

## Data flow

```
score-one:  expr ─parse─eval─build─backtest─metrics─┐
            DB pool ──load_pool──> PoolCorrelation ──┴─> gate ─> (metrics, verdict) ─> rich

generate:   GPEngine.run() ─> final_population ─> [serialize → _score_one_full]* ─>
            [passed candidates] ─> build_shortlist(top_k, max_corr, pool) ─> Table
            (GPEngine.run đồng thời persist mọi outcome + pool PnL vào DB)
```

## Khác biệt so với plan gốc (2026-06-24-phase-8-cli.md)

- GPEngine dùng `run()/GPRunResult`, KHÔNG `evolve(generations)` → `generate_many` bỏ tham
  số `generations` (engine giữ `n_generations` lúc khởi tạo).
- `generate` nâng cấp tại chỗ thay vì tạo mới (Phase 7.8 đã tạo lệnh `generate`).
- Thêm `_score_one_full` chống tính 2 lần (plan gốc để ngỏ như tối ưu tùy chọn → spec này
  đưa vào DoD).
- `load_pool()` trả kèm `dates` (xác nhận `src/storage/repository.py`) → Ambiguity #1 của
  plan gốc HẾT; dựng `PoolCorrelation` trực tiếp, không giả định trục dates chung.
- Task `calibrate` = no-op (đã wired).

## Error handling

- parse/eval lỗi trong `score_one` → verdict fail có lý do, không exception nổi lên CLI.
- signal toàn NaN → verdict fail ("signal toàn NaN").
- market-data-dir thiếu/rỗng → `typer.Exit(1)` thông báo rõ.
- `generate_many` bỏ qua `Individual.fitness is None` an toàn.

## Testing (TDD)

- `tests/unit/test_shortlist.py` (~7): rank fitness, decorrelate nội bộ + pool-aware, top_k,
  empty, không đột biến input.
- `tests/unit/test_runner_score_one.py` (~5): valid expr, parse error → fail verdict, unknown
  field → fail, pool-aware metrics bất biến theo pool, tất định.
- `tests/unit/test_runner_generate_many.py` (~3): fake GPEngine (`.run()`), bỏ
  `fitness=None`, tôn trọng top_k.
- `tests/unit/test_cli_score_one_generate.py` (~4): CliRunner, ghi panel parquet tạm; missing
  dir → exit 1, panel thật → in sharpe + verdict, expr lỗi → exit 0 in fail.
- ruff + mypy `--strict` sạch trên file mới.

## Constraints (kế thừa)

Python 3.12, full type hints, no look-ahead / no survivorship / delay-1 / stage separation,
thresholds chỉ ở `config/thresholds.py`, determinism qua seed inject, WQ operator fidelity.
Dependency rule: `src/pipeline` không import `src.llm`/`src.generation`; chỉ bị `app`/main.py
tiêu thụ.
