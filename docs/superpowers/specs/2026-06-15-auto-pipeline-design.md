# Thiết kế: Lệnh toàn trình `auto` (gộp luồng AI + GA)

Ngày: 2026-06-15
Trạng thái: đã duyệt thiết kế, chờ viết plan

## Mục tiêu

Người dùng chỉ cần **một lệnh** để chạy hết toàn trình tìm alpha, thay vì
nhớ bấm 1→2→3→7 trong menu wizard. Bỏ menu 9 lựa chọn rối; giữ nguyên engine
và các lệnh CLI phụ (chúng là bộ máy bên dưới mà lệnh toàn trình gọi lại).

Toàn trình:
```
đăng nhập → ensure fields/operators (cache nếu có) → tìm alpha
→ mô phỏng trên WQ → cải thiện nếu được → log đầy đủ → DỪNG (KHÔNG nộp)
```

Kết quả log ra để người dùng tự kiểm; nộp là bước riêng (`submit`) khi muốn.

## Quyết định đã chốt (brainstorming)

- **1 lệnh tự động toàn trình.** Hai engine (AI số 7, GA số 6) giữ riêng, chỉ
  thêm một entry point điều phối.
- **Chọn engine:** `--engine ai` (mặc định) hoặc `--engine ga`.
- **Engine AI tự brainstorm nhiều hướng** nghiên cứu (không bắt người dùng nhập
  `--direction`).
- **Điều kiện dừng:** đủ **K alpha đạt ngưỡng** HOẶC **chạm trần sim** (chặn
  chạy mãi / tránh HTTP 429).
- **Không nộp** trong lệnh này.
- **Bỏ menu wizard rối;** giữ engine + lệnh CLI phụ (login, fetch-fields...).

## Kiến trúc (Hướng A — orchestrator mỏng, tái dùng code sẵn có)

```
main.py  →  lệnh `auto`  (mỏng: dựng client + engine + gọi orchestrator + in kết quả)
                │
                ▼
src/pipeline/auto.py  →  AutoPipeline  (điều phối thuần, KHÔNG biết httpx/CLI)
                │
                ├── login + ensure fields/operators   (qua callback "prepare")
                ├── sinh nhiều hướng (engine AI)        (qua callback "propose_directions")
                └── vòng: chạy 1 hướng → gom alpha pass → kiểm tra điều kiện dừng
```

Nguyên tắc tách bạch:

- **`AutoPipeline`** chỉ nhận các hàm callback đã đóng gói (chuẩn bị dữ liệu,
  sinh hướng, chạy 1 hướng) và lo phần *vòng lặp + điều kiện dừng K-pass/trần-sim
  + thu thập kết quả*. Không import `httpx`, không gọi WQ trực tiếp → test bằng fake.
- **Lệnh `auto`** trong `main.py` lo phần *bẩn*: dựng `WQBrainClient`, gọi
  `authenticate()`, lắp `RefinementLoop`/`GeneticOptimizer` thật, rồi truyền vào
  `AutoPipeline`.

Tái dùng các mảnh sẵn có trong `main.py`: `_make_client`, `_wizard_fields`/
`_wizard_operators` (logic ensure cache), `_make_research_loop`,
`_make_llm_generator` (`generate_ideas`), logic `run_ga`.

## Giao diện `AutoPipeline` (src/pipeline/auto.py)

### Đầu vào

```python
@dataclass
class AutoPipeline:
    prepare: Callable[[], PrepareInfo]
    # đăng nhập + ensure fields/operators. Trả tóm tắt (số field, số operator).
    # Lỗi -> ném exception, pipeline dừng sạch (chưa gọi run_direction lần nào).

    propose_directions: Callable[[int], list[str]]
    # sinh tối đa N hướng nghiên cứu (engine AI). GA: trả [""] (1 hướng rỗng).

    run_direction: Callable[[str], DirectionOutcome]
    # chạy engine 1 lần cho 1 hướng. Trả: danh sách alpha pass + số sim đã dùng.
    # Engine tự tôn trọng trần sim nội bộ của nó.

    target_passes: int = 3      # K — đủ bao nhiêu alpha đạt thì dừng
    max_total_sims: int = 60    # trần cứng tổng số sim toàn pipeline
    max_directions: int = 5     # số hướng tối đa sẽ thử

    on_event: Callable[[AutoEvent], None] | None = None   # để log đầy đủ
```

### Đầu ra

```python
@dataclass
class PassedAlpha:
    expression: str
    sharpe: float | None
    fitness: float | None
    direction: str        # hướng nguồn (rỗng nếu GA)

@dataclass
class DirectionOutcome:
    passed: list[PassedAlpha]
    sims_used: int

@dataclass
class PrepareInfo:
    fields: int
    operators: int

@dataclass
class AutoResult:
    passed_alphas: list[PassedAlpha]   # gom từ mọi hướng
    directions_run: int
    total_sims: int
    stop_reason: str                   # "đủ_K_pass" | "chạm_trần_sim" | "hết_hướng"
```

### Hành vi vòng lặp

```
info = prepare()                # login + cache, emit "prepare"
directions = propose_directions(max_directions)   # emit "directions"
cho mỗi direction (kèm chỉ số i):
    nếu len(passed) >= target_passes      -> dừng ("đủ_K_pass")
    nếu total_sims  >= max_total_sims      -> dừng ("chạm_trần_sim")
    emit "direction_start"
    outcome = run_direction(direction)
    passed += outcome.passed
    total_sims += outcome.sims_used
    emit "direction_done"
hết vòng                                   -> dừng ("hết_hướng")
emit "stop"
```

**Điều kiện dừng kiểm ở ĐẦU mỗi vòng** (trước khi chạy hướng mới). Có thể vượt
trần sim một chút trong lúc một hướng đang chạy, nhưng engine có trần nội bộ
riêng nên không bùng nổ. Đánh đổi: không cắt ngang giữa lúc engine đang chạy 1 hướng.

## Đấu nối trong `main.py`

### Chữ ký lệnh

```python
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
```

### 3 callback được bọc

**`prepare()`** — tái dùng logic wizard:
```
client = _make_client(); client.authenticate()
ensure fields  (FieldRepository.ensure, dùng cache nếu có)
ensure operators (OperatorRepository.ensure)
→ PrepareInfo(fields=N, operators=M)
```

**`propose_directions(n)`**:
- engine **ai**: `_make_llm_generator(...).generate_ideas(n)` → list hướng.
- engine **ga**: trả `[""]`.

**`run_direction(direction)`**:
- engine **ai**: lắp `RefinementLoop` qua `_make_research_loop(...)`, gọi
  `loop.run(direction)`. `best_candidate` đạt (passed) → gom vào `passed`.
  `sims_used = result.sims_used`.
- engine **ga**: lắp `GeneticOptimizer`, `opt.run()`. Alpha pass = cá thể qua
  `hard_filter`. `sims_used = opt.simulations_used`.

**Chia trần sim theo hướng:** mỗi hướng cấp `per_direction = max(1, sims_còn_lại
// hướng_còn_lại)` để hướng đầu không ăn hết trần.

### `start()` đổi thành gọi thẳng `auto`

Bỏ vòng menu 9 lựa chọn. `start()` chỉ còn gọi `auto()` với mặc định (giữ lệnh
`start` cho quen tay).

## Log đầy đủ

`AutoPipeline` phát sự kiện qua `on_event`; lệnh `auto` hứng và in console (rich)
+ ghi file log (`logs/wq_alpha_*.log` qua `_setup_logging`).

```python
@dataclass
class AutoEvent:
    kind: str       # "prepare" | "directions" | "direction_start" |
                    # "direction_done" | "stop"
    message: str
    data: dict
```

| kind | Thời điểm | Console in ra |
|---|---|---|
| `prepare` | sau login + cache | `✓ đăng nhập \| fields=N \| operators=M` |
| `directions` | sau khi sinh hướng | `Sẽ thử K hướng: 1)... 2)...` |
| `direction_start` | trước mỗi hướng | `[Hướng i/total] "<direction>" — trần sim lượt này=P` |
| `direction_done` | sau mỗi hướng | `+X alpha đạt \| sim lượt=Y \| tổng pass=Z/K \| tổng sim=T` |
| `stop` | kết thúc | `Dừng: <lý do> — pass=Z, sim=T, hướng đã chạy=D` |

Cuối cùng in bảng rich các alpha đạt ngưỡng (Expression, Sharpe, Fitness, Hướng
nguồn) và nhắc: đã lưu DB (xem bằng `top`), **chưa nộp** (nộp bằng `submit`).

## Kế hoạch test (TDD)

### `tests/test_auto_pipeline.py` (mới) — fake callback, không mạng

1. **Dừng khi đủ K pass** — mỗi lượt trả 2 pass, `target_passes=3` → chạy 2 hướng,
   `stop_reason="đủ_K_pass"`, không chạy hết max_directions.
2. **Dừng khi chạm trần sim** — 0 pass, mỗi lượt 25 sim, `max_total_sims=60` →
   `stop_reason="chạm_trần_sim"`.
3. **Dừng khi hết hướng** — 2 hướng, 0 pass, trần cao → chạy đúng 2 hướng,
   `stop_reason="hết_hướng"`.
4. **Kiểm điều kiện dừng ở ĐẦU vòng** — `target_passes=1`, hướng đầu 1 pass →
   hướng 2 không được gọi (đếm số lần gọi `run_direction`).
5. **Phát đủ sự kiện** — đúng thứ tự `prepare → directions → direction_start/done…
   → stop`.
6. **`prepare` lỗi thì dừng sạch** — `prepare` ném exception → nổi ra, chưa gọi
   `run_direction` lần nào.

### Đấu nối `main.py` (nhẹ)

7. **GA callback chạy được** — `propose_directions` GA trả `[""]`; wrapper map
   alpha pass đúng. Dùng fake simulator (pattern trong `test_simulator.py`).

### Không test
- Gọi WQ/LLM thật (việc của test engine sẵn có + chạy thực tế).
- Progress bar rich.
- Đấu nối AI sâu trong `main.py` (phụ thuộc LLM; tin tưởng test `RefinementLoop`).
