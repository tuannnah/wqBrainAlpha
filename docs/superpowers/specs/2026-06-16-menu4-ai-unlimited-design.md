# Menu 4/ai chạy không giới hạn — chỉ dừng khi hết token

## Bối cảnh

Hiện tại khi người dùng vào menu `start` chọn `4) Chạy toàn trình auto` engine `ai`, pipeline dừng sau khoảng 5 ý tưởng vì các điều kiện dừng đan vào nhau:

- `target_passes=3` (đủ K alpha pass thì dừng) — `main.py:1042`
- `max_total_sims=60` (trần cứng tổng số sim) — `main.py:1042`
- `max_directions=0` (không giới hạn batch nhưng `UNLIMITED_DIRECTION_BATCH_SIZE=5` cộng với phép chia `remaining // dirs_left` ăn hết trần sim trong batch đầu) — `src/pipeline/auto.py:13`, `main.py:986-987`

Người dùng muốn: chạy đến khi LLM hết token (DeepSeek 402, claude-cli exit ≠ 0, codex hết quota) hoặc Ctrl+C.

## Phạm vi

**Chỉ menu `4/ai`**. Lệnh CLI `python main.py auto` và menu `5) Thử luồng` giữ nguyên hành vi cũ.

## Thiết kế

### 1. `src/pipeline/auto.py` — `AutoPipeline.swallow_errors`

Thêm trường `swallow_errors: bool = False` vào `AutoPipeline`. Khi `True`:

- Bọc toàn bộ thân `run()` trong try/except.
- `KeyboardInterrupt` → `stop_reason = "ctrl_c"`, phát event `stop`, trả `AutoResult` partial.
- `Exception` (bao gồm mọi lỗi LLM bubble lên) → `stop_reason = f"lỗi: {type(exc).__name__}: {str(exc)[:120]}"`, phát event `stop`, trả `AutoResult` partial.

Default `False` → tests cũ và lệnh CLI `auto` không đổi hành vi.

### 2. `main.py:_run_auto` — thêm 2 tham số

```python
def _run_auto(
    engine, region, universe, delay,
    target_passes=3, max_sims=60, max_directions=0,
    existing_client=None,
    per_direction_sims: int | None = None,  # MỚI
    swallow_errors: bool = False,           # MỚI
):
```

Hành vi:

- Nếu `per_direction_sims` được set: `per_direction_box["per_direction"]` luôn = `per_direction_sims`, KHÔNG còn chia `remaining // dirs_left`.
- `swallow_errors` được truyền xuống `AutoPipeline`.

### 3. `main.py` nhánh menu `"4"`

```python
elif choice == "4":
    engine = _menu_ask_engine()
    if engine == "ai":
        _run_auto(
            engine, state.region, state.universe, state.delay,
            target_passes=10**9,
            max_sims=10**18,
            max_directions=0,
            per_direction_sims=30,
            swallow_errors=True,
            existing_client=state.client,
        )
    else:
        _run_auto(
            engine, state.region, state.universe, state.delay,
            existing_client=state.client,
        )
```

GA giữ nguyên (không có hướng động, không cần unlimited).

### 4. Per-direction guard giữ nguyên

`RefinementLoop.max_simulations=30` + `no_improve_patience=3`: khi hướng cạn ý hoặc greedy stuck, hướng kết thúc và pipeline xin batch hướng mới. Không sửa `RefinementLoop`.

## Tests

### `tests/test_auto_pipeline.py` (bổ sung)

1. `test_swallow_errors_bat_keyboardinterrupt`: `run_direction` raise `KeyboardInterrupt` → `swallow_errors=True` ⇒ trả `AutoResult` với `stop_reason == "ctrl_c"`, các pass đã có vẫn còn.
2. `test_swallow_errors_bat_exception`: `run_direction` raise `RuntimeError("hết tiền")` → `swallow_errors=True` ⇒ `stop_reason` chứa `"lỗi"` và `"RuntimeError"`.
3. `test_swallow_errors_false_van_raise`: regression — `swallow_errors=False` (default) thì exception vẫn bubble lên (giữ tương thích với test cũ `test_prepare_loi_thi_dung_sach`).

### `tests/test_auto_command.py` (bổ sung)

4. `test_run_auto_per_direction_sims_co_dinh`: `_run_auto(..., per_direction_sims=30, max_sims=10**18)` ⇒ giá trị `per_direction` truyền vào `run_direction` luôn = 30 (capture qua fake `AutoPipeline`).
5. `test_run_auto_swallow_errors_truyen_xuong_pipeline`: `_run_auto(..., swallow_errors=True)` ⇒ `AutoPipeline` được khởi tạo với `swallow_errors=True`.

## Không làm

- KHÔNG sửa `RefinementLoop` — per-direction guard giữ nguyên.
- KHÔNG đổi default lệnh CLI `auto`.
- KHÔNG đổi menu `5`.
- KHÔNG thêm flag CLI mới.
