# Xoay vòng seed family/novel theo batch (round-robin) — thiết kế

> Spec brainstorm 2026-07-04. Phát hiện khi điều tra vì sao Auto SIM (mục 5) chạy thật chỉ thử
> được 1-3 ý tưởng rồi dừng "no_more_ideas" — nguyên nhân THẬT của lần chạy cụ thể đó là cache
> data field thiếu `close/volume/...` (đã fix riêng, xem commit `553ae8e`/`cb1e6c5`), nhưng trong
> lúc điều tra phát hiện thêm một vấn đề kiến trúc thật, độc lập, đáng sửa: seed sinh từ
> `src/generation/families.py` + `src/generation/novel_ideas.py` (150 seed sau khi sửa
> `ts_min/ts_max/ts_sum/ts_std_dev`, commit `12f78d8`/`0e1c0a3`) chỉ có **30/150 (đúng
> `pop_size`) từng được dùng, mãi mãi, bất kể chạy bao nhiêu batch**.
>
> **Đính chính (2026-07-04, lúc review cuối triển khai):** con số "150" ở trên là ước lượng lúc
> brainstorm; đếm lại thật bằng `all_seed_cores(with_llm=False)` sau khi code đã merge cho ra
> **160 seed** (families + novel_ideas). Không ảnh hưởng thiết kế — `_rotating_slice` không phụ
> thuộc số lượng cụ thể (`offset % len(items)`), đã xác minh round-robin phủ đúng 160/160 qua 7
> batch. Các con số "150"/ví dụ "batch 4" bên dưới giữ nguyên như lúc brainstorm để khỏi viết lại
> toàn bộ ví dụ; hiểu ngầm định là ước lượng, không phải giá trị cứng.

## Vấn đề

`GPIdeaSource.next_batch()` (`src/app/closed_loop_adapters.py:75-89`) tạo một `GPEngine` MỚI mỗi
lần gọi. `GPEngine.run()` (`src/gp/engine.py:295`) gọi `all_seed_cores(with_llm=...)` — trả về
TOÀN BỘ seed family+novel (150 cái, deterministic, không random) — MỚI mỗi lần, rồi gọi
`init_population(seed_cores=seed_cores, population_size=pop_size, ...)`.

`init_population()` (`src/gp/init.py:108-129`): nếu `len(valid_seeds) >= population_size`, chỉ
lấy `valid_seeds[:population_size]` — 30 seed ĐẦU TIÊN theo thứ tự cố định của
`all_seed_cores()`. Vì danh sách seed là deterministic (không phụ thuộc RNG/seed truyền vào), **kết
quả lát cắt `[:30]` luôn là đúng 30 seed y hệt ở MỌI lần gọi `GPEngine`, MỌI batch, suốt cả
phiên** — 120 seed còn lại (80%) không bao giờ được đưa vào quần thể GP để tiến hóa trực tiếp.

Đây là lãng phí thật: công sức sửa seed family (đưa từ ~0 seed dùng được lên 150) không phát huy
hết tác dụng, vì hạ tầng phía dưới vẫn chỉ khai thác một góc cố định của danh sách.

## Mục tiêu

Qua nhiều batch, toàn bộ seed hợp lệ phải lần lượt được dùng làm thành viên quần thể ban đầu —
không phụ thuộc may rủi RNG, có thể dự đoán trước (test được chính xác batch N dùng seed nào).

## Kiến trúc

Thêm tham số `seed_offset: int = 0` xuyên 3 lớp, mặc định `0` để KHÔNG đổi hành vi ở mọi call site
khác đang dùng các hàm này (CLI `generate`, `local_engine_test.py`, test hiện có):

```
GPIdeaSource.next_batch()                       (đã có sẵn bộ đếm self._batch, tăng mỗi lần gọi)
    seed_offset = self._batch * self.pop_size    (tính TRƯỚC khi tăng self._batch, khớp cách
                                                   `seed = base_seed + self._batch` đang làm)
    → GPEngine(..., seed_offset=seed_offset)
        → run() → init_population(..., seed_offset=self.seed_offset)
            → _rotating_slice(valid_seeds, seed_offset, population_size)
              thay vì luôn valid_seeds[:population_size]
```

Ví dụ 150 seed hợp lệ, `pop_size=30`: batch 0 → seed[0:30]; batch 1 → seed[30:60]; ...; batch 4 →
seed[120:150]; batch 5 → offset=150, `150 % 150 = 0` → quay lại seed[0:30]. Sau đúng 5 batch, toàn
bộ 150 seed đã được dùng làm quần thể ban đầu ít nhất 1 lần.

## Thành phần thay đổi

1. **`src/gp/init.py`**
   - Hàm mới `_rotating_slice(items: list[Node], offset: int, count: int) -> list[Node]`:
     - `items` rỗng → trả `[]` ngay (tránh chia cho 0 ở modulo).
     - `start = offset % len(items)`; nếu `start + count <= len(items)` → `items[start:start+count]`;
       ngược lại nối `items[start:] + items[:(start+count) - len(items)]` (wrap-around).
   - `init_population(...)` thêm tham số `seed_offset: int = 0`; nhánh
     `len(valid_seeds) >= population_size` đổi từ `valid_seeds[:population_size]` sang
     `_rotating_slice(valid_seeds, seed_offset, population_size)`. Nhánh "thiếu seed, lấp bằng GP
     ngẫu nhiên" (khi `len(valid_seeds) < population_size`) giữ nguyên — không liên quan tới vòng
     xoay (đằng nào cũng dùng hết toàn bộ seed hợp lệ rồi).

2. **`src/gp/engine.py`**
   - `GPEngine.__init__` thêm `seed_offset: int = 0`, lưu `self.seed_offset`.
   - `GPEngine.run()`: truyền `seed_offset=self.seed_offset` vào lời gọi `init_population(...)`.

3. **`src/app/closed_loop_adapters.py`**
   - `GPIdeaSource.next_batch()`: tính `seed_offset = self._batch * self.pop_size` (dùng
     `self._batch` TRƯỚC khi `+= 1`, cùng chỗ với dòng `seed = self.base_seed + self._batch`
     hiện có), truyền `seed_offset=seed_offset` vào `GPEngine(...)`.

## Xử lý lỗi / trường hợp biên

- `valid_seeds` rỗng: `_rotating_slice` trả `[]`, `init_population` rơi vào nhánh filler như cũ
  (không đổi).
- `seed_offset` là bội số của `len(valid_seeds)` (vd sau khi xoay hết 1 vòng): `offset % n == 0`,
  quay lại y hệt hành vi mặc định (`[:population_size]`) — đúng ý đồ "quay vòng".
- `seed_offset` rất lớn (chạy hàng trăm batch): modulo tự đưa về đúng phạm vi `[0, n)`, không cần
  giới hạn/tràn số gì thêm (Python int không tràn).
- Danh sách seed thay đổi giữa các lần gọi (vd bật `with_llm=True` khiến seed LLM biến thiên): nằm
  ngoài phạm vi spec này — không phải regression do thay đổi này gây ra, hành vi round-robin vẫn
  đúng với PHẦN deterministic (family+novel), phần LLM (nếu có) chỉ tình cờ rơi vào đâu thì rơi,
  y hệt rủi ro đã tồn tại từ trước.

## Kiểm thử (TDD — viết trước khi sửa code)

- **`tests/unit/test_gp_init.py`** (hoặc file test tương đương của `init.py`):
  - `_rotating_slice`: offset=0 → giữ nguyên `items[:count]`; offset giữa danh sách (không tràn)
    → lát cắt liền mạch đúng vị trí; offset khiến tràn cuối danh sách → nối đúng 2 đầu
    (`items[start:] + items[:phần dư]`); offset là bội số `len(items)` → y hệt offset=0; `items`
    rỗng → trả `[]` không lỗi.
  - `init_population(seed_offset=N)`: với seed_cores giả lập (vd 10 seed đơn giản), N khác nhau
    cho ra tập seed khác nhau trong quần thể trả về; N=0 giữ nguyên hành vi hiện có (regression
    test so với behaviour cũ).
- **`tests/unit/test_gp_engine.py`**: `GPEngine(seed_offset=N).run()` phải truyền đúng `N` xuống
  `init_population` — monkeypatch/spy `src.gp.engine.init_population` để assert tham số nhận
  được, tránh phải dựng toàn bộ pipeline backtest thật cho test này.
- **`tests/unit/test_closed_loop_adapters.py`**: gọi `GPIdeaSource.next_batch()` liên tiếp nhiều
  lần (fake `GPEngine`/`generate_many` như test hiện có trong file), assert `seed_offset` truyền
  cho từng lần dựng `GPEngine` tăng đúng `0, pop_size, 2*pop_size, ...`.

## Phạm vi KHÔNG làm trong spec này

- Không đụng "vấn đề A" (gate kép `src/backtest/gate.py` vs `gates.py`) — sau khi soát lại, khi
  pool rỗng hai gate gọi chung một hàm `GateEvaluator.evaluate()` với cùng input nên không thể
  lệch nhau; khi pool có alpha, `RefinementLoop` bỏ qua self-corr (hard-code `0.0`) trong khi
  `generate_many` xét pool-aware — đây là lỗ hổng nhẹ (dư thừa code, không phải nguyên nhân "loại
  oan" ứng viên đã qua gate 1), để lại làm sau nếu cần, không phải một phần của round-robin.
- Không đổi `pop_size`/`n_generations` mặc định, không thêm cấu hình CLI mới cho tính năng này —
  round-robin là hành vi tự động, không cần bật/tắt.
