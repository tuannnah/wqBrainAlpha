# Thiết kế: Cải thiện tốc độ + chất lượng vòng kín (2026-07-18)

## Bối cảnh & vấn đề

Quan sát log phiên chạy 2026-07-18 (menu 5, vòng kín):

- Mỗi `next_batch()` của `GPIdeaSource` chạy MỘT lần tiến hoá GP đầy đủ (pop 30 × 3 thế hệ
  ≈ 120–210 backtest local, tuần tự) → **3–14 phút/batch**.
- Khi một họ vừa bị đóng (vd `pv_reversal` 0/8), lọc sau-sinh làm lô rỗng →
  `max_empty_retries=8` lần tiến hoá đầy đủ → trường hợp xấu **~24+ phút** cho một batch.
- `gp_budget` (trần 3 sim Brain/phiên cho origin "gp") đã cạn nhưng GP **vẫn chạy đầy đủ**,
  ứng viên sinh ra chắc chắn bị vứt ở gate `gp_budget` — đốt CPU vô ích.
- `check_meaningful` chỉ lọc **SAU** tiến hoá (trong `generate_many`) → GP tốn backtest và
  suất NSGA-II cho biểu thức degenerate (volume-only) suốt 3 thế hệ.
- Tham số `n_jobs` của `GPEngine` **không được dùng** — đánh giá tuần tự 100%.
- Nguồn curated/alt-data bão hoà, Combiner liên tục `0 combo` → vòng kín gần trạng thái cạn
  ý tưởng nhưng không có cơ chế thoát/làm mới.

## Nguyên tắc bất di bất dịch

1. **Tốc độ không đánh đổi chất lượng**: mọi tối ưu phải *bất biến kết quả* (cùng input →
   cùng kết quả tìm kiếm) hoặc *tăng* chất lượng. Không cắt xén tính toán, không nới gate.
2. TDD từng task, mỗi task 1 commit, code/log tiếng Việt.
3. Mỗi task đủ nhỏ để một subagent thực hiện độc lập; thứ tự có phụ thuộc (A2 trước A4).

## Pha A — Tốc độ, chất lượng không đổi hoặc tăng

### A1. Tắt tiến hoá GP khi `gp_budget` cạn

- `ClosedLoop.run()` đã đếm `gp_sims_used` và chặn candidate origin `"gp"` khi
  `gp_sims_used >= max_gp_sims` (closed_loop.py:382). Khi điều kiện này chạm lần đầu, gọi
  callback mới `on_gp_budget_exhausted` (mirror pattern `on_family_closed`,
  closed_loop_adapters.py:1242).
- Wiring trong `closed_loop_adapters`: callback gọi `set_gp_budget_exhausted(True)` (chữ
  ký nhận bool — epoch mới ở B1 sẽ gọi lại với `False`) ủy quyền
  xuống chuỗi wrapper (Curated → AltData → NearMiss → Combiner → `GPIdeaSource`) — đúng
  pattern `set_saturated_families` hiện có.
- `GPIdeaSource.next_batch()` khi cờ bật: **bỏ hẳn** vòng tiến hoá, trả `[]` ngay. Các
  wrapper khác (curated/alt-data/near-miss/combiner) vẫn phục vụ bình thường.
- Chất lượng: không đổi — kết quả GP trước giờ vẫn bị vứt ở gate `gp_budget`.
- Lưu ý phụ: mất phần persist evaluation của các cá thể GP lẽ ra được tiến hoá thêm
  (nguồn dữ liệu near-miss). Chấp nhận: giá trị thấp so với chi phí; near-miss đã có kho
  evaluation tích luỹ.

### A2. Lọc meaningfulness + họ-đã-đóng TRONG GP, trước backtest

- Trong `GPEngine._evaluate_population` (gp/engine.py:223): trước khi backtest một cá thể,
  chạy `check_meaningful(ind.expr)`; nếu fail → status `"failed_gate"` mới dạng
  `"degenerate"` (không backtest, fitness None, vẫn persist lý do để avoid-list học).
- `GPEngine` nhận thêm tham số `saturated_families: set[str]` (mặc định rỗng);
  `GPIdeaSource._run_one_batch` truyền `self._saturated` vào. Cá thể thuộc họ đóng → bỏ
  như trên, không backtest.
- Lọc sau-sinh trong `generate_many` + `GPIdeaSource.next_batch` GIỮ NGUYÊN (defense in
  depth, rẻ).
- Chất lượng: **tăng** — quần thể NSGA-II không còn bị chiếm suất bởi cá thể vô nghĩa/họ
  chết; suất đó dành cho cá thể có cơ hội thật.

### A3. Cache backtest xuyên batch theo `canonical_hash`

- Cache in-memory cấp phiên (sống trong `GPIdeaSource`, truyền vào mỗi `GPEngine`):
  `canonical_hash → (daily_pnl, metrics)` — kết quả của phần THUẦN
  (eval signal → build danh mục → backtest → metrics), là hàm xác định của
  (expr, config, data) bất biến trong phiên.
- Phần phụ thuộc trạng thái (pool_corr, gate, fitness vector) **tính lại tươi mỗi lần** từ
  pnl đã cache — vì pool lớn dần trong phiên.
- Chất lượng: không đổi tuyệt đối (cache hit trả đúng kết quả backtest đã tính).
- Giới hạn bộ nhớ: pnl ~vài nghìn float64/expr; cap kích thước cache (vd 5 000 entry, LRU
  hoặc clear đơn giản khi vượt) để phiên dài không phình RAM.

### A4. Giảm `max_empty_retries` 8 → 2

- An toàn NHỜ A2: họ đóng đã bị lọc từ trong tiến hoá nên "lô rỗng" giờ nghĩa là cạn thật
  (không còn là xui vì lọc sau-sinh vứt sạch). 2 lần thử seed khác là đủ chống nhiễu.
- Lô rỗng → trả `[]` để `ClosedLoop` quyết định theo Pha B.
- Phụ thuộc: **làm SAU A2**.

## Pha B — Chất lượng + chống bão hoà

### B1. Reseed epoch tự động (chạy tới hết quota)

- Khi `idea_source.next_batch()` trả rỗng (hiện tại → `no_more_ideas`, dừng): thay vì dừng
  ngay, `ClosedLoop` mở **epoch mới** nếu còn quota Brain và chưa chạm `max_ideas`:
  - `base_seed` mới (offset lớn, vd +10 000/epoch) — quần thể GP khác hẳn.
  - **Xoay tập field**: epoch mới ưu tiên nhóm dataset khác (đặc biệt dataset ít người
    dùng — hướng originality). Cơ chế: `GPIdeaSource` nhận danh sách "field ưu tiên epoch
    này"; `init_population` seeding nghiêng về nhóm đó.
  - Reset `gp_sims_used` (ngân sách GP mới cho epoch) + gọi lại
    `set_gp_budget_exhausted(False)`.
  - **GIỮ NGUYÊN**: họ đã đóng (`closed_families`), avoid-list, dedup phiên (`seen`).
- Số epoch không giới hạn cứng — vòng dừng khi: hết quota Brain, chạm `max_ideas`, hoặc
  một epoch trọn vẹn không sinh nổi MỘT batch nào (cạn tuyệt đối).
- Log rõ: `🔄 Epoch #k: reseed (seed=…, nhóm dataset=…)`.

### B2. Đa dạng hoá field khi init quần thể

- `init_population` hiện lấy `fields` phẳng → GP hội tụ về field quen (volume, close-open).
- Thêm quota theo dataset/nhóm field khi sinh cây ngẫu nhiên: không nhóm nào chiếm quá X%
  (vd 40%) số cá thể khởi tạo của một quần thể.
- Chất lượng: tăng đa dạng khám phá; không đổi về tính đúng đắn (gate/backtest giữ nguyên).

## Pha C — Song song hoá backtest local

### C1. `ProcessPoolExecutor` cho phần thuần trong `_evaluate_population`

- Worker chỉ chạy phần thuần: eval AST → build danh mục → backtest → metrics (không chạm
  SQLite, không pool_corr). Kết quả trả về main process.
- Main process: nhận kết quả **theo thứ tự cá thể ban đầu** (submit giữ index), rồi tính
  pool_corr/gate/fitness + persist SQLite **tuần tự** — giữ determinism và an toàn SQLite
  trên Windows (spawn).
- `n_jobs` thành tham số thật (mặc định `min(4, cpu_count)`); `n_jobs=1` đi đường tuần tự
  cũ nguyên vẹn (fallback + so sánh).
- Dữ liệu `MarketData` lớn: khởi tạo worker một lần/pool (initializer nạp data), không
  pickle lại mỗi task.
- Tương tác với A3: tra cache TRƯỚC khi submit; chỉ miss mới vào pool.

### C2. Test tái lập song song ≡ tuần tự

- Test: cùng seed + cùng data → quần thể cuối (tập canonical_hash + fitness) GIỐNG HỆT giữa
  `n_jobs=1` và `n_jobs>1`. Đây là chốt chặn "không giảm chất lượng" của Pha C.

## Thực thi & kiểm chứng

- Mỗi task một subagent (TDD; file tiếng Việt → model sonnet trở lên), tôi review + chạy
  test sau mỗi task, mỗi task 1 commit. Thứ tự: A1 → A2 → A3 → A4 → B1 → B2 → C1 → C2.
- Đo trước/sau bằng `gen_batch_ms` sẵn có trong log funnel; kỳ vọng Pha A giảm ≥50% thời
  gian batch trong kịch bản "họ đóng + budget cạn", Pha C thêm ~n_jobs lần cho phần eval.
- Toàn bộ test suite hiện có (≈1 200 test) phải xanh sau mỗi task.

## Ngoài phạm vi

- Không đổi ngưỡng gate local/Brain, không đổi logic submit/Power Pool.
- Không sửa mojibake console log (cosmetic, việc khác).
- Không parallel hoá sim Brain (đã async theo quota riêng).
