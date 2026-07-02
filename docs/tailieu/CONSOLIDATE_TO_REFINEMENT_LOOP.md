# SPEC: Hợp nhất engine về RefinementLoop, gỡ HybridEngine

## Mục tiêu
Để lại **một** engine sinh alpha: `RefinementLoop` (`src/llm/loop.py`, CLI `research`).
Gỡ hoàn toàn `HybridEngine` + GA (`src/optimization/`, CLI `auto`/`start`),
nhưng **salvage** phần diversity của B trước khi xóa.

## Nguyên tắc thực thi (BẮT BUỘC)
- **TDD**: với mỗi task, viết/sửa test TRƯỚC, chạy đỏ, rồi mới sửa code cho xanh.
- **Không xóa mù**: không xóa file nào trước khi `grep` hết import/ref tới nó.
- Sau mỗi task: toàn bộ test suite phải xanh, **không còn dangling import**
  (`python -c "import src..."` cho mọi module còn lại phải chạy được; chạy `ruff`/`pyflakes` để bắt unused/undefined).
- Mỗi commit chỉ một task. Không đổi hành vi của Engine A ngoài phần salvage ở Task 2.
- **Không** đổi tham số/operator của signal logic. Đây là refactor cấu trúc, không phải tinh chỉnh alpha.

---

## Task 0 — Khảo sát phụ thuộc (không sửa code)
Mục đích: biết chính xác cái gì tham chiếu tới B trước khi đụng vào.

1. Liệt kê mọi import tới các module sắp gỡ:
   ```
   grep -rn "optimization.hybrid\|optimization\.\|GeneticOptimizer\|SynergyScorer\|HybridEngine" src/ tests/ scripts/ main.py
   grep -rn "NOVEL_ALPHAS\|novel_ideas" src/ tests/ scripts/ main.py
   grep -rn "\"auto\"\|'auto'\|\"start\"\|'start'\|--no-llm-seed" main.py src/cli* scripts/
   ```
2. Xuất ra `tailieu/_consolidation_audit.md`: bảng `symbol | file định nghĩa | nơi được import | giữ/salvage/xóa`.
3. **Dừng và in bảng này ra**. Không sang Task 1 cho tới khi bảng đầy đủ.

**Acceptance**: bảng liệt kê đủ — RefinementLoop path (giữ), GA/Hybrid path (xóa),
và mọi điểm Engine A có vô tình dùng chung gì với B (vd `SynergyScorer`, `ReferenceZoo`, `NOVEL_ALPHAS`).
Lưu ý: `decorrelation/zoo.py` (ReferenceZoo) và `scoring/` được **cả hai** dùng → **GIỮ**.

---

## Task 1 — Salvage diversity của B vào Engine A
B đóng góp 2 thứ chống "sụp về `rank(close)`/crowded family" mà A cần kế thừa:
(a) seed `NOVEL_ALPHAS` từ dataset thay thế; (b) cơ chế bơm diversity định kỳ.

### 1a. Giữ NOVEL_ALPHAS như nguồn seed của A
- `src/generation/novel_ideas.py` (`NOVEL_ALPHAS`) **không thuộc GA** — giữ nguyên file.
- Trong `loop.py`, ở bước khởi tạo direction/seed, trộn `NOVEL_ALPHAS` vào tập seed ban đầu
  (giống vai trò `_seed_pool()` của B nhưng không có GA).
- **Test trước**: `tests/test_loop_seed.py::test_seed_includes_novel_alphas`
  — assert tập seed của một `RefinementLoop` mới chứa ≥1 entry từ `NOVEL_ALPHAS`.

### 1b. Re-seed diversity định kỳ (thay cho "inject mỗi K gen" của GA)
- Thêm tham số `--reseed-every N` (mặc định 0 = tắt) vào CLI `research`.
- Khi bật: cứ mỗi N vòng refine không cải thiện được `blocking_dimension`,
  loop sinh một direction MỚI từ `LLMAlphaGenerator` (lái về dataset chưa dùng trong phiên)
  thay vì tiếp tục refine nhánh đang stuck. Đây là LLM re-seed, **không** tái lập GA.
- **Test trước**: `tests/test_loop_reseed.py::test_reseed_triggers_new_direction`
  — dùng fake generator/simulator, ép `patience` cạn, assert có gọi `generate_ideas` lần mới.

**Acceptance**: 2 test mới xanh; hành vi mặc định (`--reseed-every 0`) **không đổi** so với hiện tại.

---

## Task 2 — Gỡ HybridEngine + GA
Chỉ làm sau khi Task 1 xanh.

1. Xóa file: `src/optimization/hybrid.py` và toàn bộ module GA chỉ-phục-vụ-B
   (`GeneticOptimizer` và helper riêng của nó). **Không** xóa `scoring/`, `decorrelation/zoo.py`,
   `generation/novel_ideas.py`, `generation/families.py`, `local_select.py`, `template.py`
   (pipeline sinh-lọc offline + zoo dùng chung — giữ).
2. Nếu `SynergyScorer` **chỉ** được GA dùng (xác nhận từ bảng Task 0): xóa cùng GA.
   Nếu A cũng dùng: giữ. Theo bảng quyết định.
3. CLI: gỡ subcommand `auto` và `start` cùng cờ `--no-llm-seed`. `research` thành lệnh chính.
   Cập nhật help text/usage.
4. Xóa mọi test chỉ kiểm GA/Hybrid; chuyển test nào còn giá trị (vd test scorer dùng chung) sang chỗ phù hợp.

**Acceptance**:
- `grep -rn "HybridEngine\|GeneticOptimizer" src/ tests/ scripts/ main.py` → rỗng.
- `python main.py research --help` chạy; `python main.py auto` báo lỗi "unknown command".
- Toàn bộ test suite xanh.

---

## Task 3 — Dọn & xác minh cuối
1. `ruff check src/ tests/` (hoặc `pyflakes`) → không undefined/unused do refactor để lại.
2. `python -c "import src.llm.loop"` và import mọi module còn lại trong `src/` → không lỗi.
3. Cập nhật `tailieu/ENGINE_OVERVIEW.md`: xóa cột HybridEngine/Engine B, ghi chú
   "NOVEL_ALPHAS + reseed đã hợp nhất vào RefinementLoop (Task 1)".
4. Chạy full suite lần cuối + một smoke run `research` với fake simulator (không gọi WQ thật).

**Acceptance cuối cùng**:
- Một engine duy nhất, mọi test xanh, không dangling import, diversity của B được bảo toàn trong A,
  hành vi alpha-generation mặc định không đổi ngoài 2 cờ mới (`--reseed-every`).

## Ngoài phạm vi (KHÔNG làm trong spec này)
- Sửa depth-limit bug (spec riêng).
- Đổi `DIMENSION_HINTS`, operator, hay logic chấm điểm self-corr.
- Thay greedy → MCTS (đã có cờ `--mcts`, không động tới).
