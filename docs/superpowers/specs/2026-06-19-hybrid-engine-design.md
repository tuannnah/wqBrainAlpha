# Thiết kế: Hợp nhất Engine ai/ga thành Engine Hybrid

- Ngày: 2026-06-19
- Trạng thái: đã duyệt thiết kế, chờ rà spec
- Mục tiêu: gộp hai engine song song (AI research-loop + Genetic Algorithm) thành **một engine
  hybrid duy nhất** để tập trung phát triển, giảm bề mặt code trùng lặp.

## 1. Bối cảnh

Hiện có hai engine chạy song song qua chung `_run_auto()` + `AutoPipeline`, phân nhánh bằng
`if engine == "ai"/"ga"`:

- **AI engine**: `_auto_run_direction_ai` → `_make_research_loop` (`RefinementLoop`): LLM sinh giả
  thuyết → dịch → refine theo chiều yếu → simulate → khử tương quan zoo. Là mặc định (`Enter=ai`),
  được phát triển gần đây nhất (menu 4 ai-unlimited, LLM backend CLI).
- **GA engine**: `_auto_run_direction_ga` → `GeneticOptimizer` (`src/optimization/evolution.py`):
  tiến hóa cây AST của alpha, seed bằng `TemplateGenerator` (ngẫu nhiên).

Lệnh `run-ga --seed-llm` đã làm một dạng hybrid sơ khai (50% seed LLM + 50% template, rồi GA tiến
hóa), nên hạ tầng cơ bản đã có.

Sự trùng lặp: cả hai đều gọi `Simulator`, đều chấm điểm, đều giữ "best", đều có điểm vào riêng
(`auto --engine`, `run-ga`, prompt `Engine [ai/ga]` trong menu). Việc duy trì hai nhánh làm chậm
phát triển.

## 2. Quyết định đã chốt

| Câu hỏi | Quyết định |
|---|---|
| Hướng hợp nhất | Gộp lai (hybrid): một engine kết hợp LLM seeding + GA tiến hóa |
| Vai trò LLM | **LLM trong vòng lặp**: mỗi K thế hệ, LLM nhìn top alpha rồi bơm seed/biến thể mới |
| Cơ chế dừng | **Vô hạn** đến khi LLM hết token (402) / Ctrl+C (theo triết lý menu 4 ai-unlimited) |
| Bề mặt cũ | **Xóa hết**, chỉ còn hybrid (giữ lệnh `research` độc lập vì còn dùng) |
| Cách hiện thực | **Cách A**: mở rộng `GeneticOptimizer` + orchestrator `HybridEngine` mỏng |
| Nhịp LLM | `inject_every=3` thế hệ, `refine_top=2` cá thể |
| Trần test/CI | Giữ `--max-sims` / `--generations` tùy chọn, mặc định `None` (vô hạn) |

## 3. Kiến trúc

Một engine duy nhất thay cho cặp ai/ga. Luồng:

```
HybridEngine.run()
  ├─ seed ban đầu:  LLMAlphaGenerator.generate_ideas → generate → pool biểu thức → population
  │     (fallback TemplateGenerator nếu LLM thất bại / pool rỗng)
  ├─ vòng tiến hóa (GeneticOptimizer, generations=None → while not should_stop):
  │     ├─ đánh giá (Simulator + ScoreVector, cache theo expression)
  │     ├─ chọn lọc / crossover / mutate  (giữ nguyên toán tử GA hiện có)
  │     ├─ mỗi inject_every thế hệ → inject hook:
  │     │     ├─ lấy refine_top cá thể tốt nhất + metrics của chúng
  │     │     ├─ AlphaRefiner.refine(cá thể, metrics, chiều-yếu-nhất) → mô tả → biểu thức
  │     │     └─ thêm biến thể (qua PreFilter + khử trùng ReferenceZoo) vào quần thể
  │     └─ dừng khi: should_stop() == True
  │           (mặc định: LLM 402 / Ctrl+C; hoặc chạm max_sims/generations nếu được set)
  └─ lưu top alpha vào DB (source="hybrid")
```

"Chiều yếu nhất" của một cá thể = chiều có điểm thấp nhất trong `ScoreVector`
(`sharpe`/`fitness`/`turnover_fit`/`drawdown_fit`), khớp với `DIMENSION_HINTS` của `AlphaRefiner`.

## 4. Thành phần & ranh giới

| Thành phần | Vai trò | Trạng thái |
|---|---|---|
| `src/optimization/hybrid.py` → `HybridEngine` | Orchestrator mới: nối seed / GA / refine / zoo | **Mới** |
| `GeneticOptimizer` (`src/optimization/evolution.py`) | Lõi tìm kiếm tiến hóa | **Sửa nhỏ** |
| `LLMAlphaGenerator` (`src/llm/generator.py`) | Seed ý tưởng/biểu thức | Tái dùng |
| `AlphaRefiner` (`src/llm/refiner.py`) | Refine top theo chiều yếu | Tái dùng |
| `ReferenceZoo` (`src/decorrelation/zoo.py`) | Khử tương quan biến thể bơm vào | Tái dùng |
| `ScoreVector`/scorer (`src/scoring`) | Chấm điểm đa chiều, xác định "chiều yếu" | Tái dùng |
| `TemplateGenerator` (`src/generation/template.py`) | Fallback seed khi LLM thất bại | Tái dùng |
| `AlphaRefiner` cần `AlphaTranslator` | Dịch mô tả → biểu thức | Tái dùng |

### Sửa `GeneticOptimizer`
1. **Hook inject**: thêm callback tùy chọn `inject(scored_population) -> list[Node]` gọi mỗi
   `inject_every` thế hệ; các Node trả về được thêm vào quần thể (thay thế cá thể yếu nhất để giữ
   `population_size`). Khi `inject is None` → hành vi như cũ.
2. **Chạy vô hạn**: khi `generations is None`, vòng `for gen in range(...)` đổi thành
   `while not should_stop()`; thêm callback tùy chọn `should_stop() -> bool`. Khi `generations` là số
   nguyên → hành vi như cũ (giới hạn theo số thế hệ).
3. Cả hai thay đổi giữ tương thích ngược: mặc định `inject=None`, `should_stop=None`,
   `generations=10` như hiện tại nên test GA cũ không đổi hành vi.

### `HybridEngine` (orchestrator mới)
- Tham số: `simulator`, `prefilter`, `fields`, `llm_generator`, `refiner`, `zoo`, `scorer`,
  `template_generator` (fallback), `inject_every=3`, `refine_top=2`, `max_simulations=None`,
  `generations=None`, `simulation_settings`, `rng`.
- `run(on_generation=None, on_simulation=None, on_inject=None) -> list[Node]`:
  - Sinh seed pool ban đầu qua `llm_generator`; nếu rỗng → `template_generator`.
  - Dựng `GeneticOptimizer` với `seed_factory` lấy từ pool, truyền `inject` + `should_stop`.
  - `inject`: lấy `refine_top` cá thể tốt; với mỗi cá thể xác định chiều yếu nhất từ metrics đã
    cache → `refiner.refine(...)` → parse biểu thức → lọc PreFilter + ReferenceZoo (bỏ trùng/đồng
    dạng) → trả Node.
  - **LLM 402 (hết token)**: set cờ nội bộ `_llm_disabled = True` → từ đó bỏ qua phần inject
    (LLM-in-loop tắt), nhưng GA **vẫn tiếp tục** với quần thể hiện có. 402 KHÔNG làm `should_stop`
    trả True.
  - `should_stop`: chỉ trả True khi chạm `max_simulations` (nếu được set). Mặc định `None` → không
    bao giờ tự dừng; vòng lặp chỉ kết thúc do **Ctrl+C** (`KeyboardInterrupt` bắt ở tầng CLI, không
    qua `should_stop`).

## 5. Dọn dẹp bề mặt (đúng "hợp nhất")

**Xóa khỏi `main.py`:**
- `_menu_ask_engine()` và mọi lời gọi.
- Tham số `--engine` của lệnh `auto`; tham số `engine` của `_run_auto()` và nhánh
  `if engine == "ai"/"ga"`.
- Lệnh `run-ga` (toàn bộ hàm `run_ga`).
- `_auto_run_direction_ai`, `_auto_run_direction_ga`.
- `_run_ga_with_progress` (thay bằng progress của hybrid).
- `passed_from_ga` (import từ `src/...`); nếu không còn ai dùng, xóa luôn định nghĩa nguồn.

**Giữ nguyên:**
- Lệnh `research` độc lập + `_make_research_loop` / `RefinementLoop` (vẫn có người dùng).
- `_make_llm_generator`, `_make_invalid_field_recorder`, `_cached_symbols`, `Simulator`, PreFilter.

**Kết quả**: menu mục 4/5 và lệnh `auto` chỉ còn **một đường = hybrid**.

## 6. Tích hợp CLI / menu

- `auto`: bỏ `--engine`; luôn chạy hybrid vô hạn. Giữ `--region/universe/delay`, sim settings
  (`--decay/--truncation/--neutralization`). Thêm `--max-sims` (mặc định 0 = vô hạn) và
  `--generations` (mặc định 0 = vô hạn) cho test/CI.
- Menu mục 4 (toàn trình): bỏ `_menu_ask_engine()`, hỏi sim settings rồi gọi hybrid vô hạn
  (`swallow_errors=True`, dùng `state.client`).
- Menu mục 5 (thử luồng): bỏ `_menu_ask_engine()`, gọi hybrid với trần nhỏ (`max_sims` nhỏ,
  `generations` nhỏ) để chạy nhanh, kết thúc xác định.

## 7. Xử lý lỗi & dừng

- **LLM 402 (hết token)** trong seed hoặc refine → nuốt lỗi (log cảnh báo), tắt phần LLM-in-loop;
  GA tiếp tục chạy bằng quần thể hiện có. Nếu seed ban đầu thất bại hoàn toàn → fallback
  `TemplateGenerator`.
- **Ctrl+C** → dừng êm ở tầng CLI; vẫn lưu top alpha đã có vào DB và in bảng.
- **Sim error / invalid field** → giữ cơ chế `_make_invalid_field_recorder` + PreFilter hiện có;
  cá thể lỗi nhận điểm `NEG_INF` và bị loại như GA hiện tại.
- **Biến thể refine trùng zoo / không qua PreFilter** → bỏ qua, không bơm; không coi là lỗi.

## 8. Kiểm thử (TDD)

- `tests/test_hybrid.py` (mới):
  - Seed dùng LLM giả → quần thể ban đầu lấy từ pool LLM; pool rỗng → fallback template.
  - `inject` hook được gọi đúng mỗi `inject_every` thế hệ với `refine_top` cá thể.
  - Biến thể trùng/đồng dạng zoo bị loại, không vào quần thể.
  - LLM ném 402 ở refine → GA vẫn tiến hóa tiếp (không raise, không dừng).
  - `max_simulations` nhỏ → kết thúc xác định; trả top hiện có.
- `tests/test_evolution.py` (hoặc test GA hiện có): hook `inject` thêm Node đúng cách; chạy
  `generations=None` dừng theo `should_stop`; mặc định (không hook) hành vi không đổi.
- `tests/test_auto_command.py`: xóa test nhánh ai/ga và test gắn `run-ga`; thay bằng test đường
  hybrid duy nhất cho `_run_auto`/`auto`/menu.

## 9. Phạm vi KHÔNG làm (YAGNI)

- Không đổi lệnh `research` độc lập.
- Không thêm cấu hình giao tiếp LLM mới (dùng backend hiện tại qua `_make_deepseek`/`_make_router`).
- Không tinh chỉnh thuật toán GA (toán tử crossover/mutate giữ nguyên).
- Không thêm tham số tinh chỉnh nhịp LLM ra CLI ở vòng này (giữ trong code, mặc định 3/2).
