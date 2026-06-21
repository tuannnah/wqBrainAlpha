# Ground field thật vào ý tưởng — diệt lỗi "Field/hằng không tồn tại"

**Ngày:** 2026-06-21
**Trạng thái:** Đã duyệt thiết kế, chờ review spec → lập plan

## Vấn đề

Log ngày 21 (`logs/wq_alpha_2026-06-21.log`) cho thấy vòng `repair_to_expression` thất
bại lặp lại, **lỗi nhiều nhất là field LLM bịa ra** (`asset_growth_rate_sensitivityfactor`,
`asset_growth_rate`, `call_breakeven_10`). Mỗi lần thử lại 3 lượt rồi bỏ, tốn token/quota
mà không ra biểu thức hợp lệ. (Hai lỗi phụ `Số node > 30`, `Độ sâu > 7` — NGOÀI phạm vi
lần này, để xử lý sau.)

### Cơ chế gốc

1. `HypothesisGenerator.generate(direction)` (và `generate_ideas`) sinh giả thuyết/ý tưởng
   **tự do bằng văn xuôi**, `implementation_spec` nêu "dữ liệu" như "asset growth rate"
   mà KHÔNG ràng vào field ID có thật.
2. `expr_synth._relevant_fields` xếp hạng field theo **trùng token chữ** với text ý tưởng.
   Ý tưởng "asset growth rate" không khớp field alt-data thật → trả về field vô quan theo
   thứ tự cache gốc.
3. LLM bị bảo "diễn đạt ý tưởng này" + "chỉ dùng field liệt kê", nhưng danh sách field
   chẳng phục vụ ý tưởng → nó chọn diễn đạt ý tưởng và **bịa** tên field từ chữ.
4. Vòng repair (`expr_synth.py:201-207`): `suggest_fields` dùng prefix/token của tên bịa,
   không field thật nào khớp → **hint rỗng** → repair prompt không có grounding mới → LLM
   bịa lại y hệt (log lần 3 = lần 1).

Tức là chỉ thị "chỉ dùng field liệt kê" THUA "hãy diễn đạt đúng ý tưởng" khi danh sách
field không phục vụ được ý tưởng.

## Mục tiêu

Chặn từ gốc: **ràng ý tưởng/giả thuyết vào field có thật** ngay tại nguồn, theo cơ chế
**hybrid** — LLM đề xuất field, code validate ⊆ cache + bổ sung bằng truy hồi — rồi ghim
tập field đó xuyên suốt xuống lõi tổng hợp biểu thức làm danh sách field DUY NHẤT.

Phi mục tiêu (lần này): hint cấu trúc cho lỗi `Số node`/`Độ sâu`; embedding/semantic
search; đụng `pre_filter` (validation đã đúng).

## Kiến trúc & luồng dữ liệu

Thêm khái niệm **"palette field đã ground"** — tập field ID có thật, liên quan — sinh ở
nguồn và chảy xuyên suốt làm danh sách field DUY NHẤT cho prompt.

### Path translator (vòng chính — `RefinementLoop`)

```
research_direction
   │
   ▼
[1] retrieve_field_palette(field_repo, scope, direction)   ← MỚI: top-K field THẬT
   │      (lexical + theme, đảm bảo không rỗng → [field obj])
   ▼
HypothesisGenerator.generate(direction, palette)   ← SỬA: prompt kèm palette,
   │      LLM trả thêm "fields":[...]                       yêu cầu dùng ID có thật
   ▼
[2] ground_fields(llm_fields, palette, cache_ids)   ← MỚI: validate ⊆ cache, bỏ bịa,
   │      → Hypothesis.fields (tuple, ≥ min_k)            bổ sung từ palette nếu thiếu
   ▼
AlphaTranslator.translate(hypothesis)
   │  _describe → description (văn xuôi)
   ▼
build_symbol_context(..., pinned=hypothesis.fields)   ← SỬA: pinned → FIELDS list
   │      = đúng palette ghim + instruction "CHỈ field này, KHÔNG bịa"
   ▼
repair_to_expression(..., pinned=hypothesis.fields)   ← SỬA: lỗi field → tái-tiêm
          palette + suggest_fields đảm bảo không rỗng
```

### Path generator (`generate_ideas` / `generate`)

Đối xứng: `generate_ideas` nâng để LLM trả `(idea, fields)`, ground tương tự, rồi
`generate(idea, fields)` → `_generate_one` truyền `pinned` xuống cùng lõi.

### Điểm hội tụ & tương thích

- Cả hai path đổ vào cùng 2 hàm lõi: `build_symbol_context` (thêm `pinned`) và
  `repair_to_expression` (thêm `pinned`). Không nhân đôi logic.
- `pinned` rỗng/None → hành vi y như cũ (xếp hạng từ-vựng). Mọi test cũ + call-site chưa
  nâng cấp vẫn chạy.

## Component

### C1. `retrieve_field_palette(field_repo, scope, text, k=20, min_k=8)` — `expr_synth.py`

Tái dùng logic `_relevant_fields` nhưng (a) trả **đối tượng field** (id, type, dataset,
desc); (b) **đảm bảo không rỗng** — nếu xếp hạng từ-vựng cho < `min_k` field điểm > 0,
độn thêm field theo theme alt-data (qua `ALT_DATA_KEYWORDS` của generator hoặc tập keyword
tương đương) rồi đến thứ tự cache. Nền chung cho cả grounding lẫn augment.

### C2. Ground hypothesis — `hypothesis.py`

- `Hypothesis` thêm `fields: tuple[str, ...] = ()`; `to_dict`/`from_dict` xử lý khoá
  `fields` (đọc list, ép tuple, bỏ phần tử không phải chuỗi).
- `HypothesisGenerator.generate(direction, palette=None)`:
  - Prompt kèm palette: liệt kê `id: desc[:60]` (≤20 dòng) + câu *"implementation_spec
    PHẢI nêu field ID lấy ĐÚNG từ danh sách trên; trả thêm khoá `fields` = các ID dùng."*
  - JSON 5 khoá (4 cũ + `fields`).
  - `palette=None` → giữ prompt cũ, không yêu cầu `fields` (tương thích).
- `ground_fields(llm_fields, palette, cache_ids, min_k=2) -> tuple[str, ...]` (hàm thuần):
  - Giữ field LLM nêu **có trong `cache_ids`**; bỏ field bịa.
  - Còn < `min_k` → bổ sung từ `palette` (đã là field thật) cho đủ.
  - Khử trùng lặp, giữ thứ tự ưu tiên (LLM-hợp-lệ trước, palette sau).

### C3. Pin palette ở lõi — `expr_synth.py`

- `build_symbol_context(field_repo, operator_repo, prefilter, scope, relevance_text="",
  pinned=None)`:
  - `pinned` không rỗng → `FIELDS khả dụng` = đúng `pinned` (cắt `MAX_FIELDS_IN_PROMPT`),
    bỏ qua `_relevant_fields`; `_field_type_context` tính trên field pinned.
  - Thêm dòng ràng buộc: *"TUYỆT ĐỐI chỉ dùng field trong danh sách FIELDS trên; KHÔNG
    bịa tên field mới."*
  - `pinned` rỗng → nguyên hành vi cũ.
- `repair_to_expression(deepseek, prefilter, field_repo, scope, system, user, task,
  pinned=None)`:
  - Lỗi field bịa → hint = `suggest_fields` (không rỗng, xem C4) **+** nhắc lại trọn
    palette pinned: *"CHỈ được dùng các field: …"*.
  - Lỗi khác (node/depth/...) → giữ nguyên hành vi hiện tại (ngoài phạm vi).

### C4. `suggest_fields` không rỗng — `expr_synth.py`

Hiện trả `[]` khi không khớp prefix/token (đúng case `asset_growth_rate` → loop chết).
Sửa: nếu rỗng, fallback về `pinned` (nếu truyền vào) hoặc top-`limit` field theo thứ tự
cache, để repair luôn có gợi ý thật.

### C5. Wiring

- `translator.py`: `_to_expression` truyền `pinned=hypothesis.fields` vào
  `build_symbol_context` + `repair_to_expression`; `translate` đọc `hypothesis.fields`.
- `loop.py`: trước `hypothesis_gen.generate(...)`, gọi `retrieve_field_palette` với scope
  của loop, truyền `palette` vào `generate`.
- `generator.py`: `generate_ideas` trả `[(idea, fields)]` đã ground; `generate(idea,
  fields=None)` → `_generate_one(idea, fields)` truyền `pinned`. Giữ overload cũ.
- `main.py`: cập nhật call-site khớp chữ ký mới (line ~573-626, ~803).

## YAGNI

- Không embedding/semantic search — chỉ lexical + theme (field WQ có tên/desc mô tả tốt).
- Không đụng `pre_filter` — validation đã đúng, chỉ là cổng cuối.
- Không xử lý lỗi node/depth lần này.

## Kiểm thử (TDD — test viết trước)

### `ground_fields` (hàm thuần)
- LLM toàn field bịa → augment từ palette, độ dài ≥ `min_k`.
- LLM trộn thật+bịa → giữ thật, bỏ bịa, đủ thì không augment.
- LLM đủ field thật → trả nguyên (khử trùng lặp, đúng thứ tự).
- palette rỗng + LLM rỗng → trả `()`.

### `retrieve_field_palette`
- text khớp token field → field liên quan đứng đầu.
- text không khớp gì ("asset growth rate") → vẫn ≥ `min_k` field (độn theme), không rỗng.

### `suggest_fields`
- input `asset_growth_rate` (case log) + cache không khớp → fallback không rỗng.

### `build_symbol_context(pinned=...)`
- pinned cho trước → `FIELDS khả dụng` chứa đúng pinned + câu cấm bịa.
- `pinned=None` → output y như test cũ (regression).

### `repair_to_expression(pinned=...)`
- LLM trả field bịa lần 1 → prompt repair lần 2 chứa palette pinned + suggestion;
  lần 2 trả field thật → trả expr. (Mock `deepseek.complete` + `prefilter`.)

### `HypothesisGenerator.generate(palette)`
- mock LLM trả 5 khoá gồm `fields` → `Hypothesis.fields` được ground.
- LLM thiếu khoá `fields` → fallback augment từ palette, không vỡ.

### Regression
- `tests/test_llm.py`, `tests/test_novel_ideas.py`, `tests/test_auto_command.py` — path
  `pinned=None` không đổi hành vi.

## Edge case

- Cache rỗng/chưa nạp → `retrieve_field_palette` trả `()`; mọi thứ degrade về hành vi cũ.
- LLM trả `fields` là string thay vì list → `ground_fields` xử lý an toàn (bỏ qua / coi
  như 1 phần tử).
- Field pinned vượt `MAX_FIELDS_IN_PROMPT` → cắt còn 40.

## File chạm

- `src/llm/expr_synth.py` — `retrieve_field_palette` (mới), `build_symbol_context`
  (+`pinned`), `repair_to_expression` (+`pinned`), `suggest_fields` (fallback).
- `src/llm/hypothesis.py` — `Hypothesis.fields`, `generate(palette)`, `ground_fields`.
- `src/llm/translator.py` — truyền `pinned`.
- `src/llm/loop.py` — gọi `retrieve_field_palette`, truyền `palette`.
- `src/llm/generator.py` — `generate_ideas`/`generate` trả+truyền fields.
- `main.py` — cập nhật call-site.
- `tests/` — test mới cho từng component + regression.
