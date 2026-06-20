# Thiết kế: Hợp nhất hai bộ sinh biểu thức LLM qua module chung `expr_synth`

Ngày: 2026-06-20

## 1. Bối cảnh & vấn đề

Log chạy thật (`src.llm.generator:_generate_one`) cho thấy LLM lặp đi lặp lại
một lớp lỗi: áp operator MATRIX (`ts_zscore`, `ts_delta`, `ts_mean`...) trực
tiếp lên field VECTOR mà chưa rút về MATRIX bằng `vec_avg`/`vec_sum`:

```
LLM expr lỗi (lần 1): rank(-1 * ts_zscore(volume, 20) * ts_zscore(editorial_commentary_sentiment_2, 20))
  — Operator ts_zscore đòi input MATRIX, không nhận field VECTOR trực tiếp:
    editorial_commentary_sentiment_2 (cần vec_avg/vec_sum trước)
LLM expr lỗi (lần 2): group_neutralize(rank(ts_delta(ts_mean(aggregate_option_open_interest_2, 1), 5)) ...)
  — Operator ts_mean đòi input MATRIX ...   # LLM "sửa" bằng ts_mean → vẫn là MATRIX op, vẫn sai
```

Mỗi lần retry là một lượt gọi LLM chậm và tốn token; LLM lại sửa sai vì không
được dạy quy tắc đúng.

### Gốc rễ

Codebase có **hai bộ sinh biểu thức LLM**:

- `src/llm/translator.py` (`AlphaTranslator`): `hypothesis → 1 biểu thức`. **Đã**
  có `_field_type_context` chèn "FIELD TYPES" + "QUY TAC VECTOR" (dạy
  `vec_avg/vec_sum`) và `_suggest_fields` gợi ý field thật khi LLM bịa field.
  Dùng ở `loop.py`, `refiner.py`, `main.py`.
- `src/llm/generator.py` (`LLMAlphaGenerator`): `idea → N biểu thức` + sinh ý
  tưởng. `build_system_prompt` **không** chèn field type, **không** có quy tắc
  VECTOR, **không** gợi ý field. Dùng ở `hybrid.py` (seed GA) — đây chính là
  path đang lỗi trong log.

Prefilter (`src/simulation/pre_filter.py`) đã phát hiện đúng lỗi VECTOR→MATRIX
(bằng chứng: lỗi đang fire trong log, nên `field_types`/`matrix_only_ops` của
prefilter generator đã được nạp). Vấn đề chỉ ở **prompt** của generator thiếu
hiểu biết về field type, và **vòng repair** của generator nghèo nàn (chỉ ném
lại lý do, không gợi ý cách sửa).

Hai bộ sinh trùng lặp phần lõi: dựng ngữ cảnh prompt (symbol + field type) và
vòng prefilter-retry. Sửa riêng generator sẽ nhân đôi logic; nên **tách phần
lõi ra module chung** để chỉ còn một nơi dạy quy tắc field type.

## 2. Mục tiêu & phi mục tiêu

**Mục tiêu:**
- Generator hết lặp lỗi VECTOR→MATRIX: prompt có FIELD TYPES + quy tắc VECTOR;
  vòng repair gợi ý `vec_avg/vec_sum` và field thay thế.
- Xoá trùng lặp giữa hai bộ sinh: phần dựng ngữ cảnh + vòng repair về một nơi.
- Hành vi công khai của `AlphaTranslator` giữ nguyên (chỉ dời ruột code).

**Phi mục tiêu (YAGNI):**
- KHÔNG gộp hai lớp công khai thành một (vai trò khác nhau:
  idea→N exprs vs hypothesis→1 expr).
- KHÔNG đụng cách dựng prefilter (`local_select.py`, `main.py`).
- KHÔNG đụng path sinh ý tưởng (`generate_ideas`, `build_ideas_system_prompt`,
  `_idea_field_context`, `_feedback_context`).
- KHÔNG sửa `loop.py`/`refiner.py` (gọi qua `translate`/`_to_expression` —
  hai hàm này vẫn còn, chỉ đổi ruột).

## 3. Kiến trúc

Module mới **`src/llm/expr_synth.py`** — các hàm thuần (nhận repo/prefilter làm
tham số, không giữ state):

| Hàm | Nguồn rút ra | Nhiệm vụ |
|---|---|---|
| `build_symbol_context(field_repo, operator_repo, prefilter, scope, relevance_text="")` | translator `_symbol_context` + `_field_type_context` + `_relevant_fields` | Liệt kê operators/fields/groups + FIELD TYPES + QUY TẮC VECTOR |
| `build_syntax_constraints(prefilter)` | translator `_syntax_constraints` | Ràng buộc độ sâu/node, đối số theo vị trí |
| `suggest_fields(field_repo, scope, bad_field, limit=5)` | translator `_suggest_fields` | Gợi ý field thật gần field bịa nhất |
| `repair_to_expression(deepseek, prefilter, field_repo, scope, system, user, task)` | translator `_to_expression` (vòng lặp) | Vòng LLM→prefilter→retry, kèm hint field thay thế |

Hằng dùng chung chuyển vào module: `MAX_REPAIR_ATTEMPTS`, `FEWSHOT_EXAMPLES`
(hiện trùng ở cả hai file).

### Chọn field liên quan

Cả hai dùng cách chấm điểm token của translator (`_relevant_fields`) bên trong
`build_symbol_context`. Generator chuyển từ liệt phẳng `[:N]` sang chọn theo
điểm — **thay đổi hành vi có chủ đích** (cải thiện độ liên quan), `relevance_text`
là `idea`.

### Xử lý scope

`scope: dict | None`. `None` → gọi `field_repo.load_cached()` trơn (generator
hiện không có scope). Khác → `field_repo.load_cached(**scope)` (translator).
Hàm chung tự bao try/except chữ ký như `generator._cached_fields` đang làm.

## 4. Thay đổi theo file

- **`src/llm/expr_synth.py`** (mới): 4 hàm + 2 hằng ở trên.
- **`src/llm/translator.py`**: xoá `_field_type_context`, `_symbol_context`,
  `_syntax_constraints`, `_suggest_fields`; `_to_expression` gọi
  `expr_synth.repair_to_expression(...)`; bỏ hằng trùng, import từ `expr_synth`.
  Public `translate`/`_to_expression` giữ chữ ký.
- **`src/llm/generator.py`**: `build_system_prompt` dựng qua
  `build_symbol_context` + `build_syntax_constraints` rồi nối `_feedback_context`
  (giữ nguyên); `_generate_one` gọi `repair_to_expression(...)`; bỏ hằng trùng.
  `generate_ideas` và path ý tưởng giữ nguyên.

## 5. Luồng sau khi sửa

```
generator.generate(idea)                 translator.translate(hypothesis)
        |                                          |
   build_system_prompt                        _describe → _to_expression
        |  (symbol_context + syntax + feedback)     |  (symbol_context + syntax + avoid)
        +---------------------+--------------------+
                              v
              expr_synth.repair_to_expression(deepseek, prefilter, ...)
                  loop: complete → extract_json → prefilter.check
                        fail → extract_rejected_field → suggest_fields → hint → retry
```

Với log đang lỗi: `editorial_commentary_sentiment_2`,
`aggregate_option_open_interest_2` được đánh dấu VECTOR trong prompt + ví dụ
`ts_zscore(vec_avg(field), 20)` → LLM không áp `ts_*` thẳng; nếu lỡ sai, hint
repair chỉ đúng `vec_avg/vec_sum` thay vì để LLM mò.

## 6. Test (TDD)

**Mới `tests/test_expr_synth.py`:**
- `build_symbol_context`: có field VECTOR được chọn → output chứa "QUY TAC
  VECTOR" + dòng `FIELD TYPES`; không có VECTOR → không chèn.
- `repair_to_expression`: (a) trả expr ngay khi prefilter pass lần 1; (b) lý do
  là field bịa → lượt retry kế có hint field thay thế; (c) trả `None` sau
  `MAX_REPAIR_ATTEMPTS`.
- `suggest_fields`: field cùng tiền tố dataset xếp trên field chỉ trùng token.

**Cập nhật:**
- `tests/test_translator.py`: test trỏ private `_field_type_context/
  _symbol_context/_suggest_fields` → trỏ sang `expr_synth` hoặc kiểm qua hành vi
  `translate`. Output translator giữ nguyên nên phần lớn assert còn đúng.
- Test generator: bộ nào assert nội dung `build_system_prompt`/thứ tự field →
  cập nhật cho cách chọn theo điểm + có thêm FIELD TYPES.

**Regression:** `test_translator.py`, test generator, `test_pre_filter.py` xanh.

## 7. Rủi ro

- `load_cached` khác chữ ký giữa hai bên → hàm chung phải bao try/except scope.
- Generator đổi cách chọn field là thay đổi hành vi → cần sửa test tương ứng.
- Translator output phải bất biến (chỉ dời code) — kiểm bằng test `translate`
  hiện có.
