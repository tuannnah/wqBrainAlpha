# Task: Sửa lỗi sinh expr depth > 7 trong WQ tool

## Bối cảnh

Pipeline sinh alpha (DeepSeek) đang bị nghẽn ở PreFilter với lỗi `Độ sâu > 7`. Ba lớp wrapper ngoài cùng mà LLM hay bọc — `scale(...)` → `ts_decay_linear(...)` → `group_neutralize(..., sector)` — chiếm sẵn 3 tầng độ sâu trước khi chạm tín hiệu, nên phần lõi chỉ còn 4 tầng → tràn trần 7 gần như mỗi lần LLM ghép nhiều field.

Vòng `repair_to_expression` lại không cứu được vì `extract_rejected_field()` chỉ bóc được tên field; lỗi depth không có field → hint rỗng → LLM sinh lại biểu thức sâu y hệt → 3 lượt repair phí ~3 phút rồi trả `None`.

**Quan trọng — đọc trước khi sửa:** đừng tin các đường dẫn/chữ ký hàm bên dưới là tuyệt đối. Mở code thật, xác nhận signature hiện tại của `repair_to_expression`, `extract_rejected_field`, `build_syntax_constraints`, `tree_depth` rồi mới sửa. Tên/file dưới đây là tham chiếu từ log, không phải hợp đồng.

**Kỷ luật:** làm Task 1 và Task 2 **tách biệt**, mỗi task một commit, test xanh trước khi sang task sau. TDD: viết test fail trước, sửa cho xanh. Không đổi `max_depth` trong scope này (quyết định riêng).

---

## Task 1 — Repair hint cho lỗi depth/node (cầm máu)

### Mục tiêu
Khi PreFilter trả lỗi về độ sâu hoặc số node (không có field để bóc), thay vì gửi hint rỗng cho LLM, gửi hướng dẫn cụ thể về cách **giảm độ sâu**.

### Việc cần làm
1. Tìm nơi `reason` từ PreFilter được chuyển thành hint cho LLM (quanh `extract_rejected_field` và phần dựng prompt repair trong `expr_synth.py`).
2. Thêm nhánh phân loại lỗi: nếu `reason` khớp pattern độ sâu/node (vd chứa `Độ sâu`, `depth`, `node`, `tầng`), trả về một **depth-hint** thay cho field-hint rỗng. Đề xuất hint:
   > Biểu thức vượt giới hạn độ sâu. Hãy làm GỌN: (1) bỏ bớt lớp bọc ngoài — chỉ giữ tối đa một trong {scale, ts_decay_linear, group_neutralize}; (2) làm phẳng các tổ hợp field lồng nhau (multiply/divide/add/zscore chồng nhiều tầng) thành 1 phép kết hợp; (3) ưu tiên tín hiệu nông, ít field.
3. Giữ nguyên hành vi cũ cho lỗi có field (field-hint không đổi).

### Test (viết trước)
- `test_repair_hint_depth_error_is_nonempty`: với `reason="... — Độ sâu > 7"`, hàm dựng hint trả về chuỗi **không rỗng** và **có nhắc tới việc bỏ lớp wrapper**.
- `test_repair_hint_field_error_unchanged`: với reason chứa tên field bị loại, hint vẫn ra field-hint cũ (regression guard).
- `test_depth_classifier_matches_variants`: classifier nhận diện đúng cả `Độ sâu > 7`, `depth`, `node count` (tùy các chuỗi PreFilter thật trả ra — kiểm tra code PreFilter để lấy đúng wording).

### Acceptance
- Chạy lại một mẻ seed: log không còn cảnh "lần 1 fail → lần 2 fail y hệt depth 8"; tỉ lệ repair thành công cho lỗi depth tăng rõ. Không cần con số chính xác, chỉ cần khác biệt quan sát được + 3 test xanh.

---

## Task 2 — Tách signal khỏi config wrapper (fix gốc)

### Nguyên tắc
Theo kiến trúc đã thống nhất: **expression search và configuration search là hai stage tách biệt.** Neutralization / decay / truncation / scale là config-layer, **không nên để LLM viết vào biểu thức.** Hiện prompt đang bắt LLM tự bọc scale/decay/neutralize → vừa ngốn depth, vừa nguy cơ double-neutralize (operator trong expr chồng lên neutralization setting của sim).

### Việc cần làm
1. Mở `build_syntax_constraints` (và toàn bộ system/user prompt của bước synth) trong `expr_synth.py`. **Bỏ mọi hướng dẫn yêu cầu LLM bọc** `scale`, `ts_decay_linear`/`ts_decay_*`, `group_neutralize`/`*_neutralize`. Thay bằng ràng buộc rõ:
   > Chỉ sinh **biểu thức tín hiệu lõi**. KHÔNG bọc scale / decay / neutralize — các bước này do tầng cấu hình xử lý sau. Toàn bộ ngân sách độ sâu (≤ 7) dành cho tín hiệu.
2. Xác định nơi config thực sự được áp:
   - Nếu đã có chỗ set `neutralization` / `decay` / `truncation` trong sim config → để wrapper cho stage đó, không đụng expr.
   - Nếu autowrap đang chèn scale/decay/neutralize vào expr (vd trong `autowrap_*`) → chuyển phần này ra khỏi đường sinh expr, hoặc gate sau bằng cờ config, **một điểm áp dụng duy nhất**, không lồng trong biểu thức LLM.
3. Không tự ý bỏ `group_neutralize` mà làm mất tính trung tính ngành: chuyển nó thành **neutralization setting của sim** (vd `SECTOR`), không phải xóa hẳn. Nếu sim đã neutralize sẵn, bỏ operator trùng trong expr.

### Test (viết trước)
- `test_synth_prompt_forbids_config_wrappers`: output của `build_syntax_constraints` (hoặc prompt builder) **không** còn yêu cầu bọc scale/decay/neutralize, và **có** câu "chỉ sinh tín hiệu lõi".
- `test_signal_only_expr_depth_budget`: với một biểu thức lõi điển hình `multiply(-1, ts_zscore(ts_delta(add(F,F),4),20))`, `tree_depth` = 5 ≤ 7 (xác nhận lõi vừa ngân sách khi bỏ 3 lớp wrapper).
- `test_config_applied_single_point`: nếu chuyển wrapper sang stage config, có đúng một điểm áp dụng; expr do LLM sinh không chứa scale/decay/neutralize (assert qua AST scan trên vài mẫu output mock).

### Acceptance
- LLM sinh expr nông hơn hẳn; tỉ lệ bị chặn vì depth giảm mạnh.
- Config (neutralization/decay) vẫn được áp đúng ở stage sau, không double.
- 3 test xanh.

---

## Ngoài scope (ghi nhận, không làm ở đây)
- Nâng `max_depth` lên 8–9: chỉ cân nhắc sau, **và** phải xác nhận đó là filter local của ta chứ không phải ràng buộc WQ; nếu nâng thì kèm guard interpretability riêng để không thả expr ghép field vô nghĩa (vd breakeven quyền chọn × nợ phải trả) vào sim.
- Auto-trim wrapper tự động: chỉ cho lớp dư thừa chứng minh được (double scale), **không bao giờ** tự xóa neutralize.

## Quy ước
- Mỗi task một commit riêng, message mô tả thay đổi đơn lẻ.
- Chạy full test suite trước khi báo xong từng task.
- Báo lại diff các prompt string đã đổi để review trước khi merge.
