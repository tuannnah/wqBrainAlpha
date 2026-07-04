# FieldCollector nhầm tham số GROUP thành field dữ liệu — thiết kế fix

> Spec brainstorm 2026-07-04. Phát hiện khi chạy thật mục 5 (Auto SIM) sau khi đã sửa field
> cache (`553ae8e`/`cb1e6c5`), seed family (`12f78d8`/`0e1c0a3`), và round-robin seed
> (`33dd42c`/`74c6966`/`4fccd2f`): vòng lặp trả về `ý tưởng=0 sim=0` — TỆ HƠN các lần chạy
> trước, dù mọi fix trước đó đều đúng.

## Vấn đề

Tra DB (`evaluations` table) của lần chạy thật cho thấy 28 ứng viên GP được đánh giá, sharpe/
fitness tính ra bình thường (không lỗi eval), nhưng **23/28 (82%) bị gate loại với lý do
`fields_ok=False (field không hợp lệ)`** — ví dụ biểu thức
`group_neutralize(multiply(-1, ts_rank(close, 5)), sector)` (hoàn toàn hợp lệ, `close` và
`sector` đều đúng) vẫn bị đánh dấu `fields_ok=False`.

Nguyên nhân: `FieldCollector` (`src/lang/visitors.py:32-49`) duyệt đệ quy MỌI node con của một
`Call`, kể cả tham số thứ 2 của `group_neutralize(expr, sector)` — tham số này là **tên GROUP**
(`ArgKind.GROUP` trong registry, xem `src/operators_local/group.py:12-14`:
`signature=(ArgKind.PANEL, ArgKind.GROUP)`), được biểu diễn bằng CÙNG kiểu AST node `Field` như
field dữ liệu thật (`close`). `FieldCollector.visit_field` không phân biệt được, trả `node.name`
cho mọi `Field`, nên `sector` bị tính là "field cần tồn tại trong panel". `data.field_names()`
(MarketData panel) dĩ nhiên không có `"sector"` (nó nằm trong `data.groups`, không phải
`data.fields`) → `fields_ok = fields.issubset(data.field_names())` = `False` → gate loại oan.

`FieldCollector` được dùng để tính `fields_ok` ở **2 nơi thật sự có bug**:
`src/pipeline/runner.py:64` (dùng bởi `generate_many`/GP shortlist) và
`src/backtest/gate.py:51` (dùng bởi `RefinementLoop.run_from_seed`). Vì phần lớn seed family
(`src/generation/families.py`) bọc `group_neutralize(..., sector/industry/subindustry)`, đa số
ứng viên hợp lệ bị loại oan — giải thích trọn vẹn `ý tưởng=0` dù mọi fix trước đó đã đúng.

**Bug này đã từng bị vấp và né bằng workaround**, không phải sửa gốc: `src/scoring/
power_pool.py:24-26` có blacklist cứng `_GROUPING_FIELDS` (7 field, có `currency`) trừ ra sau
khi gọi `FieldCollector`. `src/scoring/dataset_usage.py:14` cũng có `_GROUPING_FIELDS` riêng (6
field, không có `currency`) — **nhưng đây KHÔNG chỉ là workaround cho bug này**: comment tại
`dataset_usage.py:12-13` xác nhận danh sách này mã hóa đúng quy tắc nghiệp vụ thật của WQ Brain
(field miễn trừ khi đếm dataset cho "Single Dataset Alpha", theo
`docs/worldquantbrain/docs/consultant-information/single-dataset-alphas.md`, khác danh sách
Power Pool có thêm `currency`) — field như `sector`/`country` có thể VỪA là field dữ liệu thật
VỪA dùng làm group-key, nên blacklist này có thể vẫn cần đúng vai trò của nó dù đã sửa bug AST.
**KHÔNG xóa 2 blacklist này trong phạm vi fix hiện tại** — chỉ thêm tham số registry bắt buộc
(máy móc), giữ nguyên toàn bộ logic nghiệp vụ.

## Mục tiêu

`FieldCollector` chỉ thu thập field ở đúng vị trí `ArgKind.PANEL` của mỗi operator (tín hiệu
thật) — bỏ qua vị trí WINDOW/SCALAR/GROUP (literal, không phải field tham chiếu), giống hệt cách
`Evaluator.visit_call` đã làm đúng từ trước.

## Kiến trúc

`Evaluator.visit_call` (`src/engine/evaluator.py:72-81`) đã có sẵn khuôn mẫu đúng:

```python
def visit_call(self, node: Call) -> Panel:
    spec = self._ctx.registry.get(node.op)
    eval_args: list[Panel | float | str] = []
    for arg, kind in zip(node.args, spec.signature, strict=True):
        if kind is ArgKind.PANEL:
            eval_args.append(self.evaluate(arg))
        else:  # WINDOW, SCALAR, GROUP
            eval_args.append(_literal(arg))
    ...
```

`FieldCollector` áp dụng đúng khuôn mẫu này: cần `registry` để tra `spec.signature`, zip với
`node.args`, chỉ đệ quy (`.accept(self)`) vào con có `ArgKind.PANEL`; bỏ qua hoàn toàn con ở vị
trí WINDOW/SCALAR/GROUP (không gọi `visit_field`/`visit_constant` cho chúng nữa).

## Thành phần thay đổi

1. **`src/lang/visitors.py`** — `FieldCollector.__init__(self, registry: OperatorRegistry)` bắt
   buộc (giống `Evaluator(ctx)`, không có default). `visit_call` sửa theo khuôn mẫu trên.
   `visit_field`/`visit_constant` giữ nguyên logic, chỉ khác là giờ chỉ được gọi đúng lúc (vị trí
   PANEL).

2. **2 chỗ sửa bug thật (thay đổi hành vi quan sát được):**
   - `src/pipeline/runner.py:64` — `FieldCollector().visit(node)` →
     `FieldCollector(default_registry()).visit(node)` (file đã import `default_registry` sẵn,
     dùng ở dòng 68 `EvalContext(data=data, registry=default_registry(), ...)`).
   - `src/backtest/gate.py:51` — tương tự, file đã import `default_registry` sẵn (dòng 55).

3. **Cập nhật máy móc, KHÔNG đổi logic nghiệp vụ (chỉ thêm registry để tương thích chữ ký mới):**
   - `src/scoring/power_pool.py:34` — `FieldCollector()` → `FieldCollector(default_registry())`,
     **giữ nguyên** `- _GROUPING_FIELDS`.
   - `src/scoring/dataset_usage.py:23,50` — tương tự, **giữ nguyên** logic `_GROUPING_FIELDS`.
   - `src/scoring/genius_report.py:55,71` — thêm `default_registry()`.
   - `scripts/gen_groundtruth.py:139` — thêm `default_registry()`.
   - `src/gp/engine.py:149,199` — dùng `self.registry` (đã có sẵn trên instance).

4. **Test cập nhật (thêm registry vào constructor, không đổi kỳ vọng của test cũ):**
   - `tests/unit/test_lang_visitors_depth_fields.py` (4 chỗ gọi `FieldCollector()`).
   - `tests/integration/test_storage_minibrain_integration.py` (2 chỗ).
   - `tests/integration/test_metrics_gates.py` (1 chỗ).

## Xử lý lỗi / trường hợp biên

- Operator không tồn tại trong registry: `registry.get(node.op)` tự raise `KeyError` rõ ràng
  (hành vi sẵn có, nhất quán với `Evaluator` — không cần try/except mới trong `FieldCollector`).
- `zip(node.args, spec.signature, strict=True)` (giống `Evaluator`) — nếu số args không khớp
  signature (không nên xảy ra với AST đã parse qua `parse()` với validate arity), raise
  `ValueError` rõ ràng thay vì zip âm thầm cắt cụt.
- Cây không có `Call` nào (chỉ `Field`/`Constant` gốc): `visit_field`/`visit_constant` không đổi,
  hoạt động như cũ (không phụ thuộc registry).

## Kiểm thử (TDD)

- **Test tái hiện đúng bug** (RED trước khi sửa): với registry thật (`default_registry()`, đã
  `import src.operators_local`), `FieldCollector(default_registry()).visit(parse("group_neutralize(close, sector)"))`
  phải trả `{"close"}` — KHÔNG có `"sector"`. Hiện tại (chưa sửa) trả `{"close", "sector"}`.
- **Hồi quy `FieldCollector` cũ:** 4 test trong `test_lang_visitors_depth_fields.py` (rank/add/
  ts_mean, không có GROUP) phải cho kết quả giống hệt sau khi thêm `registry` bắt buộc vào
  constructor.
- **Test tích hợp gate:** `score_local_gate("group_neutralize(rank(close), sector)", cfg, panel)`
  (hoặc `_score_one_full` tương đương ở `runner.py`) phải trả `fields_ok=True`/không còn
  `hard_failures` chứa `"fields_ok=False"` — test mới, xác nhận đúng lỗi thật đã fix, không chỉ
  fix ở tầng visitor cô lập.
- Test hiện có của `power_pool.py`/`dataset_usage.py` (nếu có) phải PASS y hệt sau khi thêm
  registry — xác nhận không đổi hành vi nghiệp vụ của 2 file này.

## Phạm vi KHÔNG làm trong spec này

- Không xóa/sửa `_GROUPING_FIELDS` ở `power_pool.py` hay `dataset_usage.py` — đây là quy tắc
  nghiệp vụ WQ Brain thật, độc lập với bug AST, để nguyên.
- Không đổi `ArgKind`/`OperatorSpec`/registry — chỉ dùng thông tin đã có sẵn.
- Không đổi `Evaluator` (đã đúng từ trước, dùng làm khuôn mẫu tham chiếu, không phải mục tiêu
  sửa).
