# GĐ3 — Ép decorrelation bằng AST-originality

> Spec triển khai GIAI ĐOẠN 3 của `tailieu/BUILD_GUIDE_AI_alpha_tool.md`.
> Tiền đề: GĐ2 xong. Nhánh: `phase2-ai-loop` (tiếp tục) hoặc nhánh con.

## Mục tiêu

Đo độ "độc đáo" của một alpha so với một **zoo tham chiếu** (Alpha101 + alpha đã
nộp) bằng tương đồng cây cú pháp (AST). Dùng điểm độc đáo làm **pre-filter rẻ,
chạy local, TRƯỚC khi simulate**: loại thẳng alpha trùng cấu trúc gần y hệt (cửa
chặn cứng), trừ điểm mềm cho phần còn lại. Mục tiêu: chống decay/correlation và
tiết kiệm quota.

Ghi rõ: AST-similarity KHÁC return-correlation thật của WQ (kiểm tra cuối khi nộp
ở GĐ7). AST chỉ là bộ lọc rẻ loại trùng hiển nhiên.

## Module (`src/decorrelation/`)

| Thành phần | Nhiệm vụ | Task |
|---|---|---|
| `similarity.py` | `subtree_canon(node)`, `largest_common_subtree(a, b) -> int`, `similarity_ratio(a, b) -> float` | T3.1, T3.2 |
| `zoo.py` | `ReferenceZoo`: nạp Alpha101 (FASTEXPR) + alpha đã nộp, parse sẵn AST; `originality(expr)`, `most_similar(expr)` | T3.3, T3.4 |
| `alpha101.py` | Hằng danh sách Alpha101 đã dịch sang FASTEXPR (tập con khả dụng với vocab hiện có) | T3.3 |
| `prefilter.py` | `OriginalityFilter(zoo, hard_ratio, ...)`: `check(expr) -> (ok, ratio, penalty)` dùng trước sim | T3.5 |
| `common_subtrees.py` | `frequent_subtrees(exprs, top_k)`: thống kê subtree phổ biến trong alpha tốt → đưa vào prompt tránh dùng lại | T3.6 |

Tái dùng `src/generation/ast_utils.py` (parse_expression, Node/Leaf, all_subtrees,
node_count, to_expression).

## Thuật toán tương đồng (T3.2)

- `subtree_canon(node)`: chuỗi chuẩn hoá của một subtree (op + canon các con theo
  thứ tự; với toán tử giao hoán có thể sort con — giữ đơn giản: theo thứ tự gốc).
- `largest_common_subtree(a, b)`: với mỗi subtree của a và b, băm canon; tìm canon
  chung; trả `node_count` lớn nhất trong các canon trùng. (xấp xỉ "nhánh con
  đẳng cấu lớn nhất" — đủ rẻ và đủ tốt để loại trùng.)
- `similarity_ratio(a, b) = largest_common_subtree(a, b) / min(node_count(a), node_count(b))`
  ∈ [0,1]. 1.0 = một cây là nhánh con của cây kia.

## Điểm độc đáo & pre-filter (T3.4, T3.5)

- `ReferenceZoo.originality(expr) = 1 - max(similarity_ratio(expr, z) for z in zoo)`.
  Zoo rỗng → originality = 1.0.
- `OriginalityFilter.check(expr)`:
  - parse lỗi → (False, 1.0, ∞) (cú pháp đã được pre-filter GĐ1 lo; ở đây an toàn).
  - ratio = max similarity vs zoo.
  - ratio ≥ `hard_ratio` (mặc định 0.9) → **cửa chặn cứng**: (False, ratio, _).
  - ngược lại → (True, ratio, penalty) với `penalty = ratio` (phạt mềm, cộng vào
    điều chuẩn ở GĐ4).
- Tích hợp hook vào `RefinementLoop._evaluate`: trước khi simulate, nếu có
  originality filter và bị chặn cứng → `record_failure("duplicate")`, không sim.
  (Mặc định loop không bật filter — bật qua tham số để giữ "mỗi cơ chế một".)

## Tránh nhánh con phổ biến (T3.6)

- `frequent_subtrees(expressions, min_size=2, top_k=10)`: đếm canon các subtree
  (kích thước ≥ min_size) xuất hiện nhiều trong tập alpha tốt; trả top_k dạng
  biểu thức người đọc được. Dùng để chèn vào prompt translator/refiner: "tránh
  lặp lại các mẫu sau".

## CLI / tích hợp

- Lệnh `python main.py originality --expr "..."`: in điểm độc đáo + alpha gần nhất
  trong zoo (demo GĐ3).
- `research --decorrelate`: bật OriginalityFilter trong vòng lặp (mặc định tắt).

## Test (TDD, local thuần — không mạng)

- `similarity`: cây giống hệt ratio=1; cây không liên quan ratio thấp; subtree
  chung lớn nhất đúng kích thước; canon ổn định.
- `zoo`: originality cao cho alpha lạ, thấp cho alpha ≈ Alpha101; most_similar đúng.
- `prefilter`: chặn cứng khi ratio ≥ hard_ratio; phạt mềm khi dưới ngưỡng; zoo rỗng
  luôn pass.
- `common_subtrees`: phát hiện đúng subtree lặp nhiều.
- `loop` (tích hợp): bật filter → alpha trùng bị loại trước sim (sim count không tăng),
  ghi failure "duplicate".

## Acceptance (guide)

- Tính được điểm độc đáo cho alpha bất kỳ so với zoo.
- Alpha trùng cấu trúc cao bị loại TRƯỚC sim.
- Toàn bộ test cũ + mới pass; không phá GĐ1/GĐ2.

## Thứ tự (mỗi bước 1 commit, TDD)

1. `similarity.py` (canon + largest_common_subtree + ratio).
2. `alpha101.py` + `zoo.py` (originality, most_similar).
3. `prefilter.py` OriginalityFilter.
4. `common_subtrees.py`.
5. Tích hợp loop (hook bật/tắt) + CLI `originality` + `research --decorrelate`.
