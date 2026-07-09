# Thiết kế: Stage Combiner — ghép nhiều tín hiệu thành một alpha mạnh hơn

Ngày: 2026-07-09
Trạng thái: đã duyệt, chuyển triển khai

## Bối cảnh & động cơ

Tool hiện sinh **alpha đơn** theo từng họ (reversal/momentum/value…), mỗi alpha là
một biểu thức FASTEXPR đơn lẻ. Alpha đơn kiểu `multiply(-2, ts_mean(subtract(close,
vwap), 10))` khó vượt ngưỡng submit (Sharpe > 1.58 / fitness > 1).

**Nền tảng lý thuyết — Luật cơ bản của quản lý danh mục (Grinold–Kahn):** ghép N tín
hiệu *có kỹ năng ngang nhau và ít tương quan với nhau* thì Information Ratio (≈ Sharpe)
của tổ hợp tăng theo ~√N. Điều kiện sống còn: các tín hiệu con **ít trùng nhau** — nếu
tương quan cao, lợi ích √N sụp về 1. Vì vậy đòn bẩy thật nằm ở **khâu chọn tín hiệu ít
tương quan**, không phải ở việc cộng bừa.

Ví dụ định lượng (ghép trọng số đều):
- 2 tín hiệu, ρ=0.3 → Sharpe tổ hợp ≈ √(2/(1+0.3)) ≈ **1.24×** tín hiệu đơn.
- 3 tín hiệu, ρ đôi một 0.3 → ≈ **1.44×**.

## Quyết định đã chốt

| Vấn đề | Lựa chọn |
|---|---|
| Kết quả cuối | Một **stage combiner tự động** trong pipeline, chạy nối tiếp sau mỗi run |
| Cách ghép | **Chuẩn hóa + trọng số đều**: `add(rank(s1), rank(s2), …)` |
| Khâu chọn | **Greedy khử tương quan theo PnL local**, ngưỡng τ=0.3, N∈[2,4] |
| Nguồn tín hiệu | Pool top-K của run hiện tại **⊕** kho alpha tốt trong DB (expr + PnL đã lưu) |
| Neutralization | **Hướng A**: nướng `group_neutralize` vào từng tín hiệu con trong biểu thức, setting combo = NONE; **tự lùi hướng B** (một neutralization chung, re-tune) khi vượt trần độ sâu |

## Kiến trúc & luồng dữ liệu

Stage mới `src/generation/combiner.py` (logic thuần, tuân dependency-rule: **không import
storage** — nhận candidate đã vật chất hóa qua tham số, giống `pool_corr.py`).

```
[sinh + chấm local như hiện tại]
        │  top-K sub-signal (expr + PnL local + dates + score)
        ▼
[gom nguồn]  pool hiện tại  ⊕  kho alpha tốt trong DB (expr + PnL đã lưu)
        ▼
[chọn greedy khử tương quan]  → vài tổ hợp, mỗi tổ hợp 2–4 tín hiệu, |rho| cặp < τ
        ▼
[dựng FASTEXPR ghép]  add(rank(s1), rank(s2), rank(s3))
        ▼
[re-tune + chấm local lại]  chỉ giữ combo VƯỢT tín hiệu con tốt nhất & qua gate local
        ▼
[đẩy vào đúng đường sim/submit hiện có]
```

Kích hoạt bằng cờ CLI `--combine` (mặc định BẬT vì chạy tự động nối tiếp); có mục menu.

## Thuật toán chọn greedy khử tương quan

```
INPUT: [(id, expr, pnl, dates, score)], τ=0.3, N_max=4, N_min=2, max_combos=5
1. Loại ứng viên PnL không hợp lệ (std=0 hoặc quá ít ngày overlap).
2. Sắp xếp giảm dần theo score local.
3. seed = ứng viên điểm cao nhất  →  combo = [seed]
4. Duyệt phần còn lại theo thứ tự điểm:
      thêm cand  CHỈ KHI  max(|rho(cand, m)| với mọi m ∈ combo) < τ
      dừng khi len(combo) == N_max
5. Nếu len(combo) ≥ N_min → xuất combo.
6. Bỏ seed đã dùng, lặp lại từ (3) tạo thêm combo (tối đa max_combos).
```

Tương quan đo trên **PnL local**, KHÔNG phải văn bản biểu thức — chống "đa dạng giả"
(hai công thức khác chữ nhưng cùng bản chất → PnL tương quan cao → bị loại đúng).

## Dựng biểu thức FASTEXPR

Dạng: `add(rank(s1), rank(s2), rank(s3))`. Dùng `rank` (bền outlier, khẩu vị WQ) để
chuẩn hóa cross-sectional mỗi ngày về [0,1], trọng số đều là công bằng.

Hai ràng buộc WQ:
1. **Trần độ sâu biểu thức**: chạy `DepthVisitor`; vượt trần → tự giảm N hoặc bỏ combo
   (không nộp biểu thức sai).
2. **Neutralization (hướng A)**: bọc `group_neutralize(si, <group_riêng>)` vào từng tín
   hiệu con *bên trong* biểu thức; setting neutralization combo = NONE (tránh trung hòa
   kép). Nếu hướng A vượt trần độ sâu → **lùi hướng B**: bỏ neutralization riêng, ghép
   thô, re-tune một neutralization chung cho combo.

Sau khi dựng: re-tune (decay/truncation) + chấm local lại; **chỉ giữ combo vượt tín hiệu
con tốt nhất trong nó VÀ qua gate local**.

## Bề mặt triển khai

**File mới:**
- `src/generation/combiner.py` — `select_decorrelated_combos()`, `build_combined_expression()`.
- Helper dùng chung `pairwise_abs_rho(pnl_a, dates_a, pnl_b, dates_b)` — tách từ
  `pool_corr.py::_rho_sorted`, dùng lại cho cả `pool_corr` lẫn `combiner`.

**File sửa (nối dây):**
- `src/pipeline/runner.py` / `closed_loop.py` — gọi combiner sau khâu chấm local, đẩy
  combo vào đường sim/submit.
- `src/storage/repository.py` — truy vấn alpha tốt trong DB (expr + PnL) làm nguồn bổ sung.
- `main.py` + `run.ps1`/menu — cờ `--combine` + mục menu.
- `config/` — `COMBINE_TAU=0.3`, `COMBINE_N_MAX=4`, `COMBINE_N_MIN=2`, `COMBINE_MAX_COMBOS=5`.

## Kiểm thử (TDD)

1. `select_decorrelated_combos`: seed đúng điểm cao nhất; loại ρ≥τ; tôn trọng N_min/N_max;
   sinh nhiều combo không trùng seed; pool rỗng/1 phần tử → không combo.
2. `pairwise_abs_rho`: align theo ngày; std=0 → bỏ; overlap thiếu → bỏ (giữ hành vi
   `_rho_sorted`).
3. `build_combined_expression`: đúng dạng `add(rank(...),…)`; hướng A nướng
   `group_neutralize`; vượt trần độ sâu → lùi B; `parse()` lại được.
4. Combiner giữ combo chỉ khi vượt tín hiệu con tốt nhất và qua gate local.
5. Integration: một run giả lập → combo xuất hiện trong danh sách gửi sim.

## Rủi ro / cạm bẫy đã lường

- **Trần độ sâu** → fallback B; nếu vẫn quá sâu, bỏ combo.
- **Đa dạng giả** → đo PnL, không đo văn bản.
- **PnL từ DB lệch trục ngày** → `pairwise_abs_rho` align theo ngày giao nhau + overlap tối thiểu.
- **created_at là UTC** (bẫy đã biết) — combiner dùng dates của PnL nên không ảnh hưởng;
  lưu ý khi truy vấn DB.
- **Double-neutralize** → hướng A đặt setting combo = NONE.
