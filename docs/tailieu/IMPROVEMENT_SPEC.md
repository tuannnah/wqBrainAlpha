# MiniBrain / wqBrainAlpha — Improvement Spec (throughput + yield)

> **Đọc bằng Claude Code trong repo `tuannnah/wqBrainAlpha`.** Đây là spec kỹ thuật để cải
> tiến vòng closed-loop (`run.bat` mục 5 / CLI `closed-loop`). Mọi khuyến nghị bên dưới bám
> vào code thật (`src/app/closed_loop_adapters.py::build_closed_loop`, `LocalTuner`,
> `GPIdeaSource`, `CombinerIdeaSource`, `CuratedIdeaSource`, `AltDataIdeaSource`,
> `CorrelationChecker`, avoid-list, calibration) và WQ Brain gate rules (self-corr ≤ 0.70,
> depth ≈ 7, Sharpe/Fitness floor).

## 0. Vấn đề người dùng (2 câu)

1. **"Chạy auto mãi không có công thức đạt"** → yield ≈ 0.
2. **"Tạo ra 1 công thức quá lâu"** → throughput thấp, thời gian/ứng viên cao.

Cả hai đều truy được về **cùng một gốc**: pipeline đang tiêu ngân sách sinh + backtest vào
họ nhân tố đã bão hoà và các biến thể trùng lặp, nên (a) hầu hết chết ở local gate trước khi
kịp sim, (b) số ít lọt sim thì không đủ Sharpe/self-corr để đạt. Đây **không** phải vấn đề
tăng tốc phần cứng; là vấn đề *chọn cái gì để sinh* và *loại rác sớm & rẻ*.

---

## 1. Bằng chứng từ 3 log phiên (2026-07-09)

| Log | #ý tưởng | Đạt | stop_reason chủ đạo | Ghi chú |
|---|---|---|---|---|
| `alphas_..._230943.csv` | **0** | 0 | — | Cả phiên **không sinh nổi 1 ứng viên** qua gate. Đây là "chạy mãi không ra" ở dạng thuần tuý. |
| `alphas_..._183046.csv` | 17 | 0 | `local_tuned` (~10, sims=1), `local_floor` (~5, sims=0), `sub_universe` (~2, sims=0) | Các dòng `local_tuned` có `wq_alpha_id` + sims=1 nhưng **`sharpe`/`fitness` để trống** → sim xong vẫn trượt. |
| `alphas_..._204308.csv` | 28 | 0 | như trên | Xuất hiện biểu thức GP **phình khổng lồ** (dòng 25) rơi `local_floor`. |

**Các mẫu hỏng cụ thể (trích thẳng từ log):**

- **Bão hoà họ nhân tố.** `ts_mean(subtract(close, vwap), 10)`, `ts_mean(subtract(close, open), 5)`,
  `ts_delta(close, ...)`, `ts_rank(close, ...)`, `ts_zscore(volume, ...)`, `ts_corr(close, volume, ...)`
  — toàn bộ nằm trong cụm **VWAP intraday-reversal + volume-anomaly** mà `memory.md` đã ghi là
  **đã mine cạn**. GP đang quay quanh chính `CuratedIdeaSource` (VWAP reversal seed).
- **Biến thể trùng lặp vô nghĩa.** Dòng 1 vs 2 chỉ khác hằng số (`4` vs `2`, `-1` vs `-0.5`) →
  `turnover`/`self_corr`/`fitness` **y hệt**. Sau `rank`/neutralize thì nhân với hằng số dương là
  **bất biến** → 2 sim = 1 thông tin. `rank(ts_delay(ts_zscore(volume,120),120))` xuất hiện ở **cả
  hai** phiên → avoid-list/cache không chặn cross-session.
- **GP phình + hằng số ngẫu nhiên.** Ví dụ dòng 25/204308 là cây `trade_when` lồng chục tầng với
  `winsorize(open, -1.9423623924877862)`, `power(..., 0.3365273576052612)`. Vừa dễ vượt depth,
  vừa vô nghĩa kinh tế, vừa tốn backtest → rồi `local_floor`. Đây là nguồn chính của "sinh 1 công
  thức quá lâu".
- **Không có toán tử hạ self-corr.** Gần như 0 dòng dùng `regression_neut`/`vector_neut`. Chỉ dùng
  `group_neutralize(sector)` — **không** khử được tương quan với thành phần crowded của pool.
- **Log schema lệch cột.** Số cột được điền khác nhau giữa các dòng (`sharpe`/`fitness` khi trống khi
  không, vị trí lệch) → **không thể phân tích thất bại một cách tự động**. Đây là nợ instrumentation
  phải trả trước khi tối ưu bất cứ thứ gì khác.

---

## 2. Chẩn đoán — gốc rễ xếp hạng theo tác động

> Xếp theo "sửa cái này giải phóng bao nhiêu yield/throughput". Mỗi mục: *bằng chứng → cơ chế → hệ quả*.

**R1. Generator bị neo vào họ đã bão hoà (tác động cao nhất).**
Bằng chứng: 100% biểu thức thuộc cụm PV/VWAP-reversal; `curated_seeds` mặc định BẬT, GP quay quanh
seed, `alt-data` mặc định TẮT. Cơ chế: trong pool bão hoà, cải thiện tín hiệu và giảm self-corr là
*cùng một lever kéo ngược chiều* (memory ghi rõ) → trần không thể vượt. Hệ quả: dù sim thành công,
self-corr/fitness vẫn trượt → yield ≈ 0. **Đây là lý do #1 "không có công thức đạt".**

**R2. Không khử tương quan cấu trúc.**
Bằng chứng: thiếu `regression_neut`/`vector_neut`. Cơ chế: mọi transform rank-preserving
(`winsorize/scale/truncate/rank/group_neutralize`) **không đổi** self-corr với pool. Hệ quả: ngay cả
ý tưởng khác họ cũng sẽ trượt gate 0.70 nếu configuration stage không có lever orthogonalize.

**R3. Sinh trùng lặp + không dedup cross-session.**
Bằng chứng: cặp hằng-số-bội, biểu thức lặp giữa 2 phiên. Cơ chế: GP mutate hằng số/scale bất biến;
avoid-list & structural-cache không chuẩn hoá (canonicalize) biểu thức. Hệ quả: tiêu quota + thời
gian backtest vào thông tin trùng. **Đây là lý do #1 "tạo 1 công thức quá lâu".**

**R4. GP bloat & random-constant mutation không kiểm soát.**
Bằng chứng: cây lồng chục tầng, float ngẫu nhiên. Cơ chế: thiếu parsimony pressure + depth guard +
constant-set rời rạc. Hệ quả: nhiều ứng viên vượt depth hoặc vô nghĩa → `local_floor`, lãng phí.

**R5. Local floor có thể vừa quá chặt vừa quá lỏng, và mù thông tin.**
Bằng chứng: nhiều `local_floor` (Sharpe<0.5 trên panel 478 mã) trong khi doc thừa nhận panel
**underestimate fitness nặng**; đồng thời các biểu thức rác vẫn lọt tới bước backtest. Cơ chế: floor
đơn-ngưỡng Sharpe trên panel hẹp có phương sai lớn. Hệ quả: có thể giết nhầm ý tưởng tốt *và* vẫn để
lọt rác cấu trúc (đáng lẽ chặn bằng syntax/depth/parsimony trước khi backtest).

**R6. Instrumentation không đủ để tự chẩn đoán.**
Bằng chứng: CSV lệch cột, thiếu trường lý do thất bại chi tiết (fail_check cụ thể, depth, timing).
Hệ quả: không đo được throughput theo stage, không biết bottleneck nằm ở đâu → tối ưu mù.

---

## 3. Kế hoạch sửa — phân pha, có acceptance criteria

### Pha 0 — Instrumentation trước (nửa ngày, bắt buộc làm đầu tiên)

Không tối ưu khi chưa đo được. Nâng cấp CSV logger (bước 6 trong `QUY_TRINH_SINH_ALPHA.md`) thành
schema **cố định, luôn điền đủ cột**, thêm:

- `stage_reached` (idea|syntax|depth|dedup|local_floor|tuned|simmed|corr_checked|passed)
- `fail_check` (mã check WQ thật khi có: LOW_SHARPE / LOW_FITNESS / IS_LADDER_SHARPE /
  LOW_SUB_UNIVERSE_SHARPE / SELF_CORR / DEPTH / SYNTAX / DUP)
- `family` (nhãn họ nhân tố suy từ field/cấu trúc: pv_reversal / momentum / fundamental_quality /
  fundamental_growth / analyst / options_iv / news_social / combiner)
- `expr_depth`, `gen_ms`, `backtest_ms`, `sim_ms`, `dedup_key` (canonical hash)
- `local_sharpe`, `brain_sharpe`, `brain_fitness` tách bạch (đừng gộp local/brain vào 1 cột)

Thêm **báo cáo cuối phiên** (in ra + ghi `logs/session_summary_*.md`): funnel theo `stage_reached`,
phân bố `fail_check`, phân bố `family`, thời gian trung vị mỗi stage, số ứng viên trùng bị chặn.

*Acceptance:* chạy 1 phiên → đọc summary trả lời được ngay "ứng viên chết ở đâu, vì sao, tốn bao lâu".

### Pha 1 — Chặn rác sớm & rẻ (throughput; 1 ngày)

Mục tiêu: cắt "tạo 1 công thức quá lâu" bằng cách **không bao giờ backtest những gì chắc chắn hỏng**.
Thứ tự gate phải là **rẻ → đắt**: `syntax/arity → depth → canonical-dedup → parsimony → local backtest → sim`.

1. **Canonical dedup (R3).** Chuẩn hoá biểu thức trước khi đánh giá: (a) gấp hằng số & bỏ scale dương
   dư thừa ngoài `rank`/neutralize; (b) sắp xếp các toán hạng giao hoán (`add`,`multiply`,`min`,`max`)
   theo thứ tự chuẩn; (c) hash cây AST đã chuẩn hoá → `dedup_key`. Chặn nếu `dedup_key` đã có trong
   **avoid-list bền vững cross-session** (dùng bảng SQLite theo email đã có). *Acceptance:* cặp
   `multiply(4,X)` và `multiply(2,X)` cho **cùng** `dedup_key`; biểu thức lặp giữa 2 phiên bị chặn ở
   stage `dedup`, không tốn backtest.
2. **Depth guard trước backtest (R4).** Tính depth của **core + wrapper stack dự kiến**
   (`scale∘ts_decay_linear∘neutralize` = 3 tầng) và loại nếu > ~7 **trước** khi backtest. Lỗi depth
   sửa bằng *bỏ wrapper/làm phẳng core*, KHÔNG swap field (theo skill).
3. **Parsimony pressure trong GP (R4).** Thêm phạt kích thước cây vào fitness GP; giới hạn depth khi
   sinh; **rời rạc hoá tập hằng số** (vd chỉ cho phép {-2,-1,-0.5,0.5,1,2} và các cửa sổ thời gian
   {5,10,20,40,60,120,250}) thay vì float ngẫu nhiên. *Acceptance:* độ dài biểu thức trung vị giảm
   ≥30%; không còn float 15 chữ số trong log.

### Pha 2 — Chuyển họ nhân tố (yield; đây là đòn quyết định — 2–3 ngày)

Mục tiêu: sửa "không có công thức đạt" bằng **thoát cụm PV/VWAP-reversal đã cạn** (R1).

1. **Đổi mặc định generation.** Trong `build_closed_loop`: hạ tỉ trọng `CuratedIdeaSource` PV-reversal;
   **bật `AltDataIdeaSource` mặc định** (option8 IV/HV, socialmedia8 sentiment) — đây là con đường
   orthogonality thật với pool đã bão hoà PV. Thêm nguồn **fundamental** mà memory/skill gợi ý:
   gross-profitability, operating cash-flow yield, asset growth, analyst estimate revision.
   Nhớ: fundamental field **bắt buộc `ts_backfill`** (sparse, NaN giữa các kỳ báo cáo) — nếu thiếu là
   alpha chết. Verify field LIVE qua `get_datafields`, đừng tin cứng `VERIFIED_FIELDS`.
2. **Family-aware budget & exhaustion guard.** Dùng nhãn `family` (Pha 0) để **giới hạn số ứng viên
   mỗi họ mỗi phiên**; khi một họ đã sinh N ứng viên mà 0 lọt qua self-corr proxy → **đóng họ đó
   trong phiên** và chuyển ngân sách sang họ khác. Đây là "exhaustion recognition" tự động hoá.
3. **Hypothesis-first cho LLM (DeepSeek).** Ép prompt sinh theo 4 phần (observation → theoretical
   basis → economic mechanism → specification) và **tiêm avoid-list + danh sách họ đã bão hoà** vào
   prompt để LLM không tái sinh reversal. Nhãn phần nào literature-grounded vs engineering.
   *Acceptance:* ≥60% ứng viên/phiên **không** thuộc `pv_reversal`; log summary cho thấy ≥3 họ khác nhau.

### Pha 3 — Configuration stage đúng lever (yield/self-corr; 1–2 ngày)

Tách hẳn **expression search** khỏi **configuration search** (skill rule 7).

1. **self-corr là objective hạng nhất, không phải hậu kiểm (R2).** Ở LocalTuner/configuration stage,
   khi self-corr là chiều ràng buộc, thêm nhánh cấu hình dùng **`regression_neut`/`vector_neut`** chống
   lại thành phần crowded (vd neutralize theo tín hiệu reversal đại diện của pool). Đây là toán tử
   **duy nhất** hạ được self-corr — đừng kỳ vọng winsorize/scale/truncate làm điều đó.
2. **Verify self-corr bằng checker THẬT của WQ**, không chỉ proxy local (proxy phân kỳ với checker
   thật). Poll `GET /alphas/{id}/correlations/self` tới khi có JSON (đã có `CorrelationChecker`).
   Nhắm ≤ 0.5 để lọt Power Pool, không chỉ ≤ 0.70.
3. **Bounded > unbounded, đừng smooth tín hiệu nhanh.** Ưu tiên `ts_rank` hơn `ts_zscore` cho ổn định
   regime; nếu turnover *là* alpha thì không decay/hump.

### Pha 4 — Hiệu chỉnh local floor (giảm false-negative; 1 ngày)

- Thay floor Sharpe đơn-ngưỡng (R5) bằng ngưỡng **percentile theo họ** hoặc ngưỡng đã hiệu chỉnh theo
  `brain ≈ local × 1.28` (doc) + biên an toàn cho phương sai panel hẹp. Ghi lại **cả** local_sharpe và
  ngưỡng áp dụng để audit.
- Vì panel underestimate fitness: floor chỉ nên loại phần *rác rõ ràng*, việc lọc tinh để cho sim thật.
  Đảm bảo rác cấu trúc đã bị chặn ở Pha 1 *trước* backtest, không phải ở đây.
- *Acceptance:* tỉ lệ chết ở `local_floor` giảm, trong khi tỉ lệ ứng viên đã-sim mà đạt tăng (đo bằng
  funnel Pha 0).

---

## 4. Bảng ưu tiên nhanh (nếu chỉ làm được vài thứ)

| Ưu tiên | Việc | Sửa vấn đề | Chi phí |
|---|---|---|---|
| **1** | Bật alt-data + fundamental, hạ curated PV-reversal (Pha 2.1) | Yield (không có alpha đạt) | Trung bình |
| **2** | Canonical dedup + avoid-list cross-session (Pha 1.1) | Throughput (quá lâu) | Thấp |
| **3** | Instrumentation funnel (Pha 0) | Chẩn đoán mọi thứ khác | Thấp |
| **4** | `regression_neut`/`vector_neut` ở config stage (Pha 3.1) | Self-corr gate | Trung bình |
| **5** | Parsimony + depth guard + hằng số rời rạc (Pha 1.2–1.3) | Throughput + tính hợp lệ | Thấp |

---

## 5. Cần điều tra trong repo trước khi code (đừng hardcode giả định)

- `src/app/closed_loop_adapters.py::build_closed_loop` — thứ tự & mặc định các IdeaSource, chỗ đổi
  tỉ trọng curated/alt-data.
- `GPIdeaSource` — nơi thêm parsimony pressure, depth cap, tập hằng số rời rạc.
- `LocalTuner` — nơi tách config stage & chèn nhánh `regression_neut`/`vector_neut`.
- Avoid-list & structural-cache — có canonicalize chưa? bảng SQLite theo email có bền cross-session không?
- CSV logger (bước 6) — chuẩn hoá schema, thêm trường Pha 0.
- `CorrelationChecker` — xác nhận đang verify bằng checker thật, ngưỡng đang là bao nhiêu.
- `get_datafields` — danh sách field fundamental/alt-data thực sự khả dụng cho account hiện tại.

## 6. Kế hoạch kiểm chứng (bắt buộc — người dùng làm việc theo single-variable methodology)

1. **Đổi một biến mỗi lần**, chạy 1 phiên closed-loop, so `session_summary` trước/sau.
2. Chỉ tiêu throughput: thời gian trung vị/ứng viên, % ứng viên bị chặn *trước* backtest, số sim/phiên.
3. Chỉ tiêu yield: % ứng viên đã-sim đạt gate, số họ nhân tố khác nhau, self-corr trung vị (checker thật).
4. Regression test FASTEXPR: bộ test parser/operator (`./venv/Scripts/python.exe -m pytest`) phải xanh —
   canonicalize/dedup không được đổi ngữ nghĩa biểu thức.
5. **Không auto-submit.** Submit vẫn là hành động thủ công cần người dùng đồng ý (bước 7).

---

### Ghi chú FASTEXPR nhanh (tránh sai âm thầm)
- `ts_delay(x,d)` — KHÔNG dùng `delay`.
- Fundamental **luôn** `ts_backfill(field, d)`.
- Chỉ `regression_neut`/`vector_neut` hạ self-corr; rank/scale/winsorize/truncate thì không.
- Depth ≈ 7: wrapper stack ăn 3 tầng, core còn ~4. Lỗi depth → làm phẳng core, không swap field.
- Verify field LIVE trước khi dùng; field bịa → WQ loại → cho vào blacklist.
