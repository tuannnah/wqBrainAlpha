# Review & Kế hoạch cải thiện tool sinh alpha (cho Claude Code thực hiện)

> Ngày: 2026-07-19 · Phạm vi: `wqBrainAlpha` (MiniBrain) · Người review: Claude (Cowork)
>
> **Mục đích doc này:** đưa cho Claude Code chạy trực tiếp. Mỗi workstream (WS) có:
> nguyên nhân gốc (kèm `file:line` để tự xác minh), các task TDD, và tiêu chí nghiệm thu.
> Làm theo THỨ TỰ WS1 → WS2 → WS3 → WS4 (WS1+WS2 gỡ nút throughput trước, có đòn bẩy
> ngay; WS3 mở rộng nguồn; WS4 cần dữ liệu live nên để sau).
>
> **Nguyên tắc bắt buộc (theo skill worldquant-brain + convention repo):**
> - TDD: viết/chỉnh test TRƯỚC, rồi code cho xanh. `pytest -q` phải giữ 1543+ passed.
> - KHÔNG hardcode ngưỡng ở call site — mọi số vào `config/thresholds.py`.
> - Tách 2 khâu: tìm signal-core vs cấu hình (neut/decay/trunc). Đừng trộn.
> - Depth budget MAX_DEPTH=7: wrapper `scale(ts_decay(group_neut(...)))` đã ăn 3 tầng.
> - Self-corr ≤ 0.70 là hard gate; chỉ `regression_neut`/`vector_neut` gỡ được.

---

## Bối cảnh (đã xác minh trong source)

Tool đã trưởng thành: closed-loop GP → LLM refine → SIM Brain → feedback, 1543 test pass,
đã nộp alpha thật. 4 nút thắt dưới đây rút ra từ ĐỌC CODE + log chẩn đoán của chính repo
(`logs/diag_combiner_20260712/13.md`, `PROGRESS.md`).

---

## WS1 — Combiner ra ~0 combo: nút là ĐỘ SÂU, không phải tương quan (ưu tiên #1)

### Nguyên nhân gốc (đã xác minh)

- `select_decorrelated_combos` (`src/generation/combiner.py`) seed greedy theo `score`
  (fitness) GIẢM DẦN: `ranked = sorted(signals, key=lambda s: s.score, reverse=True)`.
- Tín hiệu fitness cao nhất trong DB là các biểu thức GP KHỔNG LỒ. Xem
  `logs/diag_combiner_20260713.md` tầng 0: sub-expr #1 fitness 0.7685 là
  `winsorize(subtract(trade_when(ts_std_dev(min(winsorize(open,...),...` — sâu ~12–15 tầng,
  còn kèm hằng số float ngẫu nhiên (`2.8711...`) → dấu hiệu overfit.
- Hệ quả: `select` vẫn tạo được 5 combo THÔ, nhưng ở `combine_stage` (`src/pipeline/
  combine_stage.py`) TẤT CẢ chết ở nhánh `_bump(drop_stats,"depth")`: sau khi
  `build_combined_expression` bọc `rank()` + cây `add` cân bằng, combo N=4 cần MỖI component
  ≤ ~4 tầng, nhưng greedy chọn ĐÚNG các component sâu nhất trước. Log 07-13 combo #1:
  *"RỚT: depth — không dựng được biểu thức lọt trần MAX_DEPTH=7"*.
- `COMBINER_MAX_COMPONENT_DEPTH=4` (`config/thresholds.py`) đã lọc bớt, nhưng nó chỉ lọc
  TRƯỚC greedy chứ KHÔNG đổi thứ tự seed: seed vẫn là fitness cao nhất trong tập đã lọc, và
  4 component mỗi cái depth=4 gộp lại vẫn có thể vượt 7. Nút thật: **combiner tối ưu sai mục
  tiêu — nó chọn theo fitness thay vì theo "khả năng ghép được" (combinability).**

### Tasks

1. **T1.1 — Xếp ứng viên theo combinability, không phải fitness thô.**
   Trong `select_decorrelated_combos`, đổi khóa sort seed/candidate sang lexicographic:
   `key=(depth_bucket_asc, score_desc)` — ưu tiên NÔNG trước, trong cùng bucket độ sâu mới
   xét fitness. Tính depth qua `_depth_of` (đã có). Điều này khiến seed là tín hiệu nông,
   sạch (dễ lọt trần) thay vì monster.
   - TDD: test mới trong `tests/unit/test_combiner.py` — cho 1 tín hiệu fitness cao nhưng
     depth=6 và 2 tín hiệu fitness vừa depth=2: combo phải chọn 2 tín hiệu nông, KHÔNG chọn
     monster.

2. **T1.2 — Dự trù độ sâu theo N thực tế của combo.**
   Cây `add` cân bằng tốn `ceil(log2(N))` tầng + `rank` 1 tầng. Với N=4 → 3 tầng wrapper
   (đúng như comment hiện tại); N=3 → cũng 2 tầng add. Đặt trần component ĐỘNG theo N dự
   kiến: `max_component_depth = MAX_DEPTH - 1 - ceil(log2(n_max))`. Hiện `COMBINER_MAX_
   COMPONENT_DEPTH=4` cố định cho N=4; hãy để `combine_stage` suy nó từ `n_max` thay vì hằng
   số, và thử N nhỏ hơn (3, rồi 2) khi N=4 không lọt — `build_combined_expression` đã có
   vòng giảm N, nhưng nó chỉ bỏ component CUỐI (điểm thấp nhất) chứ không thử lại greedy với
   n_max nhỏ hơn từ đầu.
   - TDD: test combo với các bộ depth khác nhau, assert component-depth-cap suy đúng theo N.

3. **T1.3 — Ưu tiên component ĐÃ chuẩn hóa để tiết kiệm 1 tầng.**
   `_standardize` bỏ qua `rank()` nếu gốc đã là `rank`/`zscore`. Cho combiner ưu tiên (trong
   khóa sort) tín hiệu gốc đã là `rank/zscore/ts_rank` → tiết kiệm tầng bọc, tăng tỉ lệ lọt.
   - TDD: assert giữa 2 tín hiệu cùng depth, bản đã-rank được xếp trước.

4. **T1.4 — Instrument: log phân bố depth của pool combiner mỗi run.**
   Thêm 1 dòng INFO: `combiner pool: n=.. depth p50=.. p90=.. shallow(≤cap)=..%`. Giúp
   nghiệm thu là fix đã đổi phân bố đầu vào chứ không chỉ đổi code.

### Nghiệm thu WS1

- Trên tập tín hiệu của `diag_combiner`, số combo QUA `build_combined_expression` (không
  rớt depth) tăng từ 0 lên ≥ 1. Chạy lại `tools/diag_combiner.py`, `drop_stats["depth"]`
  giảm rõ rệt.
- `pytest -q` xanh; test combiner mới cover T1.1–T1.3.

---

## WS2 — GP đẻ core overfit, sâu, KHÔNG ghép được (đòn bẩy kép với WS1)

### Nguyên nhân gốc (đã xác minh)

- `GPEngine.run` (`src/gp/engine.py:540-542`) trả về best như sau:
  `best = max(evaluated_final, key=lambda i: i.fitness.sharpe_deflated)`. → "best" đưa xuống
  downstream/combiner LUÔN là cá thể sharpe cao nhất, tức thường là **cây sâu nhất, overfit
  nhất** (kèm hằng số float ngẫu nhiên). Đây chính là nguồn monster mà WS1 phải lọc.
- NSGA-II (`src/gp/selection.py`) tối ưu Pareto 6 chiều, trong đó `complexity_penalty` chỉ
  là 1 chiều. Một cá thể sharpe cao + sâu KHÔNG bị dominate bởi cá thể nông hơn (trừ khi nông
  hơn thắng ở MỌI chiều) → monster sống trên Pareto front. Ngoài ra `complexity_penalty =
  complexity / 50` (`src/gp/fitness_vec.py:67`) là ĐẾM NODE, không phải ĐỘ SÂU — mà combiner
  ràng buộc theo ĐỘ SÂU.
- Sinh cây: `MAX_NODES=30` được ràng buộc, nhưng ĐỘ SÂU không bị chặn cứng lúc sinh
  (`src/gp/init.py`) → một cây 30 node dạng chuỗi có thể sâu ~15.

### Tasks

1. **T2.1 — "best" nạp cho combiner phải combinable.**
   Thêm hàm chọn best-cho-combiner: trong top-K theo `sharpe_deflated`, ưu tiên cá thể có
   `depth ≤ COMBINER_MAX_COMPONENT_DEPTH`. Giữ `best_by_sharpe` cũ cho báo cáo, nhưng khi
   feed combiner/DB-good-signals dùng bản combinable-aware. (Xem `GPRunResult`; thêm field
   `best_combinable` hoặc trả thêm danh sách shallow-good.)
   - TDD: `tests/unit/test_gp_engine.py` — population có 1 monster sharpe cao depth 8 và 1 cá
     thể sharpe hơi thấp hơn depth 3: best-cho-combiner chọn cái depth 3.

2. **T2.2 — Chặn ĐỘ SÂU ngay lúc sinh + biến dị (grow/mutation).**
   Thêm hằng `GP_MAX_CORE_DEPTH` vào `config/thresholds.py` (đề xuất = `MAX_DEPTH - 3 = 4`,
   khớp WS1 để core GP luôn ghép được). Ràng buộc trong `src/gp/init.py` (`random_tree`) và
   `src/gp/variation.py` (subtree mutation/crossover) để không sinh/không nhận cây vượt trần.
   Lý do: rẻ hơn nhiều so với sinh-rồi-reject (comment RC5 trong thresholds đã nêu nguyên tắc
   này cho MAX_NODES — làm y hệt cho depth).
   - TDD: `tests/unit/test_gp_init.py` + `test_gp_variation.py` — assert MỌI cây sinh/biến dị
     có `DepthVisitor().visit(...) ≤ GP_MAX_CORE_DEPTH`.

3. **T2.3 — Thêm chiều depth vào parsimony (soft) HOẶC đổi complexity_penalty sang depth.**
   Tối thiểu: bổ sung `depth_penalty = depth / MAX_DEPTH` vào `FitnessVector` (chiều thứ 7,
   minimize) để Pareto ép nông. Cân nhắc A/B: đây là thay đổi hành vi tiến hóa, đo trước khi
   giữ. Nếu ngại tăng chiều Pareto, gộp: `complexity_penalty = max(node/50, depth/MAX_DEPTH)`.
   - TDD: cập nhật `test_gp_fitness_vec.py` cho chiều/công thức mới; test dominance ví dụ.

4. **T2.4 — Rời rạc hằng số float (giảm overfit).**
   `_random_scalar` (`src/gp/init.py:33`) — ép về tập rời rạc nhỏ (vd {-2,-1,-0.5,0.5,1,2}
   hoặc số nguyên nhỏ) thay vì float liên tục. Log cho thấy `2.8711043923756...` = dấu hiệu
   overfit + không diễn giải được. (PROGRESS đã có "hằng số GP rời rạc" ở Pha 1 — xác minh đã
   áp CHƯA; nếu rồi, siết tập giá trị hẹp hơn.)
   - TDD: assert `_random_scalar` chỉ trả giá trị trong tập cho phép.

### Nghiệm thu WS2

- Sau 1 phiên GP local: p90 độ sâu core giảm; % core có `depth ≤ 4` tăng lên đa số.
- Combiner (WS1) nhận pool nông hơn → số combo lọt tăng thêm.
- `pytest -q` xanh.

---

## WS3 — Phá family lock-in (thoát vùng PV-reversal bão hòa)

### Nguyên nhân gốc (đã xác minh)

- Seed frontier/alt-data đã có (`src/generation/frontier_seeds.py`, `alt_data_seeds.py`,
  `fundamental_seeds.py`, `hypothesis_seeds.py`) và định tuyến qua nhánh sim-thẳng
  `LocalTunerRefiner._sim_direct` khi `local_usable(expr,data)==False`
  (`src/app/closed_loop_adapters.py:275,279,487`).
- Nhưng PROGRESS (Session 16) ghi rõ: *"B1-xoay/B2 inert với panel PV 6 field — cần field
  alt-data + filler trong pop mới kích hoạt"*. Tức GP local KHÔNG dùng được field alt-data
  (panel chỉ có 6 field PV) → mọi thứ hội tụ về họ PV reversal đã bão hòa → alpha mới fail
  self-corr với pool.
- Family budget/exhaustion đã có (`src/pipeline/closed_loop.py:211-438`,
  `on_family_closed`) nhưng đó là để ĐÓNG họ cạn ngân sách, KHÔNG đảm bảo SÀN đa dạng cho
  họ mới mỗi batch.

### Tasks

1. **T3.1 — Sàn quota đa dạng mỗi batch.**
   Trong `closed_loop.py`, thêm ràng buộc: mỗi batch dành ≥ `FRONTIER_MIN_FRACTION` (đề xuất
   0.3, vào `config/thresholds.py`) slot cho seed KHÔNG thuộc họ price/volume-reversal
   (frontier/alt-data/fundamental/hypothesis), đi thẳng `_sim_direct`, bất kể panel local có
   field hay không. Mục tiêu: PV-reversal không chiếm trọn batch.
   - TDD: `tests/unit/test_closed_loop.py` — batch có sẵn nhiều candidate PV: assert ≥30%
     slot đi cho nguồn non-PV.

2. **T3.2 — Xoay seed theo độ bão hòa pool (saturation-aware).**
   Đọc pool hiện tại; nếu họ X đã chiếm >K% pool (đo qua `src/scoring/dataset_usage.py`
   hoặc AST family_fn), HẠ ưu tiên seed họ X ở batch kế. Tránh tiếp tục đào vùng đã mine.
   - TDD: seed rotation ưu tiên họ ít xuất hiện trong pool.

3. **T3.3 — Xác minh field alt-data LIVE trước khi seed (cardinal rule #1).**
   Đảm bảo `tools/verify_frontier_fields.py` / `verify_datasets.py` chạy và blacklist field
   không tồn tại; seed dùng field đã verify. (Đã có hạ tầng — chỉ cần chắc nó chạy trong
   đường closed-loop, không phải chỉ chạy tay.)
   - TDD: seed chứa field chưa-verify bị loại/blacklist, không xuống sim.

### Nghiệm thu WS3

- 1 phiên menu-5: ≥60% ý tưởng KHÔNG thuộc pv_reversal; ≥3 họ khác nhau được sim
  (khớp acceptance đang treo trong PROGRESS).
- Self-corr trung bình của candidate giảm so với baseline PV-only.

---

## WS4 — Tăng độ tin cậy calibration local↔Brain (giảm giết oan / phí quota)

### Nguyên nhân gốc (đã xác minh)

- Gate pre-sim dựa vào Sharpe local với hệ số `CALIBRATION_LOCAL_TO_BRAIN=1.28`
  (`config/thresholds.py`). Nhưng PROGRESS Session 09: *"ρ=0.308 local không tin"* — dưới
  `CALIBRATION_RHO_BAR=0.5`. Ranking local yếu → floor local giết oan alpha tốt hoặc thả rác.
- `CalibrationHarness` (`src/calibration/harness.py`) re-score expr từ DB, đo Spearman ρ của
  **Sharpe local vs Sharpe Brain**. Hai điểm cải thiện: (a) chỉ đo Sharpe, không đo trục nộp
  thật (fitness); (b) mẫu ghép ít + config re-score phải khớp config Brain mới hợp lệ.

### Tasks (một phần cần DỮ LIỆU LIVE — đánh dấu ⚠)

1. **T4.1 — Đổi mục tiêu tương quan sang ĐIỂM-NỘP, không phải Sharpe thô.**
   Trong harness, ngoài ρ(Sharpe) hãy đo thêm ρ của `submit_score = min(sharpe/SUBMIT_
   SHARPE_REF, fitness/SUBMIT_FITNESS_REF)` (khớp `combine_stage._submit_score`). Gate/floor
   nên tin trục dự đoán "qua ngưỡng nộp" hơn là Sharpe đơn.
   - TDD: `tests/unit/test_calibration_*` — thêm case tính ρ trên submit_score.

2. **T4.2 — Calibration theo họ (per-family).**
   Đo ρ và hệ số local→Brain RIÊNG cho từng họ (PV vs alt-data vs fundamental). Hệ số 1.28
   đo trên winner PV có thể sai cho fundamental (sparse, ts_backfill). Lưu bảng hệ số theo họ
   vào DB/`config`; `calibrated_floor` nhận `family` để suy floor đúng.
   - TDD: harness trả dict theo family; `calibrated_floor(family=...)` dùng hệ số đúng.

3. **T4.3 — ⚠ Thu thập mẫu ghép + báo cáo.**
   Cần USER chạy menu-5 để tích thêm cặp (local, Brain). Thêm lệnh/`report` in ρ tổng và
   theo họ, n mẫu, và cảnh báo khi n<30 (ρ chưa đáng tin). KHÔNG siết floor tự động khi ρ còn
   thấp — chỉ báo cáo. Sau khi ρ≥0.5, mới cho phép hạ `PRE_SIM_TARGET_BRAIN_SHARPE` để tiết
   kiệm quota.
   - Nghiệm thu bước này là SỐ LIỆU, không phải code — ghi vào `logs/`.

### Nghiệm thu WS4

- `calibrate`/`report` in ρ tổng + theo họ + n mẫu + cờ cảnh báo n nhỏ.
- Không đổi ngưỡng gate cho tới khi ρ≥`CALIBRATION_RHO_BAR` trên n đủ lớn (kỷ luật:
  đừng tin ranking local khi chưa chứng minh).

---

## Thứ tự & rủi ro

| WS | Đòn bẩy | Rủi ro | Phụ thuộc |
|----|---------|--------|-----------|
| WS1 Combiner depth | Cao (gỡ 0-combo) | Thấp (logic thuần, test kỹ) | — |
| WS2 GP parsimony | Cao (cải thiện nguồn cho WS1) | Trung (đổi hành vi tiến hóa → A/B T2.3) | tăng lực cho WS1 |
| WS3 Family lock-in | Cao (nguồn mới) | Trung (cần verify field live) | — |
| WS4 Calibration | Trung (tiết kiệm quota) | Thấp code, nhưng cần DATA live | cần menu-5 |

**Khuyến nghị chạy:** WS1 → WS2 (hai cái này ăn khớp: WS2 làm core nông hơn → WS1 ghép được
nhiều hơn), commit + `pytest -q`. Rồi WS3. WS4 làm phần code (T4.1/T4.2) song song, nhưng
CHỜ user chạy menu-5 để có số cho T4.3 trước khi đụng ngưỡng.

**Sau mỗi WS:** `pytest -q` (giữ ≥1543 passed), cập nhật `PROGRESS.md` 1 entry, và với WS3/WS4
cần user chạy menu-5 nghiệm thu live (không tự tuyên bố xong khi chưa có số liệu thật).
