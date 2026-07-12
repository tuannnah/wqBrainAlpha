# Chẩn đoán CombinerIdeaSource — 0 combo

Chạy lúc: 2026-07-13

> **Lưu ý phạm vi**: script này CHỈ tái hiện nhánh tín hiệu lấy từ DB (`repo.good_signals_for_combine`, `source="db"`). Ở production, `CombinerIdeaSource.next_batch()` còn trộn thêm tín hiệu `source="run"` phát sinh ngay trong batch hiện tại — nên các combo #N liệt kê dưới đây KHÔNG nhất thiết trùng với combo mà production từng dựng trên cùng batch đó; đây là chẩn đoán offline trên một tập con tín hiệu, không phải replay chính xác 1-1.

## Bước 0 — Dựng môi trường (DB / panel / config)

- Tài khoản active (`.wq_account`): `tuananhpo13@gmail.com`
- DB URL: `sqlite:///wq_alpha_tuananhpo13_gmail_com.db`
- MarketData dir: `data\market_yf`
- Config local gate: neutralization=MARKET (yêu cầu=MARKET), decay=4, truncation=0.08, delay=1, region=USA, universe=TOP3000
- Panel: 2618 ngày, groups có sẵn = ['sector']

## Tầng 0 — `repo.good_signals_for_combine(limit=50)`

- Số tín hiệu lấy được: **50** (limit=50)
- Phân bố fitness: n=50 min=0.2234 p25=0.2392 median=0.2756 p75=0.3376 max=0.7685 mean=0.3178

10 tín hiệu đầu (đã sort fitness giảm dần):

| # | fitness | expr |
|---|---|---|
| 1 | 0.7685 | `winsorize(subtract(trade_when(ts_std_dev(min(winsorize(open, -1.9423623924877862), log(returns)), 5), subtract(sign(hump(vwap, -0.3570296940325539)), min(max(volume, volume), log(high))), ts_zscore(subtract(winsorize(open, 1.8135773868025726), ts_sum(volume, 60)), 120)), ts_rank(ts_std_dev(ts_sum(abs(volume), 10), 10), 60)), 2.8711043923756696)` |
| 2 | 0.6486 | `group_neutralize(volume, sector)` |
| 3 | 0.5833 | `group_neutralize(divide(subtract(ts_mean(divide(open, vwap), 20), ts_rank(max(volume, low), 120)), subtract(ts_corr(multiply(open, volume), power(close, 1.2160545479872145), 20), multiply(winsorize(vwap, -2.8542600547822197), ts_mean(low, 60)))), sector)` |
| 4 | 0.5254 | `group_neutralize(multiply(-1, rank(close)), sector)` |
| 5 | 0.5141 | `group_neutralize(multiply(-1, rank(open)), sector)` |
| 6 | 0.5119 | `ts_rank(ts_mean(winsorize(zscore(ts_std_dev(open, 20)), -2.727073635360348), 60), 60)` |
| 7 | 0.4367 | `group_neutralize(ts_std_dev(returns, 20), sector)` |
| 8 | 0.3969 | `group_neutralize(abs(multiply(ts_std_dev(volume, 20), ts_rank(close, 60))), sector)` |
| 9 | 0.3697 | `sign(ts_mean(subtract(high, low), 20))` |
| 10 | 0.3668 | `group_neutralize(multiply(-1, rank(ts_std_dev(rank(ts_std_dev(returns, 120)), 120))), sector)` |

## Tầng 1 — `select_decorrelated_combos(tau=0.3, n_min=2, n_max=4, max_combos=5)`

- Số combo THÔ (trước dựng biểu thức/gate): **5**
  - combo #1: 4 tín hiệu — `winsorize(subtract(trade_when(ts_std_dev(min(winsorize(open, -1.9423623924877862), log(returns)), 5), subtract(sign(hump(vwap, -0.3570296940325539)), min(max(volume, volume), log(high))), ts_zscore(subtract(winsorize(open, 1.8135773868025726), ts_sum(volume, 60)), 120)), ts_rank(ts_std_dev(ts_sum(abs(volume), 10), 10), 60)), 2.8711043923756696)`, `ts_rank(ts_mean(winsorize(zscore(ts_std_dev(open, 20)), -2.727073635360348), 60), 60)`, `group_neutralize(abs(multiply(ts_std_dev(volume, 20), ts_rank(close, 60))), sector)`, `sign(ts_mean(subtract(high, low), 20))`
  - combo #2: 4 tín hiệu — `group_neutralize(volume, sector)`, `group_neutralize(divide(subtract(ts_mean(divide(open, vwap), 20), ts_rank(max(volume, low), 120)), subtract(ts_corr(multiply(open, volume), power(close, 1.2160545479872145), 20), multiply(winsorize(vwap, -2.8542600547822197), ts_mean(low, 60)))), sector)`, `add(multiply(2, multiply(-1, ts_mean(subtract(close, vwap), 10))), group_neutralize(rank(ts_delta(close, 120)), sector))`, `multiply(-1, ts_decay_linear(rank(sign(returns)), 5))`
  - combo #3: 4 tín hiệu — `group_neutralize(multiply(-1, rank(close)), sector)`, `rank(ts_corr(rank(ts_zscore(volume, 10)), volume, 20))`, `multiply(-1, ts_delta(subtract(close, ts_mean(close, 20)), 10))`, `rank(ts_zscore(power(ts_backfill(ts_std(low, 120), 120), 1.8372428844748763), 20))`
  - combo #4: 4 tín hiệu — `group_neutralize(multiply(-1, rank(open)), sector)`, `multiply(-1, ts_mean(subtract(close, vwap), 10))`, `group_neutralize(multiply(-1, rank(ts_std_dev(ts_delta(ts_zscore(close, 250), 1), 120))), sector)`, `divide(volume, ts_std(low, 120))`
  - combo #5: 4 tín hiệu — `group_neutralize(ts_std_dev(returns, 20), sector)`, `multiply(multiply(-1, rank(ts_backfill(ts_std(subtract(high, low), 60), 5))), rank(ts_zscore(returns, 10)))`, `multiply(vwap, ts_mean(subtract(open, vwap), 20))`, `group_neutralize(power(high, -2.3440390798578923), sector)`

## Tầng 2/3/4 — dựng biểu thức, chấm gate (pool=1350 thành viên), so fitness với sub-expr tốt nhất

- `repo.load_pool()` trả **1350** thành viên (PoolPnlModel).

### combo #1 — 4 tín hiệu

- sub-expr (fitness=0.7685, source=db): `winsorize(subtract(trade_when(ts_std_dev(min(winsorize(open, -1.9423623924877862), log(returns)), 5), subtract(sign(hump(vwap, -0.3570296940325539)), min(max(volume, volume), log(high))), ts_zscore(subtract(winsorize(open, 1.8135773868025726), ts_sum(volume, 60)), 120)), ts_rank(ts_std_dev(ts_sum(abs(volume), 10), 10), 60)), 2.8711043923756696)`
- sub-expr (fitness=0.5119, source=db): `ts_rank(ts_mean(winsorize(zscore(ts_std_dev(open, 20)), -2.727073635360348), 60), 60)`
- sub-expr (fitness=0.3969, source=db): `group_neutralize(abs(multiply(ts_std_dev(volume, 20), ts_rank(close, 60))), sector)`
- sub-expr (fitness=0.3697, source=db): `sign(ts_mean(subtract(high, low), 20))`
- **RỚT: depth** — không dựng được biểu thức lọt trần độ sâu MAX_DEPTH=7.

### combo #2 — 4 tín hiệu

- sub-expr (fitness=0.6486, source=db): `group_neutralize(volume, sector)`
- sub-expr (fitness=0.5833, source=db): `group_neutralize(divide(subtract(ts_mean(divide(open, vwap), 20), ts_rank(max(volume, low), 120)), subtract(ts_corr(multiply(open, volume), power(close, 1.2160545479872145), 20), multiply(winsorize(vwap, -2.8542600547822197), ts_mean(low, 60)))), sector)`
- sub-expr (fitness=0.3398, source=db): `add(multiply(2, multiply(-1, ts_mean(subtract(close, vwap), 10))), group_neutralize(rank(ts_delta(close, 120)), sector))`
- sub-expr (fitness=0.3338, source=db): `multiply(-1, ts_decay_linear(rank(sign(returns)), 5))`
- Biểu thức ghép: `add(rank(volume), rank(divide(subtract(ts_mean(divide(open, vwap), 20), ts_rank(max(volume, low), 120)), subtract(ts_corr(multiply(open, volume), power(close, 1.2160545479872145), 20), multiply(winsorize(vwap, -2.8542600547822197), ts_mean(low, 60))))))` (dùng 2/4 sub-expr)
- **RỚT: gate** — verdict.passed=False. Lý do: self_corr 0.860 >= SELF_CORR_MAX 0.7
  - self_corr thật với pool: 0.8601 (thành viên pool evaluation_id=90, expr=`group_neutralize(multiply(-1, rank(close)), sector)`)
  - thành viên pool trùng nhất là một alpha KHÁC trong pool (không phải sub-expr của combo này) — nghi ngờ pool bão hòa (saturation) chứ không chỉ tự trùng đầu vào.
  - |rho| combo vs sub-expr `group_neutralize(volume, sector)`: 0.3160
  - |rho| combo vs sub-expr `group_neutralize(divide(subtract(ts_mean(divide(open, vwap), 20), ts_rank(max(volume, low), 120)), subtract(ts_corr(multiply(open, volume), power(close, 1.2160545479872145), 20), multiply(winsorize(vwap, -2.8542600547822197), ts_mean(low, 60)))), sector)`: 0.6317
  - |rho| combo vs sub-expr `add(multiply(2, multiply(-1, ts_mean(subtract(close, vwap), 10))), group_neutralize(rank(ts_delta(close, 120)), sector))`: 0.1111
  - |rho| combo vs sub-expr `multiply(-1, ts_decay_linear(rank(sign(returns)), 5))`: 0.0338

### combo #3 — 4 tín hiệu

- sub-expr (fitness=0.5254, source=db): `group_neutralize(multiply(-1, rank(close)), sector)`
- sub-expr (fitness=0.3134, source=db): `rank(ts_corr(rank(ts_zscore(volume, 10)), volume, 20))`
- sub-expr (fitness=0.3088, source=db): `multiply(-1, ts_delta(subtract(close, ts_mean(close, 20)), 10))`
- sub-expr (fitness=0.3047, source=db): `rank(ts_zscore(power(ts_backfill(ts_std(low, 120), 120), 1.8372428844748763), 20))`
- Biểu thức ghép: `add(add(rank(multiply(-1, rank(close))), rank(ts_corr(rank(ts_zscore(volume, 10)), volume, 20))), rank(multiply(-1, ts_delta(subtract(close, ts_mean(close, 20)), 10))))` (dùng 3/4 sub-expr)
- **RỚT: gate** — verdict.passed=False. Lý do: self_corr 0.702 >= SELF_CORR_MAX 0.7
  - self_corr thật với pool: 0.7021 (thành viên pool evaluation_id=5595, expr=`multiply(rank(ts_sum(returns, 5)), rank(ts_std_dev(ts_delta(close, 1), 120)))`)
  - thành viên pool trùng nhất là một alpha KHÁC trong pool (không phải sub-expr của combo này) — nghi ngờ pool bão hòa (saturation) chứ không chỉ tự trùng đầu vào.
  - |rho| combo vs sub-expr `group_neutralize(multiply(-1, rank(close)), sector)`: 0.6910
  - |rho| combo vs sub-expr `rank(ts_corr(rank(ts_zscore(volume, 10)), volume, 20))`: 0.2791
  - |rho| combo vs sub-expr `multiply(-1, ts_delta(subtract(close, ts_mean(close, 20)), 10))`: 0.4381
  - |rho| combo vs sub-expr `rank(ts_zscore(power(ts_backfill(ts_std(low, 120), 120), 1.8372428844748763), 20))`: 0.0439

### combo #4 — 4 tín hiệu

- sub-expr (fitness=0.5141, source=db): `group_neutralize(multiply(-1, rank(open)), sector)`
- sub-expr (fitness=0.3388, source=db): `multiply(-1, ts_mean(subtract(close, vwap), 10))`
- sub-expr (fitness=0.3041, source=db): `group_neutralize(multiply(-1, rank(ts_std_dev(ts_delta(ts_zscore(close, 250), 1), 120))), sector)`
- sub-expr (fitness=0.3024, source=db): `divide(volume, ts_std(low, 120))`
- Biểu thức ghép: `add(rank(multiply(-1, rank(open))), rank(multiply(-1, ts_mean(subtract(close, vwap), 10))))` (dùng 2/4 sub-expr)
- **RỚT: gate** — verdict.passed=False. Lý do: self_corr 0.769 >= SELF_CORR_MAX 0.7
  - self_corr thật với pool: 0.7693 (thành viên pool evaluation_id=90, expr=`group_neutralize(multiply(-1, rank(close)), sector)`)
  - thành viên pool trùng nhất là một alpha KHÁC trong pool (không phải sub-expr của combo này) — nghi ngờ pool bão hòa (saturation) chứ không chỉ tự trùng đầu vào.
  - |rho| combo vs sub-expr `group_neutralize(multiply(-1, rank(open)), sector)`: 0.7632
  - |rho| combo vs sub-expr `multiply(-1, ts_mean(subtract(close, vwap), 10))`: 0.6641
  - |rho| combo vs sub-expr `group_neutralize(multiply(-1, rank(ts_std_dev(ts_delta(ts_zscore(close, 250), 1), 120))), sector)`: 0.1302
  - |rho| combo vs sub-expr `divide(volume, ts_std(low, 120))`: 0.0396

### combo #5 — 4 tín hiệu

- sub-expr (fitness=0.4367, source=db): `group_neutralize(ts_std_dev(returns, 20), sector)`
- sub-expr (fitness=0.3152, source=db): `multiply(multiply(-1, rank(ts_backfill(ts_std(subtract(high, low), 60), 5))), rank(ts_zscore(returns, 10)))`
- sub-expr (fitness=0.2996, source=db): `multiply(vwap, ts_mean(subtract(open, vwap), 20))`
- sub-expr (fitness=0.2794, source=db): `group_neutralize(power(high, -2.3440390798578923), sector)`
- **RỚT: depth** — không dựng được biểu thức lọt trần độ sâu MAX_DEPTH=7.

## Tổng kết theo tầng

- Combo thô (Tầng 1): 5
- RỚT Tầng 2 (depth): 2
- RỚT Tầng 3 (gate) do self_corr pool: 3
- RỚT Tầng 3 (gate) lý do khác: 0
- RỚT Tầng 4 (không vượt trội): 0
- QUA HẾT: 0
- self_corr trung bình của các combo rớt vì pool-corr: 0.7772 (ngưỡng SELF_CORR_MAX=0.70), min=0.7021 max=0.8601

## Kết luận

**Cả 5/5 combo thô đều chết — do HAI tầng lọc cộng hưởng, không phải một tầng duy nhất:**

1. **Tầng 2 (depth)**: 2/5 combo không dựng nổi biểu thức lọt MAX_DEPTH=7 — chết trước khi kịp tới gate.
2. **Tầng 3 (gate, self_corr pool)**: trong số combo VƯỢT qua Tầng 2, **3/3** (100%) rớt vì `self_corr >= SELF_CORR_MAX=0.70` so với `repo.load_pool()`.

   self_corr đo được: 0.8601, 0.7021, 0.7693 (đều vượt ngưỡng 0.70, cách xa từ +0.002 đến +0.160).

**Kiểm chứng giả thuyết trong brief** ("combo tương quan với chính sub-signal của nó trong pool") — ĐÃ ĐO TRỰC TIẾP, KHÔNG đúng hoàn toàn như hình dung ban đầu: trong 3 ca rớt vì self_corr, **0** ca thành viên pool trùng nhất CHÍNH LÀ một sub-expr của combo đó, còn **3** ca thành viên trùng nhất là một alpha KHÁC hẳn trong pool (không nằm trong combo). Tức nguyên nhân thật sự rộng hơn giả thuyết ban đầu: **pool 1350 thành viên đã BÃO HÒA** (dày đặc các biến thể `group_neutralize(rank(...))`/`multiply(-1, rank(...))` trên price/volume) — combo `add(rank(s1), rank(s2))` mới dựng, dù 2 sub-expr chọn qua greedy đã <0.30 tương quan VỚI NHAU, vẫn gần như chắc chắn rơi vào bán kính 0.70 của MỘT thành viên nào đó trong 1350 alpha đã có sẵn — vì combiner chỉ khử tương quan trong nội bộ ~50 ứng viên (`select_decorrelated_combos`), KHÔNG hề kiểm tra trước tương quan với TOÀN BỘ pool đã tích lũy.


### Fix đề xuất cho Task 2 (theo thứ tự ưu tiên, dựa trên số liệu đo được)

1. **Pool-decorrelation SỚM, trước khi tốn công dựng+chấm combo**: sau `build_combined_expression`, tính `PoolCorrelation(pool).max_corr(candidate_pnl, dates)` NGAY (rẻ hơn `_score_one_full` đầy đủ) và bỏ combo sớm nếu vượt ngưỡng — tiết kiệm nhưng KHÔNG tự nó tăng số combo qua được (đây chỉ là early-exit, không phải fix triệt để).
2. **Fix triệt để hơn — chọn tín hiệu ít tương quan với CẢ POOL, không chỉ với nhau**: `select_decorrelated_combos` hiện chỉ dùng `pairwise_abs_rho` GIỮA các candidate. Sửa để mỗi candidate (hoặc mỗi combo ứng viên) phải có `PoolCorrelation(pool).max_corr(...) < SELF_CORR_MAX` TRƯỚC khi được chọn vào combo — loại các signal 'phổ biến' (gần giống nhiều alpha đã pass) khỏi vai trò thành phần combo ngay từ khâu chọn, thay vì phát hiện muộn ở gate.
3. **Loại trừ sub-expr CHÍNH nó khỏi pool khi chấm gate** (vẫn nên làm dù không phải nguyên nhân chính trong lần đo này — 0/3 ca là tự trùng thật): `good_signals_for_combine` trả thêm evaluation_id, `combine_stage` dựng `pool_excl = {k: v for k, v in pool.items() if k not in combo_evaluation_ids}` trước khi gọi gate.
4. **Tầng depth**: ưu tiên sub-expr độ sâu thấp khi greedy chọn combo (đo `_depth_of` mỗi candidate trước, không chỉ sau khi build thất bại) để giảm tỉ lệ 2/5 chết vì depth trước khi tới gate.

KHÔNG hạ `SELF_CORR_MAX` — đó là ngưỡng an toàn thật ánh xạ tới self-correlation checker của Brain, hạ ngưỡng sẽ tạo alpha nộp lên chắc chắn bị Brain từ chối.
