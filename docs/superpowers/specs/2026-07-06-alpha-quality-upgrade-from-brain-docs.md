# Nâng cấp chất lượng alpha từ docs WorldQuant Brain (2026-07-06)

Tổng hợp từ việc đọc toàn bộ `docs/worldquantbrain/docs` (74 file) bằng 4 sub-agent
song song. Mục tiêu: biến kiến thức docs thành nâng cấp CỤ THỂ cho engine MiniBrain
(TOP3000 US, Delay 1). Đây là tài liệu tham chiếu + roadmap; phần đã triển khai được
đánh dấu ✅.

## 1. Bảng ngưỡng submission THẬT (khắc vào code, đổi theo cuộc thi)

Nguồn: `interpret-results/*`, `consultant-information/consultant-submission-tests.md`,
`consultant-information/finding-consultant-alphas.md`.

| Test | Ngưỡng (Delay 1) | Lever sửa |
|---|---|---|
| Max weight 1 mã | `< 10%` (USA consultant khuyến nghị `< 8%`) | tăng truncation 0.05–0.1, `winsorize`, tăng breadth |
| Concentration / breadth | không mã nào ~30% tổng weight; đủ số mã có weight/năm | tăng coverage, rank/scale rộng, `ts_backfill` |
| Self-correlation | PnL corr `< 0.7` với pool user (cửa sổ 4 năm); HOẶC Sharpe cao hơn ≥10% mọi alpha corr>0.7 | `regression_neut`/`vector_neut`; dataset lạ |
| Prod-correlation (consultant) | như trên nhưng so **toàn bộ** alpha đã nộp trên Brain | chỉ check được qua API trước nộp |
| Fitness | `> 1.0` | `Fitness = Sharpe·√(|Return|/max(Turnover,0.125))` |
| Sharpe (toàn kỳ) | user `> 1.25`; consultant `> 1.58` | — |
| Turnover band | `1% < TO < 70%` (SuperAlpha 2%–40%) | decay/hump |
| IS-Ladder Sharpe | FAIL=**1.58**; PASS năm 2–5=**2.38**, 6=2.22, 7=2.06, 8=1.90, 9=1.74, 10=1.59; **TO<30% ⇒ PASS×0.85** | Sharpe bền ở cửa sổ gần |
| Sub-Universe | `sub_sharpe ≥ 0.75·√(sub_size/univ_size)·alpha_sharpe`; TOP3000→TOP1000 ≈ **0.433·alpha_sharpe** | tránh size-multiplier, decay tách liquid |
| Super-Universe | Sharpe universe lớn hơn `≥ 0.7·Sharpe` | ổn định cross-universe |
| RankSharpe | `> 0.5·Sharpe` hoặc `> 0.15` (2 năm gần) | tránh lệch một phía |
| Bias (forward) | không look-ahead | delay-1 đúng |

Công thức phụ: `Sharpe = √252·IR`, `IR = mean(PnL)/std(PnL)`; `Return = ann.PnL/(½ book)`,
book \$20M; `Margin = PnL/tổng dollar traded`; `Drawdown = max peak-to-trough/(½ book)`.

Chú ý kiến trúc: submission thật do **WQ Brain tự chấm** khi SIM (`status=="passed"`).
Gate LOCAL của MiniBrain là **pre-filter tiết kiệm quota** — càng khớp bảng trên càng
ít đốt quota vào alpha Brain sẽ loại. Panel local `market_yf` **chỉ có PV** nên alpha
dataset-thay-thế thuộc **path Brain sim**, không chấm local được.

## 2. Neutralization & risk-neutralization (đòn bẩy Sharpe + self-corr)

- **Chọn group theo CATEGORY tín hiệu**, không hardcode SUBINDUSTRY:
  Price-Volume/Option → MARKET/SECTOR; Fundamental/Analyst/Earnings → INDUSTRY;
  News/Social/Sentiment → SUBINDUSTRY. (`advanced-topics/neutralization.md`)
- **TOP3000 là universe thanh khoản** → ưu tiên **nhóm lớn** (INDUSTRY/SECTOR) để mỗi
  nhóm đủ số cổ phiếu; SUBINDUSTRY chỉ hợp News/Sentiment. Đây là mâu thuẫn có chủ đích
  với default SUBINDUSTRY hiện tại — xử lý bằng mapping theo category thay vì lật cứng.
- **Risk-neutralization** (Slow/Fast/Slow+Fast=RAM, Statistical, Crowding): tạo return
  trực giao market/industry/style → giảm self-corr, tăng robustness. Chọn set theo
  turnover (thấp→Slow, cao→Fast); RAM tốt cho fundamental có mẫu số giá (bóc momentum
  lẫn trong `eps/close`). Turnover TĂNG sau risk-neut → phải đo lại band.
- **Retained-Sharpe ratio** = `sharpe_sau_riskneut/sharpe_raw`: alpha giữ nhiều Sharpe
  hơn = ít phụ thuộc style factor, ít corr → dùng làm tie-breaker (ưu tiên ≥0.5).
- **Double-neut đúng cách**: KHÔNG lồng hai `group_neutralize` (triệt tiêu một phần);
  dùng `densify(group_cartesian_product(sector, sta1_top1000c50))`.
- `winsorize(x, std=4)` để qua gate weight-concentration. ✅ (operator đã có sẵn local)

## 3. Chống crowding: dataset thay thế + cấu trúc lạ (originality)

Bài học cốt lõi (`understanding-data/*`, `examples/*`, `discover-brain/*`):
**"alpha tốt sống trong GAP/GATE/RESIDUAL, không phải LEVEL"**. Độ mới đến từ nguồn
dữ liệu ít khai thác + cấu trúc bậc cao, không từ đổi operator trên `close/returns`.

Nguồn ít crowd: earnings4 (`ern4_*`), option6 (`opt6_*`, IV skew/slope/term), sentiment1
(`snt1_*`), analyst-estimate (`est_*`), news vector (`nws*`, `vec_*`), supply-chain (pv13).

Ba khuôn mẫu cấu trúc giá trị cao:
1. **GAP/spread hai chuỗi đồng họ**: `(A_có − A_không)`, `ts_zscore(A) − ts_zscore(B)`
   (vd IV có/không earnings; forecast − implied; op-cashflow − capex).
2. **GATE bằng `trade_when`**: bật tín hiệu theo event/confidence/coverage
   (`trade_when(condition, signal, -1)`) → profile lạ, giảm self-corr + turnover.
3. **RESIDUAL bằng `vector_neut`**: `vector_neut(signal, beta)` orthogonalize thủ công
   với beta/factor phổ biến → hạ self-corr trực tiếp.

Ràng buộc: field mới PHẢI verify tồn tại thật trước khi dùng (cardinal rule #1). Cho tới
khi verify được `ern4_*`/`opt6_slope*`/`est_*` qua đăng nhập thật, ta áp 3 khuôn mẫu trên
lên **`VERIFIED_FIELDS` đã có** (option-IV, news, social, analyst-sentiment, supply-chain).

## 4. Anti-overfit & chất lượng consultant

- Khóa tham số `ts_*` vào {5,20,60,120,252} (tránh 14/37...); hạn chế `if_else` lồng;
  **cấm thêm "noise term" chỉ để hạ self-corr** (`consultant-dos-and-don-ts.md`).
- Alpha phải có **nền tảng kinh tế giải thích trong 1 phút**; Power Pool bắt buộc mô tả
  Idea/Rationale ≥100 ký tự.
- `ts_backfill`/`group_backfill` đảm bảo coverage (không tính vào giới hạn 8 operator
  của Power Pool).
- Lái **đa dạng theme/dataset/region ("pyramid")** để tăng payout + giảm prod-corr.

## 5. Profile sinh alpha theo lớp cấp vốn (roadmap)

- **Single-Dataset (ATOM)**: mọi field (trừ 6 grouping) cùng 1 dataset; chỉ cần **2Y
  Sharpe ≥ 2.38** (×0.85 nếu TO<30%), không cần full ladder — dễ pass, ít overfit.
- **Power Pool**: ≤8 operator (không đếm backfill), ≤3 datafield không-grouping,
  Sharpe≥1.0, pool-corr<0.5, ưu tiên low-turnover + universe thanh khoản.

## 6. Roadmap ưu tiên (impact × feasibility)

| # | Hạng mục | Nguồn | Trạng thái |
|---|---|---|---|
| 1 | IS-Ladder robustness gate (Sharpe cửa sổ trượt) | A2#1, A3#1 | Commit 1 (phiên này) |
| 2 | Novel-ideas v2: gap/gate/residual trên VERIFIED_FIELDS | A4 | Commit 2 (phiên này) |
| 3 | Neutralization theo category (bỏ SUBINDUSTRY cứng) | A1#1/#2, A4#6 | Commit 2 (một phần) |
| 4 | Sub-Universe stability gate (TOP1000 heuristic) | A2#3, A3#5 | TODO |
| 5 | Hash-cache chống SIM trùng (nếu chưa có) | A2#5 | TODO (kiểm result_cache) |
| 6 | Self-corr exception Sharpe+10% | A2#9, A3#6 | TODO |
| 7 | Anti-overfit: khóa param GP {5,20,60,120,252} | A3#4 | TODO |
| 8 | Profile Single-Dataset (ATOM) + Power Pool gates | A3#2/#3 | TODO (power_pool đã có một phần) |
| 9 | Neutralization là trục config-sweep mặc định | A1#11, A2#8 | TODO |
| 10 | Risk-neut set + retained-Sharpe ratio (path Brain) | A1#3/#4 | TODO |

(A1..A4 = 4 sub-agent: neutralization / submission-tests / consultant-quality / dataset-seed.)
