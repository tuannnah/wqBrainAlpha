# MiniBrain tích hợp vào tool hiện tại — Design Spec

> **Ngày:** 2026-06-23
> **Nguồn:** `docs/design/MINIBRAIN_DESIGN.md` (master spec) + codebase hiện tại (`src/`, `main.py`).
> **Mục tiêu:** Tích hợp dần MiniBrain (local pre-filter cho WQ Brain) vào tool sẵn có,
> tái dùng tối đa module cũ, xóa phần bị thay thế. Vẫn dùng AI (DeepSeek / claude-cli /
> codex-cli) để sinh alpha. Code mới đi thẳng nhánh `main`, mỗi phase = 1 nhánh → merge → push.

---

## 1. Bối cảnh & quyết định đã chốt

Tool hiện tại đã có gần đủ hạ tầng: đăng nhập WQ, fetch metadata, DB (SQLite/PG), sim API +
rate limit, pipeline AI sinh alpha (`RefinementLoop`), scoring 6 chiều, submit + correlation
check, menu PowerShell, dashboard. **Cái còn thiếu so với MINIBRAIN_DESIGN là tầng "local
backtester"**: parser FASTEXPR → evaluator → portfolio → backtest → metrics local →
pool-correlation local → calibration → GP engine.

Giá trị: hiện mỗi lần kiểm tra ý tưởng đều **đốt 1 lượt sim Brain** (chậm, có quota).
MiniBrain biến đa số bộ lọc thành **cục bộ, miễn phí, không quota** — chỉ alpha vượt local
gate mới được đốt sim thật.

**Quyết định (từ brainstorming với user):**

| # | Quyết định |
|---|---|
| D1 | Phạm vi: **full MiniBrain** tích hợp dần — local pre-filter + pool-corr + calibration + GP. |
| D2 | Market data: **kéo từ WQ Brain API** (`/datasets/{id}/data`) cho PIT-faithful, khớp universe TOP3000/Delay-1. Chấp nhận chậm để đổi lấy ρ cao. |
| D3 | Engine sinh alpha: **build GP mới theo spec B13** (typed AST, depth cap ≈7, NSGA-II, fitness 6 chiều correlation-aware). AI (LLM) đóng vai **seed + refine**, không bị bỏ. |
| D4 | Code layout: **code mới đi thẳng `src/`**, tái dùng cái nào dùng được, **xóa** phần thừa. Mỗi phase = 1 nhánh git → test xanh → merge `main` → push. |
| D5 | Thứ tự build: **theo đúng Part E** của master spec (Phase 0 → 9). |
| D6 | Cải tiến các engine liên quan tới **sinh alpha** theo MINIBRAIN_DESIGN (stage separation, correlation-aware, deflated/regime fitness). |
| D7 | Xóa file rác ở root (`new 1.txt`, `llm_bridge_run.log`). |
| D8 | `src/generation/ast_utils.py` + `template.py` xóa **sau khi** `src/lang/` + `src/gp/` thay thế xong (Phase 1 + 7 verify rồi xóa). |
| D9 | **Bỏ đường cũ** (LLM → sim Brain trực tiếp). Chỉ còn **một pipeline thống nhất** đi qua local pre-filter; mọi candidate phải vượt local gate mới được đốt sim. Đường cũ giữ tạm làm fallback **chỉ tới hết Phase 2**, **gỡ tại Phase 3** khi local gate đã chạy được (nếu gỡ ngay từ Phase 0 thì tool không còn đường nào chạy trong lúc build). |

---

## 2. Kiến trúc tổng thể

Sau tích hợp, tool có **một pipeline alpha thống nhất** (D9 — đã bỏ đường cũ), dùng chung lõi
hạ tầng (login / AI / DB / submit):

```
nguồn candidate (LLM hypothesis→translator  ·  GP  ·  NOVEL/factor families)
  → parse FASTEXPR → eval local (T,N) → portfolio + backtest local → metrics local
  → gate (hard + pool_corr local 0.70)            ← LỌC CỤC BỘ, MIỄN PHÍ, KHÔNG QUOTA
  → CHỈ alpha vượt gate mới đốt SIM Brain → score 6 chiều → submit
  → calibrate ρ định kỳ · GP sinh batch lớn
```

**Không còn nhánh "LLM → sim Brain trực tiếp".** Mọi candidate — dù từ LLM, GP, hay
NOVEL/factor families — đều **bắt buộc** đi qua local gate trước khi tiêu tốn sim. Sim Brain
chỉ là bước xác nhận cuối cho thiểu số đã vượt local.

**Chuyển đổi từ đường cũ:** `src/llm/loop.py` (`RefinementLoop`) hiện gọi thẳng
`simulator.simulate(...)` sau prefilter cú pháp. Tại **Phase 3**, refactor để chèn
`score_local_gate(expr, cfg)` thành **cổng bắt buộc** (không phải tùy chọn) trước mọi lần
`simulate`; local hard-fail → bỏ, không đốt quota. Trong Phase 0–2 (local chưa xong), đường
cũ vẫn chạy như fallback để tool không chết; hết Phase 3 thì đường cũ biến mất hoàn toàn.

**Điều phối:** giữ `main.py` menu + `run.ps1`. Thêm 4 entry: `fetch-market`, `score-one`,
`calibrate`, `generate`.

**Dependency rule (theo B1):** `src/lang`, `src/operators_local`, `src/engine`,
`src/backtest` **KHÔNG** được import từ `src/llm`, `src/storage`, `src/submission`. Việc ghép
keo nằm ở `src/llm/loop.py` và các entry CLI mới (nhận data source + repository qua tham số).

---

## 3. Mapping module cũ → trạng thái

### 3.1 GIỮ NGUYÊN (tái dùng)
`main.py`, `run.ps1`, `login.bat`; `src/data/{client,fields,operators,warm_cache,universe_matrix}.py`;
`src/storage/{db,models,repository,migrate}.py`; `src/simulation/*` (simulator, rate_limiter,
oos, sweep, config, pre_filter); `src/llm/{deepseek_client,cli_client,agent_bridge,router,
hypothesis,translator,refiner,marathon,referee,mcts,alignment,generator,expr_synth,errors,
jsonutil}.py`; `src/submission/{manager,correlation}.py`; `src/scoring/{vector,filter,regime,
metrics,scorer,complexity,regularized}.py`; `src/decorrelation/{zoo,similarity,alpha101}.py`;
`src/generation/{families,novel_ideas,local_select,alpha_logger}.py`; `src/optimization/bayesian.py`;
`src/pipeline/auto.py`; `dashboard/`; `tests/` (test cũ giữ để regression).

### 3.2 SỬA (mở rộng, không thay thế)
| Module | Sửa | Phase |
|---|---|---|
| `src/llm/loop.py` | **Bỏ đường cũ** (D9): `score_local_gate` thành cổng bắt buộc trước mọi `simulate`; gỡ nhánh gọi thẳng sim | 3 |
| `src/storage/models.py` + `migrate.py` | Thêm bảng `pool_pnl`, `dead_field`, `brain_record` + cột mở rộng `evaluation` (config_json, per_year_json, self_corr_max, fail_reasons, seed) | 5 |
| `src/storage/repository.py` | Thêm `load_pool`, `record_evaluation_with_pnl`, `add_dead_field`, `get_brain_records`, `result_cache_get/put` | 5 |
| `src/scoring/filter.py` | Thêm `evaluate_local(metrics, self_corr)` đọc `config/thresholds.py` | 4 |
| `config/` | Thêm `settings.py` + `thresholds.py` | 0 |
| `main.py` | Thêm entry `fetch-market` / `score-one` / `calibrate` / `generate` | 0/3/4.5/7 |
| `dashboard/` | Thêm tab Calibration ρ + Pool-corr map | 9 |
| `src/generation/families.py`, `novel_ideas.py` | Tái dùng làm nguồn cho `src/gp/seeds.py` | 7 |

### 3.3 XÓA
| Mục | Khi nào | Thay bằng |
|---|---|---|
| `new 1.txt`, `llm_bridge_run.log` | Ngay (Phase 0) | — |
| `src/generation/ast_utils.py`, `tests/test_ast_utils.py` | Sau Phase 1 (khi `src/lang/` thay xong, đã migrate caller) | `src/lang/{ast,parser}.py` |
| `src/generation/template.py`, `tests/test_template.py` | Sau Phase 7 (khi `src/gp/{seeds,init}.py` thay xong) | `src/gp/` |

### 3.4 THÊM MỚI
| Module | Phase | Trách nhiệm |
|---|---|---|
| `config/settings.py`, `config/thresholds.py` | 0 | Settings + tất cả ngưỡng gate (1 nơi) |
| `src/data/market_fetch.py` | 0 | Kéo OHLCV + universe + sector từ WQ `/datasets/{id}/data` → parquet partitioned |
| `src/data/market_panel.py` | 0 | `MarketData` (frozen, slots) + `MarketDataSource` Protocol + `ParquetSource` adapter |
| `src/data/universe.py` | 0 | Per-day universe mask + sector groups |
| `src/lang/{ast,registry,parser}.py`, `src/lang/grammar.lark` | 1 | FASTEXPR → typed AST + visitors (Depth/Field/Hash/Complexity/Serializer) |
| `src/operators_local/{arithmetic,cross_sectional,timeseries,group,neutralization,conditional}.py` | 2 | Impl operator WQ-faithful + `@register` |
| `src/engine/{evaluator,subexpr_cache}.py` | 2 | AST → signal (T,N), LRU sub-expr cache |
| `src/backtest/{config,portfolio,backtester,metrics_local,pool_corr}.py` | 3/4/6 | PortfolioConfig + PortfolioBuilder + Backtester delay-1 + MetricsCalculator + PoolCorrelation |
| `src/calibration/{harness,report,loader}.py` | 4.5 | CalibrationHarness + Spearman ρ + loader brain_record |
| `src/gp/{individual,seeds,init,variation,fitness_vec,selection,engine}.py` | 7 | Typed GP, NSGA-II, fitness 6 chiều correlation-aware |

> **Lưu ý đặt tên:** dùng `src/operators_local/` (không `src/operators/`) để tránh đụng
> `src/data/operators.py` (metadata operators của WQ). `metrics_local.py` để tách với
> `src/scoring/metrics.py` cũ (metrics từ sim Brain).

---

## 4. Hợp đồng các thành phần mới (theo Part B master spec)

Tất cả lớp/Protocol/dataclass theo đúng B3–B13. Tóm tắt hợp đồng chính:

- **`MarketData`** (B3): frozen+slots, `dates(T,) assets(N,) fields{str:(T,N)} universe(T,N)bool
  returns(T,N) groups{str:(T,N)int}`; out-of-universe = NaN; `__post_init__` validate shape/axis.
- **`MarketDataSource`** Protocol (B3): `load(start,end,universe)->MarketData`, `available_fields()`.
- **AST** (B4): sealed `Constant/Field/Call` + `NodeVisitor` Protocol; visitors mỗi class 1 trách nhiệm.
- **`OperatorRegistry`/`OperatorSpec`** (B5): single source of truth (parser+evaluator+GP); `@register` decorator.
- **`Evaluator`** (B6): NaN-propagate, cross-sectional per-row in-universe, time-series chỉ đọc rows ≤ t (no look-ahead), sub-expr cache theo canonical hash.
- **`PortfolioConfig`/`PortfolioBuilder`/`Backtester`** (B7): stage separation; delay-1 `pnl_t = nansum(w_{t-1} * ret_t)`.
- **`AlphaMetrics`/`MetricsCalculator`** (B8): sharpe/annual_return/turnover/max_dd/fitness/per_year_sharpe/weight_concentration; `fitness = sharpe*sqrt(|annual_return|/max(turnover,0.125))` — chỉ dùng ranking tương đối.
- **`GateEvaluator`/`GateVerdict`** (B8): hard gates (cú pháp/depth≤cap/field hợp lệ/self_corr<0.70/concentration≤cap) + soft scores. Ngưỡng chỉ từ `config/thresholds.py`.
- **`PoolCorrelation`** (B9): `max_corr(candidate_pnl, dates)->(max|ρ|, worst_id)` trên dates chung.
- **`CalibrationHarness`/`CalibrationReport`** (B10): re-score local trên ≥50 alpha đã sim Brain, báo `spearman_sharpe` (headline), `self_corr_agreement`, `decile_hit_rate`, `by_year`. Bar tin cậy ρ ≥ 0.5–0.6.
- **DB schema** (B11): bảng `expression`, `evaluation`, `pool_pnl`, `dead_field`, `brain_record`; lưu cả thất bại + seed.
- **Cache 3 tầng** (B12): field (parquet) / sub-expr (LRU) / result (DB).
- **GP** (B13): `FitnessVector` 6 chiều (sharpe_deflated, per_year_min_sharpe, turnover_penalty, complexity_penalty, pool_corr_penalty, pop_corr_penalty); seed từ factor families; cores only; NSGA-II; persist mọi outcome.

---

## 5. Tích hợp với AI sinh alpha (cải tiến engine — D6)

- **Seed GP từ LLM:** `src/gp/seeds.py` lấy nguồn từ (a) `src/generation/families.py` (factor
  families), (b) `src/generation/novel_ideas.py` (NOVEL_ALPHAS), (c) **LLM** qua
  `HypothesisGenerator → AlphaTranslator` (parse về AST mới). Pure-random GP bị cấm (lãng phí).
- **Stage separation (B7/Gap#8):** LLM/GP chỉ sinh **signal core**; neutralization/decay/
  truncation/scale/delay áp ở `PortfolioConfig` riêng. Cải tiến `RefinementLoop` để không
  emit expression đã wrap sẵn.
- **Correlation-aware (Gap#4/R4):** GP fitness có `pool_corr_penalty` + `pop_corr_penalty`
  ngay từ đầu — chống sinh quần thể bão hòa, trùng PnL.
- **Backend AI giữ nguyên đa lựa chọn:** DeepSeek (v4-flash/pro) / claude-cli (opus 4.8) /
  codex-cli / agent-bridge qua `LLM_BACKEND` (đã có `src/llm/router.py`).
- **Local gate là cổng bắt buộc (D9):** mọi candidate (LLM, GP, NOVEL/factor families) đều
  qua `score_local_gate` → chỉ cái vượt mới đốt sim Brain. Không còn nhánh nào đốt sim mà
  bỏ qua local gate.

---

## 6. Quy trình git & sub-agent (D4)

**Mỗi phase = 1 nhánh:** `git checkout -b phase-N-<tên>` → code (TDD) → test xanh + ruff →
merge vào `main` → push → cập nhật `PROGRESS.md`.

**Sub-agent (superpower):**
- Phase tuần tự P0→P1→P2→P3 (phụ thuộc nhau) — 1 sub-agent/phase.
- Trong phase có module độc lập thì song song: Phase 2 (`operators_local/*` ×6),
  Phase 7 (`seeds/variation/fitness_vec/selection` sau khi `individual.py` xong).
- Test + tài liệu viết song song với code.

**File tiến độ:** `PROGRESS.md` ở root (skill `session-journal`) — append cuối mỗi phase.

---

## 7. Testing (per-phase, theo Part D rule 10)

- **Unit** cho từng module mới.
- **Golden** cho operator semantics (B6): assert panel output cho input nhỏ cố định; test
  no-look-ahead (đổi rows > t không làm đổi row t); đối chiếu vài giá trị Brain khi có.
- **Integration:** 1 lần `parse → eval → portfolio → backtest → metrics` trên fixture panel.
- **Calibration (Phase 4.5):** Spearman ρ trên ≥50 alpha đã sim Brain (lấy từ DB `wq_alpha_*.db`
  hiện có — đã có alpha thật đã sim). Đây là dữ liệu vàng sẵn có, tận dụng được ngay.
- Giữ suite xanh trước khi sang phase kế. `mypy --strict` + `ruff` clean.

---

## 8. Rủi ro & giảm thiểu (kế thừa A4)

| # | Rủi ro | Giảm thiểu |
|---|---|---|
| R1 | Local không tương quan Brain | Phase 4.5 gate ρ; tận dụng alpha đã sim trong DB |
| R2 | Market data không khớp PIT/universe | Kéo trực tiếp từ WQ API; universe mask per-day; tài liệu hóa convention |
| R3 | Look-ahead / survivorship | Mask per-day; ts ops chỉ đọc ≤ t; PIT cho fundamentals |
| R4 | GP bão hòa | Fitness correlation-aware + NSGA-II từ ngày đầu |
| R5 | Operator drift FASTEXPR | Pin semantics B5/B6 + golden test; tra `worldquant-brain` skill |
| R8 | Bất định | Seed inject + ghi DB |
| R-new | Vỡ regression code cũ khi xóa `ast_utils`/`template` | Migrate caller trước; chạy full test cũ trước khi xóa |

---

## 9. Phase plan (theo Part E)

| Phase | Output | Nhánh git |
|---|---|---|
| 0 | config + market_fetch + market_panel + universe + xóa rác | `phase-0-data-foundation` |
| 1 | lang/{ast,registry,parser,grammar} + visitors; xóa ast_utils sau verify | `phase-1-parser` |
| 2 | operators_local/* + engine/{evaluator,subexpr_cache} + golden tests | `phase-2-operator-engine` |
| 3 | backtest/{config,portfolio,backtester} — **MVP**; **gỡ đường cũ** trong loop.py (D9), local gate thành cổng bắt buộc; demo + review | `phase-3-backtester` |
| 4 | backtest/metrics_local + scoring/filter.evaluate_local + thresholds | `phase-4-metrics-gates` |
| 4.5 | calibration/{harness,report,loader}; đo ρ trên DB | `phase-4.5-calibration` |
| 5 | storage migrate + repository mở rộng + result_cache | `phase-5-database` |
| 6 | backtest/pool_corr + wire gate 0.70 | `phase-6-pool-corr` |
| 7 | gp/* (GP engine) + tích hợp loop; xóa template sau verify | `phase-7-gp-engine` |
| 8 | pipeline shortlist + 4 entry CLI vào main.py menu | `phase-8-cli` |
| 9 | dashboard tab calibration + pool-corr; submit assistant | `phase-9-dashboard` |

**MVP = Phase 0–3:** parse 1 alpha viết tay → eval → backtest → đọc Sharpe + equity curve
trên data thật. Dừng, demo, review trước khi xây tầng tin cậy (calibration) và tầng quy mô
(DB, pool-corr, GP).
