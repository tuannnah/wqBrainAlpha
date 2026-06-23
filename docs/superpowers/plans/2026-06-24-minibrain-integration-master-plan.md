# MiniBrain Integration — Master Plan (10 phase)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (khuyến nghị) hoặc superpowers:executing-plans để thực thi từng phase. Mỗi phase có plan riêng (`docs/superpowers/plans/2026-06-24-phase-N-*.md`) với task ở mức step-by-step; file này là **chỉ mục + danh sách coding task mức task** cho toàn bộ.

**Goal:** Tích hợp MiniBrain (local pre-filter) vào tool WQ Brain sẵn có — thêm tầng local
backtester (parse→eval→backtest→metrics→pool-corr→calibration→GP), bỏ đường cũ, mọi candidate
qua local gate trước khi đốt sim.

**Architecture:** Tái dùng lõi (login/fetch/DB/sim/AI/submit) trong `src/`. Thêm các package
`src/{lang,operators_local,engine,backtest,calibration,gp}` + `config/{settings,thresholds}`.
Tuân dependency rule: lang/operators_local/engine/backtest KHÔNG import gp/storage/llm.

**Tech Stack:** Python 3.12, numpy, pandas, pyarrow, lark, SQLAlchemy, joblib, pytest, ruff, mypy.

**Spec nguồn:** `docs/superpowers/specs/2026-06-23-minibrain-into-existing-tool-design.md`
+ `docs/design/MINIBRAIN_DESIGN.md` (Part B chi tiết class/schema).

## Global Constraints

- Python 3.12; cú pháp hiện đại (`match`, `X | None`, `type` alias, `@dataclass(frozen=True, slots=True)`, `Protocol`).
- Full type hints; `mypy --strict` clean; `ruff` clean; không unused import.
- **No look-ahead:** time-series ops chỉ đọc rows ≤ t; thiếu lịch sử → NaN.
- **No survivorship:** universe mask per-day; out-of-universe = NaN (không phải 0).
- **Delay-1:** `pnl_t = nansum(weights_{t-1} * returns_t)`.
- **Stage separation:** expression = signal core; neut/decay/trunc/scale/delay ở `PortfolioConfig`.
- **Thresholds chỉ ở `config/thresholds.py`** — không hardcode gate number ở call site.
- **Determinism:** randomness qua seed inject; ghi seed vào DB.
- **WQ operator fidelity:** tra skill `worldquant-brain` trước khi viết FASTEXPR/operator.
- **TDD:** test trước, đỏ → code tối thiểu → xanh → commit. Mỗi phase = 1 nhánh git → merge → push.
- **Per-phase ritual:** Design → Implement → Explain → Review (test+ruff+mypy) → Gate → Journal (`PROGRESS.md`).

---

## Phase 0 — Data foundation  (nhánh `phase-0-data-foundation`)

> Plan chi tiết step-by-step: `docs/superpowers/plans/2026-06-24-phase-0-data-foundation.md`

| # | Task | Files | DoD |
|---|---|---|---|
| 0.1 | Dọn rác + nhánh | xóa `new 1.txt`, `llm_bridge_run.log` (đã xóa); tạo nhánh | nhánh tạo, working tree sạch |
| 0.2 | `config/thresholds.py` | Create `config/thresholds.py` | Tất cả ngưỡng (MAX_DEPTH=7, SELF_CORR_MAX=0.70, TURNOVER_FLOOR=0.125, WEIGHT_CONCENTRATION_CAP, SHARPE_MIN, PER_YEAR_SHARPE_MIN, TURNOVER_BAND, CALIBRATION_RHO_BAR=0.5) là module-level constants có test |
| 0.3 | `config/settings.py` mở rộng | Modify `config/settings.py` | Thêm `market_data_dir`, `global_seed`; test load mặc định |
| 0.4 | Type aliases | Create `src/local_types.py` | `Panel/Mask/Dates/Assets` numpy.typing aliases |
| 0.5 | `MarketData` panel | Create `src/data/market_panel.py` | frozen+slots; `__post_init__` validate shape/axis; `field()`, `years()`; out-of-universe=NaN; unit test |
| 0.6 | `MarketDataSource` port | Create `src/data/market_source.py` | Protocol `load(start,end,universe)->MarketData`, `available_fields()` |
| 0.7 | `universe.py` | Create `src/data/universe.py` | build per-day mask + sector groups từ raw; test mask thay đổi theo ngày |
| 0.8 | `ParquetSource` adapter | Create `src/data/adapters/parquet_source.py` | đọc parquet partitioned → MarketData; test round-trip ghi/đọc |
| 0.9 | `market_fetch.py` (WQ→parquet) | Create `src/data/market_fetch.py` | probe WQ API lấy OHLCV+universe+sector cho cửa sổ nhỏ; ghi parquet; **dùng client thật, có thể skip nếu offline** |
| 0.10 | fixture panel | Create `tests/conftest.py` (hoặc mở rộng) | fixture `small_panel` (real-shaped, reproducible seed) cho mọi test sau |

**Sub-agent:** 0.2–0.8 song song được phần lớn (độc lập); 0.9 phụ thuộc 0.5–0.8; 0.10 phụ thuộc 0.5.
**Rủi ro chính:** 0.9 — WQ không cấp bulk OHLCV sạch (Gap #3). Nếu probe thất bại, Phase 0 vẫn
done với adapter + fixture; 0.9 chuyển sang spike riêng (tài liệu hóa cách lấy data thực tế).

---

## Phase 1 — Parser  (nhánh `phase-1-parser`)

| # | Task | Files | DoD |
|---|---|---|---|
| 1.1 | AST nodes | Create `src/lang/ast.py` | `Node`(ABC) + `Constant/Field/Call` (frozen,slots) + `NodeVisitor` Protocol; `accept`/`children` |
| 1.2 | Registry | Create `src/lang/registry.py` | `ArgKind/OpCategory` enum, `OperatorSpec`, `OperatorRegistry`, `@register`; `gp_function_set()` |
| 1.3 | Grammar | Create `src/lang/grammar.lark` | FASTEXPR-subset: field, number, call(args), toán tử `+ - * /` |
| 1.4 | Parser + transformer | Create `src/lang/parser.py` | `parse(str)->Node`; lỗi rõ cho op/arity/field sai; `python -m src.lang.parser "<expr>"` chạy |
| 1.5 | DepthVisitor | Create `src/lang/visitors.py` | `DepthVisitor->int` khớp depth tính tay |
| 1.6 | FieldCollector | `src/lang/visitors.py` | `FieldCollector->set[str]` |
| 1.7 | Serializer | `src/lang/visitors.py` | `Serializer->str` round-trip với parser |
| 1.8 | CanonicalHasher | `src/lang/visitors.py` | hash ổn định sau canonicalize (sort commutative, normalize literal) |
| 1.9 | ComplexityVisitor | `src/lang/visitors.py` | node count / weighted complexity |
| 1.10 | Migrate + xóa ast_utils | Modify callers, Delete `src/generation/ast_utils.py`, `tests/test_ast_utils.py` | caller (`decorrelation/zoo`, `scoring/complexity`...) chuyển sang AST mới; full test cũ xanh trước khi xóa |

**Sub-agent:** 1.5–1.9 song song sau khi 1.1 xong. 1.10 cuối cùng (verify rồi xóa).
**Consumes:** `OperatorRegistry` từ 1.2 dùng trong 1.4. **Produces:** `parse`, AST, visitors cho Phase 2.

---

## Phase 2 — Operator Engine  (nhánh `phase-2-operator-engine`)

| # | Task | Files | DoD |
|---|---|---|---|
| 2.1 | EvalContext + Evaluator khung | Create `src/engine/evaluator.py` | `EvalContext(data,registry,cache)`, `Evaluator(NodeVisitor[Panel])`; mask universe; literal broadcast |
| 2.2 | SubexprCache | Create `src/engine/subexpr_cache.py` | LRU theo canonical hash; test hit/miss |
| 2.3 | arithmetic ops | Create `src/operators_local/arithmetic.py` | `+ - * / log abs sign power max min` + golden test |
| 2.4 | cross_sectional ops | Create `src/operators_local/cross_sectional.py` | `rank winsorize scale zscore` per-row in-universe + golden |
| 2.5 | timeseries ops | Create `src/operators_local/timeseries.py` | `ts_mean/std/delta/delay/rank/zscore/corr/decay_linear/backfill`; no-look-ahead test; `ts_delay` không `delay` |
| 2.6 | group ops | Create `src/operators_local/group.py` | `group_neutralize` (wrapper, gp_usable=False) |
| 2.7 | neutralization ops | Create `src/operators_local/neutralization.py` | `regression_neut`, `vector_neut` (chỉ 2 op giảm self-corr) |
| 2.8 | conditional ops | Create `src/operators_local/conditional.py` | `trade_when`, `hump` |
| 2.9 | Wire evaluator + integration | Modify `src/engine/evaluator.py`, Create `tests/integration/test_eval.py` | parse→eval ra `(T,N)` đúng NaN-propagation trên fixture |

**Sub-agent:** 2.3–2.8 (6 file operator) song song sau khi 2.1 xong. Mỗi file golden test riêng.
**Consumes:** AST/registry (P1), MarketData (P0). **Produces:** `Evaluator.evaluate(node)->Panel` cho P3.
**Quan trọng:** tra skill `worldquant-brain` cho semantics từng op (NaN, ts_rank normalize, decay weights).

---

## Phase 3 — Backtester (MVP)  (nhánh `phase-3-backtester`)

| # | Task | Files | DoD |
|---|---|---|---|
| 3.1 | PortfolioConfig | Create `src/backtest/config.py` | `Neutralization` enum + `PortfolioConfig` (neut/decay/trunc/scale/delay) |
| 3.2 | PortfolioBuilder | Create `src/backtest/portfolio.py` | `build(signal,cfg,data)->weights`: decay→neutralize→truncate→scale→delay; dollar-neutral; truncation cap |
| 3.3 | Backtester | Create `src/backtest/backtester.py` | `BacktestResult`; `run(weights,data)`: `pnl_t=nansum(w_{t-1}*ret_t)`; delay-1 test |
| 3.4 | Integration MVP | Create `tests/integration/test_backtest_mvp.py` | alpha viết tay → equity curve + Sharpe trên data thật; **demo + review** |
| 3.5 | Gỡ đường cũ (D9) | Modify `src/llm/loop.py` | `score_local_gate(expr,cfg)` thành cổng bắt buộc trước mọi `simulate`; gỡ nhánh sim trực tiếp; test loop bỏ candidate local-fail |

**Consumes:** Evaluator (P2). **Produces:** `BacktestResult`, `score_local_gate` cho P4/loop.
**Đây là MVP** — dừng, demo equity curve, review trước khi sang P4.

---

## Phase 4 — Metrics + Gates  (nhánh `phase-4-metrics-gates`)

| # | Task | Files | DoD |
|---|---|---|---|
| 4.1 | AlphaMetrics + MetricsCalculator | Create `src/backtest/metrics_local.py` | sharpe/annual_return/turnover/max_dd/fitness/per_year_sharpe/weight_concentration; `fitness=sharpe*sqrt(\|ann\|/max(turn,0.125))` |
| 4.2 | per-year Sharpe | `src/backtest/metrics_local.py` | dict{year:sharpe} dùng `data.years()` |
| 4.3 | GateVerdict + GateEvaluator | Create `src/backtest/gates.py` | hard gates (depth/field/self_corr/concentration) + soft scores; ngưỡng từ thresholds |
| 4.4 | filter.evaluate_local | Modify `src/scoring/filter.py` | wrap GateEvaluator cho loop dùng |
| 4.5 | Integration | `tests/integration/test_metrics_gates.py` | parse→eval→backtest→metrics→gate end-to-end |

**Consumes:** BacktestResult (P3). **Produces:** `AlphaMetrics`, `GateVerdict` cho P4.5/P6/P7.

---

## Phase 4.5 — Calibration  (nhánh `phase-4.5-calibration`)

| # | Task | Files | DoD |
|---|---|---|---|
| 4.5.1 | BrainRecord loader | Create `src/calibration/loader.py` | đọc alpha đã sim từ DB `wq_alpha_*.db` (expr + brain sharpe/fitness/turnover/self_corr) |
| 4.5.2 | CalibrationReport | Create `src/calibration/report.py` | dataclass: n, spearman_sharpe, spearman_fitness, self_corr_agreement, decile_hit_rate, by_year |
| 4.5.3 | CalibrationHarness | Create `src/calibration/harness.py` | re-score local + Spearman ρ; gate ρ≥CALIBRATION_RHO_BAR |
| 4.5.4 | CLI calibrate + report | Modify `main.py` | `calibrate` chạy harness, in báo cáo; đo ρ trên ≥50 alpha thật |

**Consumes:** MetricsCalculator (P4), DB hiện có. **Produces:** ρ — gate tin cậy cả tool.
**Highest value-per-line.** Nếu ρ thấp → fix data fidelity (P0) / operator semantics (P2), không phải GP.

---

## Phase 5 — Database  (nhánh `phase-5-database`)

| # | Task | Files | DoD |
|---|---|---|---|
| 5.1 | Models mới | Modify `src/storage/models.py` | thêm `ExpressionModel`, `EvaluationModel`(+config_json/per_year_json/self_corr_max/fail_reasons/seed), `PoolPnlModel`, `DeadFieldModel`, `BrainRecordModel` |
| 5.2 | Migration | Modify `src/storage/migrate.py` | tạo bảng mới idempotent (create_all + _migrate_add_columns đã có) |
| 5.3 | Repository mở rộng | Modify `src/storage/repository.py` | `upsert_expression`, `record_evaluation` (pass&fail), `load_pool`, `add_dead_field`, `result_cache_get/put`, `top_n` |
| 5.4 | result_cache | Create `src/cache/result_cache.py` | DB-backed `hash+config+window→AlphaMetrics`; test hit skip re-eval |
| 5.5 | dead-field blacklist wire | Modify loop/gen | field bị Brain reject → blacklist, không đề xuất lại |

**Consumes:** AlphaMetrics (P4). **Produces:** pool + result cache cho P6/P7.

---

## Phase 6 — Pool correlation  (nhánh `phase-6-pool-corr`)

| # | Task | Files | DoD |
|---|---|---|---|
| 6.1 | PoolCorrelation | Create `src/backtest/pool_corr.py` | `max_corr(candidate_pnl,dates)->(max\|ρ\|,worst_id)` trên dates chung; test pair có ρ biết trước |
| 6.2 | Wire pool vào gate | Modify `src/backtest/gates.py` | hard gate self_corr<0.70 dùng PoolCorrelation từ DB pool |
| 6.3 | Lưu PnL khi pass | Modify `src/storage/repository.py` + loop | alpha pass → ghi `pool_pnl` (dates+pnl blob) |

**Consumes:** load_pool (P5), BacktestResult (P3). **Produces:** self-corr local cho gate + GP.

---

## Phase 7 — GP Engine  (nhánh `phase-7-gp-engine`)

| # | Task | Files | DoD |
|---|---|---|---|
| 7.1 | Individual | Create `src/gp/individual.py` | wrap Expression + cached FitnessVector |
| 7.2 | FitnessVector | Create `src/gp/fitness_vec.py` | 6 chiều: sharpe_deflated, per_year_min_sharpe, turnover_penalty, complexity_penalty, pool_corr_penalty, pop_corr_penalty |
| 7.3 | Seeds | Create `src/gp/seeds.py` | từ `families.py`+`novel_ideas.py`+LLM (hypothesis→translator→parse); cores only |
| 7.4 | Init | Create `src/gp/init.py` | ramped half-and-half + seeding; depth cap ≤7 |
| 7.5 | Variation | Create `src/gp/variation.py` | typed crossover, point/subtree/hoist mutation; validity repair; canonical-hash dedup |
| 7.6 | Selection | Create `src/gp/selection.py` | NSGA-II Pareto / fitness-sharing correlation-aware |
| 7.7 | GPEngine | Create `src/gp/engine.py` | `evolve(generations)`: eval(joblib+cache)→score→select→vary→elitism; persist mọi outcome |
| 7.8 | Tích hợp loop + xóa template | Modify `src/llm/loop.py`, Delete `src/generation/template.py`, `tests/test_template.py` | GP sinh batch qua local gate; verify rồi xóa template |

**Sub-agent:** 7.2–7.6 song song sau 7.1. **Consumes:** evaluator/metrics/pool_corr/repo.
**Đo:** population self-correlation bị chặn (diversity giữ). Persist pass&fail + seed.

---

## Phase 8 — Short-list + CLI  (nhánh `phase-8-cli`)

| # | Task | Files | DoD |
|---|---|---|---|
| 8.1 | shortlist | Create `src/pipeline/shortlist.py` | rank + decorrelate → candidate list pool-aware |
| 8.2 | runner | Create `src/pipeline/runner.py` | `score_one(expr,cfg)->(AlphaMetrics,GateVerdict)`, `generate_many(...)` |
| 8.3 | CLI entries | Modify `main.py` | `score-one`, `generate`, `calibrate` vào menu PowerShell |

**Consumes:** GPEngine (P7), harness (P4.5). **Produces:** shortlist xuất ra để sim Brain.

---

## Phase 9 — Dashboard / submit assistant  (nhánh `phase-9-dashboard`)

| # | Task | Files | DoD |
|---|---|---|---|
| 9.1 | Tab Calibration | Modify `dashboard/` | hiển thị ρ Spearman + by_year |
| 9.2 | Tab Pool-corr | Modify `dashboard/` | bản đồ self-correlation pool |
| 9.3 | Export shortlist | Modify `dashboard/` + `src/pipeline/shortlist.py` | xuất shortlist để sim; submit helper (không hardcode secret) |

**Consumes:** runs + metrics DB. **Produces:** UI inspect + export.

---

## Self-review (spec coverage)

- Local pre-filter (D1) → P1–P4 + P6. ✔
- Market data WQ (D2) → P0 (0.9). ✔ (rủi ro bulk-fetch ghi rõ)
- GP mới B13 (D3) → P7. ✔
- Code mới `src/`, nhánh/merge/push (D4) → mọi phase. ✔
- Thứ tự Part E (D5) → P0→P9. ✔
- Cải tiến engine sinh alpha (D6) → P3.5 (gỡ đường cũ) + P7 (correlation-aware). ✔
- Xóa rác (D7) → 0.1. ✔
- Xóa ast_utils/template (D8) → 1.10 + 7.8. ✔
- Bỏ đường cũ (D9) → 3.5. ✔
