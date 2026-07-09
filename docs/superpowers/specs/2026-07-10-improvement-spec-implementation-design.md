# Triển khai IMPROVEMENT_SPEC — thiết kế bám code thật

> Ánh xạ `docs/tailieu/IMPROVEMENT_SPEC.md` (5 pha) vào code hiện tại. Điều tra §5 đã xong
> (3 sub-agent). Doc này chốt *làm gì ở đâu*, các quyết định correctness, và acceptance mỗi pha.
> Phạm vi đã chốt với user: **cả 5 pha, tuần tự**, mỗi pha commit + verify riêng, dừng giữa
> các pha để user chạy phiên so sánh (single-variable §6). TDD bắt buộc, commit tiếng Việt.

## Nền tảng đã có (không viết lại)

- Parser lark → AST (`src/lang/parser.py`, `ast.py`); `DepthVisitor`/`ComplexityVisitor`/
  `CanonicalHasher`/`Serializer` (`src/lang/visitors.py`).
- `CanonicalHasher` đã: sort toán hạng giao hoán (add/multiply/max/min), normalize literal,
  hash AST. **Chưa** gấp hằng số.
- `canonical_hash` lưu SQLite `expressions.canonical_hash UNIQUE` (`models.py:152`) — bền
  cross-session; DB tách theo email (`db.py`).
- GP: depth cap cứng 7, parsimony mềm (NSGA-II), hoist chống bloat (`src/gp/`).
- `CorrelationChecker` poll checker THẬT `/correlations/self`, ngưỡng 0.70 (`correlation.py:11`).
- `regression_neut`/`vector_neut` đã implement (`operators_local/neutralization.py`) — chưa
  dùng làm lever hạ self-corr trong tuner.
- Test pytest phủ parser/canonical/gp/tuner (xem §5(c) điều tra).

## Quyết định correctness đã chốt

**Canonical fold (Pha 1):** strip MỌI `multiply`/`divide` bởi hằng **DƯƠNG** ở tầng canonical
hash (giữ nguyên biểu thức thực thi — chỉ đổi *dedup key*). Giữ hằng ÂM (đổi dấu = alpha khác:
WQ normalize book long/short nên scale dương bất biến sau normalize, scale âm đảo PnL). Bằng
chứng: `logs/alphas_2026-07-09_183046.csv:2-3` cặp `multiply(4,X)`/`multiply(2,X)` cho
`turnover/self_corr/fitness` y hệt. Golden test khóa bất biến này (Pha 1 acceptance).

---

## Pha 0 — Instrumentation

Gốc: `IdeaOutcome` (`closed_loop.py:38`) thiếu trường; `_reasons` từ `hard_filter` bị vứt
(`closed_loop_adapters.py:188`); CSV nuốt `sharpe`/`fitness` khi `passed=False`
(`run_alpha_log.py:50`); không có `session_summary`.

1. Mở rộng `IdeaOutcome`: `stage_reached`, `fail_check`, `family`, `expr_depth`, `gen_ms`,
   `backtest_ms`, `sim_ms`, `dedup_key`, `local_sharpe`; đổi `sharpe/fitness` →
   `brain_sharpe/brain_fitness`. Default để tương thích ngược.
2. `LocalTunerRefiner`: giữ `_reasons` → `fail_check` (LOW_SHARPE/SELF_CORR/DEPTH/DUP/...);
   set `stage_reached` mỗi điểm return; đo `perf_counter` 3 mốc (gen/backtest/sim).
3. `RunAlphaLogger`: schema cố định mới, **luôn điền đủ cột** (metric ghi bất kể passed).
4. `SessionSummary` (`src/reporting/session_summary.py`): funnel theo `stage_reached`, phân bố
   `fail_check`/`family`, median thời gian/stage, số dup bị chặn → in + `logs/session_summary_*.md`.

*Acceptance:* 1 phiên → summary trả lời "chết ở đâu, vì sao, tốn bao lâu".

## Pha 1 — Chặn rác sớm & rẻ (throughput)

Thứ tự gate hiện: `string-dedup → backtest → local_floor → sub_universe → sim`. Đích:
`syntax/arity → depth → canonical-dedup → parsimony → backtest → sim`.

1. **Canonical fold** trong `CanonicalHasher` (quyết định trên) + golden test bất biến.
2. **Dedup dùng hash trước backtest**: `ClosedLoop` dùng `dedup_key` (canonical) thay `cand.expr`
   thô ở `seen` (`closed_loop.py:180`); avoid-list nạp theo hash. Chặn ở `stage=dedup`, 0 backtest.
3. **Depth guard trước backtest**: tính depth core + wrapper stack dự kiến; > ~7 → loại
   `stage=depth`. Sửa depth = làm phẳng core (skill), KHÔNG swap field.
4. **Parsimony + hằng số rời rạc trong GP**: phạt kích thước mạnh hơn; scalar rời rạc
   {-2,-1,-0.5,0.5,1,2} thay `uniform` float (`src/gp/init.py:64,81`, `variation.py:132`).

*Acceptance:* `multiply(4,X)`≡`multiply(2,X)` cùng `dedup_key`; dup cross-session chặn ở
`dedup`; độ dài biểu thức median giảm ≥30%; không còn float 15 chữ số.

## Pha 2 — Chuyển họ nhân tố (yield)

1. `build_closed_loop`: `include_alt_data=True` mặc định; thêm nguồn fundamental
   (gross-profitability, cash-flow yield, asset growth, analyst revision) — `ts_backfill` bắt
   buộc; verify field LIVE qua `get_datafields`, không tin cứng `VERIFIED_FIELDS`.
2. **Family-aware budget + exhaustion guard** trong vòng kín: gán `family` (Pha 0), giới hạn
   ứng viên/họ/phiên; họ sinh N mà 0 qua self-corr proxy → đóng họ, chuyển ngân sách.
3. **Hypothesis-first cho LLM**: ép `generate_ideas` theo 4 phần (đã có ở `hypothesis.py`,
   nối vào đường generator); tiêm avoid-list + họ đã bão hoà vào prompt.

*Acceptance:* ≥60% ứng viên/phiên KHÔNG thuộc `pv_reversal`; summary cho thấy ≥3 họ.

## Pha 3 — Configuration stage đúng lever (self-corr)

1. `LocalTuner`: thêm nhánh cấu hình bọc `regression_neut`/`vector_neut` chống thành phần
   crowded — toán tử DUY NHẤT hạ self-corr. self-corr là objective hạng nhất khi là ràng buộc.
2. Verify self-corr bằng checker THẬT (đã có `CorrelationChecker`); nhắm ≤0.5 (Power Pool).
3. Ưu tiên `ts_rank` hơn `ts_zscore`; turnover là alpha thì không decay/hump.

## Pha 4 — Hiệu chỉnh local floor

- Thay floor đơn-ngưỡng 0.5 (`thresholds.py:38`) bằng percentile-theo-họ hoặc calibrated
  `brain≈local×1.28`; ghi cả `local_sharpe` + ngưỡng áp dụng để audit.
- Floor chỉ loại rác rõ ràng; rác cấu trúc đã chặn ở Pha 1 trước backtest.

*Acceptance:* tỉ lệ chết `local_floor` giảm, tỉ lệ đã-sim-mà-đạt tăng (đo bằng funnel Pha 0).

---

## Kiểm chứng (mỗi pha)

1. `./venv/Scripts/python.exe -m pytest` xanh (canonicalize/dedup KHÔNG đổi ngữ nghĩa biểu thức).
2. 1 phiên closed-loop → so `session_summary` trước/sau (single-variable §6).
3. KHÔNG auto-submit — submit là thủ công (bước 7).

## Ghi chú FASTEXPR
- `ts_delay(x,d)` không `delay`. Fundamental luôn `ts_backfill`. Chỉ `regression_neut`/
  `vector_neut` hạ self-corr. Depth ≈7: wrapper ăn 3 tầng, core ~4 — lỗi depth làm phẳng core,
  không swap field. Verify field LIVE trước khi dùng.
