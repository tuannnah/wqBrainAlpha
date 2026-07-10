# MiniBrain — Progress log

## Current state
- **Phase:** **HOÀN TẤT CẢ 5 PHA (0→4) của `docs/tailieu/IMPROVEMENT_SPEC.md`** — code+test
  xong trên `main` [2026-07-10 Session 08]. Design:
  `docs/superpowers/specs/2026-07-10-improvement-spec-implementation-design.md`.
  - Pha 0 (instrumentation): IdeaOutcome +9 trường, RunAlphaLogger luôn điền đủ,
    diagnostics+session_summary, LocalTunerRefiner giữ _reasons + timing.
  - Pha 1 (throughput): canonical fold scale dương (golden), dedup hash trước backtest +
    avoided_hashes cross-session, depth guard, hằng số GP rời rạc.
  - Pha 2 (yield): alt-data+fundamental mặc định ON (field verify LIVE), family budget +
    exhaustion guard, tiêm họ bão hoà vào prompt LLM.
  - Pha 3 (self-corr): tune() bọc regression_neut(expr, rank(volume)) hạ self-corr (golden
    chứng minh trực giao). Phần phụ ts_rank/turnover để lại làm A/B.
  - Pha 4 (floor): calibrated_floor(target/1.28) thay hằng cứng 0.5; audit local_sharpe+ngưỡng.
- **Next: CẦN USER CHẠY MENU-5 để đo phiên thật** — không code thêm gì cho tới khi có số liệu.
  Đọc `logs/session_summary_*.md` để nghiệm thu acceptance từng pha (funnel/độ dài/đa dạng họ/
  self-corr/tỉ lệ local_floor). Có thể A/B `--no-alt-data` để đo đóng góp yield.
- **Acceptance TẤT CẢ pha còn treo (cần menu-5 QR-login):** baseline funnel; median độ dài -30%;
  >=60% không pv_reversal + >=3 họ; self-corr giảm; tỉ lệ local_floor giảm. Fundamental cores +
  regression_neut chưa sim thật (MCP create_simulation trả 400 = lỗi wrapper, KHÔNG phải field;
  sim thật qua Simulator repo khi chạy menu-5).
- **Tinh chỉnh để lại làm A/B (không áp mù):** complexity_penalty (Pha 1), ts_rank>ts_zscore +
  turnover-alpha không decay (Pha 3) — chờ số liệu phiên thật.
- **Phase (trước):** Đã (1) nâng chất lượng alpha từ docs WQ, (2) CHẠY THẬT Auto SIM + cải
  thiện. **e2e ĐÃ có alpha đạt chất lượng submit** — 3 alpha (`rKlkG9O8`/`kq0RY2G8`/`E5E3NKZJ`,
  VWAP intraday-reversal) Sharpe ~1.5, `failed_checks=[]`, self-corr 0.49/0.47/0.50 < 0.70.
  Chưa nộp (submissions=0) vì (a) bug correlation-poll ĐÃ FIX, (b) submit cần người dùng đồng ý.
- **Session:** người dùng đã đăng nhập lại QR (23:38). Dry-run `submit` chạy thông suốt
  (correlation-poll đã fix): e2e HOÀN CHỈNH (khám phá -> sim -> verify self-corr -> chọn
  alpha sẵn sàng nộp). Người dùng CHỌN chưa nộp, GIỮ SẴN SÀNG — alpha `rKlkG9O8` (Sharpe
  1.57, self-corr 0.49, `failed_checks=[]`) nằm trong DB, nộp sau bằng `submit --no-dry-run`.
- **E2e chạy dài tin cậy hơn (Session 05):** giải 2 nút throughput — `_neutralize` 72s→5s
  (`223be99`) + `pool_corr.max_corr` 62s→7s (`2b327f2`, lộ khi profile POOL THẬT) → **batch
  pool-thật 113s→58s (~2x)**; pre-sim floor calibrate (`c476a65`); LLM refine không sập phiên
  (`ed0e34f`). **Đo LIVE:** sim-throughput giờ bị chặn bởi POOL BÃO HÒA (gate `độc đáo<0.35`
  loại ~11/16 candidate — vùng VWAP đã mine trong pool 268), KHÔNG phải tốc độ; quota rất tiết
  kiệm. Muốn nhiều alpha submit mới → mở rộng dataset (novel-v2, path-Brain). `_truncate` 12.5s
  còn để roadmap.
- **Next (đề xuất):** merge nhánh `alpha-quality-from-brain-docs` vào `main` khi duyệt; (tùy
  chọn) chạy lại `closed-loop` đo throughput+quota thực; nộp `rKlkG9O8` khi muốn.
- **Done:** Phase 0-8 theo `docs/tailieu/BUILD_GUIDE_AI_alpha_tool.md` (đăng nhập, data
  fields/operators, GP engine, backtest/metrics/gate local, LLM refine, short-list
  decorrelate). Nhánh `closed-loop-integration` (DB cầu Brain SIM <-> MiniBrain, orchestrator
  `ClosedLoop`, adapter `GPIdeaSource`/`RefinementLoopRefiner`, CalibrationTracker) đã merge
  vào `main` + push (`fbe4b2b..70e3e54`). Menu run.bat còn đúng 5 mục: đăng nhập (tự
  ensure fields/operators) / tải lại fields / tải lại operators / test engine cục bộ (không
  cần đăng nhập) / Auto SIM (vòng kín thật). Dọn code chết `_auto_prepare`/`AutoPipeline`/
  `login.bat`; gộp trùng lặp CLI `closed-loop` + menu qua `_run_closed_loop_session`.
- **In progress:** Không có việc dở dang. Vừa đối chiếu mục 5 (Auto SIM) với toàn bộ tài liệu
  thiết kế vòng kín (`docs/superpowers/specs/2026-06-26-ai-minibrain-closed-loop-design.md` +
  6 plan phase) + vá gap phát hiện quota (`QuotaExceededError`, xem entry bên dưới) — sẵn sàng
  bắt đầu kiểm thử thật.
- **Next step:** (a) Cân nhắc merge nhánh `alpha-quality-from-brain-docs` vào `main` sau khi
  người dùng duyệt; (b) tiếp roadmap trong
  `docs/superpowers/specs/2026-07-06-alpha-quality-upgrade-from-brain-docs.md` (Sub-Universe
  gate, ngoại lệ self-corr Sharpe+10%, profile Single-Dataset/Power-Pool, hash-cache chống
  SIM trùng); (c) vẫn giữ mục chạy thật Auto SIM (mục 5) end-to-end với WQ Brain SIM/quota
  thật — theo dõi `QuotaExceededError` với response thật + `RefinementLoop.repo.save_*`.
- **Blockers / open risks:** Không có blocker biết tới ở thời điểm này. LLM_BACKEND hiện là
  `deepseek` (đã xác nhận API còn hoạt động khi chạy mục 4 thật trong phiên này). Feedback (d)
  "AI học từ SIM" (`AlphaTranslator.avoid_subtrees`/zoo) chỉ refresh giữa các phiên, không
  refresh động trong lúc 1 phiên Auto SIM dài đang chạy — chấp nhận được cho v1, ghi nhận để
  cải thiện sau nếu cần.
- **MVP (Phases 1–3) reached:** yes (từ lâu).
- **Calibration ρ (Spearman, Sharpe):** không đo trong phiên này — xem lệnh `calibrate`.

## Entries
<!-- append-only; newest at the bottom -->

### [2026-07-09] Session 07 — Mở đường ALT-DATA đi thẳng Brain (đòn bẩy độ mới) + fix pre-filter
- **Phase:** Sau Phase 8, nhánh `alpha-quality-from-brain-docs`. Mục tiêu người dùng: đọc
  docs+log, research để cải thiện chất lượng tool, chạy menu-5 kiểm thử, đọc lại định hướng.
- **Chẩn đoán:** Tool đã sinh alpha đạt chuẩn nộp nhưng **khóa cứng vào 1 họ PV reversal đã
  bão hòa** (mọi ý tưởng run gần nhất = close-vwap/close-open → alpha mới fail self-corr). Nút
  cũ "không verify được field alt-data" (cardinal rule #1) — nay ĐÃ ĐĂNG NHẬP nên gỡ được qua
  API. Cũng thấy Brain free-tier sim 8-20′/cái (timeout 1200s cắt oan ~50% ở run 07-08).
- **Done:**
  - **`b56cd3a` Đường alt-data đi thẳng Brain:** verify field thật (get_datafields):
    option8 `implied_volatility_call/put/mean_*` + `historical_volatility_*`, socialmedia8
    `snt_social_value/volume`. Module `src/generation/alt_data_seeds.py` (6 core GAP/reversal
    xen kẽ option↔social, `ts_backfill` field option sparse) + `neutralization_for_expr`
    (option→SECTOR, social/news→SUBINDUSTRY, analyst→INDUSTRY). `AltDataIdeaSource` +
    **nhánh sim-thẳng** trong `LocalTunerRefiner._sim_direct` (khi `local_usable==False`: BỎ
    tune/floor local — panel không có field — sim Brain 1 lần với neut theo category; trích
    `_finalize` dùng chung). Cờ `--alt-data` (build_closed_loop `include_alt_data`, đặt NGOÀI
    CÙNG idea-source để phiên ngắn vẫn chạm alt-data). TDD 3 file mới + wiring.
  - **`b6a3cb0` Fix pre-filter chặn oan:** run thật lộ `count_positional_arity` bỏ param có
    `=` khỏi cap → `ts_backfill(x,lookback=d,k=1)` cap=1 → chặn OAN `ts_backfill(x,22)` (HỢP
    LỆ Brain). Đổi `count_max_arity` (tổng param). Ảnh hưởng cả rank/winsorize/ts_decay.
  - **Chạy thật menu-5** (`closed-loop --alt-data`): **3 sim Brain thật, không timeout 11-13′**
    — XÁC NHẬN đường alt-data e2e cho CẢ 2 dataset: social `-ts_mean(snt_social_value,5)`
    Sharpe -0.48/TO 0.22, social `-ts_delta` Sharpe -0.19/TO 0.37 (SAI DẤU), option
    `iv_mean_90−iv_mean_30` Sharpe +0.17/TO 0.30 (ĐÚNG DẤU, sau khi fix arity qua pre-filter +
    neut SECTOR). Chưa pass nhưng tín hiệu CÓ THẬT. Bài học: **refiner alt-data CHƯA tune sign**.
- **Decisions:** Chỉ dùng field ĐÃ verify LIVE (KHÔNG dùng opt6_*/pcr_*/snt1_* trong
  VERIFIED_FIELDS — dataset option6/sentiment1 KHÔNG có cho account này). Không GATE bằng
  comparison (`greater`/`less` chưa đăng ký local) — dùng scaling nhân thay. Refiner alt-data
  chưa tune sign — hardcode dấu seed (nên có core sai dấu).
- **Next step (định hướng tiếp):** (a) **lật dấu / tune nhẹ cho alt-data** (bằng chứng: fade
  sai chiều → thêm biến thể `+` hoặc coordinate-descent sign trong `_sim_direct`); (b) refiner
  alt-data hiện KHÔNG chọn được sign/param tối ưu — cân nhắc mini-sweep dấu×decay 1-2 sim; (c)
  **throughput sim** vẫn là nút phụ (8-20′/sim tuần tự; timeout lãng phí quota khi >20′) — cân
  nhắc multi-simulation Brain (create_multi_simulation) hoặc reap-on-timeout; (d) mở rộng seed
  alt-data (analyst4 revision/surprise, option skew term-structure, news18) khi cần thêm độ mới.
- **Blockers/mở:** Alt-data đi thẳng Brain KHÔNG có pre-sim local floor (không chấm local
  được) → mỗi core tốn 1 sim thật; cần seed CHẤT (đúng dấu) để không phí quota. Session Brain
  ~2-3h hết hạn cần QR terminal thật.
- **Tests:** `pytest -q` **1094 passed**, 1 fail psycopg có sẵn. ruff sạch. 2 commit code +
  test trên nhánh. Menu-5 chạy thật xác nhận (log `wq_altdata_test.log`).

### [2026-07-07] Session 05 — E2e tự chạy dài TIN CẬY hơn (throughput + pre-sim floor + LLM robust)
- **Phase:** Sau Phase 8, nhánh `alpha-quality-from-brain-docs`. Theo yêu cầu người dùng:
  giải nút throughput GP + calibrate pre-sim floor (để e2e tự chạy dài tin cậy hơn).
- **Done:**
  - **Throughput (commit `223be99`):** profile 1 GP batch (pop20×gen2) = 117s; hotspot áp đảo
    `portfolio.py::_neutralize` = 72s/61% (double-loop Python ngày×nhóm gọi `np.nanmean` 2.1
    TRIỆU lần). Vectorize bằng `np.bincount` bucket `row*G+code` → **batch 117s→53.6s (2.2x)**;
    tương đương số học (test_backtest_portfolio + golden + mvp xanh).
  - **Pre-sim floor (commit `c476a65`):** calibrate local≈Brain/1.28 (winner local 1.23 vs
    Brain 1.57); bật `PRE_SIM_LOCAL_SHARPE_FLOOR=0.5` qua `functools.partial(score_local_gate,
    min_sharpe=…)` trong `_run_closed_loop_session` → bỏ sim alpha local Sharpe<0.5 (rác đã
    quan sát: 0.39/-0.02/-0.13). Bảo thủ nên không đói loop (winner local 1.23 vẫn qua).
  - **LLM robustness (commit `ed0e34f`):** `_refine_loop` gọi `refiner.refine` (LLM) trước đây
    KHÔNG catch → claude-cli exit≠0 1 call là crash cả phiên dài. Bọc try/except: lỗi LLM tạm
    thời → bỏ qua bước (cand=None), chỉ re-raise KeyboardInterrupt. TDD `_RaisingRefiner`.
- **Decisions:** Không vectorize `_truncate`/`_decay`/`argsort` (nhỏ hơn + rủi ro numerics cao
  hơn) — ghi roadmap. Floor 0.5 bảo thủ thay vì siết mạnh để ưu tiên không đói loop.
- **In progress:** Không.
- **Blockers / open risks:** Chưa chạy lại phiên dài thật với 3 cải thiện này (session Brain
  cần QR khi hết hạn). `_truncate`/`_decay` vẫn là nút phụ (~19s/batch).
- **Next step:** (tùy chọn) chạy lại `closed-loop` để đo throughput+quota thực tế; merge nhánh.
- **Tests:** `pytest -q` **1027 passed**, 1 fail psycopg có sẵn. ruff sạch. 3 commit.


### [2026-07-02] Session 01 (journal) — Rút gọn menu run.bat còn 5 mục + dọn code thừa + merge main
- **Phase:** Sau Phase 8 — hoàn tất tích hợp `closed-loop-integration` vào `main` + đơn giản
  hoá UX khởi động (`run.bat`).
- **Done:**
  - Tách commit riêng phần dọn tài liệu tồn đọng (`tailieu/` → `docs/tailieu/`, thêm
    `docs/design/`, `docs/worldquantbrain/`, `skill/`) — không liên quan task, giữ lịch sử
    sạch.
  - Xóa code chết: `_auto_prepare` (main.py, không nơi nào gọi), `src/pipeline/auto.py`
    (`AutoPipeline`/`PrepareInfo`, chỉ dùng trong test riêng của nó — xóa cả file lẫn test),
    `login.bat` (thay thế hoàn toàn bởi mục 1 trong menu `run.bat`).
  - Gộp trùng lặp: `closed_loop_cmd` (CLI) và `_menu_closed_loop` từng lặp gần hệt logo dựng
    `ParquetSource`/`_make_research_loop(marathon=True)`/`build_closed_loop` — rút thành
    `_run_closed_loop_session()` dùng chung trong `main.py`.
  - Thêm `src/app/local_engine_test.py` (composition-root layer, cùng tầng
    `closed_loop_adapters.py`): 1 lượt test cục bộ hoàn toàn — `GPIdeaSource.next_batch()`
    sinh nhanh 1 ứng viên đã chấm local → `AlphaRefiner.refine()` gọi **LLM thật** → chấm lại
    qua `score_one()`. Cố ý **không** dùng `ClosedLoop`/`repo.record_brain_sim` vì bảng đó
    dành cho kết quả SIM thật từ Brain — ghi dữ liệu giả (không `wq_alpha_id` thật) vào đây
    sẽ làm sai lệch calibration ρ về sau. TDD: viết `tests/unit/test_local_engine_test.py`
    (4 case: happy path, LLM không sinh được biểu thức, GP rỗng, lỗi bất ngờ gói vào
    `result.error` thay vì crash) trước khi hoàn thiện module.
  - Viết lại menu `main.py` (`_print_menu`/`start`) còn đúng 5 mục theo yêu cầu: 1) đăng
    nhập — nay **tự** gọi `FieldRepository.ensure()`/`OperatorRepository.ensure()` (fetch
    nếu thiếu, cache thì thôi) thay vì chỉ hiện số; 2)/3) tải lại fields/operators (ghi đè
    cache, không đổi); 4) **mới** — "Test engine", không đòi đăng nhập, đọc
    `read_active_account()`/`.wq_account` để mở đúng DB, kiểm cache fields/operators, tự dò
    thư mục MarketData (`settings.market_data_dir` hoặc quét `data/*/returns.parquet`), gọi
    `local_engine_test.run_local_engine_test()`; 5) "Auto SIM" (đổi tên từ mục 7 cũ) — vòng
    kín thật qua `_run_closed_loop_session`. Xóa `_menu_research`/`_menu_marathon` (không còn
    slot trong 5 mục) — CLI `research`/`marathon` độc lập vẫn giữ nguyên cho debug/script.
  - Cập nhật `README.md` mục "Cách dùng nhanh" khớp menu mới.
- **Decisions:**
  - Không xóa các CLI command độc lập khác (`research`, `marathon`, `generate`, `submit`,
    `calibrate`, `sweep-config`, ...) dù không còn nằm trong menu — vẫn dùng cho debug/script
    ngoài `run.bat` và có test riêng; chỉ xóa code **chắc chắn chết** (không nơi nào gọi) và
    code chỉ phục vụ menu cũ đã bỏ.
  - "Test engine" (mục 4) dùng LLM **thật** (không fake) theo yêu cầu người dùng, để bắt lỗi
    sát với luồng thật của mục 5, nhưng KHÔNG gọi WQ Brain API/tốn sim quota — chỉ
    `score_one()` local.
  - Trước khi động vào `main.py`, commit tách riêng các thay đổi tài liệu/skill tồn đọng
    không liên quan (theo lựa chọn của người dùng) để lịch sử merge sạch, không lẫn 2 việc.
- **In progress:** Không có.
- **Blockers / open risks:** Chưa chạy thật mục 5 (Auto SIM) qua đăng nhập thật trong phiên
  này (tốn quota Brain thật, cần xác nhận người dùng trước khi chạy) — chỉ xác nhận mục 4
  (local, LLM thật, không SIM Brain) chạy sạch không lỗi.
- **Next step:** Nếu người dùng muốn, chạy thật mục 5 (Auto SIM) với đăng nhập thật để xác
  nhận toàn luồng tới tận SIM Brain thật.
- **Tests:** Xanh toàn bộ trước và sau merge — `pytest -q`: 879 passed, 1 fail có sẵn
  (`tests/test_db_postgres.py::test_make_engine_postgres_backend`, thiếu module `psycopg`,
  không liên quan, đã xanh từ trước phiên này). `ruff check --select F401,F811,F821` sạch
  trên các file sửa. Đã merge `closed-loop-integration` → `main` (merge commit) và
  `git push origin main` thành công (`fbe4b2b..70e3e54`).

### [2026-07-06] Session 04 — Chạy thật Auto SIM (mục 5) + đánh giá e2e + cải thiện
- **Phase:** Sau Phase 8 — LẦN CHẠY THẬT ĐẦU TIÊN của closed-loop qua đăng nhập thật (đúng
  Next step tồn đọng). Nhánh `alpha-quality-from-brain-docs`.
- **Done (đánh giá e2e bằng dữ liệu THẬT):**
  - Auth non-interactive OK (session `.wq_session` còn hạn; `.env` có mật khẩu). Crash `✅`
    cp1252 CHỈ khi gọi python trực tiếp — `run.ps1`+`main.py` đã set UTF-8 nên KHÔNG phải bug.
  - Chẩn đoán 240 sim: **67% error** lịch sử (áp đảo auth-expiry cũ, nay đã có
    `AuthExpiredError`); `failed_checks` THẬT của Brain = LOW_SHARPE/LOW_FITNESS/**IS_LADDER_
    SHARPE**/LOW_2Y_SHARPE/LOW_SUB_UNIVERSE_SHARPE → xác nhận Brain thật sự enforce IS-Ladder.
  - **Sim lại winner core (VWAP intraday-reversal) dưới luật hiện tại: status=passed,
    failed_checks=[], Sharpe 1.57, fitness 0.73** → pipeline CÓ alpha qua hết is.checks;
    fitness>1 KHÔNG phải hard-check account này. Gate còn lại: self-corr (endpoint
    `/correlations/self` trả HTTP 200 body RỖNG — WQ tính bất đồng bộ, cần poll; chưa verify).
  - **Calibration xác nhận local≈Brain/1.28** (winner local Sharpe 1.23 vs Brain 1.57).
  - **Nút thắt phát hiện:** (1) throughput thấp (~3 sim/30ph; refine claude-cli chậm + đôi khi
    phản tác dụng, hạ Sharpe 1.57→1.46); (2) GP seed ngẫu nhiên bỏ lỡ họ VWAP tốt → run sinh
    core rác (Sharpe <0.4); (3) submissions=0 (loop chỉ khám phá).
- **Cải thiện đã commit:**
  - `b57ae66` pre-sim floor OPT-IN (`min_sharpe`/`require_is_ladder`, mặc định TẮT — tránh đói
    loop khi ρ chưa calibrate; local IS-Ladder dùng ngưỡng Brain nên hiện quá strict, để soft).
  - `d481fe1` seed họ `reversal` bằng 9 core intraday-reversal ĐÃ KIỂM CHỨNG (close↔vwap,
    close↔open + tổ hợp) — PV-only, chấm local được → GP tin cậy khám phá vùng tốt.
- **In progress:** phiên `wq_autosim_v2.log` (PID mới, market_yf, pop20×gen2, patience2,
  base_seed0, seed intraday) đang chạy — đánh giá xem seed cải thiện có ra alpha tốt ổn định.
- **Blockers / open risks:** self-corr chưa verify được (endpoint cần poll — `CorrelationChecker`
  chưa xử lý 200+body rỗng → crash JSON). Submit là hành động KHÔNG đảo ngược → cần người dùng
  đồng ý, KHÔNG auto-submit. Đạt "2h→1 submit alpha" tin cậy là bài toán nghiên cứu (vượt
  Sharpe~1.58 + IS-Ladder), không chỉ tinh chỉnh tham số.
- **Next step:** Đánh giá `wq_autosim_v2.log`; nếu seed intraday cho alpha passed ổn định →
  fix `CorrelationChecker` (poll 200+empty) để verify self-corr, rồi hỏi người dùng về submit.
- **Tests:** pytest xanh (1021→ +tests mới), 1 fail psycopg có sẵn. 3 commit code trên nhánh.

### [2026-07-06] Session 03 — Nâng cấp chất lượng alpha từ docs WQ Brain (4 sub-agent)
- **Phase:** Sau Phase 8 — đợt cải thiện chất lượng alpha độc lập, nhánh
  `alpha-quality-from-brain-docs` (chưa merge). Không đụng luồng ClosedLoop/quota.
- **Done:**
  - Đọc toàn bộ `docs/worldquantbrain/docs` (74 file) bằng **4 sub-agent Explore song song**
    theo 4 nhóm: neutralization/risk, submission-tests/settings, consultant-quality,
    dataset/seed. Tổng hợp thành spec
    `docs/superpowers/specs/2026-07-06-alpha-quality-upgrade-from-brain-docs.md` (bảng ngưỡng
    submission THẬT + đòn bẩy neutralization + khuôn mẫu chống-crowding + roadmap 10 mục) và
    memory `reference_brain_submission_thresholds`. Commit `1490530`.
  - **Commit `912a797` — IS-Ladder robustness gate:** module thuần `src/backtest/is_ladder.py`
    (`ladder_decision` + `is_ladder_verdict`) xét Sharpe cửa sổ trượt N=2..10 năm gần nhất
    (FAIL<1.58, PASS thang 2.38..1.59, turnover<30% ×0.85); ngưỡng ở `config/thresholds.py`;
    wire vào `MetricsCalculator` (thêm field `is_ladder_passed/detail` CÓ DEFAULT -> không phá
    17 constructor `AlphaMetrics` cũ) + soft-score `is_ladder` của `GateEvaluator`. Bắt alpha
    suy thoái gần đây mà Sharpe-tổng bỏ sót. TDD 12 case.
  - **Commit `5d12a0e` — novel-ideas v2:** thêm 6 alpha cấu trúc GAP/GATE/RESIDUAL
    (`vector_neut` residualize = lever DUY NHẤT hạ self-corr, gap zscore, `trade_when` gate)
    trên `VERIFIED_FIELDS` (KHÔNG bịa field mới, theo cardinal rule #1). Danh sách riêng
    `NOVEL_ALPHAS_V2` + `all_novel_alphas()`; `seed_cores_from_novel_ideas` nay seed 16 core
    (10 v1 + 6 v2), 6/6 parse qua registry thật. TDD `tests/test_novel_ideas_v2.py`.
- **Decisions:**
  - Panel local `market_yf` CHỈ có PV -> alpha dataset-thay-thế thuộc path Brain, không chấm
    local được; vì thế originality cắm ở `novel_ideas.py`/seed (ground field), KHÔNG hardcode
    field vào panel. Bài học cốt lõi từ docs: "alpha tốt sống trong GAP/GATE/RESIDUAL, không
    phải LEVEL" -> ưu tiên `vector_neut` (hạ self-corr — nguyên nhân reject #1).
  - IS-Ladder để SOFT (không chặn `passed`) đúng triết lý gate hiện tại: Brain là trọng tài
    cuối, gate local chỉ pre-filter tiết kiệm quota.
  - KHÔNG lật cứng `DEFAULT_NEUTRALIZATION` SUBINDUSTRY->INDUSTRY (mâu thuẫn commit `c33983b`
    có chủ đích); docs khuyên neutralization THEO CATEGORY — đã có `sweep-config` quét
    neutralization sẵn, nên để roadmap thay vì đổi default. KHÔNG làm ngoại lệ self-corr
    Sharpe+10% trong phiên này (checker chỉ trả max-corr, cần Sharpe từng alpha + format
    response thật chưa chắc — rủi ro trên đường submission thật).
- **In progress:** Không (nhánh sạch, chờ người dùng duyệt merge).
- **Blockers / open risks:** Field trong v2 dựa `VERIFIED_FIELDS` đã verify trước đó; nếu DB
  đổi tên field cần verify lại. Nhánh chưa merge `main`.
- **Next step:** Xem `Current state`.
- **Tests:** `venv` pytest: **1021 passed**, 1 fail có sẵn (`test_db_postgres` thiếu psycopg,
  không liên quan). `ruff --select F401,F811,F821` sạch trên file sửa. 3 commit trên nhánh
  `alpha-quality-from-brain-docs`.

### [2026-07-02] Session 02 — Đối chiếu mục 5 (Auto SIM) với tài liệu thiết kế + vá gap quota
- **Phase:** Sau Phase 8 — kiểm tra hoàn thiện `ClosedLoop` (mục 5) trước khi bắt đầu kiểm thử
  thật với WQ Brain.
- **Done:** Đọc lại toàn bộ `docs/superpowers/specs/2026-06-26-ai-minibrain-closed-loop-design.md`
  + 6 plan phase (1/2/3/4A/4B/4C) và đối chiếu từng phần với code hiện tại (`ClosedLoop`,
  `RefinementLoopRefiner`, `GPIdeaSource`, `CalibrationTracker`, `pool_corr_fn`,
  `RefinementLoop.repo.save_alpha/save_simulation`). Xác nhận: data flow chính, 3/4 feedback
  (avoid-list, calibrate ρ, pool self-corr Brain tầng-2 qua `/correlations/self`) đã wiring
  đầy đủ; kết quả SIM ghi đúng vào cả `BrainSimLinkModel` lẫn `AlphaModel`/`SimulationModel`
  (nên `top`/`submit` thấy được alpha do Auto SIM tìm ra).
- **Decisions:** Vá phòng ngừa gap "phát hiện hết quota ngày không chính xác" (tự tài liệu
  Phase 4C ghi "best-effort, chốt sau lần chạy thật đầu tiên" — nhưng chưa có lần chạy thật
  nào). Thêm `QuotaExceededError` (`src/simulation/simulator.py`, song song `AuthExpiredError`)
  nhận diện 429 dai dẳng hoặc header `X-Ratelimit-Remaining<=0` trên response
  `POST /simulations`, wire vào `RefinementLoopRefiner.refine_and_sim`
  (`src/app/closed_loop_adapters.py`) — cả hai loại lỗi đều ánh xạ sang `QuotaExhausted` để
  `ClosedLoop` dừng gọn. Chọn vá TRƯỚC khi test thật (theo yêu cầu người dùng) thay vì chờ
  quan sát hành vi thật rồi mới sửa — vì logic chỉ verify qua fake response, CHƯA verify với
  response thật của WQ khi hết quota (ghi rõ ở Next step để theo dõi khi chạy thật).
- **In progress:** Không.
- **Blockers / open risks:** Feedback (d) "AI học từ SIM" chỉ refresh avoid-subtree/zoo giữa
  các phiên, không động trong 1 phiên dài — chấp nhận cho v1. `QuotaExceededError` logic mới
  chưa verify với response thật của WQ Brain.
- **Next step:** Chạy thật mục 5 (Auto SIM) — xem `Current state` phía trên.
- **Tests:** `pytest -q`: 883 passed (879 + 4 mới: 3 `test_simulator.py` + 1
  `test_closed_loop_adapters.py`), 1 fail có sẵn không liên quan (psycopg). `ruff --select
  F401,F811,F821` sạch. Commit `265f2e8`, đã push `main` (`0ebc68d..265f2e8`).

### [2026-07-10] Session 08 — IMPROVEMENT_SPEC Pha 0: Instrumentation funnel
- **Phase:** Bắt đầu triển khai `docs/tailieu/IMPROVEMENT_SPEC.md` (5 pha, tuần tự — user
  chốt). Design ánh xạ spec→code: `docs/superpowers/specs/2026-07-10-improvement-spec-implementation-design.md`.
  Pha 0 (instrumentation) XONG code+test; các pha 1-4 chưa bắt đầu.
- **Điều tra trước (3 sub-agent, §5):** nền tảng đã có nhiều — `CanonicalHasher` sort giao
  hoán+normalize literal+hash AST (CHƯA fold hằng số), lưu SQLite `expressions.canonical_hash
  UNIQUE` bền cross-session; GP depth cap 7 + parsimony mềm NSGA-II; `CorrelationChecker` poll
  checker THẬT 0.70; `regression_neut`/`vector_neut` đã implement (chưa dùng làm lever). Thiếu:
  fail_check bị vứt (`closed_loop_adapters.py` cũ `_reasons`), CSV nuốt sharpe khi failed,
  không session_summary, dedup dùng string thô (chạy SAU backtest).
- **Done (verified, pytest xanh):**
  - `IdeaOutcome` +9 trường Pha 0 (stage_reached/fail_check/family/expr_depth/gen_ms/
    backtest_ms/sim_ms/dedup_key/local_sharpe), default tương thích ngược. `sharpe`/`fitness`
    giữ nguyên = brain metric (giảm blast-radius; CSV tách cột local_/brain_).
  - `RunAlphaLogger` schema cố định 27 cột, LUÔN điền đủ, ghi brain metric BẤT KỂ passed
    (khác log cũ nuốt sharpe → không phân tích tự động được).
  - `src/reporting/diagnostics.py`: `fail_check_from_reasons` (reasons→mã LOW_SHARPE/…),
    `classify_family` (suy họ từ field). `src/reporting/session_summary.py`: funnel theo
    stage, phân bố fail_check/family, median timing, dup — render markdown + ghi file.
  - `LocalTunerRefiner` giữ `_reasons` (trước bị vứt) + đo perf_counter + điền field mọi
    đường return. `ClosedLoop` nhận `session_summary` (record + record_dup_blocked). `main.py`
    dựng SessionSummary → ghi `logs/session_summary_*.md` + in cuối phiên.
- **Decisions:** (1) Canonical fold (Pha 1) sẽ **strip mọi multiply/divide hằng DƯƠNG** khỏi
  dedup key (giữ hằng ÂM = đổi dấu = alpha khác); user ủy quyền, sẽ khóa bằng golden test ở
  Pha 1. (2) Giữ tên trường `sharpe`/`fitness` trên IdeaOutcome thay vì đổi tên — tránh vỡ
  mọi consumer; việc tách chỉ ở tầng CSV. (3) TDD từng bước, mỗi task 1 commit.
- **In progress:** Pha 1 (canonical dedup + depth guard trước backtest + parsimony/hằng số
  rời rạc) — chưa bắt đầu.
- **Blockers / open risks:** "Chạy 1 phiên baseline funnel" (acceptance Pha 0, single-variable
  §6) cần WQ Brain QR-login trong terminal thật — KHÔNG chạy được qua Claude Code. User cần tự
  chạy menu-5 để lấy `session_summary` baseline trước khi so sánh Pha 1+.
- **Next step:** Pha 1.1 — thêm constant-folding scale dương vào `CanonicalHasher` + golden
  test bất biến PnL; rồi chuyển dedup sang dùng hash TRƯỚC backtest ở `ClosedLoop`.
- **Tests:** `pytest -q`: 1181 passed, 1 fail pre-existing (psycopg thiếu, không liên quan).
  Thêm test: `test_idea_outcome_fields`, `test_run_alpha_log` (viết lại), `test_session_summary`,
  `test_diagnostics`, `test_local_refiner_instrumentation`, +1 case `test_closed_loop`.
  Commits: `cf525ce` (design) → `3dd94f3` → `d03f75c` → `94459e3`.

### [2026-07-10] Session 08 (tiếp) — IMPROVEMENT_SPEC Pha 1: Chặn rác sớm & rẻ (throughput)
- **Phase:** Pha 1 (throughput) XONG cả 4 mục trên `main`. Đích: đổi thứ tự gate sang rẻ→đắt
  (syntax/depth/canonical-dedup TRƯỚC backtest) + cắt bloat/biến thể trùng.
- **Done (verified, pytest xanh 1198 passed, 1 fail postgres pre-existing):**
  - **1.1 Canonical fold (`d235515`):** `CanonicalHasher._fold_positive_scale_at_root` bóc
    multiply/divide hằng DƯƠNG toàn-alpha Ở GỐC → `multiply(4,X)≡multiply(2,X)≡X`. Golden
    `tests/golden/test_fold_scale_invariant.py` chứng minh bất biến positions trên
    PortfolioBuilder THẬT (3 config) — cơ sở: `_scale` chia L1-norm nên (k·s)/Σ|k·s|=s/Σ|s|.
    KHÔNG fold hằng âm / divide(k,X) phi tuyến / scale chôn trong add (trọng số tương đối).
  - **1.2 Dedup hash trước backtest (`fa094df`):** `ClosedLoop.dedup_key_fn` (inject, giữ B1)
    — seen chứa canonical key, nạp `repo.avoided_hashes()` (hash cross-session, mới). Biến
    thể scale chặn ở dedup, 0 backtest. `build_closed_loop` tiêm CanonicalHasher thật.
  - **1.3 Depth guard (`b9a5b39`):** loại cây trần > MAX_DEPTH(7) ngay đầu `refine_and_sim`,
    stage=depth/fail_check=DEPTH, 0 sim (trước cả `_tune`).
  - **1.4 Hằng số rời rạc (`45695b6`):** `DISCRETE_SCALARS {-2,-1,-0.5,0.5,1,2}` thay
    `rng.uniform` ở init + mutate resample rời rạc thay Gaussian.
- **Decisions:** (1) Fold CHỈ ở gốc (whole-alpha), KHÔNG toàn cục — phân tích lại thấy fold
  toàn cục gộp nhầm 2 VERIFIED_CORES (`add(multiply(2,C1),...)` vs `add(multiply(1,C1),...)`
  là trọng số tương đối khác = alpha khác). Thu hẹp về tập chứng minh được. (2) Tinh chỉnh hệ
  số `complexity_penalty` (parsimony mạnh hơn) KHÔNG làm ở đây — để làm biến A/B đo phiên thật
  (§6), tránh áp mù hại NSGA-II Pareto; depth-guard + hằng rời rạc + hoist đã cắt bloat an toàn.
- **In progress:** Pha 2 (chuyển họ nhân tố — yield) chưa bắt đầu.
- **Blockers / open risks:** Cùng Pha 0 — cần USER chạy menu-5 baseline. Acceptance Pha 1
  ("median độ dài giảm ≥30%") cần đo phiên thật; phần "không còn float 15 chữ số" đã đảm bảo
  bằng DISCRETE_SCALARS.
- **Next step:** Pha 2.1 — `include_alt_data=True` mặc định + thêm nguồn fundamental (ts_backfill,
  verify field LIVE qua get_datafields) vào idea source của closed-loop.
- **Tests:** `pytest -q` 1198 passed, 1 fail pre-existing (psycopg). Thêm:
  `test_fold_scale_invariant` (golden), `test_gp_discrete_scalars`, +cases hasher/brain_sim_link/
  closed_loop/local_refiner_instrumentation. Commits `d235515`→`45695b6`.

### [2026-07-10] Session 08 (tiếp) — IMPROVEMENT_SPEC Pha 2: Chuyển họ nhân tố (yield)
- **Phase:** Pha 2 (yield — đòn quyết định) XONG cả 3 mục trên `main`. User đã QR-login lại
  nên verify được field LIVE (gỡ ràng buộc chính).
- **Done (verified, pytest 1210 passed, 1 fail postgres pre-existing):**
  - **2.1 alt-data default + fundamental (`cd0a801`):** `include_alt_data`+`include_fundamental`
    mặc định True; gộp ALT_DATA_CORES + FUNDAMENTAL_CORES vào 1 batch đầu đi thẳng Brain. CLI
    `--no-alt-data` để A/B. `FUNDAMENTAL_CORES` (src/generation/fundamental_seeds.py) trên field
    fundamental6 ĐÃ VERIFY LIVE (assets/cashflow_op/revenue/operating_income/sales_growth):
    gross-profitability, cash-flow yield, asset growth (fade), accruals, sales growth — mọi
    field ts_backfill(,66), neutralization INDUSTRY.
  - **2.2 family budget + exhaustion (`0af6cbe`):** ClosedLoop `family_fn`+`max_per_family`(8):
    đóng họ khi cạn ngân sách mà 0 pass -> chuyển sang họ orthogonal. build_closed_loop tiêm
    classify_family.
  - **2.3 prompt họ bão hoà (`e2d62f4`):** LLMAlphaGenerator.set_saturated_families tiêm vào
    prompt; ClosedLoop.on_family_closed callback phát set họ đóng.
- **Decisions:** (1) Flip alt-data/fundamental mặc định True (spec §2.1) NHƯNG thêm --no-alt-data
  giữ khả năng A/B single-variable §6. (2) Field fundamental verify LIVE qua get_datafields
  (fundamental6, coverage 0.5) — KHÔNG bịa (cardinal rule #1); MCP create_simulation tool trả
  400 cả với rank(close) => lỗi MCP wrapper, KHÔNG phải field; sim thật chạy qua Simulator repo
  khi user chạy menu-5. (3) on_family_closed để ngỏ, không nối vào đường research (LocalTuner
  mặc định không có idea_generator) — tránh over-reach.
- **In progress:** Pha 3 (config stage đúng lever self-corr) chưa bắt đầu.
- **Blockers / open risks:** Seed social trước đó SAI DẤU (memory) — theo dõi khi chạy thật.
  Fundamental cores CHƯA sim thật trên Brain (MCP tool 400) — cần menu-5 xác nhận Sharpe/
  failed_checks. Acceptance Pha 2 (">=60% ứng viên KHÔNG pv_reversal, >=3 họ") cần đo phiên thật.
- **Next step:** Pha 3.1 — thêm nhánh cấu hình bọc regression_neut/vector_neut vào LocalTuner
  để hạ self-corr (toán tử DUY NHẤT làm được điều này; self-corr là ràng buộc hạng nhất).
- **Tests:** `pytest -q` 1210 passed, 1 fail pre-existing. Thêm: test_fundamental_seeds,
  cases test_closed_loop_adapters (alt-data/fundamental default), test_closed_loop (family
  budget/callback), test_generator (saturated families). Commits `cd0a801`→`e2d62f4`.

### [2026-07-10] Session 08 (tiếp) — IMPROVEMENT_SPEC Pha 3+4: self-corr lever + floor calibrated
- **Phase:** Pha 3 (config stage đúng lever) + Pha 4 (floor calibrated) XONG trên `main`.
  **HOÀN TẤT CẢ 5 PHA (0→4) của IMPROVEMENT_SPEC.**
- **Done (verified, pytest 1218 passed, 1 fail postgres pre-existing):**
  - **3.1 regression_neut lever (`407b50d`):** tune() Giai đoạn 3 thử bọc regression_neut(best,
    risk_factor) trừ thành phần crowded -> hạ self-corr Brain (toán tử DUY NHẤT). Bất biến đơn
    điệu (chỉ nhận nếu điểm không tệ hơn). neut_risk_factors inject; main dùng rank(volume).
    Golden test_regression_neut_orthogonal: residual trực giao risk factor (corr<1e-6).
  - **4 floor calibrated (`11b0614`):** calibrated_floor(target/1.28) suy floor từ MỤC TIÊU
    Brain sharpe thay hằng cứng 0.5. stop_reason ghi kèm ngưỡng (local_floor(<0.50)) +
    local_sharpe để audit. stage_reached giữ 'local_floor' -> funnel không đổi.
- **Decisions:** (1) Pha 3 phần phụ (ưu tiên ts_rank>ts_zscore, turnover-alpha không decay) để
  lại làm biến A/B đo thật — micro-opt cần đo, không phải lever chính; regression_neut là lever
  quyết định đã xong. (2) Floor giữ giá trị 0.5 nhưng nay DERIVED từ target 0.64/1.28 — chỉnh 1
  chỗ (target) thay vì hằng rải rác; nâng target để siết quota.
- **In progress:** Không còn. Cả 5 pha code+test xong.
- **Blockers / open risks:** TẤT CẢ acceptance cần USER chạy menu-5 để đo phiên thật (baseline
  funnel; median độ dài -30%; >=60% không pv_reversal + >=3 họ; self-corr giảm nhờ regression_
  neut; tỉ lệ chết local_floor giảm). Fundamental cores + regression_neut chưa sim thật trên
  Brain (MCP create_simulation tool trả 400 — lỗi wrapper, sim thật qua Simulator repo menu-5).
- **Next step:** USER chạy menu-5 (1 phiên) -> đọc logs/session_summary_*.md so baseline; nếu
  đạt, cân nhắc chạy A/B --no-alt-data để đo đóng góp yield. Tùy chọn: tinh chỉnh complexity_
  penalty (Pha 1) + ts_rank>ts_zscore (Pha 3) dựa trên số liệu phiên thật.
- **Tests:** `pytest -q` 1218 passed, 1 fail pre-existing. Thêm: test_local_tuner_tune (3 case
  regression_neut), test_regression_neut_orthogonal (golden), test_calibrated_floor. Commits
  `407b50d`→`11b0614`.
