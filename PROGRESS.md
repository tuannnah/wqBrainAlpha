# MiniBrain — Progress log

## Current state
- **Phase:** Sau Phase 8 (short-list + CLI) — vòng kín đã merge `main`. Đang bổ sung một
  đợt **nâng cấp chất lượng alpha rút từ docs WorldQuant Brain** (nhánh
  `alpha-quality-from-brain-docs`, CHƯA merge): đọc toàn bộ `docs/worldquantbrain/docs`
  bằng 4 sub-agent song song -> spec tổng hợp + 2 nâng cấp code có test.
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
