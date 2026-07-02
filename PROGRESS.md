# MiniBrain — Progress log

## Current state
- **Phase:** Sau Phase 8 (short-list + CLI) — vòng kín AI+MiniBrain (ClosedLoop) đã merge
  vào `main`; menu tương tác `run.bat` vừa được rút gọn còn 5 mục.
- **Done:** Phase 0-8 theo `docs/tailieu/BUILD_GUIDE_AI_alpha_tool.md` (đăng nhập, data
  fields/operators, GP engine, backtest/metrics/gate local, LLM refine, short-list
  decorrelate). Nhánh `closed-loop-integration` (DB cầu Brain SIM <-> MiniBrain, orchestrator
  `ClosedLoop`, adapter `GPIdeaSource`/`RefinementLoopRefiner`, CalibrationTracker) đã merge
  vào `main` + push (`fbe4b2b..70e3e54`). Menu run.bat còn đúng 5 mục: đăng nhập (tự
  ensure fields/operators) / tải lại fields / tải lại operators / test engine cục bộ (không
  cần đăng nhập) / Auto SIM (vòng kín thật). Dọn code chết `_auto_prepare`/`AutoPipeline`/
  `login.bat`; gộp trùng lặp CLI `closed-loop` + menu qua `_run_closed_loop_session`.
- **In progress:** Không có việc dở dang — vừa merge + push xong.
- **Next step:** Chạy thật mục 5 (Auto SIM) với đăng nhập thật để xác nhận end-to-end với
  WQ Brain SIM/quota thật (mục 4 test engine cục bộ đã xác nhận wiring sạch, nhưng chưa thử
  SIM Brain thật qua đường menu mới).
- **Blockers / open risks:** Không có blocker biết tới ở thời điểm này. LLM_BACKEND hiện là
  `deepseek` (đã xác nhận API còn hoạt động khi chạy mục 4 thật trong phiên này).
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
