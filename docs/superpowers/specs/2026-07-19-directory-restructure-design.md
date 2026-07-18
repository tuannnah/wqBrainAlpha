# Cấu trúc lại thư mục dự án + tách main.py

**Ngày:** 2026-07-19
**Trạng thái:** Đã duyệt (chờ viết plan)

## Bối cảnh

Thư mục gốc dự án đang lẫn lộn nhiều loại file (entrypoint, config, DB, docs review cũ),
`scripts/` và `tools/` không có quy ước rõ ràng, `docs/` có tầng con khó nhớ, và `main.py`
đã phình tới 2133 dòng (1 Typer CLI app gồm ~23 lệnh + khối menu tương tác), gây khó tìm/khó
sửa. Mục tiêu: gọn gàng, dễ kiểm soát, không đổi hành vi runtime.

Gồm 2 dự án con độc lập, làm tuần tự trong 1 plan:
- **Dự án A** — cấu trúc thư mục (di chuyển file, rủi ro thấp)
- **Dự án B** — tách `main.py` thành các module theo nhóm nghiệp vụ (refactor code, rủi ro
  trung bình vì có 14 file test import trực tiếp từ `main`)

## Dự án A — Cấu trúc thư mục

### Gốc repo (sau khi dọn)
Chỉ còn: `main.py`, `README.md`, `PROGRESS.md` (bắt buộc ở gốc theo quy ước của skill
`session-journal` — không di chuyển), `pytest.ini`, `requirements.txt`, `run.bat`, `run.ps1`,
`.gitignore`, `.env`/`.env.example`, cùng các thư mục `config/`, `data/`, `docs/`, `logs/`,
`scripts/`, `skill/`, `src/`, `tests/`, `tools/`, `venv/`, `dashboard/`.

`run_loop5.sh` → chuyển vào `scripts/`.

### DB files → `data/db/`
Di chuyển `wq_alpha_tuananhpo13_gmail_com.db(+.db-shm/.db-wal)` và
`wq_alpha_phtrang1229_gmail_com.db` vào `data/db/`. Cần sửa code (không chỉ di chuyển file):

- `src/storage/db.py`: `DEFAULT_SQLITE_URL = "sqlite:///wq_alpha.db"` →
  `"sqlite:///data/db/wq_alpha.db"`; `active_database_url()` trả về
  `f"sqlite:///data/db/wq_alpha_{_email_slug(email)}.db"`. Thêm tạo thư mục
  `data/db/` nếu chưa tồn tại trước khi SQLAlchemy kết nối (SQLite không tự tạo parent dir).
- `config/settings.py`: default `database_url` phải khớp **y hệt** chuỗi mới ở trên (so sánh
  bằng `==` với `DEFAULT_SQLITE_URL` trong `db.py`).
- `main.py` (lệnh `migrate_sqlite`, hiện dòng 90): default `source` URL → dùng chuỗi mới
  (nên tham chiếu `db.DEFAULT_SQLITE_URL` thay vì hardcode lại).
- `scripts/run_groundtruth.py`, `scripts/persist_groundtruth.py`: `DB_URL` hardcode →
  `"sqlite:///data/db/wq_alpha_phtrang1229_gmail_com.db"`.
- `tests/test_db_per_account.py`, `tests/test_migrate.py`: cập nhật các assert path.
- Di chuyển vật lý 4 file `.db*` hiện có vào `data/db/` (đảm bảo không có tiến trình nào đang
  giữ WAL/SHM mở khi move — đóng mọi phiên `main.py`/dashboard trước khi thực hiện).

`dashboard/app.py` và `tools/diag_combiner.py` chỉ gọi `make_engine()`/`active_database_url()`
nên tự động ăn theo, không cần sửa.

### `scripts/` vs `tools/` — quy ước
Không gộp, chỉ làm rõ bằng docstring/README ngắn ở đầu mỗi thư mục:
- `scripts/` = script chạy một lần để chuẩn bị dữ liệu/ground-truth: `fetch_yfinance_panel.py`,
  `gen_groundtruth.py`, `generate_alphas.py`, `generate_novel.py`, `persist_groundtruth.py`,
  `run_groundtruth.py`, `verified_cores_provenance.py`, `run_loop5.sh` (mới chuyển vào).
- `tools/` = tiện ích chẩn đoán/verify dùng lặp lại trong quá trình phát triển:
  `diag_combiner.py`, `verify_datasets.py`, `verify_frontier_fields.py`.

### `docs/` — gọn tầng con
- `docs/tailieu/review20260710/` và `docs/tailieu/review20260711/` → gộp vào
  `docs/tailieu/old/` (đã xong việc, chỉ còn giá trị tham khảo lịch sử).
- `docs/worldquantbrain/` → đổi tên thành `docs/wq_scraped_docs/` (rõ ràng đây là dữ liệu
  scrape, tránh trùng tên với `skill/worldquant-brain/`). Cập nhật đường dẫn output trong
  `docs/worldquantbrain/wq_docs_scraper.py` (và bất kỳ nơi nào khác tham chiếu đường dẫn cũ)
  cho khớp tên thư mục mới.
- `docs/design/`, `docs/superpowers/{plans,specs}/`, `docs/tailieu/IMPROVEMENT_SPEC.md`,
  `docs/tailieu/old/` (nội dung hiện có): giữ nguyên vị trí.

### Ngoài phạm vi
- `skill/` (skill bundle phân phối riêng), `venv/`, `data/market_yf*/` — không đụng vào.
- `src/app/closed_loop_adapters.py` (90KB) cũng là ứng viên tách nhỏ trong tương lai nhưng
  **không** nằm trong phạm vi lần này — cần một thiết kế riêng.

## Dự án B — Tách `main.py`

`main.py` hiện là 1 Typer CLI app gồm 2 khối: ~23 lệnh CLI và khối menu tương tác. Tách theo
nhóm nghiệp vụ vào `src/app/cli/`, `main.py` chỉ còn là entrypoint mỏng: `_setup_logging`,
khởi tạo `typer.Typer()`, import và gắn command từ các module con (~50-80 dòng).

### Bảng ánh xạ module (theo dòng hiện tại trong `main.py`)

| Module đích | Nội dung chuyển sang |
|---|---|
| `src/app/cli/auth.py` | `prompt_credentials` (52), `_make_client` (65), `login` (75) |
| `src/app/cli/fields.py` | `probe_fields` (110), `warm_cache_cmd` (135), `fetch_fields` (177), `cache_status` (213), `fetch_operators` (235), `list_fields` (256) |
| `src/app/cli/simulate.py` | `simulate` (299), `sweep_config` (341), `_cached_symbols` (404), `_local_operator_arity` (436), `_make_invalid_field_recorder` (450), `_make_validated_simulator` (461), `_portfolio_config_from_opts` (479) |
| `src/app/cli/generate.py` | `generate` (499), `score_one_cmd` (575) |
| `src/app/cli/closed_loop.py` | `_resolve_base_seed` (626), `_run_reseed_until_quota` (635), `_local_neutralization` (652), `_closed_loop_configs` (668), `_run_closed_loop_session` (685), `closed_loop_cmd` (858) |
| `src/app/cli/llm.py` | `_make_deepseek` (931), `run_deepseek_smoke` (954), `describe_deepseek_smoke_error` (981), `check_deepseek` (993), `_make_router` (1024), `_make_llm_generator` (1036), `llm_generate` (1464), `llm_ideas` (1494) |
| `src/app/cli/research.py` | `_make_pool_corr_fn` (1051), `_make_pnl_fn` (1069), `_make_research_loop` (1104), `_render_research_result` (1192), `resolve_direction` (1236), `research` (1249), `_run_research_with_progress` (1321) |
| `src/app/cli/marathon.py` | `_render_marathon_report` (1352), `_marathon_direction_provider` (1369), `_marathon_on_event` (1388), `_run_marathon_session` (1402), `marathon` (1435) |
| `src/app/cli/report.py` | `top` (1509), `originality` (1552), `genius_report_cmd` (1649) |
| `src/app/cli/submit.py` | `submit` (1584) |
| `src/app/cli/migrate.py` | `migrate_sqlite` (89), `calibrate` (2039) |
| `src/app/menu.py` | `_MenuState` (1692), `_menu_counts` (1712), `_menu_login` (1723), `_menu_fields` (1755), `_menu_operators` (1771), `_find_market_data_dir` (1784), `_menu_test_engine` (1795), `_menu_auto_sim` (1893), `_menu_view_submit` (1916), `_print_menu` (1962), `start` (2003) |
| `main.py` (giữ lại) | `_setup_logging` (41), khởi tạo `app = typer.Typer()`, import + `app.command()` cho từng lệnh |

Mỗi module CLI con export các Typer command function; `main.py` import và gắn bằng
`app.command(name=...)(fn)` hoặc để mỗi module tự có `@app.command()` nếu dùng chung 1
instance `app` truyền vào — quyết định kỹ thuật cụ thể để writing-plans chọn theo pattern
Typer phù hợp nhất (single shared `app` instance là hướng đơn giản nhất, tránh sub-app
phức tạp không cần thiết cho quy mô này).

### Rủi ro & xử lý: import trong test

14 file test hiện `import` trực tiếp từ `main`:
`tests/test_main_login.py`, `tests/test_menu_counts.py`, `tests/test_closed_loop_seed.py`,
`tests/test_closed_loop_reseed.py`, `tests/test_auto_command.py`, `tests/test_pnl_fn.py`,
`tests/test_pool_corr_fn.py`, `tests/test_logging_setup.py`, `tests/test_deepseek_check.py`,
`tests/test_marathon_command.py`, `tests/test_research_direction.py`,
`tests/unit/test_cli_closed_loop.py`, `tests/unit/test_cli_score_one_generate.py`,
`tests/unit/test_calibrate_command.py`.

Mỗi file cần sửa import trỏ sang module mới tương ứng (không dùng shim/re-export tương thích
ngược trong `main.py` — sửa import thẳng ở nơi dùng, đúng theo quy ước "không hack tương
thích ngược" của dự án).

### Thứ tự thực hiện & kiểm chứng
Tách từng nhóm một (bắt đầu từ nhóm ít phụ thuộc chéo nhất, ví dụ `auth.py` hoặc `report.py`),
sau mỗi lần tách: cập nhật import ở test liên quan, chạy `pytest -q` cho các test đó rồi chạy
toàn bộ suite trước khi sang nhóm tiếp theo. Không đổi logic bên trong hàm — chỉ đổi vị trí +
import. Sau khi xong toàn bộ, `grep -rn "from main import\|^import main"` trong `tests/`
và `src/` phải không còn kết quả nào ngoài việc `main.py` tự import các module con của nó.

## Ngoài phạm vi (không làm trong plan này)
- Không tách `src/app/closed_loop_adapters.py`.
- Không đổi schema DB hay logic nghiệp vụ bất kỳ đâu.
- Không đổi hành vi CLI (tên lệnh, option, output) — đây thuần là di chuyển code/file.

## Kiểm chứng hoàn tất
- `pytest -q` toàn bộ suite xanh.
- `python main.py --help` liệt kê đủ 23 lệnh như trước.
- Chạy thử `run.bat` / menu tương tác (`start()`) không lỗi.
- `git status` sạch, không còn file rác ở gốc ngoài danh sách đã chốt.
