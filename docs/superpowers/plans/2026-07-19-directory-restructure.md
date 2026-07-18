# Cấu trúc lại thư mục + tách main.py — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dọn gọn thư mục dự án (gốc, `scripts/`/`tools/`, `docs/`, `.db`) và tách `main.py`
(2133 dòng, 1 Typer CLI app) thành các module nhỏ theo nhóm nghiệp vụ dưới `src/app/cli/` +
`src/app/menu.py`, không đổi hành vi runtime.

**Architecture:** Dự án A (4 task) di chuyển file/thư mục và sửa các chỗ code hardcode
đường dẫn liên quan. Dự án B (14 task) tách `main.py` theo DAG phụ thuộc đã xác định
(`common` → 8 module lá → `research` → `closed_loop`/`marathon` → `menu`), mỗi task tự
kiểm chứng bằng test trước khi sang task kế — không có bước "dọn cuối cùng" gộp nhiều thay
đổi chưa kiểm chứng.

**Tech Stack:** Python 3, Typer (CLI), pytest, ruff (lint/unused-import cleanup), SQLAlchemy
+ SQLite, loguru, rich.

## Global Constraints

- TDD bắt buộc: sửa test trước (RED), sau đó mới sửa code (GREEN) — áp dụng cho cả refactor
  thuần túy (ở đây "RED" là import/monkeypatch trỏ sai chỗ gây lỗi ngay).
- Code/commit message/comment bằng tiếng Việt.
- Mỗi task = 1 commit riêng.
- Không tạo shim tương thích ngược (không re-export tên cũ từ `main.py`) — sửa thẳng nơi
  dùng.
- Không đổi hành vi CLI (tên lệnh, option, output) hay logic nghiệp vụ — thuần di chuyển vị
  trí code/file.
- Sau MỌI task: `pytest -q` toàn bộ suite phải xanh trước khi commit.

---

## Common Header dùng cho mọi module CLI mới (Dự án B)

Tất cả file mới trong `src/app/cli/` bắt đầu bằng khối này (rồi mới tới phần
cross-import riêng của từng module, rồi tới các hàm được chuyển sang):

```python
"""<mô tả ngắn 1 dòng, tiếng Việt, theo nội dung module>"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.data.client import WQBrainClient
from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository, count_max_arity
from src.data.universe_matrix import iter_scopes
from src.data.warm_cache import warm_cache
from src.simulation.simulator import Simulator
from src.storage.db import init_db, make_engine, make_session_factory
from src.storage.migrate import migrate_all, _same_database
from src.storage.repository import AlphaRepository, InvalidFieldRepository
from src.llm.marathon import MarathonReport, run_marathon

console = Console()
```

Sau khi dán các hàm được chuyển vào cuối file, chạy `ruff check --fix <file>` để tự động
xoá import không dùng tới (mỗi module chỉ giữ lại phần header nó thực sự cần). Không xoá
import thủ công trước khi chạy ruff — để ruff làm, tránh xoá nhầm thứ vẫn được dùng bên
trong hàm.

**Quy tắc cắt hàm khỏi `main.py`:** tìm hàm bằng `def <tên>(` (không dùng số dòng cứng vì
số dòng dịch chuyển sau mỗi task) — cắt từ dòng `@app.command(...)` phía trên nó (nếu có)
cho tới hết thân hàm (dòng trống trước `def`/`@app.command` tiếp theo ở top-level), dán
nguyên văn (không sửa logic bên trong) vào file đích, bỏ dòng `@app.command(...)` (command
sẽ được gắn thủ công trong `main.py` ở bước wiring).

---

# DỰ ÁN A — Cấu trúc thư mục

### Task A1: Chuyển DB SQLite vào `data/db/`

**Files:**
- Modify: `src/storage/db.py`
- Modify: `config/settings.py`
- Modify: `main.py` (lệnh `migrate-sqlite`, tìm `def migrate_sqlite(`)
- Modify: `scripts/run_groundtruth.py`
- Modify: `scripts/persist_groundtruth.py`
- Test: `tests/test_db_per_account.py`

**Interfaces:**
- Produces: `src.storage.db.DEFAULT_SQLITE_URL == "sqlite:///data/db/wq_alpha.db"`;
  `src.storage.db.active_database_url()` trả `sqlite:///data/db/wq_alpha_<slug>.db`;
  `src.storage.db.make_engine()` tự tạo thư mục cha nếu chưa có.

- [ ] **Step 1: Sửa test (RED) — cập nhật 3 assertion path**

Trong `tests/test_db_per_account.py`, sửa 3 dòng assert:

```python
def test_env_email_derives_per_account_db(tmp_path, restore_settings):
    settings.wq_email = "Tuan.Anh+wq@Gmail.com"
    settings.database_url = db.DEFAULT_SQLITE_URL
    acc = tmp_path / ".wq_account"
    assert db.active_database_url(acc) == "sqlite:///data/db/wq_alpha_tuan_anh_wq_gmail_com.db"


def test_account_file_fallback_when_env_empty(tmp_path, restore_settings):
    settings.wq_email = ""
    settings.database_url = db.DEFAULT_SQLITE_URL
    acc = tmp_path / ".wq_account"
    acc.write_text("foo@bar.com", encoding="utf-8")
    assert db.active_database_url(acc) == "sqlite:///data/db/wq_alpha_foo_bar_com.db"


def test_env_email_overrides_account_file(tmp_path, restore_settings):
    settings.wq_email = "env@x.com"
    settings.database_url = db.DEFAULT_SQLITE_URL
    acc = tmp_path / ".wq_account"
    acc.write_text("file@y.com", encoding="utf-8")
    assert db.active_database_url(acc) == "sqlite:///data/db/wq_alpha_env_x_com.db"
```

Thêm test mới cuối file kiểm tra `make_engine` tự tạo thư mục cha:

```python
def test_make_engine_tao_thu_muc_cha_neu_chua_co(tmp_path):
    target = tmp_path / "newsub" / "test.db"
    assert not target.parent.exists()
    engine = db.make_engine(f"sqlite:///{target}")
    engine.dispose()
    assert target.parent.exists()
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `pytest tests/test_db_per_account.py -v`
Expected: 3 test cũ FAIL (assert path lệch — code vẫn trả `sqlite:///wq_alpha_...db` không
có `data/db/`), test mới FAIL (`FileNotFoundError` từ SQLite vì thư mục `newsub/` chưa có).

- [ ] **Step 3: Sửa `src/storage/db.py`**

```python
DEFAULT_SQLITE_URL = "sqlite:///data/db/wq_alpha.db"
```

```python
def active_database_url(account_file: Path = ACCOUNT_FILE) -> str:
    url = settings.database_url
    if url != DEFAULT_SQLITE_URL:
        return url
    email = (settings.wq_email or read_active_account(account_file)).strip()
    if not email:
        return url
    return f"sqlite:///data/db/wq_alpha_{_email_slug(email)}.db"


def make_engine(database_url: str | None = None) -> Engine:
    url = database_url or active_database_url()
    if url.startswith("sqlite") and ":memory:" not in url:
        db_path = Path(url[len("sqlite:///"):])
        db_path.parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, future=True, connect_args=connect_args)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine
```

- [ ] **Step 4: Sửa `config/settings.py` dòng `database_url`**

```python
    database_url: str = "sqlite:///data/db/wq_alpha.db"
```

- [ ] **Step 5: Sửa `main.py` — default của lệnh `migrate-sqlite`**

Tìm `def migrate_sqlite(`, sửa dòng `source`:

```python
    source: str = typer.Option("sqlite:///data/db/wq_alpha.db", help="URL DB nguồn (SQLite)"),
```

- [ ] **Step 6: Sửa `scripts/run_groundtruth.py` và `scripts/persist_groundtruth.py`**

Cả hai file, đổi dòng `DB_URL`:

```python
DB_URL = "sqlite:///data/db/wq_alpha_phtrang1229_gmail_com.db"
```

- [ ] **Step 7: Chạy test, xác nhận PASS**

Run: `pytest tests/test_db_per_account.py -v`
Expected: PASS toàn bộ (kể cả test mới).

- [ ] **Step 8: Di chuyển vật lý các file `.db` hiện có**

Đảm bảo không có tiến trình nào đang mở DB (đóng mọi phiên `main.py`/dashboard đang chạy)
trước khi chạy:

```bash
mkdir -p data/db
mv wq_alpha_tuananhpo13_gmail_com.db data/db/
mv wq_alpha_tuananhpo13_gmail_com.db-shm data/db/ 2>/dev/null || true
mv wq_alpha_tuananhpo13_gmail_com.db-wal data/db/ 2>/dev/null || true
mv wq_alpha_phtrang1229_gmail_com.db data/db/
```

- [ ] **Step 9: Chạy toàn bộ test suite**

Run: `pytest -q`
Expected: PASS toàn bộ (không chỉ file test đã sửa — các test khác dùng `sqlite:///:memory:`
hoặc `tmp_path` nên không bị ảnh hưởng bởi đổi default path).

- [ ] **Step 10: Commit**

```bash
git add src/storage/db.py config/settings.py main.py scripts/run_groundtruth.py scripts/persist_groundtruth.py tests/test_db_per_account.py
git commit -m "refactor(db): chuyển DB SQLite mặc định vào data/db/"
```

(File `.db` vật lý đã gitignore, không cần `git add`.)

---

### Task A2: Quy ước `scripts/` vs `tools/` + chuyển `run_loop5.sh`

**Files:**
- Create: `scripts/README.md`
- Create: `tools/README.md`
- Modify: di chuyển `run_loop5.sh` → `scripts/run_loop5.sh`

- [ ] **Step 1: Di chuyển `run_loop5.sh`**

```bash
git mv run_loop5.sh scripts/run_loop5.sh
```

(Đã xác nhận không có file nào khác trong repo tham chiếu tới đường dẫn `run_loop5.sh` cũ,
nên không cần sửa gì thêm.)

- [ ] **Step 2: Tạo `scripts/README.md`**

```markdown
# scripts/

Script chạy một lần để chuẩn bị dữ liệu/ground-truth (không phải thư viện để import).
Chạy trực tiếp bằng `python scripts/<tên>.py` hoặc `venv/Scripts/python -m scripts.<tên>`.

Khác với `tools/` (tiện ích chẩn đoán/verify dùng lặp lại trong quá trình phát triển).
```

- [ ] **Step 3: Tạo `tools/README.md`**

```markdown
# tools/

Tiện ích chẩn đoán/verify dùng lặp lại trong quá trình phát triển (diagnose combiner,
verify datasets/fields...). Chạy trực tiếp bằng `python tools/<tên>.py`.

Khác với `scripts/` (script chạy một lần để chuẩn bị dữ liệu/ground-truth).
```

- [ ] **Step 4: Chạy test suite (không kỳ vọng gì đổi, chỉ xác nhận không vỡ)**

Run: `pytest -q`
Expected: PASS (không file test nào import theo đường dẫn cũ của `run_loop5.sh` — đây là
shell script, không phải module Python).

- [ ] **Step 5: Commit**

```bash
git add scripts/README.md tools/README.md scripts/run_loop5.sh
git commit -m "docs(scripts): thêm quy ước scripts/ vs tools/, chuyển run_loop5.sh vào scripts/"
```

---

### Task A3: Gộp `docs/tailieu/review2026071{0,1}/` vào `old/` + sửa path trong code

**Files:**
- Modify: di chuyển 2 thư mục con
- Modify: `scripts/verified_cores_provenance.py`

- [ ] **Step 1: Di chuyển 2 thư mục**

```bash
git mv docs/tailieu/review20260710 docs/tailieu/old/review20260710
git mv docs/tailieu/review20260711 docs/tailieu/old/review20260711
```

- [ ] **Step 2: Sửa `scripts/verified_cores_provenance.py`**

Dòng 6 (docstring module) và dòng 37 (`CLAIMED_LABEL`), đổi
`docs/tailieu/review20260710/` → `docs/tailieu/old/review20260710/`:

```python
docs/tailieu/old/review20260710/IMPROVEMENT_SPEC_v2.md mục C7). Đừng tin nhãn cũ mù quáng —
```

```python
    "(xem docs/tailieu/old/review20260710/IMPROVEMENT_SPEC_v2.md muc C7), dung tin nhan cu mu quang"
```

- [ ] **Step 3: Chạy test suite**

Run: `pytest -q`
Expected: PASS (không có test nào assert nội dung `CLAIMED_LABEL`/docstring này theo path
cũ — xác nhận bằng `grep -rn "CLAIMED_LABEL\|review20260710" tests/` trước khi commit, nếu
có test nào match thì cập nhật path trong test đó tương tự Step 2).

- [ ] **Step 4: Commit**

```bash
git add docs/tailieu scripts/verified_cores_provenance.py
git commit -m "docs(tailieu): gộp review20260710/20260711 vào old/, sửa path tham chiếu"
```

---

### Task A4: Đổi tên `docs/worldquantbrain/` → `docs/wq_scraped_docs/` + sửa tham chiếu sống

**Files:**
- Modify: di chuyển thư mục
- Modify: `docs/wq_scraped_docs/wq_docs_scraper.py`
- Modify: `src/submission/power_pool_quota.py`
- Modify: `src/simulation/config.py`
- Modify: `src/simulation/simulator.py`
- Modify: `src/scoring/genius_report.py`
- Modify: `src/scoring/power_pool_theme.py`
- Modify: `src/scoring/power_pool.py`
- Modify: `src/scoring/dataset_usage.py`

Các tham chiếu trong `PROGRESS.md`, `docs/superpowers/plans/*.md`,
`docs/superpowers/specs/*.md`, `.superpowers/research/*.md` **KHÔNG sửa** — đó là bản ghi
lịch sử (PROGRESS.md append-only theo quy ước skill `session-journal`; các spec/plan cũ ghi
lại quyết định tại thời điểm đó, sửa lại sẽ làm sai lệch lịch sử).

- [ ] **Step 1: Di chuyển thư mục**

```bash
git mv docs/worldquantbrain docs/wq_scraped_docs
```

- [ ] **Step 2: Sửa `docs/wq_scraped_docs/wq_docs_scraper.py`**

Dòng 14, 22, 23 (lệnh mẫu trong docstring) và dòng 164 (comment):

```python
       python docs/wq_scraped_docs/wq_docs_scraper.py --api
```
```python
       python docs/wq_scraped_docs/wq_docs_scraper.py --login
       python docs/wq_scraped_docs/wq_docs_scraper.py --crawl
```
```python
    # docs/wq_scraped_docs/wq_docs_scraper.py -> gốc dự án là 2 cấp trên.
```

- [ ] **Step 3: Sửa 7 file comment/docstring tham chiếu path cũ**

`src/submission/power_pool_quota.py:3`:
```python
Nguồn tiêu chí: docs/wq_scraped_docs/docs/consultant-information/power-pool-alphas.md
```

`src/simulation/config.py:23`:
```python
# docs/wq_scraped_docs/docs/advanced-topics/{statistical,crowding,ram}-risk-neutralized-alphas.md.
```

`src/simulation/simulator.py:115`:
```python
# docs/wq_scraped_docs/docs/_/brain-api.md dòng 266 (tài liệu chính thức /simulations POST):
```

`src/simulation/simulator.py:343`:
```python
        1. `docs/wq_scraped_docs/docs/_/brain-api.md` (dòng 213-337, tài liệu BRAIN API chính
```

`src/scoring/genius_report.py:2`:
```python
KHÔNG phải gate nộp. Nguồn: docs/wq_scraped_docs/docs/consultant-information/brain-genius.md
```

`src/scoring/power_pool_theme.py:27`:
```python
# docs/wq_scraped_docs/docs/advanced-topics/*-risk-neutralized-alphas.md). Token lạ -> bỏ qua.
```

`src/scoring/power_pool.py:5`:
```python
Nguồn tiêu chí: docs/wq_scraped_docs/docs/consultant-information/power-pool-alphas.md."""
```

`src/scoring/dataset_usage.py:4`:
```python
docs/wq_scraped_docs/docs/consultant-information/single-dataset-alphas.md và đính chính
```

- [ ] **Step 4: Xác nhận không còn tham chiếu path cũ trong code sống**

Run: `grep -rn "docs/worldquantbrain" src/ scripts/ tools/ tests/`
Expected: không kết quả nào (0 dòng).

- [ ] **Step 5: Chạy test suite**

Run: `pytest -q`
Expected: PASS (các dòng sửa chỉ là comment/docstring, không ảnh hưởng logic).

- [ ] **Step 6: Commit**

```bash
git add docs/wq_scraped_docs src/submission/power_pool_quota.py src/simulation/config.py src/simulation/simulator.py src/scoring/genius_report.py src/scoring/power_pool_theme.py src/scoring/power_pool.py src/scoring/dataset_usage.py
git commit -m "docs(worldquantbrain): đổi tên docs/worldquantbrain -> docs/wq_scraped_docs"
```

---

# DỰ ÁN B — Tách `main.py`

Thứ tự bắt buộc theo DAG phụ thuộc: `common` trước tiên; rồi `auth`/`fields`/`simulate`/
`generate`/`submit`/`report`/`migrate`/`llm` (thứ tự tự do); rồi `research`; rồi
`closed_loop`/`marathon` (thứ tự tự do); cuối cùng `menu`.

Mỗi task dùng chung khuôn "Common Header" ở đầu tài liệu này.

---

### Task B1: `src/app/cli/common.py`

**Files:**
- Create: `src/app/cli/__init__.py` (rỗng)
- Create: `src/app/cli/common.py`
- Modify: `main.py`
- Test: `tests/test_auto_command.py`, `tests/test_marathon_command.py` (chỉ sửa dòng
  monkeypatch liên quan tới các tên chuyển sang `common.py` — xử lý đầy đủ ở Task B7/B10 khi
  `simulate`/`research`/`marathon` thật sự chuyển; ở task này CHƯA cần sửa 2 file test đó).

**Interfaces:**
- Produces: `src.app.cli.common._make_client() -> WQBrainClient`,
  `_cached_symbols(session_factory)`, `_local_operator_arity() -> dict[str, int]`,
  `_make_invalid_field_recorder(session_factory, region, universe)`,
  `_make_validated_simulator(client, pf, session_factory, region, universe)`,
  `_portfolio_config_from_opts(neutralization, decay, truncation, delay)`.

- [ ] **Step 1: Tạo `src/app/cli/__init__.py` rỗng**

- [ ] **Step 2: Tạo `src/app/cli/common.py`**

Dán Common Header (đầu tài liệu), sửa docstring dòng 1 thành:
`"""Helper CLI dùng chung nhiều nhóm lệnh (client factory, cache field/operator, config từ option)."""`

Sau đó dán nguyên văn (cắt khỏi `main.py`, giữ nguyên logic):
- Hàm `_make_client` (tìm `def _make_client(` trong `main.py`)
- Hàm `_cached_symbols` (tìm `def _cached_symbols(`)
- Hàm `_local_operator_arity` (tìm `def _local_operator_arity(`)
- Hàm `_make_invalid_field_recorder` (tìm `def _make_invalid_field_recorder(`)
- Hàm `_make_validated_simulator` (tìm `def _make_validated_simulator(`)
- Hàm `_portfolio_config_from_opts` (tìm `def _portfolio_config_from_opts(`)

(6 hàm này nằm liền nhau trong `main.py` gốc, giữa `sweep_config` và `generate` — cắt cả
khối một lần.)

- [ ] **Step 3: Xoá 6 hàm đó khỏi `main.py`**

Xoá từ dòng `def _make_client(` (không có decorator `@app.command`, nó là helper thường)
tới hết thân `_portfolio_config_from_opts`. Giữ nguyên `login()` phía trên (nó gọi
`_make_client()` — sẽ sửa ở Task B2) và `generate()` phía dưới.

- [ ] **Step 4: Chạy `ruff check --fix src/app/cli/common.py`**

Run: `venv/Scripts/python -m ruff check --fix src/app/cli/common.py`
Expected: xoá các import không dùng tới trong 6 hàm trên (vd `math`, `random`, `os`,
`MarathonReport`, `run_marathon`, `migrate_all`, `_same_database`, `Table`, `logger` nếu
không hàm nào trong file dùng tới).

- [ ] **Step 5: `main.py` tạm thời lỗi (chưa gọi `_make_client` được) — sửa ngay trong task
  này để giữ trạng thái xanh**

Thêm import ở đầu `main.py` (sau khối import hiện có):

```python
from src.app.cli import common as cli_common
```

Thay TẤT CẢ lời gọi `_make_client()` còn lại trong `main.py` (ở `login`, `probe_fields`,
`warm_cache_cmd`, `fetch_fields`, `fetch_operators`, `simulate`, `sweep_config`,
`closed_loop_cmd`, `research`, `marathon`, `submit`, `_menu_login`) thành
`cli_common._make_client()`. Tương tự thay mọi lời gọi còn lại trong `main.py` của
`_cached_symbols(...)` → `cli_common._cached_symbols(...)`,
`_local_operator_arity()` → `cli_common._local_operator_arity()`,
`_make_validated_simulator(...)` → `cli_common._make_validated_simulator(...)`,
`_portfolio_config_from_opts(...)` → `cli_common._portfolio_config_from_opts(...)`.

(Đây là bước tạm thời — các lời gọi này sẽ tự nhiên "chuyển nhà" cùng hàm cha ở các task
B2–B13 kế tiếp; sửa ở đây chỉ để `main.py` không vỡ ngay bây giờ.)

- [ ] **Step 6: Chạy toàn bộ test suite**

Run: `pytest -q`
Expected: PASS toàn bộ.

- [ ] **Step 7: Commit**

```bash
git add src/app/cli/__init__.py src/app/cli/common.py main.py
git commit -m "refactor(cli): tách helper dùng chung (_make_client, _cached_symbols...) sang src/app/cli/common.py"
```

---

### Task B2: `src/app/cli/auth.py`

**Files:**
- Create: `src/app/cli/auth.py`
- Modify: `main.py`
- Test: `tests/test_main_login.py`

**Interfaces:**
- Consumes: `src.app.cli.common._make_client`
- Produces: `src.app.cli.auth.prompt_credentials(...)`, `src.app.cli.auth.login(...)`

- [ ] **Step 1: Sửa test (RED)**

`tests/test_main_login.py`, đổi dòng import:

```python
from src.app.cli.auth import prompt_credentials
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `pytest tests/test_main_login.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.cli.auth'`.

- [ ] **Step 3: Tạo `src/app/cli/auth.py`**

Common Header (docstring: `"""Lệnh đăng nhập WQ Brain."""`) +
`from src.app.cli.common import _make_client` + dán nguyên văn `prompt_credentials` và
`login` (bỏ `@app.command()` phía trên `login`), sửa lời gọi `_make_client()` bên trong
`login` thành `_make_client()` (giữ nguyên tên vì đã import trực tiếp, KHÔNG cần tiền tố
`cli_common.` ở đây — khác với Task B1 vốn chỉ tạm thời dùng tiền tố trong `main.py`).

- [ ] **Step 4: Xoá `prompt_credentials` + `login` (kèm `@app.command()`) khỏi `main.py`**

- [ ] **Step 5: Chạy `ruff check --fix src/app/cli/auth.py`**

- [ ] **Step 6: Wiring trong `main.py`**

Thêm import: `from src.app.cli import auth as cli_auth`
Thêm đăng ký lệnh (đặt gần đầu phần "đăng ký command" — có thể tạm đặt ngay sau
`app = typer.Typer(...)`, các task sau sẽ bổ sung thêm dòng cạnh nó):

```python
app.command()(cli_auth.login)
```

Xoá lời gọi tạm `cli_common._make_client()` còn sót lại KHÔNG áp dụng ở đây vì `login` đã
chuyển hẳn (nó dùng `_make_client` nội bộ trong `auth.py`, không còn nằm trong `main.py`).

- [ ] **Step 7: Chạy test, xác nhận PASS**

Run: `pytest tests/test_main_login.py -v`
Expected: PASS.

- [ ] **Step 8: Chạy toàn bộ suite + xác nhận CLI còn đủ lệnh**

Run: `pytest -q`
Run: `python main.py --help`
Expected: suite PASS; `--help` vẫn liệt kê lệnh `login` cùng các lệnh khác chưa tách.

- [ ] **Step 9: Commit**

```bash
git add src/app/cli/auth.py main.py tests/test_main_login.py
git commit -m "refactor(cli): tách lệnh login sang src/app/cli/auth.py"
```

---

### Task B3: `src/app/cli/fields.py`

**Files:**
- Create: `src/app/cli/fields.py`
- Modify: `main.py`

**Interfaces:**
- Consumes: `src.app.cli.common._make_client`
- Produces: `probe_fields`, `warm_cache_cmd`, `fetch_fields`, `cache_status`,
  `fetch_operators`, `list_fields`

- [ ] **Step 1: Tạo `src/app/cli/fields.py`**

Common Header (docstring: `"""Lệnh quản lý fields/operators (probe, fetch, cache, list)."""`)
+ `from src.app.cli.common import _make_client` + dán nguyên văn 6 hàm (bỏ decorator
`@app.command(...)` trên mỗi hàm): `probe_fields`, `warm_cache_cmd`, `fetch_fields`,
`cache_status`, `fetch_operators`, `list_fields`.

- [ ] **Step 2: Xoá 6 hàm đó (kèm decorator) khỏi `main.py`**

- [ ] **Step 3: `ruff check --fix src/app/cli/fields.py`**

- [ ] **Step 4: Wiring trong `main.py`**

```python
from src.app.cli import fields as cli_fields
```
```python
app.command("probe-fields")(cli_fields.probe_fields)
app.command("warm-cache")(cli_fields.warm_cache_cmd)
app.command("fetch-fields")(cli_fields.fetch_fields)
app.command("cache-status")(cli_fields.cache_status)
app.command("fetch-operators")(cli_fields.fetch_operators)
app.command("list-fields")(cli_fields.list_fields)
```

- [ ] **Step 5: Chạy toàn bộ suite + `python main.py --help`**

Run: `pytest -q`
Run: `python main.py --help`
Expected: PASS; đủ lệnh `probe-fields`, `warm-cache`, `fetch-fields`, `cache-status`,
`fetch-operators`, `list-fields`.

- [ ] **Step 6: Commit**

```bash
git add src/app/cli/fields.py main.py
git commit -m "refactor(cli): tách lệnh fields/operators sang src/app/cli/fields.py"
```

---

### Task B4: `src/app/cli/simulate.py`

**Files:**
- Create: `src/app/cli/simulate.py`
- Modify: `main.py`
- Test: `tests/test_auto_command.py`

**Interfaces:**
- Consumes: `src.app.cli.common._make_client`
- Produces: `simulate`, `sweep_config`

- [ ] **Step 1: Sửa test (RED) — `tests/test_auto_command.py`**

Đổi import và toàn bộ `monkeypatch.setattr(main, ...)` trong
`test_simulate_command_truyen_day_du_sim_config` từ target `main` sang module
`src.app.cli.simulate`:

```python
import main
from src.app.cli import simulate as cli_simulate
```

Trong `test_simulate_command_truyen_day_du_sim_config`, đổi tất cả
`monkeypatch.setattr(main, "X", ...)` thành `monkeypatch.setattr(cli_simulate, "X", ...)`
(giữ nguyên tên `X`: `init_db`, `make_engine`, `make_session_factory`, `_make_client`,
`Simulator`, `AlphaRepository`), và đổi lời gọi cuối `main.simulate(...)` thành
`cli_simulate.simulate(...)`.

`test_research_truyen_fixed_sim_config_xuong_loop_builder` và
`test_wiring_theme_ap_top1000_cho_hom_nay_trong_lich` **giữ nguyên** ở task này (chưa đụng
tới `research`/theme-config).

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `pytest tests/test_auto_command.py::test_simulate_command_truyen_day_du_sim_config -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.cli.simulate'`.

- [ ] **Step 3: Tạo `src/app/cli/simulate.py`**

Common Header (docstring: `"""Lệnh simulate/sweep-config."""`) +
`from src.app.cli.common import _make_client` + dán nguyên văn `simulate` và `sweep_config`
(bỏ decorator).

- [ ] **Step 4: Xoá `simulate` + `sweep_config` khỏi `main.py`**

- [ ] **Step 5: `ruff check --fix src/app/cli/simulate.py`**

- [ ] **Step 6: Wiring trong `main.py`**

```python
from src.app.cli import simulate as cli_simulate
```
```python
app.command()(cli_simulate.simulate)
app.command("sweep-config")(cli_simulate.sweep_config)
```

- [ ] **Step 7: Chạy test, xác nhận PASS**

Run: `pytest tests/test_auto_command.py -v`
Expected: PASS toàn bộ file (bao gồm 2 test chưa sửa, vẫn dùng `main.research` — sẽ đổi ở
Task B10).

- [ ] **Step 8: Chạy toàn bộ suite + `python main.py --help`**

- [ ] **Step 9: Commit**

```bash
git add src/app/cli/simulate.py main.py tests/test_auto_command.py
git commit -m "refactor(cli): tách lệnh simulate/sweep-config sang src/app/cli/simulate.py"
```

---

### Task B5: `src/app/cli/generate.py`

**Files:**
- Create: `src/app/cli/generate.py`
- Modify: `main.py`

**Interfaces:**
- Consumes: `src.app.cli.common._cached_symbols`, `src.app.cli.common._portfolio_config_from_opts`
- Produces: `generate`, `score_one_cmd`

- [ ] **Step 1: Tạo `src/app/cli/generate.py`**

Common Header (docstring: `"""Lệnh generate/score-one."""`) +
`from src.app.cli.common import _cached_symbols, _portfolio_config_from_opts` + dán nguyên
văn `generate` và `score_one_cmd` (bỏ decorator).

- [ ] **Step 2: Xoá `generate` + `score_one_cmd` khỏi `main.py`**

- [ ] **Step 3: `ruff check --fix src/app/cli/generate.py`**

- [ ] **Step 4: Wiring trong `main.py`**

```python
from src.app.cli import generate as cli_generate
```
```python
app.command()(cli_generate.generate)
app.command("score-one")(cli_generate.score_one_cmd)
```

- [ ] **Step 5: Chạy toàn bộ suite + `python main.py --help`**

- [ ] **Step 6: Commit**

```bash
git add src/app/cli/generate.py main.py
git commit -m "refactor(cli): tách lệnh generate/score-one sang src/app/cli/generate.py"
```

---

### Task B6: `src/app/cli/submit.py`

**Files:**
- Create: `src/app/cli/submit.py`
- Modify: `main.py`

**Interfaces:**
- Consumes: `src.app.cli.common._make_client`
- Produces: `submit`

- [ ] **Step 1: Tạo `src/app/cli/submit.py`**

Common Header (docstring: `"""Lệnh submit alpha lên Brain."""`) +
`from src.app.cli.common import _make_client` + dán nguyên văn `submit` (bỏ decorator).

- [ ] **Step 2: Xoá `submit` khỏi `main.py`**

- [ ] **Step 3: `ruff check --fix src/app/cli/submit.py`**

- [ ] **Step 4: Wiring trong `main.py`**

```python
from src.app.cli import submit as cli_submit
```
```python
app.command()(cli_submit.submit)
```

- [ ] **Step 5: Chạy toàn bộ suite (đặc biệt `tests/test_submission*.py`,
  `tests/unit/test_cli_score_one_generate.py` nếu có phần submit) + `python main.py --help`**

- [ ] **Step 6: Commit**

```bash
git add src/app/cli/submit.py main.py
git commit -m "refactor(cli): tách lệnh submit sang src/app/cli/submit.py"
```

---

### Task B7: `src/app/cli/report.py`

**Files:**
- Create: `src/app/cli/report.py`
- Modify: `main.py`

**Interfaces:**
- Produces: `top`, `originality`, `genius_report_cmd`

- [ ] **Step 1: Tạo `src/app/cli/report.py`**

Common Header (docstring: `"""Lệnh báo cáo: top alpha, originality, genius report."""`) +
dán nguyên văn `top`, `originality`, `genius_report_cmd` (bỏ decorator; `submit` nằm xen
giữa `originality` và `genius_report_cmd` trong `main.py` gốc nhưng đã chuyển ở Task B6 nên
giờ `originality` và `genius_report_cmd` liền kề nhau).

- [ ] **Step 2: Xoá `top`, `originality`, `genius_report_cmd` khỏi `main.py`**

- [ ] **Step 3: `ruff check --fix src/app/cli/report.py`**

- [ ] **Step 4: Wiring trong `main.py`**

```python
from src.app.cli import report as cli_report
```
```python
app.command()(cli_report.top)
app.command()(cli_report.originality)
app.command("genius-report")(cli_report.genius_report_cmd)
```

- [ ] **Step 5: Chạy toàn bộ suite + `python main.py --help`**

- [ ] **Step 6: Commit**

```bash
git add src/app/cli/report.py main.py
git commit -m "refactor(cli): tách lệnh top/originality/genius-report sang src/app/cli/report.py"
```

---

### Task B8: `src/app/cli/migrate.py`

**Files:**
- Create: `src/app/cli/migrate.py`
- Modify: `main.py`
- Test: `tests/unit/test_calibrate_command.py` (KHÔNG cần sửa — dùng `from main import app`,
  vẫn hoạt động vì `app` ở lại `main.py`; chỉ chạy lại để xác nhận).

**Interfaces:**
- Produces: `migrate_sqlite`, `calibrate`

- [ ] **Step 1: Tạo `src/app/cli/migrate.py`**

Common Header (docstring: `"""Lệnh migrate DB và calibrate."""`) + dán nguyên văn
`migrate_sqlite` và `calibrate` (bỏ decorator).

- [ ] **Step 2: Xoá `migrate_sqlite` + `calibrate` khỏi `main.py`**

- [ ] **Step 3: `ruff check --fix src/app/cli/migrate.py`**

- [ ] **Step 4: Wiring trong `main.py`**

```python
from src.app.cli import migrate as cli_migrate
```
```python
app.command("migrate-sqlite")(cli_migrate.migrate_sqlite)
app.command("calibrate")(cli_migrate.calibrate)
```

- [ ] **Step 5: Chạy toàn bộ suite (đặc biệt `tests/unit/test_calibrate_command.py`,
  `tests/test_migrate.py`) + `python main.py --help`**

- [ ] **Step 6: Commit**

```bash
git add src/app/cli/migrate.py main.py
git commit -m "refactor(cli): tách lệnh migrate-sqlite/calibrate sang src/app/cli/migrate.py"
```

---

### Task B9: `src/app/cli/llm.py`

**Files:**
- Create: `src/app/cli/llm.py`
- Modify: `main.py`
- Test: `tests/test_deepseek_check.py`

**Interfaces:**
- Consumes: `src.app.cli.common._cached_symbols`, `src.app.cli.common._local_operator_arity`
- Produces: `_make_deepseek`, `run_deepseek_smoke`, `describe_deepseek_smoke_error`,
  `check_deepseek`, `_make_router`, `_make_llm_generator`, `llm_generate`, `llm_ideas`

- [ ] **Step 1: Sửa test (RED) — `tests/test_deepseek_check.py`**

```python
from src.app.cli import llm as main
```

(giữ nguyên toàn bộ phần còn lại của file — alias `as main` khiến mọi `main.run_deepseek_smoke(...)`
và `main.describe_deepseek_smoke_error(...)` hiện có tiếp tục chạy đúng mà không cần sửa
từng lời gọi.)

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `pytest tests/test_deepseek_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.cli.llm'`.

- [ ] **Step 3: Tạo `src/app/cli/llm.py`**

Common Header (docstring: `"""Lệnh LLM: deepseek smoke-check, llm-generate/ideas."""`) +
`from src.app.cli.common import _cached_symbols, _local_operator_arity` + dán nguyên văn (2
khối không liền nhau trong `main.py` gốc, dán chung vào 1 file theo thứ tự này):
`_make_deepseek`, `run_deepseek_smoke`, `describe_deepseek_smoke_error`, `check_deepseek`
(bỏ decorator trên `check_deepseek`), `_make_router`, `_make_llm_generator`, rồi
`llm_generate`, `llm_ideas` (bỏ decorator trên cả hai).

- [ ] **Step 4: Xoá 8 hàm đó khỏi `main.py`**

- [ ] **Step 5: `ruff check --fix src/app/cli/llm.py`**

- [ ] **Step 6: Wiring trong `main.py`**

```python
from src.app.cli import llm as cli_llm
```
```python
app.command("check-deepseek")(cli_llm.check_deepseek)
app.command("llm-generate")(cli_llm.llm_generate)
app.command("llm-ideas")(cli_llm.llm_ideas)
```

- [ ] **Step 7: Chạy test, xác nhận PASS**

Run: `pytest tests/test_deepseek_check.py -v`
Expected: PASS.

- [ ] **Step 8: Chạy toàn bộ suite + `python main.py --help`**

- [ ] **Step 9: Commit**

```bash
git add src/app/cli/llm.py main.py tests/test_deepseek_check.py
git commit -m "refactor(cli): tách lệnh LLM (deepseek/llm-generate/llm-ideas) sang src/app/cli/llm.py"
```

---

### Task B10: `src/app/cli/research.py`

**Files:**
- Create: `src/app/cli/research.py`
- Modify: `main.py`
- Test: `tests/test_pool_corr_fn.py`, `tests/test_pnl_fn.py`, `tests/test_research_direction.py`,
  `tests/test_auto_command.py`

**Interfaces:**
- Consumes: `src.app.cli.common._cached_symbols`, `_local_operator_arity`,
  `_make_validated_simulator`, `_make_client`; `src.app.cli.llm._make_router`,
  `_make_llm_generator`
- Produces: `_make_pool_corr_fn`, `_make_pnl_fn`, `_make_research_loop`,
  `_render_research_result`, `resolve_direction`, `research`, `_run_research_with_progress`

- [ ] **Step 1: Sửa test (RED) — 3 file alias đơn giản**

`tests/test_pool_corr_fn.py`:
```python
from src.app.cli.research import _make_pool_corr_fn
```

`tests/test_pnl_fn.py`:
```python
from src.app.cli.research import _make_pnl_fn
```

`tests/test_research_direction.py`:
```python
import src.app.cli.research as main
```

(alias `as main` giữ nguyên mọi `main.resolve_direction(...)` bên dưới.)

- [ ] **Step 2: Sửa test (RED) — `tests/test_auto_command.py`**

Đổi `test_research_truyen_fixed_sim_config_xuong_loop_builder`: thêm import
`from src.app.cli import research as cli_research`, đổi mọi
`monkeypatch.setattr(main, "X", ...)` trong test này thành
`monkeypatch.setattr(cli_research, "X", ...)` (giữ nguyên tên `X`: `init_db`, `make_engine`,
`make_session_factory`, `_cached_symbols`, `_make_client`, `_make_research_loop`,
`_run_research_with_progress`, `_render_research_result`), đổi lời gọi cuối
`main.research(...)` thành `cli_research.research(...)`.

`test_wiring_theme_ap_top1000_cho_hom_nay_trong_lich` không đụng gì (không dùng `main`).

`import main` ở đầu file: giữ lại (vẫn cần cho
`test_simulate_command_truyen_day_du_sim_config` đã xong ở Task B4 dùng `cli_simulate`
riêng — kiểm tra `import main` còn được dùng chỗ nào khác trong file này; nếu không còn
dùng, xoá dòng `import main`).

- [ ] **Step 3: Chạy test, xác nhận FAIL**

Run: `pytest tests/test_pool_corr_fn.py tests/test_pnl_fn.py tests/test_research_direction.py tests/test_auto_command.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.cli.research'`.

- [ ] **Step 4: Tạo `src/app/cli/research.py`**

Common Header (docstring: `"""Lệnh research (vòng nghiên cứu chính sinh alpha)."""`) +

```python
from src.app.cli.common import (
    _cached_symbols,
    _local_operator_arity,
    _make_client,
    _make_validated_simulator,
)
from src.app.cli.llm import _make_router, _make_llm_generator
```

Dán nguyên văn `_make_pool_corr_fn`, `_make_pnl_fn`, `_make_research_loop`,
`_render_research_result`, `resolve_direction`, `research` (bỏ decorator),
`_run_research_with_progress`.

- [ ] **Step 5: Xoá 7 hàm đó khỏi `main.py`**

- [ ] **Step 6: `ruff check --fix src/app/cli/research.py`**

- [ ] **Step 7: Wiring trong `main.py`**

```python
from src.app.cli import research as cli_research
```
```python
app.command()(cli_research.research)
```

- [ ] **Step 8: Chạy test, xác nhận PASS**

Run: `pytest tests/test_pool_corr_fn.py tests/test_pnl_fn.py tests/test_research_direction.py tests/test_auto_command.py -v`
Expected: PASS toàn bộ.

- [ ] **Step 9: Chạy toàn bộ suite + `python main.py --help`**

- [ ] **Step 10: Commit**

```bash
git add src/app/cli/research.py main.py tests/test_pool_corr_fn.py tests/test_pnl_fn.py tests/test_research_direction.py tests/test_auto_command.py
git commit -m "refactor(cli): tách lệnh research sang src/app/cli/research.py"
```

---

### Task B11: `src/app/cli/closed_loop.py`

**Files:**
- Create: `src/app/cli/closed_loop.py`
- Modify: `main.py`
- Test: `tests/test_closed_loop_seed.py`, `tests/test_closed_loop_reseed.py`

**Interfaces:**
- Consumes: `src.app.cli.common._cached_symbols`, `_portfolio_config_from_opts`,
  `_make_client`; `src.app.cli.research._make_research_loop`
- Produces: `_resolve_base_seed`, `_run_reseed_until_quota`, `_local_neutralization`,
  `_closed_loop_configs`, `_run_closed_loop_session`, `closed_loop_cmd`

- [ ] **Step 1: Sửa test (RED)**

`tests/test_closed_loop_seed.py`:
```python
import src.app.cli.closed_loop as main
```

`tests/test_closed_loop_reseed.py`:
```python
import src.app.cli.closed_loop as main
```

(alias `as main` giữ nguyên toàn bộ lời gọi `main._resolve_base_seed(...)`,
`main._local_neutralization(...)`, `main._closed_loop_configs(...)`,
`main._run_closed_loop_session(...)`, `main._run_reseed_until_quota(...)` hiện có.)

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `pytest tests/test_closed_loop_seed.py tests/test_closed_loop_reseed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.cli.closed_loop'`.

- [ ] **Step 3: Tạo `src/app/cli/closed_loop.py`**

Common Header (docstring: `"""Lệnh closed-loop (vòng kín local-search + Brain sim)."""`) +

```python
from src.app.cli.common import _cached_symbols, _make_client, _portfolio_config_from_opts
from src.app.cli.research import _make_research_loop
```

Dán nguyên văn `_resolve_base_seed`, `_run_reseed_until_quota`, `_local_neutralization`,
`_closed_loop_configs`, `_run_closed_loop_session`, `closed_loop_cmd` (bỏ decorator trên
`closed_loop_cmd`).

- [ ] **Step 4: Xoá 6 hàm đó khỏi `main.py`**

- [ ] **Step 5: `ruff check --fix src/app/cli/closed_loop.py`**

- [ ] **Step 6: Wiring trong `main.py`**

```python
from src.app.cli import closed_loop as cli_closed_loop
```
```python
app.command("closed-loop")(cli_closed_loop.closed_loop_cmd)
```

- [ ] **Step 7: Chạy test, xác nhận PASS**

Run: `pytest tests/test_closed_loop_seed.py tests/test_closed_loop_reseed.py -v`
Expected: PASS toàn bộ.

- [ ] **Step 8: Chạy toàn bộ suite (đặc biệt `tests/unit/test_cli_closed_loop.py` — dùng
  `from main import app`, không sửa, chỉ xác nhận vẫn PASS qua CliRunner) +
  `python main.py --help`**

- [ ] **Step 9: Commit**

```bash
git add src/app/cli/closed_loop.py main.py tests/test_closed_loop_seed.py tests/test_closed_loop_reseed.py
git commit -m "refactor(cli): tách lệnh closed-loop sang src/app/cli/closed_loop.py"
```

---

### Task B12: `src/app/cli/marathon.py`

**Files:**
- Create: `src/app/cli/marathon.py`
- Modify: `main.py`
- Test: `tests/test_marathon_command.py`

**Interfaces:**
- Consumes: `src.app.cli.common._cached_symbols`, `_local_operator_arity`, `_make_client`;
  `src.app.cli.llm._make_llm_generator`; `src.app.cli.research.resolve_direction`,
  `_make_research_loop`
- Produces: `_render_marathon_report`, `_marathon_direction_provider`,
  `_marathon_on_event`, `_run_marathon_session`, `marathon`

- [ ] **Step 1: Sửa test (RED) — `tests/test_marathon_command.py`**

```python
import src.app.cli.marathon as main
```

(alias `as main` giữ nguyên `main.marathon(...)`, `main.MarathonReport(...)`,
`monkeypatch.setattr(main, "X", ...)` với `X` = `init_db`, `make_engine`,
`make_session_factory`, `_cached_symbols`, `_make_client`, `_make_research_loop`,
`run_marathon`.)

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `pytest tests/test_marathon_command.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.cli.marathon'`.

- [ ] **Step 3: Tạo `src/app/cli/marathon.py`**

Common Header (docstring: `"""Lệnh marathon (chạy nhiều hướng nghiên cứu liên tiếp)."""`) +

```python
from src.app.cli.common import _cached_symbols, _local_operator_arity, _make_client
from src.app.cli.llm import _make_llm_generator
from src.app.cli.research import _make_research_loop, resolve_direction
```

Dán nguyên văn `_render_marathon_report`, `_marathon_direction_provider`,
`_marathon_on_event`, `_run_marathon_session`, `marathon` (bỏ decorator).

- [ ] **Step 4: Xoá 5 hàm đó khỏi `main.py`**

- [ ] **Step 5: `ruff check --fix src/app/cli/marathon.py`**

- [ ] **Step 6: Wiring trong `main.py`**

```python
from src.app.cli import marathon as cli_marathon
```
```python
app.command()(cli_marathon.marathon)
```

- [ ] **Step 7: Chạy test, xác nhận PASS**

Run: `pytest tests/test_marathon_command.py -v`
Expected: PASS.

- [ ] **Step 8: Chạy toàn bộ suite + `python main.py --help`**

- [ ] **Step 9: Commit**

```bash
git add src/app/cli/marathon.py main.py tests/test_marathon_command.py
git commit -m "refactor(cli): tách lệnh marathon sang src/app/cli/marathon.py"
```

---

### Task B13: `src/app/menu.py`

**Files:**
- Create: `src/app/menu.py`
- Modify: `main.py`
- Test: `tests/test_menu_counts.py`

**Interfaces:**
- Consumes: `src.app.cli.common._cached_symbols`, `_local_operator_arity`,
  `_portfolio_config_from_opts`, `_make_client`; `src.app.cli.llm._make_router`;
  `src.app.cli.closed_loop._run_closed_loop_session`
- Produces: `_MenuState`, `_menu_counts`, `_menu_login`, `_menu_fields`, `_menu_operators`,
  `_find_market_data_dir`, `_menu_test_engine`, `_menu_auto_sim`, `_menu_view_submit`,
  `_print_menu`, `start`

- [ ] **Step 1: Sửa test (RED) — `tests/test_menu_counts.py`**

```python
import src.app.menu as main
```

(alias `as main` giữ nguyên `main._MenuState()`, `main._menu_counts(...)`,
`main._menu_view_submit(...)`, `main._menu_auto_sim(...)`,
`monkeypatch.setattr(main, "_find_market_data_dir", ...)`,
`monkeypatch.setattr(main, "_run_closed_loop_session", ...)`.)

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `pytest tests/test_menu_counts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.menu'`.

- [ ] **Step 3: Tạo `src/app/menu.py`**

Common Header (docstring: `"""Menu tương tác dòng lệnh (lựa chọn 1-6)."""`) +

```python
from src.app.cli.common import (
    _cached_symbols,
    _local_operator_arity,
    _make_client,
    _portfolio_config_from_opts,
)
from src.app.cli.llm import _make_router
from src.app.cli.closed_loop import _run_closed_loop_session
```

Dán nguyên văn `_MenuState`, `_menu_counts`, `_menu_login`, `_menu_fields`,
`_menu_operators`, `_find_market_data_dir`, `_menu_test_engine`, `_menu_auto_sim`,
`_menu_view_submit`, `_print_menu`, `start` (bỏ decorator `@app.command()` trên `start`).

- [ ] **Step 4: Xoá 11 khối đó (kể cả class `_MenuState`) khỏi `main.py`**

- [ ] **Step 5: `ruff check --fix src/app/menu.py`**

- [ ] **Step 6: Wiring trong `main.py`**

```python
from src.app import menu as cli_menu
```
```python
app.command()(cli_menu.start)
```

- [ ] **Step 7: Chạy test, xác nhận PASS**

Run: `pytest tests/test_menu_counts.py -v`
Expected: PASS.

- [ ] **Step 8: Chạy toàn bộ suite + `python main.py --help`**

Expected: `--help` liệt kê đủ 23 lệnh (kiểm tra bằng mắt so với danh sách gốc đã ghi ở đầu
tài liệu spec).

- [ ] **Step 9: Commit**

```bash
git add src/app/menu.py main.py tests/test_menu_counts.py
git commit -m "refactor(cli): tách menu tương tác sang src/app/menu.py"
```

---

### Task B14: Dọn cuối `main.py` + xác nhận toàn cục

**Files:**
- Modify: `main.py`

- [ ] **Step 1: `ruff check --fix main.py`**

Xoá các import ở đầu `main.py` không còn dùng tới (giờ `main.py` chỉ còn `_setup_logging`,
khởi tạo `app`/`console`/`LOG_DIR`, và khối import + wiring của 12 module con).

- [ ] **Step 2: Xác nhận không còn hàm nghiệp vụ nào sót lại trong `main.py`**

Run: `grep -n "^def \|^class " main.py`
Expected: chỉ còn `_setup_logging` (và có thể `console`/`app` không phải hàm nên không xuất
hiện) — không còn `def login(`, `def simulate(`, v.v.

- [ ] **Step 3: Xác nhận không còn `import main`/`from main import` sai chỗ trong test**

Run: `grep -rln "^import main\|^from main import" tests/`
Expected: chỉ còn `tests/unit/test_calibrate_command.py`, `tests/unit/test_cli_closed_loop.py`,
`tests/unit/test_cli_score_one_generate.py` (cả 3 dùng `from main import app`, hợp lệ vì
`app` vẫn ở `main.py`).

- [ ] **Step 4: Chạy toàn bộ suite**

Run: `pytest -q`
Expected: PASS toàn bộ, không warning import thừa.

- [ ] **Step 5: Xác nhận CLI đầy đủ + menu chạy được**

Run: `python main.py --help`
Expected: liệt kê đủ 23 lệnh.

Chạy thử `run.bat` (hoặc `python main.py start`), chọn thoát ngay (mục cuối menu) — xác
nhận menu hiển thị đúng, không traceback khi khởi động.

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "refactor(cli): dọn import thừa trong main.py sau khi tách toàn bộ command"
```

---

## Kiểm chứng hoàn tất toàn kế hoạch

- `pytest -q` xanh sau task cuối cùng (cả Dự án A lẫn B).
- `python main.py --help` liệt kê đủ 23 lệnh như trước khi tách.
- `main.py` chỉ còn `_setup_logging` + khởi tạo `app`/`console` + import/wiring 12 module.
- Gốc repo chỉ còn các file/thư mục đã liệt kê trong spec (không còn file rác).
- `git status` sạch sau mỗi commit.
