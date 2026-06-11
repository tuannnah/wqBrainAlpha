# Task cho Claude Code: Đăng nhập WQ Brain + Lấy Data Field (fetch một lần vào DB)

> Đây là task spec để Claude Code thực thi. Mục tiêu: đăng nhập WorldQuant Brain (lưu session để khỏi login lại), fetch toàn bộ data field **một lần** vào SQLite, các lần sau **load thẳng từ DB không gọi API**, và có cơ chế **reload** khi WQ cập nhật dữ liệu mới.

---

## Nguyên tắc thiết kế (đọc kỹ trước khi code)

1. **Chỉ cache metadata, KHÔNG cache dữ liệu thị trường.** Ta chỉ tải danh mục field (`id`, mô tả, type, dataset). Giá trị thật của field nằm trên server WQ, chỉ dùng khi simulation. DB rất nhẹ (vài MB).
2. **Một file SQLite duy nhất** chứa tất cả. Bảng `data_fields` phân biệt theo `(region, universe, delay)`.
3. **Mỗi tổ hợp `(region, universe, delay)` là một cache độc lập** — WQ trả bộ field khác nhau cho mỗi tổ hợp.
4. **Fetch một lần → load từ DB mãi.** Chỉ gọi API lại khi: chưa có cache, cache quá hạn (TTL), hoặc user `force_reload`.
5. **Session persistence:** đăng nhập một lần, lưu cookie ra file, tái dùng. Tự re-auth khi 401.
6. **Trước khi viết logic parse response, gọi API thật và log nguyên JSON** để xác nhận đúng tên trường. Không đoán format.
7. Log đầy đủ bằng `loguru`. Không hardcode credentials — đọc từ `.env`.

---

## Cấu trúc cần tạo

```
src/
├── config/settings.py          # Pydantic settings đọc .env
├── data/
│   ├── client.py               # WQBrainClient: auth + session
│   └── fields.py               # FieldRepository: fetch-once + cache
├── storage/
│   ├── models.py               # SQLAlchemy: DataField, FetchState
│   └── db.py                   # engine, session, init_db
tests/
├── test_client.py
└── test_fields.py
main.py                         # CLI: login, fetch-fields, cache-status
.env.example
```

---

## Task 1 — Settings (`src/config/settings.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    wq_email: str
    wq_password: str
    database_url: str = "sqlite:///wq_alpha.db"
    cache_ttl_days: int = 30

    class Config:
        env_file = ".env"

settings = Settings()
```

`.env.example`:
```env
WQ_EMAIL=your_email@example.com
WQ_PASSWORD=your_password
DATABASE_URL=sqlite:///wq_alpha.db
CACHE_TTL_DAYS=30
```

Thêm `.env`, `.wq_session`, `*.db`, `__pycache__/` vào `.gitignore`.

---

## Task 2 — Storage models (`src/storage/models.py`)

```python
from sqlalchemy import Column, String, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class DataField(Base):
    __tablename__ = "data_fields"
    # Khóa kép: cùng field id nhưng khác tổ hợp = dòng khác nhau
    id          = Column(String, primary_key=True)
    region      = Column(String, primary_key=True)
    universe    = Column(String, primary_key=True)
    delay       = Column(Integer, primary_key=True)
    description = Column(Text)
    type        = Column(String)        # MATRIX / VECTOR / GROUP
    dataset_id  = Column(String)
    cached_at   = Column(DateTime)

class FetchState(Base):
    __tablename__ = "fetch_state"
    key          = Column(String, primary_key=True)  # "data_fields:USA:TOP3000:1"
    entity       = Column(String)                    # "data_fields" | "operators"
    region       = Column(String)
    universe     = Column(String)
    delay        = Column(Integer)
    total_count  = Column(Integer)
    fetched_at   = Column(DateTime)
    status       = Column(String)                    # "complete" | "partial"
```

`src/storage/db.py`:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config.settings import settings
from src.storage.models import Base

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)
```

---

## Task 3 — WQ Brain Client (`src/data/client.py`)

Xử lý đăng nhập + session persistence + tự re-auth.

```python
import json
from pathlib import Path
import httpx
from loguru import logger

SESSION_FILE = Path(".wq_session")

class AuthError(Exception): ...
class BiometricRequired(Exception): ...

class WQBrainClient:
    BASE_URL = "https://api.worldquantbrain.com"

    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self.client = httpx.Client(base_url=self.BASE_URL, timeout=30.0)
        self._load_session()

    def _load_session(self):
        if SESSION_FILE.exists():
            data = json.loads(SESSION_FILE.read_text())
            for k, v in data.items():
                self.client.cookies.set(k, v)
            logger.info("Đã nạp session từ file")

    def _save_session(self):
        data = {c.name: c.value for c in self.client.cookies.jar}
        SESSION_FILE.write_text(json.dumps(data))
        SESSION_FILE.chmod(0o600)

    def is_session_valid(self) -> bool:
        try:
            r = self.client.get("/users/self/")
            return r.status_code == 200
        except Exception:
            return False

    def authenticate(self, force: bool = False):
        if not force and self.is_session_valid():
            logger.success("Session còn hạn, bỏ qua đăng nhập")
            return
        logger.info("Đang đăng nhập WQ Brain...")
        r = self.client.post("/authentication", auth=(self.email, self.password))
        if r.status_code == 201:
            self._save_session()
            logger.success("Đăng nhập thành công")
        elif r.status_code == 401:
            w_auth = r.headers.get("WWW-Authenticate", "")
            if "persona" in w_auth.lower() or "biometric" in w_auth.lower():
                logger.warning(f"Cần xác thực biometric. Mở link trong trình duyệt: {w_auth}")
                raise BiometricRequired(w_auth)
            raise AuthError(f"Sai thông tin đăng nhập: {r.text}")
        else:
            raise AuthError(f"Lỗi auth {r.status_code}: {r.text}")

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        r = self.client.request(method, path, **kwargs)
        if r.status_code == 401:
            logger.warning("Session hết hạn giữa chừng, re-auth...")
            self.authenticate(force=True)
            r = self.client.request(method, path, **kwargs)  # retry 1 lần
        return r

    def get(self, path, **kw):  return self.request("GET", path, **kw)
    def post(self, path, **kw): return self.request("POST", path, **kw)
```

**QUAN TRỌNG:** Trước khi tin format, chạy `python main.py login` rồi `python main.py probe-fields` (xem Task 5) để **log nguyên response JSON** của `/data-fields`, xác nhận tên trường thật (`results`, `count`, cấu trúc item). WQ có thể đã đổi.

---

## Task 4 — Field Repository (`src/data/fields.py`)

Trái tim của cơ chế fetch-once.

```python
from datetime import datetime, timedelta
from loguru import logger
from src.config.settings import settings
from src.storage.models import DataField, FetchState

class FieldRepository:
    def __init__(self, client, db):
        self.client = client
        self.db = db

    def _key(self, region, universe, delay) -> str:
        return f"data_fields:{region}:{universe}:{delay}"

    def _is_cached(self, region, universe, delay) -> bool:
        state = self.db.get(FetchState, self._key(region, universe, delay))
        if state is None or state.status != "complete":
            return False
        age = datetime.utcnow() - state.fetched_at
        if age > timedelta(days=settings.cache_ttl_days):
            logger.info(f"Cache quá hạn ({age.days} ngày)")
            return False
        return True

    # ---- HÀM CHÍNH app gọi ----
    def get_fields(self, region, universe, delay, force_reload=False):
        if not force_reload and self._is_cached(region, universe, delay):
            logger.info("Load data fields từ DB (không gọi API)")
            return self._load_from_db(region, universe, delay)
        logger.info("Fetch data fields từ WQ API...")
        return self._fetch_and_store(region, universe, delay)

    def _load_from_db(self, region, universe, delay):
        return (self.db.query(DataField)
                .filter_by(region=region, universe=universe, delay=delay).all())

    def _fetch_and_store(self, region, universe, delay):
        fields = self._fetch_all_pages(region, universe, delay)
        self._replace_in_db(fields, region, universe, delay)
        self._update_state(region, universe, delay, len(fields))
        logger.success(f"Đã lưu {len(fields)} fields vào DB")
        return fields

    def _fetch_all_pages(self, region, universe, delay, page_size=50):
        all_fields, offset = [], 0
        while True:
            r = self.client.get("/data-fields", params={
                "region": region, "universe": universe, "delay": delay,
                "limit": page_size, "offset": offset,
            })
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            for item in results:
                all_fields.append(DataField(
                    id=item["id"],
                    region=region, universe=universe, delay=delay,
                    description=item.get("description", ""),
                    type=item.get("type", ""),
                    dataset_id=(item.get("dataset") or {}).get("id", ""),
                    cached_at=datetime.utcnow(),
                ))
            offset += page_size
            if offset >= data.get("count", 0):
                break
        return all_fields

    def _replace_in_db(self, fields, region, universe, delay):
        # Xóa cũ của đúng tổ hợp rồi ghi mới (replace, không append)
        (self.db.query(DataField)
         .filter_by(region=region, universe=universe, delay=delay).delete())
        self.db.add_all(fields)
        self.db.commit()

    def _update_state(self, region, universe, delay, count):
        key = self._key(region, universe, delay)
        state = self.db.get(FetchState, key) or FetchState(key=key)
        state.entity = "data_fields"
        state.region, state.universe, state.delay = region, universe, delay
        state.total_count = count
        state.fetched_at = datetime.utcnow()
        state.status = "complete"
        self.db.merge(state)
        self.db.commit()

    def get_state(self, region, universe, delay):
        return self.db.get(FetchState, self._key(region, universe, delay))
```

---

## Task 5 — CLI (`main.py`)

```python
import typer
from rich.console import Console
from rich.table import Table
from src.config.settings import settings
from src.storage.db import init_db, SessionLocal
from src.data.client import WQBrainClient
from src.data.fields import FieldRepository

app = typer.Typer()
console = Console()

def _client():
    return WQBrainClient(settings.wq_email, settings.wq_password)

@app.command()
def login(force: bool = False):
    """Đăng nhập (dùng session cũ nếu còn hạn)."""
    c = _client()
    c.authenticate(force=force)
    console.print("[green]OK[/green]")

@app.command()
def probe_fields(region: str = "USA", universe: str = "TOP3000", delay: int = 1):
    """Gọi /data-fields THẬT và in nguyên JSON 1 trang để kiểm tra format."""
    c = _client(); c.authenticate()
    r = c.get("/data-fields", params={
        "region": region, "universe": universe, "delay": delay,
        "limit": 5, "offset": 0,
    })
    console.print_json(r.text)

@app.command()
def fetch_fields(region: str = "USA", universe: str = "TOP3000",
                 delay: int = 1, reload: bool = False):
    """Fetch một lần (bỏ qua nếu đã cache). --reload để ép tải lại."""
    init_db()
    c = _client(); c.authenticate()
    db = SessionLocal()
    repo = FieldRepository(c, db)
    fields = repo.get_fields(region, universe, delay, force_reload=reload)
    console.print(f"[green]{len(fields)} fields[/green] cho {region}/{universe}/delay{delay}")

@app.command()
def cache_status():
    """Xem trạng thái cache hiện có."""
    init_db()
    db = SessionLocal()
    from src.storage.models import FetchState
    rows = db.query(FetchState).all()
    t = Table("Tổ hợp", "Số field", "Cập nhật", "Trạng thái")
    for s in rows:
        t.add_row(f"{s.region}/{s.universe}/{s.delay}",
                  str(s.total_count),
                  s.fetched_at.strftime("%Y-%m-%d %H:%M") if s.fetched_at else "-",
                  s.status)
    console.print(t)

if __name__ == "__main__":
    app()
```

---

## Trình tự thực thi (Claude Code làm theo đúng thứ tự)

1. Tạo cấu trúc thư mục + `.env.example` + `.gitignore`.
2. Viết `settings.py`, `models.py`, `db.py`.
3. Viết `client.py`.
4. **Chạy `python main.py login`** — xác nhận đăng nhập + lưu session thành công. Nếu gặp biometric, log link và dừng (user xử lý thủ công, chạy lại).
5. **Chạy `python main.py probe-fields`** — in nguyên JSON, **kiểm tra tên trường thật**. Nếu khác với code (vd không phải `results`/`count`), sửa `_fetch_all_pages` cho khớp TRƯỚC khi đi tiếp.
6. Viết `fields.py` (đã khớp format thật từ bước 5).
7. **Chạy `python main.py fetch-fields`** — fetch một lần, kiểm tra số field lưu vào DB.
8. **Chạy lại `python main.py fetch-fields`** — lần này phải log "Load data fields từ DB (không gọi API)", KHÔNG gọi API.
9. **Chạy `python main.py fetch-fields --reload`** — ép tải lại, log fetch từ API.
10. **Chạy `python main.py cache-status`** — bảng trạng thái hiển thị đúng.
11. Viết test.

---

## Acceptance criteria

- [ ] `python main.py login` → đăng nhập thành công, tạo file `.wq_session`.
- [ ] Chạy `login` lần 2 → log "Session còn hạn, bỏ qua đăng nhập", KHÔNG gọi `/authentication`.
- [ ] `probe-fields` in được JSON thật của `/data-fields`.
- [ ] `fetch-fields` lần 1 → fetch API, lưu ≥ vài trăm field vào bảng `data_fields`, ghi `fetch_state`.
- [ ] `fetch-fields` lần 2 → **load từ DB, KHÔNG gọi API** (kiểm tra qua log).
- [ ] `fetch-fields --reload` → fetch lại từ API, ghi đè (không nhân đôi số dòng).
- [ ] `cache-status` → in bảng đúng số field và thời gian.
- [ ] Tự re-auth khi session hết hạn giữa chừng (test bằng cách xóa cookie giả lập 401).
- [ ] `.wq_session` có quyền `0600`, nằm trong `.gitignore`.

---

## Test gợi ý (`tests/`)

```python
# test_fields.py — mock client, không gọi WQ thật
def test_load_from_db_when_cached(monkeypatch):
    """Khi cache hợp lệ, get_fields KHÔNG được gọi API."""
    # 1. Seed fetch_state status=complete, fetched_at=now
    # 2. Spy vào client.get → assert KHÔNG được gọi
    # 3. get_fields(..., force_reload=False) trả data từ DB

def test_fetch_when_not_cached(monkeypatch):
    """Khi chưa cache, get_fields gọi API và lưu DB."""

def test_reload_replaces_not_appends(monkeypatch):
    """force_reload=True ghi đè, không nhân đôi số field."""

def test_ttl_expired_triggers_fetch(monkeypatch):
    """fetched_at quá CACHE_TTL_DAYS → coi là chưa cache, fetch lại."""
```

---

## Lưu ý cuối

- **Operators** áp dụng y hệt cơ chế này (fetch một lần vào DB, reload khi cần) — làm sau khi fields chạy ổn, dùng lại pattern `FetchState`.
- Nếu chỉ chạy **một region/universe duy nhất**, vẫn giữ cột `(region, universe, delay)` — không tốn gì mà sau này mở rộng dễ.
- Khi WQ thêm dataset mới, user chỉ cần `fetch-fields --reload` — không xóa DB thủ công.
