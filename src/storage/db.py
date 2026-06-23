"""Engine + session factory cho SQLite (bật WAL để giảm khóa khi chạy song song)."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from src.storage.models import Base

# DB SQLite mặc định. Khi đang dùng đúng URL này, ta tách file theo email đăng
# nhập (mỗi tài khoản 1 file). URL tùy biến (vd Postgres) thì giữ nguyên.
DEFAULT_SQLITE_URL = "sqlite:///wq_alpha.db"
# Email của lần đăng nhập gần nhất (ghi bởi lệnh login) — dùng chọn DB khi .env
# để trống WQ_EMAIL (luồng nhập email tương tác).
ACCOUNT_FILE = Path(".wq_account")


def _email_slug(email: str) -> str:
    """Chuẩn hoá email thành phần tên file an toàn (a-z0-9 và '_')."""
    slug = re.sub(r"[^a-z0-9]+", "_", email.strip().lower()).strip("_")
    return slug or "default"


def read_active_account(account_file: Path = ACCOUNT_FILE) -> str:
    """Đọc email tài khoản active từ file; trả '' nếu không có."""
    try:
        return account_file.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def write_active_account(email: str, account_file: Path = ACCOUNT_FILE) -> None:
    """Ghi email tài khoản active (gọi sau khi login thành công)."""
    try:
        account_file.write_text(email.strip(), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - lỗi ổ đĩa hiếm
        logger.warning("Không lưu được tài khoản active: {}", exc)


def active_database_url(account_file: Path = ACCOUNT_FILE) -> str:
    """URL DB hiệu lực, tách theo email đăng nhập.

    Ưu tiên `settings.wq_email` (.env); nếu trống đọc `.wq_account`. Chỉ tách khi
    đang dùng SQLite mặc định — URL tùy biến (DATABASE_URL Postgres...) giữ nguyên.
    """
    url = settings.database_url
    if url != DEFAULT_SQLITE_URL:
        return url
    email = (settings.wq_email or read_active_account(account_file)).strip()
    if not email:
        return url
    return f"sqlite:///wq_alpha_{_email_slug(email)}.db"


def make_engine(database_url: str | None = None) -> Engine:
    url = database_url or active_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, future=True, connect_args=connect_args)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


# Kiểu cột SQLite cho ALTER TABLE ADD COLUMN (chỉ cần với cột thêm sau).
_SQLITE_TYPE = {"INTEGER": "INTEGER", "FLOAT": "FLOAT", "DATETIME": "DATETIME"}


def _migrate_add_columns(engine: Engine) -> None:
    """Thêm cột còn thiếu cho bảng đã tồn tại (DB cũ) — idempotent, chỉ ADD COLUMN."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in Base.metadata.tables.values():
            if table.name not in existing_tables:
                continue  # create_all đã tạo mới với đủ cột
            have = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in have:
                    continue
                col_type = _SQLITE_TYPE.get(
                    column.type.__class__.__name__.upper(), "TEXT"
                )
                conn.execute(
                    text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}')
                )


def init_db(engine: Engine | None = None) -> Engine:
    engine = engine or make_engine()
    Base.metadata.create_all(engine)
    if engine.url.get_backend_name() == "sqlite":
        _migrate_add_columns(engine)
    return engine


def make_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    engine = engine or make_engine()
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)
