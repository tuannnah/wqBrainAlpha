"""Engine + session factory cho SQLite (bật WAL để giảm khóa khi chạy song song)."""

from __future__ import annotations

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from src.storage.models import Base


def make_engine(database_url: str | None = None) -> Engine:
    url = database_url or settings.database_url
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
