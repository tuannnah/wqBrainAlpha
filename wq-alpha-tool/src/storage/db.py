"""Engine + session factory cho SQLite (bật WAL để giảm khóa khi chạy song song)."""

from __future__ import annotations

from sqlalchemy import create_engine, event
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


def init_db(engine: Engine | None = None) -> Engine:
    engine = engine or make_engine()
    Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    engine = engine or make_engine()
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)
