"""Test make_engine nhận URL Postgres mà không áp cấu hình riêng SQLite."""

from __future__ import annotations

from src.storage.db import make_engine


def test_make_engine_postgres_backend():
    # create_engine là lazy: không kết nối thật, chỉ phân giải dialect.
    engine = make_engine("postgresql+psycopg://u:p@localhost:5432/wq")
    assert engine.url.get_backend_name() == "postgresql"
    # Không set check_same_thread (đó là cờ riêng của SQLite).
    assert "check_same_thread" not in engine.url.query


def test_make_engine_sqlite_van_hoat_dong():
    engine = make_engine("sqlite:///:memory:")
    assert engine.url.get_backend_name() == "sqlite"
