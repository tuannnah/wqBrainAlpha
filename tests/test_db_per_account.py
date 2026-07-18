"""DB tách theo email đăng nhập: mỗi tài khoản một file SQLite riêng.

Quy tắc:
- Chỉ tách khi đang dùng SQLite mặc định (`DEFAULT_SQLITE_URL`). URL tùy biến
  (vd Postgres qua DATABASE_URL) giữ nguyên để không phá deployment.
- Nguồn email: ưu tiên `settings.wq_email` (.env), nếu trống đọc `.wq_account`
  (email của lần đăng nhập gần nhất, ghi bởi lệnh login).
"""

from __future__ import annotations

import pytest

from config.settings import settings
from src.storage import db


@pytest.fixture
def restore_settings():
    old_email, old_url = settings.wq_email, settings.database_url
    yield
    settings.wq_email, settings.database_url = old_email, old_url


def test_default_no_email_uses_default_db(tmp_path, restore_settings):
    settings.wq_email = ""
    settings.database_url = db.DEFAULT_SQLITE_URL
    acc = tmp_path / ".wq_account"
    assert db.active_database_url(acc) == db.DEFAULT_SQLITE_URL


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


def test_custom_url_is_left_unchanged(tmp_path, restore_settings):
    settings.wq_email = "a@b.com"
    settings.database_url = "postgresql://localhost/wq"
    acc = tmp_path / ".wq_account"
    assert db.active_database_url(acc) == "postgresql://localhost/wq"


def test_write_then_read_account_roundtrip(tmp_path):
    acc = tmp_path / ".wq_account"
    db.write_active_account("  Me@Example.com  ", acc)
    assert db.read_active_account(acc) == "Me@Example.com"


def test_read_account_missing_file_returns_empty(tmp_path):
    assert db.read_active_account(tmp_path / "nope") == ""


def test_make_engine_tao_thu_muc_cha_neu_chua_co(tmp_path):
    target = tmp_path / "newsub" / "test.db"
    assert not target.parent.exists()
    engine = db.make_engine(f"sqlite:///{target}")
    engine.dispose()
    assert target.parent.exists()
