"""Test giá trị mặc định của Settings (không phụ thuộc .env)."""

from __future__ import annotations

from config.settings import Settings


def test_default_cli_timeout_la_300():
    """Trần timeout mỗi lượt gọi CLI mặc định 300s (180s cũ hơi ngắn cho opus+effort)."""
    s = Settings(_env_file=None)
    assert s.llm_cli_timeout_s == 300
