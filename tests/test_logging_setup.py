"""Test cấu hình logging: không ghi đè log production khi chạy test.

Bối cảnh: `logs/wq_alpha_<date>.log` là log production. Loguru dùng handler
toàn cục, nên trước đây chạy pytest làm dính file sink → mọi logger.error của
test (fixture foo_bar, rank(a,b,c)...) đổ vào log production, gây "nhiễu" khó
soi lỗi thật. `_setup_logging` phải bỏ file sink khi biến môi trường
WQ_NO_FILE_LOG được đặt (conftest đặt sẵn cho cả phiên test).
"""

from __future__ import annotations

from pathlib import Path

import main


def _spy_sinks(monkeypatch) -> list:
    added: list = []
    monkeypatch.setattr(main.logger, "remove", lambda *a, **k: None)
    monkeypatch.setattr(main.logger, "add", lambda sink, *a, **k: added.append(sink) or 0)
    return added


def test_setup_logging_bo_file_sink_khi_co_env(monkeypatch):
    """WQ_NO_FILE_LOG đặt → không add sink dạng đường dẫn file."""
    added = _spy_sinks(monkeypatch)
    monkeypatch.setenv("WQ_NO_FILE_LOG", "1")
    main._setup_logging()
    assert not any(isinstance(s, (str, Path)) for s in added), added
    # Vẫn còn ít nhất sink stream (stderr) để xem log lúc chạy.
    assert added, "phải còn sink stderr"


def test_setup_logging_van_co_file_sink_khi_khong_env(monkeypatch):
    """Run thật (không có env) → vẫn ghi file log như cũ."""
    added = _spy_sinks(monkeypatch)
    monkeypatch.delenv("WQ_NO_FILE_LOG", raising=False)
    main._setup_logging()
    assert any(isinstance(s, (str, Path)) for s in added), added
