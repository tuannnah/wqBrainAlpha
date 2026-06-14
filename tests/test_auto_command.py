"""Test _run_auto truyền scope cụ thể (chống lỗi OptionInfo lọt vào cache key).

Regression: trước đây start() gọi lệnh Typer auto() như hàm thường -> region/
universe/delay là typer OptionInfo -> cache key sai -> gọi API -> HTTP 400.
"""

from __future__ import annotations

import main


class _FakeClient:
    authenticated = True

    def authenticate(self, *a, **k):
        return None


class _FakeFieldRepo:
    scope = None

    def __init__(self, *a, **k):
        pass

    def ensure(self, region, universe, delay, **k):
        _FakeFieldRepo.scope = (region, universe, delay)
        return ([1, 2, 3], False)


class _FakeOpRepo:
    def __init__(self, *a, **k):
        pass

    def ensure(self, force=False):
        return ([1], False)


class _FakeGen:
    def generate_ideas(self, n):
        return []


def test_run_auto_truyen_scope_cu_the(monkeypatch):
    _FakeFieldRepo.scope = None
    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(main, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(main, "FieldRepository", _FakeFieldRepo)
    monkeypatch.setattr(main, "OperatorRepository", _FakeOpRepo)
    monkeypatch.setattr(main, "_make_llm_generator", lambda sf, pf: _FakeGen())

    result = main._run_auto(
        "ai", "USA", "TOP3000", 1,
        target_passes=3, max_sims=1, max_directions=0,
    )

    # Scope tới ensure phải là chuỗi/số cụ thể, KHÔNG phải OptionInfo.
    assert _FakeFieldRepo.scope == ("USA", "TOP3000", 1)
    # Không hướng -> dừng sạch, không gọi API.
    assert result.stop_reason == "hết_hướng"
