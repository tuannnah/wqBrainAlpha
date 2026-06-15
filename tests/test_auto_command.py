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


def test_run_ga_with_progress_noi_callback_va_tra_ket_qua():
    """_run_ga_with_progress phải gọi opt.run với on_generation/on_simulation
    (để hiện tiến trình) và trả đúng kết quả của opt.run."""
    from src.optimization.evolution import GenerationStats

    captured = {}

    class _FakeOpt:
        def run(self, on_generation=None, on_simulation=None):
            captured["has_gen"] = on_generation is not None
            captured["has_sim"] = on_simulation is not None
            # Mô phỏng vài lần gọi callback để chắc không vỡ (truy cập đúng field).
            on_simulation(1, "rank(close)", 0.5)
            on_generation(GenerationStats(0, 1.23, 0.4, "rank(close)"))
            return ["node1", "node2"]

    result = main._run_ga_with_progress(_FakeOpt(), total=3)

    assert result == ["node1", "node2"]
    assert captured == {"has_gen": True, "has_sim": True}
