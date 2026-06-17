"""Test _run_auto truyền scope cụ thể (chống lỗi OptionInfo lọt vào cache key).

Regression: trước đây start() gọi lệnh Typer auto() như hàm thường -> region/
universe/delay là typer OptionInfo -> cache key sai -> gọi API -> HTTP 400.
"""

from __future__ import annotations

import main
from src.pipeline.auto import AutoResult
from src.simulation.config import SimConfig


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


def test_run_auto_ai_mac_dinh_khong_gioi_han_huong(monkeypatch):
    captured = {}

    class _FakePipe:
        def __init__(self, **kwargs):
            captured["max_directions"] = kwargs["max_directions"]
            self.prepare = kwargs["prepare"]

        def run(self):
            self.prepare()
            return AutoResult([], directions_run=0, total_sims=0, stop_reason="hết_hướng")

    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(main, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(main, "FieldRepository", _FakeFieldRepo)
    monkeypatch.setattr(main, "OperatorRepository", _FakeOpRepo)
    monkeypatch.setattr(main, "_make_llm_generator", lambda sf, pf: _FakeGen())
    monkeypatch.setattr(main, "AutoPipeline", _FakePipe)

    main._run_auto("ai", "USA", "TOP3000", 1, target_passes=3, max_sims=1)

    assert captured["max_directions"] == 0


def test_run_auto_builds_sim_config_for_ai_builder(monkeypatch):
    captured = {}

    class _FakePipe:
        def __init__(self, **kwargs):
            self.prepare = kwargs["prepare"]

        def run(self):
            self.prepare()
            return AutoResult([], directions_run=0, total_sims=0, stop_reason="het_huong")

    def _fake_run_builder(client_box, sf, region, universe, delay, per_direction_box, sim_config):
        captured["scope"] = (region, universe, delay)
        captured["sim_config"] = sim_config

        def run(direction: str):
            raise AssertionError("run_direction should not be called")

        return run

    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(main, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(main, "FieldRepository", _FakeFieldRepo)
    monkeypatch.setattr(main, "OperatorRepository", _FakeOpRepo)
    monkeypatch.setattr(main, "_make_llm_generator", lambda sf, pf: _FakeGen())
    monkeypatch.setattr(main, "AutoPipeline", _FakePipe)
    monkeypatch.setattr(main, "_auto_run_direction_ai", _fake_run_builder)

    main._run_auto(
        "ai", "EUR", "TOP1200", 0,
        target_passes=3, max_sims=1, max_directions=0,
        decay=6, truncation=0.12, neutralization="industry",
    )

    assert captured["scope"] == ("EUR", "TOP1200", 0)
    assert captured["sim_config"] == SimConfig(
        region="EUR",
        universe="TOP1200",
        delay=0,
        decay=6,
        truncation=0.12,
        neutralization="INDUSTRY",
    )


def test_run_auto_per_direction_sims_co_dinh(monkeypatch):
    """Khi per_direction_sims được set, hệ chia trần sim biến mất:
    per_direction_box["per_direction"] LUÔN bằng giá trị truyền vào,
    bất kể max_sims còn bao nhiêu."""
    captured_per_direction: list[int] = []

    class _FakePipe:
        def __init__(self, **kwargs):
            self.prepare = kwargs["prepare"]
            self.propose_directions = kwargs["propose_directions"]
            self.run_direction = kwargs["run_direction"]

        def run(self):
            self.prepare()
            # Giả lập: AutoPipeline phát "directions" (cập nhật dirs_total trong _run_auto)
            kwargs_on_event = None
            # _run_auto truyền on_event qua AutoPipeline; ta gọi nó trực tiếp qua state.
            # Để đơn giản, chỉ gọi run_direction và đọc per_direction trong fake builder.
            self.run_direction("h1")
            self.run_direction("h2")
            return AutoResult([], directions_run=2, total_sims=0, stop_reason="hết_hướng")

    def _fake_run_builder(client_box, sf, region, universe, delay, per_direction_box, sim_config):
        def run(direction: str):
            captured_per_direction.append(per_direction_box["per_direction"])
            from src.pipeline.auto import DirectionOutcome
            return DirectionOutcome(passed=[], sims_used=10)
        return run

    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(main, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(main, "FieldRepository", _FakeFieldRepo)
    monkeypatch.setattr(main, "OperatorRepository", _FakeOpRepo)
    monkeypatch.setattr(main, "_make_llm_generator", lambda sf, pf: _FakeGen())
    monkeypatch.setattr(main, "AutoPipeline", _FakePipe)
    monkeypatch.setattr(main, "_auto_run_direction_ai", _fake_run_builder)

    main._run_auto(
        "ai", "USA", "TOP3000", 1,
        target_passes=10**9, max_sims=10**18, max_directions=0,
        per_direction_sims=30,
    )

    # Cả 2 hướng đều thấy per_direction = 30 (KHÔNG còn chia remaining // dirs_left).
    assert captured_per_direction == [30, 30]


def test_run_auto_swallow_errors_truyen_xuong_pipeline(monkeypatch):
    captured = {}

    class _FakePipe:
        def __init__(self, **kwargs):
            captured["swallow_errors"] = kwargs.get("swallow_errors")
            self.prepare = kwargs["prepare"]

        def run(self):
            self.prepare()
            return AutoResult([], directions_run=0, total_sims=0, stop_reason="hết_hướng")

    monkeypatch.setattr(main, "init_db", lambda e: e)
    monkeypatch.setattr(main, "make_engine", lambda: None)
    monkeypatch.setattr(main, "make_session_factory", lambda e: (lambda: None))
    monkeypatch.setattr(main, "_make_client", lambda: _FakeClient())
    monkeypatch.setattr(main, "FieldRepository", _FakeFieldRepo)
    monkeypatch.setattr(main, "OperatorRepository", _FakeOpRepo)
    monkeypatch.setattr(main, "_make_llm_generator", lambda sf, pf: _FakeGen())
    monkeypatch.setattr(main, "AutoPipeline", _FakePipe)

    main._run_auto(
        "ai", "USA", "TOP3000", 1,
        target_passes=3, max_sims=1,
        swallow_errors=True,
    )

    assert captured["swallow_errors"] is True


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
