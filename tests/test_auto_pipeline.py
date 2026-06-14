"""Test AutoPipeline bằng fake callback — không gọi mạng."""

from __future__ import annotations

import pytest

from src.pipeline.auto import (
    AutoPipeline,
    DirectionOutcome,
    PassedAlpha,
    PrepareInfo,
)


def _pa(expr: str, direction: str = "") -> PassedAlpha:
    return PassedAlpha(expression=expr, sharpe=1.5, fitness=1.1, direction=direction)


def test_dung_khi_het_huong():
    calls = {"run": 0}

    def prepare() -> PrepareInfo:
        return PrepareInfo(fields=10, operators=5)

    def propose(n: int) -> list[str]:
        return ["h1", "h2"]

    def run_direction(direction: str) -> DirectionOutcome:
        calls["run"] += 1
        return DirectionOutcome(passed=[], sims_used=1)

    pipe = AutoPipeline(
        prepare=prepare,
        propose_directions=propose,
        run_direction=run_direction,
        target_passes=99,
        max_total_sims=999,
        max_directions=5,
    )
    result = pipe.run()

    assert calls["run"] == 2           # chạy đúng 2 hướng được đề xuất
    assert result.directions_run == 2
    assert result.total_sims == 2
    assert result.stop_reason == "hết_hướng"
    assert result.passed_alphas == []


def test_dung_khi_du_k_pass():
    calls = {"run": 0}

    def run_direction(direction: str) -> DirectionOutcome:
        calls["run"] += 1
        return DirectionOutcome(passed=[_pa(f"e{calls['run']}a"), _pa(f"e{calls['run']}b")], sims_used=3)

    pipe = AutoPipeline(
        prepare=lambda: PrepareInfo(10, 5),
        propose_directions=lambda n: ["h1", "h2", "h3", "h4", "h5"],
        run_direction=run_direction,
        target_passes=3,
        max_total_sims=999,
        max_directions=5,
    )
    result = pipe.run()

    assert calls["run"] == 2           # mỗi hướng 2 pass; sau hướng 2 đã có 4 >= 3 -> dừng
    assert len(result.passed_alphas) == 4
    assert result.stop_reason == "đủ_K_pass"
    assert result.directions_run == 2


def test_kiem_dieu_kien_dung_o_dau_vong():
    calls = {"run": 0}

    def run_direction(direction: str) -> DirectionOutcome:
        calls["run"] += 1
        return DirectionOutcome(passed=[_pa("only")], sims_used=1)

    pipe = AutoPipeline(
        prepare=lambda: PrepareInfo(10, 5),
        propose_directions=lambda n: ["h1", "h2", "h3"],
        run_direction=run_direction,
        target_passes=1,
        max_total_sims=999,
        max_directions=3,
    )
    result = pipe.run()

    assert calls["run"] == 1           # hướng đầu đủ 1 pass -> hướng 2 KHÔNG được gọi
    assert result.stop_reason == "đủ_K_pass"


def test_dung_khi_cham_tran_sim():
    calls = {"run": 0}

    def run_direction(direction: str) -> DirectionOutcome:
        calls["run"] += 1
        return DirectionOutcome(passed=[], sims_used=25)

    pipe = AutoPipeline(
        prepare=lambda: PrepareInfo(10, 5),
        propose_directions=lambda n: ["h1", "h2", "h3", "h4", "h5"],
        run_direction=run_direction,
        target_passes=99,
        max_total_sims=60,
        max_directions=5,
    )
    result = pipe.run()

    # Hướng 1 (25) + hướng 2 (50): chưa chạm; đầu vòng 3 tổng=50<60 vẫn chạy -> 75.
    # Đầu vòng 4: 75 >= 60 -> dừng. Vậy chạy 3 hướng.
    assert calls["run"] == 3
    assert result.total_sims == 75
    assert result.stop_reason == "chạm_trần_sim"


def test_phat_du_su_kien():
    events = []

    pipe = AutoPipeline(
        prepare=lambda: PrepareInfo(10, 5),
        propose_directions=lambda n: ["h1", "h2"],
        run_direction=lambda d: DirectionOutcome(passed=[], sims_used=1),
        target_passes=99,
        max_total_sims=999,
        max_directions=5,
        on_event=lambda ev: events.append(ev),
    )
    pipe.run()

    kinds = [e.kind for e in events]
    assert kinds == [
        "prepare",
        "directions",
        "direction_start",
        "direction_done",
        "direction_start",
        "direction_done",
        "stop",
    ]
    # sự kiện stop mang lý do dừng
    assert events[-1].data.get("stop_reason") == "hết_hướng"


def test_prepare_loi_thi_dung_sach():
    calls = {"run": 0, "propose": 0}

    def prepare() -> PrepareInfo:
        raise RuntimeError("login hỏng")

    def run_direction(d):
        calls["run"] += 1
        return DirectionOutcome(passed=[], sims_used=1)

    def propose(n):
        calls["propose"] += 1
        return ["h1"]

    pipe = AutoPipeline(
        prepare=prepare,
        propose_directions=propose,
        run_direction=run_direction,
    )

    with pytest.raises(RuntimeError, match="login hỏng"):
        pipe.run()

    assert calls["run"] == 0        # chưa chạy hướng nào
    assert calls["propose"] == 0    # cũng chưa sinh hướng


from src.pipeline.auto import passed_from_ga


class _FakeSimResult:
    """Giả lập result của simulator cho hard_filter + score."""
    def __init__(self, sharpe, fitness, turnover, drawdown, status="passed"):
        self._m = {"sharpe": sharpe, "fitness": fitness, "turnover": turnover,
                   "returns": 0.1, "drawdown": drawdown, "margin": 0.002}
        self.status = status

    def metrics(self):
        return dict(self._m)


def test_passed_from_ga_loc_alpha_dat_nguong():
    # alpha tốt (đạt ngưỡng filter mặc định) + alpha tệ (trượt)
    good_expr = "rank(close)"
    bad_expr = "rank(open)"
    results = {
        good_expr: _FakeSimResult(sharpe=1.8, fitness=1.3, turnover=0.25, drawdown=0.08),
        bad_expr: _FakeSimResult(sharpe=0.2, fitness=0.1, turnover=0.9, drawdown=0.5),
    }

    passed = passed_from_ga([good_expr, bad_expr], results)

    assert [p.expression for p in passed] == [good_expr]
    assert passed[0].direction == ""           # GA không có hướng
    assert passed[0].sharpe == 1.8
    assert passed[0].fitness == 1.3
