"""Orchestrator toàn trình: điều phối thuần, không biết httpx/CLI.

Nhận 3 callback (prepare, propose_directions, run_direction) + cấu hình dừng.
Lo vòng lặp + điều kiện dừng (K-pass / trần sim / hết hướng) + thu thập kết quả.
Test được bằng fake callback, không gọi mạng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PassedAlpha:
    expression: str
    sharpe: float | None
    fitness: float | None
    direction: str  # hướng nguồn (rỗng nếu GA)


@dataclass
class DirectionOutcome:
    passed: list[PassedAlpha]
    sims_used: int


@dataclass
class PrepareInfo:
    fields: int
    operators: int


@dataclass
class AutoEvent:
    kind: str       # prepare | directions | direction_start | direction_done | stop
    message: str
    data: dict = field(default_factory=dict)


@dataclass
class AutoResult:
    passed_alphas: list[PassedAlpha]
    directions_run: int
    total_sims: int
    stop_reason: str


@dataclass
class AutoPipeline:
    prepare: Callable[[], PrepareInfo]
    propose_directions: Callable[[int], list[str]]
    run_direction: Callable[[str], DirectionOutcome]
    target_passes: int = 3
    max_total_sims: int = 60
    max_directions: int = 5
    on_event: Callable[[AutoEvent], None] | None = None

    def _emit(self, kind: str, message: str, **data) -> None:
        if self.on_event is not None:
            self.on_event(AutoEvent(kind=kind, message=message, data=data))

    def run(self) -> AutoResult:
        passed: list[PassedAlpha] = []
        total_sims = 0
        directions_run = 0
        stop_reason = "hết_hướng"

        info = self.prepare()
        self._emit(
            "prepare",
            f"✓ đăng nhập | fields={info.fields} | operators={info.operators}",
            fields=info.fields,
            operators=info.operators,
        )

        directions = self.propose_directions(self.max_directions)
        self._emit(
            "directions",
            f"Sẽ thử {len(directions)} hướng",
            directions=list(directions),
        )

        total = len(directions)
        for i, direction in enumerate(directions, start=1):
            if len(passed) >= self.target_passes:
                stop_reason = "đủ_K_pass"
                break
            if total_sims >= self.max_total_sims:
                stop_reason = "chạm_trần_sim"
                break

            self._emit(
                "direction_start",
                f"[Hướng {i}/{total}] {direction!r}",
                index=i,
                total=total,
                direction=direction,
            )
            outcome = self.run_direction(direction)
            passed.extend(outcome.passed)
            total_sims += outcome.sims_used
            directions_run += 1
            self._emit(
                "direction_done",
                f"+{len(outcome.passed)} alpha đạt | sim lượt={outcome.sims_used} "
                f"| tổng pass={len(passed)}/{self.target_passes} | tổng sim={total_sims}",
                index=i,
                added=len(outcome.passed),
                sims_used=outcome.sims_used,
                total_passed=len(passed),
                total_sims=total_sims,
            )

        if len(passed) >= self.target_passes:
            stop_reason = "đủ_K_pass"

        self._emit(
            "stop",
            f"Dừng: {stop_reason} — pass={len(passed)}, sim={total_sims}, hướng đã chạy={directions_run}",
            stop_reason=stop_reason,
            total_passed=len(passed),
            total_sims=total_sims,
            directions_run=directions_run,
        )

        return AutoResult(
            passed_alphas=passed,
            directions_run=directions_run,
            total_sims=total_sims,
            stop_reason=stop_reason,
        )
