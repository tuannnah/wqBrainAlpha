"""Orchestrator toàn trình: điều phối thuần, không biết httpx/CLI.

Nhận 3 callback (prepare, propose_directions, run_direction) + cấu hình dừng.
Lo vòng lặp + điều kiện dừng (K-pass / trần sim / hết hướng) + thu thập kết quả.
Test được bằng fake callback, không gọi mạng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

UNLIMITED_DIRECTION_BATCH_SIZE = 5


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
    swallow_errors: bool = False

    def _emit(self, kind: str, message: str, **data) -> None:
        if self.on_event is not None:
            self.on_event(AutoEvent(kind=kind, message=message, data=data))

    def _build_result(
        self,
        passed: list[PassedAlpha],
        directions_run: int,
        total_sims: int,
        stop_reason: str,
    ) -> AutoResult:
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

        unlimited = self.max_directions <= 0
        batch_size = UNLIMITED_DIRECTION_BATCH_SIZE if unlimited else self.max_directions
        while True:
            if len(passed) >= self.target_passes:
                stop_reason = "đủ_K_pass"
                break
            if total_sims >= self.max_total_sims:
                stop_reason = "chạm_trần_sim"
                break

            try:
                directions = self.propose_directions(batch_size)
            except KeyboardInterrupt:
                if not self.swallow_errors:
                    raise
                return self._build_result(passed, directions_run, total_sims, "ctrl_c")
            except Exception as exc:
                if not self.swallow_errors:
                    raise
                reason = f"lỗi: {type(exc).__name__}: {str(exc)[:120]}"
                return self._build_result(passed, directions_run, total_sims, reason)

            self._emit(
                "directions",
                f"Sẽ thử {len(directions)} hướng",
                directions=list(directions),
            )
            if not directions:
                stop_reason = "hết_hướng"
                break

            total_label = "∞" if unlimited else str(len(directions))
            for direction in directions:
                if len(passed) >= self.target_passes:
                    stop_reason = "đủ_K_pass"
                    break
                if total_sims >= self.max_total_sims:
                    stop_reason = "chạm_trần_sim"
                    break

                index = directions_run + 1
                self._emit(
                    "direction_start",
                    f"[Hướng {index}/{total_label}] {direction!r}",
                    index=index,
                    total=0 if unlimited else len(directions),
                    direction=direction,
                )
                try:
                    outcome = self.run_direction(direction)
                except KeyboardInterrupt:
                    if not self.swallow_errors:
                        raise
                    return self._build_result(passed, directions_run, total_sims, "ctrl_c")
                except Exception as exc:
                    if not self.swallow_errors:
                        raise
                    reason = f"lỗi: {type(exc).__name__}: {str(exc)[:120]}"
                    return self._build_result(passed, directions_run, total_sims, reason)
                passed.extend(outcome.passed)
                total_sims += outcome.sims_used
                directions_run += 1
                self._emit(
                    "direction_done",
                    f"+{len(outcome.passed)} alpha đạt | sim lượt={outcome.sims_used} "
                    f"| tổng pass={len(passed)}/{self.target_passes} | tổng sim={total_sims}",
                    index=index,
                    added=len(outcome.passed),
                    sims_used=outcome.sims_used,
                    total_passed=len(passed),
                    total_sims=total_sims,
                )

            if not unlimited or stop_reason != "hết_hướng":
                break

        if len(passed) >= self.target_passes:
            stop_reason = "đủ_K_pass"

        return self._build_result(passed, directions_run, total_sims, stop_reason)


def passed_from_ga(expressions, results) -> list[PassedAlpha]:
    """Lọc các biểu thức GA đạt ngưỡng hard-filter -> PassedAlpha (direction='').

    expressions: danh sách expr ứng viên (theo thứ tự tốt→kém).
    results: dict expr -> sim result (có .metrics() và .status).
    """
    from src.scoring.filter import passes as hard_filter
    from src.scoring.metrics import normalize

    out: list[PassedAlpha] = []
    for expr in expressions:
        result = results.get(expr)
        if result is None:
            continue
        ok, _ = hard_filter(result)
        if result.status == "passed" and ok:
            m = normalize(result)
            out.append(
                PassedAlpha(expression=expr, sharpe=m["sharpe"], fitness=m["fitness"], direction="")
            )
    return out
