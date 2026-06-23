"""Driver marathon: chạy liên tục nhiều hướng nghiên cứu cho tới khi LLM hết quota.

Tách bạch khỏi RefinementLoop để test được không cần LLM/sim thật:
- `direction_provider()` -> str: lấy hướng nghiên cứu kế tiếp (LLM tự đề xuất).
- `run_direction(direction)` -> LoopResult: chạy một vòng tinh chỉnh cho hướng đó.

Quy tắc dừng: gặp QuotaExhaustedError ở bất kỳ đâu -> DỪNG hẳn (hết quota). Lỗi
tạm thời (RuntimeError khác) -> retry tối đa `max_retries` lần rồi BỎ hướng và đi
tiếp. Lỗi không phải RuntimeError -> ném ra ngoài (bug thật, không nuốt).
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.llm.errors import QuotaExhaustedError


@dataclass
class MarathonReport:
    directions_completed: int = 0
    directions_skipped: int = 0
    total_sims: int = 0
    total_zoo_added: int = 0
    stop_reason: str = ""  # "quota" khi hết quota


def _emit(on_event, kind, direction, payload):
    if on_event is not None:
        on_event(kind, direction, payload)


def run_marathon(direction_provider, run_direction, *, max_retries: int = 2, on_event=None) -> MarathonReport:
    report = MarathonReport()
    while True:
        try:
            direction = direction_provider()
        except QuotaExhaustedError as exc:
            report.stop_reason = "quota"
            _emit(on_event, "quota", None, exc)
            logger.info("Marathon dừng: hết quota khi sinh hướng ({})", exc)
            break

        # Chạy hướng này, retry lỗi tạm thời, bỏ hướng nếu vượt ngưỡng retry.
        attempt = 0
        while True:
            try:
                result = run_direction(direction)
            except QuotaExhaustedError as exc:
                report.stop_reason = "quota"
                _emit(on_event, "quota", direction, exc)
                logger.info("Marathon dừng: hết quota khi chạy hướng ({})", exc)
                return report
            except RuntimeError as exc:
                attempt += 1
                if attempt > max_retries:
                    report.directions_skipped += 1
                    _emit(on_event, "skip", direction, exc)
                    logger.warning("Bỏ hướng sau {} lần lỗi tạm: {}", attempt, exc)
                    break
                _emit(on_event, "retry", direction, exc)
                logger.warning("Lỗi tạm (lần {}), retry hướng: {}", attempt, exc)
                continue
            else:
                report.directions_completed += 1
                report.total_sims += getattr(result, "sims_used", 0)
                report.total_zoo_added += getattr(result, "zoo_added", 0)
                _emit(on_event, "done", direction, result)
                break
    return report
