"""Tải sẵn (warm) toàn bộ datafields + operators cho nhiều scope, resume được.

Tận dụng FieldRepository/OperatorRepository (đã có cache + TTL) và retry 429 sẵn
ở WQBrainClient. Mỗi scope tự lưu trạng thái nên chạy lại chỉ làm phần còn thiếu.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from loguru import logger

from src.data.fields import FieldFetchError, FieldRepository
from src.data.operators import OperatorRepository

Scope = tuple[str, str, int]


@dataclass
class WarmCacheReport:
    fetched: int = 0          # số scope fetch mới thành công (có field)
    skipped: int = 0          # số scope đã cache còn hạn -> bỏ qua
    no_access: int = 0        # số scope không quyền/empty -> đánh dấu no_access
    errors: list = field(default_factory=list)  # list[tuple[Scope, str]]
    operators: int = 0        # số operator đã đảm bảo trong cache


def warm_cache(
    field_repo: FieldRepository,
    operator_repo: OperatorRepository,
    scopes: Iterable[Scope],
    *,
    force: bool = False,
    sleep_s: float = 2.0,
    sleep_func: Callable[[float], None] | None = None,
    on_event: Callable[[str, Scope], None] | None = None,
) -> WarmCacheReport:
    """Duyệt scopes, fetch field còn thiếu, đánh dấu no_access cho scope không quyền.

    force=True: bỏ qua cache, tải lại tất cả (kể cả scope đã no_access).
    sleep_s: nghỉ giữa các scope CÓ gọi API (giảm rủi ro 429).
    on_event(kind, scope): callback tiến độ; kind in
        {"fetched","skip_cached","skip_no_access","no_access","error"}.
    """
    sleep_func = sleep_func or _time.sleep
    report = WarmCacheReport()

    ops, _ = operator_repo.ensure(force=force)
    report.operators = len(ops)

    def _emit(kind: str, scope: Scope) -> None:
        if on_event is not None:
            on_event(kind, scope)

    for scope in scopes:
        region, universe, delay = scope

        if not force:
            state = field_repo.get_state(region, universe, delay)
            if state is not None and state.status == "no_access":
                report.no_access += 1
                _emit("skip_no_access", scope)
                continue

        try:
            fields, fetched = field_repo.ensure(region, universe, delay, force=force)
        except FieldFetchError as exc:
            if getattr(exc, "status_code", None) in (401, 403):
                field_repo.mark_no_access(region, universe, delay)
                report.no_access += 1
                _emit("no_access", scope)
            else:
                report.errors.append((scope, str(exc)))
                _emit("error", scope)
                logger.warning("warm-cache lỗi {}: {}", scope, exc)
            continue

        if not fetched:
            report.skipped += 1
            _emit("skip_cached", scope)
            continue

        if not fields:
            field_repo.mark_no_access(region, universe, delay)
            report.no_access += 1
            _emit("no_access", scope)
        else:
            report.fetched += 1
            _emit("fetched", scope)

        sleep_func(sleep_s)

    return report
