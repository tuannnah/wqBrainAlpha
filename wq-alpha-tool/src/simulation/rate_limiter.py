"""Điều tiết nhịp gọi WQ Brain: giới hạn đồng thời + delay tối thiểu giữa 2 POST."""

from __future__ import annotations

import threading
import time

from tenacity import retry, retry_if_result, stop_after_attempt, wait_exponential

RETRYABLE_STATUS = {429, 500, 503}


def is_retryable_response(resp) -> bool:
    """Dùng cho tenacity.retry_if_result — retry khi status thuộc nhóm tạm thời."""
    status = getattr(resp, "status_code", None)
    return status in RETRYABLE_STATUS


def with_backoff(max_attempts: int = 5):
    """Decorator retry exponential backoff cho hàm trả về httpx.Response."""
    return retry(
        retry=retry_if_result(is_retryable_response),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )


class RateLimiter:
    """Giới hạn số simulation đồng thời và delay tối thiểu giữa 2 POST /simulations."""

    def __init__(
        self,
        max_concurrent: int = 3,
        min_delay: float = 1.0,
        sleep_func=time.sleep,
        time_func=time.monotonic,
    ):
        self.max_concurrent = max_concurrent
        self.min_delay = min_delay
        self._sleep = sleep_func
        self._time = time_func
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._last_post = 0.0

    def acquire(self) -> None:
        self._semaphore.acquire()

    def release(self) -> None:
        self._semaphore.release()

    def wait_for_slot(self) -> None:
        """Đảm bảo cách lần POST trước ít nhất min_delay giây."""
        with self._lock:
            now = self._time()
            elapsed = now - self._last_post
            if elapsed < self.min_delay:
                self._sleep(self.min_delay - elapsed)
            self._last_post = self._time()

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        self.wait_for_slot()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
