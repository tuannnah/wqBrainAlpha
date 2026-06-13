"""Simulator: POST /simulations → poll → GET /alphas/{id} → metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.data.client import WQBrainClient
from src.simulation.rate_limiter import RateLimiter

SIM_DEFAULTS: dict[str, Any] = {
    "type": "REGULAR",
    "settings": {
        "instrumentType": "EQUITY",
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "decay": 0,
        "neutralization": "SUBINDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "unitHandling": "VERIFY",
        "nanHandling": "OFF",
        "language": "FASTEXPR",
        "visualization": False,
    },
}

# Các metric quan tâm trong block `is` của alpha.
_METRIC_KEYS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin")
# Metric Out-of-Sample (block `os`) — trọng tài cuối chống overfit IS (T5.6).
_OS_KEYS = ("sharpe", "fitness")


@dataclass
class SimulationResult:
    expression: str
    alpha_id: str | None = None
    status: str = "error"  # passed/failed/error
    sharpe: float | None = None
    fitness: float | None = None
    turnover: float | None = None
    returns: float | None = None
    drawdown: float | None = None
    margin: float | None = None
    os_sharpe: float | None = None
    os_fitness: float | None = None
    raw: dict = field(default_factory=dict)

    def metrics(self) -> dict[str, float | None]:
        return {k: getattr(self, k) for k in _METRIC_KEYS}


class SimulationError(RuntimeError):
    pass


class Simulator:
    POLL_INTERVAL = 3.0
    TIMEOUT_SECONDS = 300.0

    def __init__(
        self,
        client: WQBrainClient,
        rate_limiter: RateLimiter | None = None,
        sleep_func=time.sleep,
        time_func=time.monotonic,
    ):
        self.client = client
        self.rate_limiter = rate_limiter or RateLimiter()
        self._sleep = sleep_func
        self._time = time_func

    def _build_body(self, expression: str, settings: dict | None) -> dict:
        body = {
            "type": SIM_DEFAULTS["type"],
            "settings": dict(SIM_DEFAULTS["settings"]),
            "regular": expression,
        }
        if settings:
            body["settings"].update(settings)
        return body

    def simulate(self, expression: str, settings: dict | None = None) -> SimulationResult:
        body = self._build_body(expression, settings)

        with self.rate_limiter:
            resp = self.client.post("/simulations", json=body)

        if resp.status_code not in (200, 201):
            logger.error("POST /simulations lỗi {}: {}", resp.status_code, resp.text)
            return SimulationResult(expression=expression, status="error", raw={"error": resp.text})

        location = resp.headers.get("Location")
        if not location:
            return SimulationResult(
                expression=expression, status="error", raw={"error": "thiếu Location header"}
            )

        try:
            progress = self._poll(location)
        except SimulationError as exc:
            logger.error("Simulation lỗi/timeout: {}", exc)
            return SimulationResult(expression=expression, status="error", raw={"error": str(exc)})

        alpha_id = progress.get("alpha")
        if not alpha_id:
            return SimulationResult(
                expression=expression, status="error", raw=progress
            )

        return self._fetch_metrics(expression, alpha_id)

    def _poll(self, location: str) -> dict:
        deadline = self._time() + self.TIMEOUT_SECONDS
        while True:
            resp = self.client.get(location)
            retry_after = resp.headers.get("Retry-After")
            if resp.status_code in (200, 201):
                payload = resp.json()
                status = (payload.get("status") or "").upper()
                if status in ("COMPLETE", "WARNING"):
                    return payload
                if status in ("ERROR", "FAILED"):
                    raise SimulationError(f"status={status}")
            elif resp.status_code >= 400:
                raise SimulationError(f"poll HTTP {resp.status_code}")

            if self._time() >= deadline:
                raise SimulationError("timeout khi poll simulation")

            delay = float(retry_after) if retry_after else self.POLL_INTERVAL
            self._sleep(delay)

    def _fetch_metrics(self, expression: str, alpha_id: str) -> SimulationResult:
        resp = self.client.get(f"/alphas/{alpha_id}")
        if resp.status_code not in (200, 201):
            return SimulationResult(
                expression=expression,
                alpha_id=alpha_id,
                status="error",
                raw={"error": resp.text},
            )
        payload = resp.json()
        is_block = payload.get("is") or {}
        metrics = {k: is_block.get(k) for k in _METRIC_KEYS}

        # Block `os` (Out-of-Sample) — trọng tài cuối, có thể thiếu (T5.6).
        os_block = payload.get("os") or {}
        os_metrics = {f"os_{k}": os_block.get(k) for k in _OS_KEYS}

        # Status xác định bởi checks (PASS/FAIL) nếu có, mặc định 'passed'.
        checks = is_block.get("checks") or []
        failed = any((c.get("result") == "FAIL") for c in checks if isinstance(c, dict))
        status = "failed" if failed else "passed"

        return SimulationResult(
            expression=expression,
            alpha_id=alpha_id,
            status=status,
            raw=payload,
            **metrics,
            **os_metrics,
        )
