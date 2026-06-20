"""Simulator: POST /simulations → poll → GET /alphas/{id} → metrics."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.data.client import WQBrainClient
from src.generation.ast_utils import Leaf, Node, all_subtrees, parse_expression
from src.simulation.rate_limiter import RateLimiter

# WQ trả "Invalid data field <id>." khi field được liệt kê nhưng không simulate được.
_INVALID_FIELD_RE = re.compile(r"Invalid data field (\w+)")
# WQ trả "Operator <op> does not support event inputs" khi field 'event' (dataset
# news/earnings/analyst…) bị đưa vào operator chuẩn — không suy ra được từ type cache.
_EVENT_OP_RE = re.compile(r"Operator (\w+) does not support event inputs")
# Reason từ PreFilter khi field/hằng không nằm trong catalog (tiền-kiểm).
_REJECTED_FIELD_RE = re.compile(r"Field/hằng không tồn tại: (\S+)")

# Tên GROUP cho group_neutralize/group_* — là tham số phân nhóm, KHÔNG phải data
# field nên không bao giờ là "field chết". Loại khỏi trích lỗi để không blacklist
# nhầm (gốc rễ 'sector' lọt blacklist, chặn oan group_neutralize sau này).
GROUP_FIELDS = frozenset(
    {"market", "sector", "industry", "subindustry", "country", "exchange"}
)


def extract_rejected_field(reason: str) -> str | None:
    """Trích field id từ reason tiền-kiểm 'Field/hằng không tồn tại: X'; None nếu khác."""
    if not reason:
        return None
    m = _REJECTED_FIELD_RE.search(reason)
    return m.group(1) if m else None


def extract_invalid_field(error: str) -> str | None:
    """Trích field id từ thông điệp lỗi WQ 'Invalid data field X'; None nếu khác."""
    if not error:
        return None
    m = _INVALID_FIELD_RE.search(error)
    return m.group(1) if m else None


def extract_event_fields(error: str, expression: str) -> list[str]:
    """Trích các field 'event' gây lỗi: là input field-leaf TRỰC TIẾP của operator
    mà WQ báo 'does not support event inputs'. Trả [] nếu lỗi khác hoặc parse hỏng."""
    if not error or not expression:
        return []
    m = _EVENT_OP_RE.search(error)
    if not m:
        return []
    op = m.group(1)
    try:
        tree = parse_expression(expression)
    except Exception:
        return []
    out: list[str] = []
    for node in all_subtrees(tree):
        if isinstance(node, Node) and node.op == op:
            for child in node.children:
                if (
                    isinstance(child, Leaf)
                    and isinstance(child.value, str)
                    and child.value not in GROUP_FIELDS
                ):
                    out.append(child.value)
    return list(dict.fromkeys(out))  # khử trùng, giữ thứ tự


class AuthExpiredError(RuntimeError):
    """Phiên xác thực hết hạn và re-auth thất bại lặp lại — dừng sớm để tránh phí
    quota (thay vì mô phỏng hỏng hàng loạt như sự cố session chết kéo dài)."""


def _is_auth_error(status_code: int, text: str) -> bool:
    return status_code in (401, 403) or "authentication credentials" in (text or "").lower()


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


def _error_detail(payload: dict) -> str:
    """Trích lý do lỗi từ payload simulation ERROR/FAILED của WQ Brain.

    WQ Brain thường đặt mô tả ở `message`; đôi khi nằm trong list lỗi con
    (`children`) hoặc trong `regular`/`detail`. Trả chuỗi rỗng nếu không thấy
    để caller fallback về mình status."""
    for key in ("message", "detail", "error"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    children = payload.get("children")
    if isinstance(children, list):
        msgs = [
            str(c.get("message")).strip()
            for c in children
            if isinstance(c, dict) and c.get("message")
        ]
        if msgs:
            return "; ".join(msgs)
    return ""


class Simulator:
    POLL_INTERVAL = 3.0
    TIMEOUT_SECONDS = 300.0
    # Số lần POST /simulations lỗi xác thực LIÊN TIẾP trước khi bỏ cuộc (client đã
    # tự re-auth 1 lần; còn 401 nghĩa là session chết) — chặn phí quota kéo dài.
    MAX_CONSECUTIVE_AUTH_FAILURES = 3

    def __init__(
        self,
        client: WQBrainClient,
        rate_limiter: RateLimiter | None = None,
        sleep_func=time.sleep,
        time_func=time.monotonic,
        on_invalid_field=None,
        pre_sim_validator=None,
    ):
        self.client = client
        self.rate_limiter = rate_limiter or RateLimiter()
        self._sleep = sleep_func
        self._time = time_func
        # callback(field_id) khi WQ báo field 'chết'/'event' — để blacklist tránh sinh lại.
        self.on_invalid_field = on_invalid_field
        # callback(expr)->(ok, reason): chặn biểu thức field-bịa TRƯỚC khi tốn 1 lượt API.
        self.pre_sim_validator = pre_sim_validator
        self._consecutive_auth_failures = 0

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
        if self.pre_sim_validator is not None:
            ok, reason = self.pre_sim_validator(expression)
            if not ok:
                logger.warning("Bỏ sim (tiền-kiểm): {} | expr={}", reason, expression)
                bad = extract_rejected_field(reason)
                if bad and self.on_invalid_field is not None:
                    self.on_invalid_field(bad)
                return SimulationResult(
                    expression=expression, status="error",
                    raw={"error": f"pre-sim reject: {reason}"},
                )
        body = self._build_body(expression, settings)

        with self.rate_limiter:
            resp = self.client.post("/simulations", json=body)

        if resp.status_code not in (200, 201):
            logger.error("POST /simulations lỗi {}: {}", resp.status_code, resp.text)
            if _is_auth_error(resp.status_code, resp.text):
                self._consecutive_auth_failures += 1
                if self._consecutive_auth_failures >= self.MAX_CONSECUTIVE_AUTH_FAILURES:
                    raise AuthExpiredError(
                        f"Xác thực thất bại {self._consecutive_auth_failures} lần liên tiếp "
                        "— dừng để tránh phí quota. Hãy đăng nhập lại (re-auth)."
                    )
            else:
                self._consecutive_auth_failures = 0
            return SimulationResult(expression=expression, status="error", raw={"error": resp.text})

        self._consecutive_auth_failures = 0  # POST thành công -> reset bộ đếm auth.
        location = resp.headers.get("Location")
        if not location:
            return SimulationResult(
                expression=expression, status="error", raw={"error": "thiếu Location header"}
            )

        try:
            progress = self._poll(location)
        except SimulationError as exc:
            logger.error("Simulation lỗi/timeout: {} | expr={}", exc, expression)
            if self.on_invalid_field is not None:
                bad_field = extract_invalid_field(str(exc))
                if bad_field:
                    self.on_invalid_field(bad_field)
                for event_field in extract_event_fields(str(exc), expression):
                    self.on_invalid_field(event_field)
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
                    detail = _error_detail(payload)
                    msg = f"status={status}: {detail}" if detail else f"status={status}"
                    raise SimulationError(msg)
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
