"""Simulator: POST /simulations → poll → GET /alphas/{id} → metrics."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.data.client import WQBrainClient
from src.lang.ast import Call, Field
from src.lang.parser import parse_expression
from src.lang.visitors import all_subtrees
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
        if isinstance(node, Call) and node.op == op:
            for child in node.args:
                if isinstance(child, Field) and child.name not in GROUP_FIELDS:
                    out.append(child.name)
    return list(dict.fromkeys(out))  # khử trùng, giữ thứ tự


class AuthExpiredError(RuntimeError):
    """Phiên xác thực hết hạn và re-auth thất bại lặp lại — dừng sớm để tránh phí
    quota (thay vì mô phỏng hỏng hàng loạt như sự cố session chết kéo dài)."""


class QuotaExceededError(RuntimeError):
    """WQ Brain báo hết quota simulation ngày — KHÁC AuthExpiredError (không phải lỗi xác
    thực). Dừng sớm thay vì coi mỗi lần POST /simulations là "sim lỗi" rồi cứ thử ý tưởng
    khác (vòng kín sẽ không bao giờ dừng gọn nếu không phân biệt hai trường hợp này)."""


def _parse_positive_number(raw: str | None) -> float | None:
    """Parse chuỗi header thành float; trả None nếu thiếu/không parse được (an toàn,
    không đoán mò khi header lạ)."""
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _is_auth_error(status_code: int, text: str) -> bool:
    return status_code in (401, 403) or "authentication credentials" in (text or "").lower()


def _is_quota_exhausted(resp) -> bool:
    """Hết quota SIM ngày: POST /simulations vẫn 429 (WQBrainClient đã tự retry theo
    Retry-After ở tầng dưới nhưng vẫn bị chặn -> chặn dai dẳng, không phải rate-limit tạm
    thời), hoặc header X-Ratelimit-Remaining báo 0 (tài liệu WQ: 'Remaining simulations for
    the day'). Không tự tin đọc header thiếu -> mặc định KHÔNG coi là hết quota."""
    if resp.status_code == 429:
        return True
    remaining = resp.headers.get("X-Ratelimit-Remaining")
    if remaining is None:
        return False
    try:
        return int(remaining) <= 0
    except ValueError:
        return False


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
    failed_checks: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    # Lý do gốc (text) khi bị `pre_sim_validator` loại TRƯỚC khi chạm Brain — None nghĩa là
    # kết quả này ĐÃ đi qua API thật (kể cả khi status="error" vì lỗi Brain). Phân biệt "chưa
    # tốn quota" khỏi "sim thật rớt" (Task 3, spec C2: đừng gộp 2 trường hợp vào 1 status).
    presim_reason: str | None = None

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
    # WQ Brain đôi lúc xử lý sim > 10 phút (tải/queue cao). Bằng chứng live 2026-07-08: core
    # ĐÃ KIỂM CHỨNG (~1.57) bị bỏ oan ở 600s vì Brain chưa kịp COMPLETE -> metric rỗng, mất cơ
    # hội xác nhận alpha tốt. Nâng 600->1200s để sim chậm kịp hoàn tất (đánh đổi: timeout thật
    # chờ lâu hơn, chấp nhận được vì sim rỗng làm hỏng cả vòng săn alpha đạt chuẩn nộp).
    TIMEOUT_SECONDS = 1200.0
    # Số lần POST /simulations lỗi xác thực LIÊN TIẾP trước khi bỏ cuộc (client đã
    # tự re-auth 1 lần; còn 401 nghĩa là session chết) — chặn phí quota kéo dài.
    MAX_CONSECUTIVE_AUTH_FAILURES = 3
    # Cap sleep tự-giãn rate-limit (giây): chống footgun nếu X-Ratelimit-Reset là epoch tuyệt đối.
    MAX_RATE_LIMIT_SLEEP = 300.0

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

    def _respect_rate_limit(self, resp) -> None:
        """Tự-giãn PHÒNG NGỪA: đọc X-Ratelimit-Remaining trên response POST /simulations
        THÀNH CÔNG; nếu đã về 0 (hay thấp hơn), chủ động sleep theo X-Ratelimit-Reset (hoặc
        Retry-After nếu Reset vắng mặt) TRƯỚC khi tiếp tục — thay vì đợi tới khi bị 429 mới
        xử lý (đó là việc của `_is_quota_exhausted`, xảy ra SAU khi đã bị chặn). Header
        thiếu hoặc không parse được -> bỏ qua, không đoán mò (an toàn)."""
        remaining = _parse_positive_number(resp.headers.get("X-Ratelimit-Remaining"))
        if remaining is None or remaining > 0:
            return
        wait_s = _parse_positive_number(resp.headers.get("X-Ratelimit-Reset"))
        if wait_s is None:
            wait_s = _parse_positive_number(resp.headers.get("Retry-After"))
        if wait_s is None:
            return
        # AN TOÀN: nếu Brain trả Reset là timestamp EPOCH tuyệt đối (không phải giây tương đối),
        # sleep(wait_s) sẽ chờ ~vô tận -> cap ở MAX_RATE_LIMIT_SLEEP. Reset quota thực tế chỉ vài
        # giây tới ~1 phút; cap 300s vừa đủ rộng vừa chặn footgun epoch.
        wait_s = min(wait_s, self.MAX_RATE_LIMIT_SLEEP)
        logger.info(
            "X-Ratelimit-Remaining=0 -> chủ động sleep {}s trước khi tiếp tục (tránh bị 429).",
            wait_s,
        )
        self._sleep(wait_s)

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
                    presim_reason=reason,
                )
        body = self._build_body(expression, settings)

        with self.rate_limiter:
            resp = self.client.post("/simulations", json=body)

        if resp.status_code not in (200, 201):
            logger.error("POST /simulations lỗi {}: {}", resp.status_code, resp.text)
            if _is_quota_exhausted(resp):
                raise QuotaExceededError(
                    f"WQ Brain hết quota simulation ngày (HTTP {resp.status_code}, "
                    f"X-Ratelimit-Remaining={resp.headers.get('X-Ratelimit-Remaining')})."
                )
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

        self._respect_rate_limit(resp)  # phòng ngừa: sleep trước khi poll nếu quota đã cạn.

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
        failed_check_names = [
            c.get("name") for c in checks
            if isinstance(c, dict) and c.get("result") == "FAIL" and c.get("name")
        ]

        return SimulationResult(
            expression=expression,
            alpha_id=alpha_id,
            status=status,
            failed_checks=failed_check_names,
            raw=payload,
            **metrics,
            **os_metrics,
        )
