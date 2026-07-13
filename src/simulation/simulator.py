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


# Giới hạn số phần tử/mảng multi-simulation (POST /simulations với body là MẢNG) — nguồn:
# docs/worldquantbrain/docs/_/brain-api.md dòng 266 (tài liệu chính thức /simulations POST):
# "Multiple simulations can be run by posting an array of length 2..10 of the above simulation
# objects." Mảng > 10 phần tử phải chia nhiều request (simulate_many tự chunk).
MULTI_SIM_MAX = 10

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
        return self._simulate_one(expression, settings)

    def _simulate_one(self, expression: str, settings: dict | None) -> SimulationResult:
        """Đường sim ĐƠN đầy đủ: POST /simulations (body 1 object) -> poll Location -> GET
        /alphas/{id}. Tách khỏi `simulate()` (đã qua pre_sim_validator) để `simulate_many` tái
        dùng cho phần tử lẻ (chunk 1 job — WQ Brain KHÔNG chấp nhận mảng multi-sim 1 phần tử,
        xem docstring `simulate_many`) và cho fallback tuần tự khi multi-sim lỗi."""
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

    def simulate_many(self, jobs: list[tuple[str, dict | None]]) -> list[SimulationResult]:
        """Mô phỏng NHIỀU biểu thức trong 1 lần chờ, dùng tính năng multi-simulation của WQ
        Brain (POST /simulations với body là MẢNG payload) thay vì N lần POST/poll tuần tự.

        NGUỒN xác nhận định dạng (điều tra Task 6, KHÔNG đoán mò):
        1. `docs/worldquantbrain/docs/_/brain-api.md` (dòng 213-337, tài liệu BRAIN API chính
           thức, mục `/simulations` POST) — nguồn CHUẨN, trích dẫn nguyên văn:
           - "Multiple simulations can be run by posting an array of length 2..10 of the above
             simulation objects. The user requires the MULTI_SIMULATION permission... Also the
             settings for the simulation must be compatible, they must have the same simulation
             type, instrument type, region, delay and language."
           - Payload mẫu: `[{"type":"REGULAR",...}, {"type":"REGULAR",...}, ...]` — MỖI phần tử
             có CÙNG hình dạng với body sim đơn (`{"type","settings","regular"}`).
           - Response THÀNH CÔNG giống sim đơn: `201 Created` + header `Location: /simulations/
             <id>` — nhưng `<id>` này là simulation CHA (parent).
           - "Progress of multi-simulations is tracked by a parent simulation object. A child
             simulation is created for each of the multi-simulation objects. The list of child
             simulation ids are available when the parent simulation is complete."
           - Poll `GET /simulations/<parent id>` trả cùng schema sim đơn (`status`, `progress`,
             `Retry-After` header khi chưa xong) CỘNG thêm field `"children": [<id>, <id>, ...]`
             khi cha đã có danh sách con. Chi tiết từng con: `GET /simulations/<child id>` —
             CÙNG schema/luồng poll với sim đơn (status COMPLETE/WARNING -> field `alpha`).
        2. `wqb_mcp/tools/simulation_tools.py::create_multi_simulation` + `_wait_for_multi
           simulation_completion` (server MCP `wqb-mcp` cài local tại
           `C:\\Users\\PC\\.venvs\\wqb-mcp\\Lib\\site-packages\\wqb_mcp\\tools\\simulation_tools.py`,
           đối chiếu THỰC TẾ 1 cài đặt đã chạy production) — xác nhận lại đúng luồng trên: POST
           `f"{base_url}/simulations"` với `json=multisimulation_data` (list các dict cùng hình
           dạng sim đơn), đọc header `Location` làm parent, poll parent tới khi JSON body có
           `children` (list) khác rỗng, rồi với mỗi `child_id` — GET `f"{base_url}/simulations/
           {child_id}"` (hoặc dùng thẳng nếu `child_id` đã là URL đầy đủ) lặp lại CHÍNH XÁC vòng
           poll của sim đơn (đọc `Retry-After`, `status`, cuối cùng lấy `alpha` rồi GET
           `/alphas/{alpha_id}`). Tool này tự giới hạn 2-8 biểu thức/lần (quy ước RIÊNG của
           tool, KHÔNG phải giới hạn cứng của API — giới hạn cứng là 2..10 theo nguồn (1)).

        Hành vi (khớp `Simulator.simulate` đường đơn, xem interface Task 6):
        - Giữ NGUYÊN THỨ TỰ `jobs` trong kết quả trả về.
        - Job bị `pre_sim_validator` chặn (field/toán tử bịa) -> `SimulationResult(presim_
          reason=...)` NGAY, KHÔNG chiếm slot trong mảng payload gửi Brain (giống đường đơn).
        - Job hợp lệ còn lại được CHIA THÀNH CÁC CHUNK ≤ `MULTI_SIM_MAX` (10, theo nguồn (1)):
          chunk 1 phần tử dùng ĐƯỜNG ĐƠN (`_simulate_one`, mảng 1 phần tử KHÔNG hợp lệ với API);
          chunk ≥2 phần tử dùng multi-sim thật (1 lần POST mảng, 1 lần chờ chung).
        - 429/quota hoặc auth hết hạn ở bước POST multi-sim -> raise `QuotaExceededError`/
          `AuthExpiredError` GIỐNG HỆT đường đơn (không nuốt, không fallback — hết quota là
          tín hiệu dừng thật).
        - Lỗi KHÁC ở tầng multi-sim (POST hỏng theo cách khác, thiếu Location, mismatch số
          children, timeout poll cha...) -> FALLBACK: sim TUẦN TỰ từng job trong chunk đó qua
          `_simulate_one` (log warning, KHÔNG chết phiên) — an toàn hơn bỏ cuộc cả batch.
        - `TIMEOUT_SECONDS` dùng CHUNG cho cả cha lẫn các con của 1 chunk (tính 1 deadline khi
          bắt đầu POST chunk; poll con nối tiếp phần thời gian còn lại, không reset đồng hồ)."""
        if not jobs:
            return []
        results: list[SimulationResult | None] = [None] * len(jobs)
        eligible_idx: list[int] = []
        eligible_jobs: list[tuple[str, dict | None]] = []
        for i, (expr, settings) in enumerate(jobs):
            if self.pre_sim_validator is not None:
                ok, reason = self.pre_sim_validator(expr)
                if not ok:
                    logger.warning("Bỏ sim (tiền-kiểm, multi-sim): {} | expr={}", reason, expr)
                    bad = extract_rejected_field(reason)
                    if bad and self.on_invalid_field is not None:
                        self.on_invalid_field(bad)
                    results[i] = SimulationResult(
                        expression=expr, status="error",
                        raw={"error": f"pre-sim reject: {reason}"}, presim_reason=reason,
                    )
                    continue
            eligible_idx.append(i)
            eligible_jobs.append((expr, settings))

        pos = 0
        while pos < len(eligible_jobs):
            chunk = eligible_jobs[pos: pos + MULTI_SIM_MAX]
            chunk_idx = eligible_idx[pos: pos + MULTI_SIM_MAX]
            pos += MULTI_SIM_MAX
            if len(chunk) == 1:
                expr, settings = chunk[0]
                chunk_results = [self._simulate_one(expr, settings)]
            else:
                chunk_results = self._simulate_batch(chunk)
            for gi, r in zip(chunk_idx, chunk_results):
                results[gi] = r

        return results  # type: ignore[return-value]  # mọi ô đã được điền ở trên

    def _simulate_batch(self, chunk: list[tuple[str, dict | None]]) -> list[SimulationResult]:
        """1 chunk (2..MULTI_SIM_MAX job) -> 1 request multi-sim. Lỗi tầng multi-sim (không
        phải quota/auth) -> fallback sim TUẦN TỰ từng job qua `_simulate_one`."""
        try:
            return self._post_multi_sim(chunk)
        except (QuotaExceededError, AuthExpiredError):
            raise  # tín hiệu dừng thật — KHÔNG fallback (sim tuần tự cũng sẽ hết quota).
        except Exception as exc:
            logger.warning(
                "Multi-sim lỗi ({}) — fallback sim TUẦN TỰ {} biểu thức.", exc, len(chunk),
            )
            return [self._simulate_one(expr, settings) for expr, settings in chunk]

    def _post_multi_sim(self, chunk: list[tuple[str, dict | None]]) -> list[SimulationResult]:
        exprs = [expr for expr, _ in chunk]
        body = [self._build_body(expr, settings) for expr, settings in chunk]

        with self.rate_limiter:
            resp = self.client.post("/simulations", json=body)

        if resp.status_code not in (200, 201):
            logger.error("POST /simulations (multi) lỗi {}: {}", resp.status_code, resp.text)
            if _is_quota_exhausted(resp):
                raise QuotaExceededError(
                    f"WQ Brain hết quota simulation ngày (multi-sim, HTTP {resp.status_code}, "
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
            raise SimulationError(f"multi-sim POST lỗi HTTP {resp.status_code}")

        self._consecutive_auth_failures = 0
        location = resp.headers.get("Location")
        if not location:
            raise SimulationError("multi-sim: thiếu Location header")

        self._respect_rate_limit(resp)

        deadline = self._time() + self.TIMEOUT_SECONDS
        children = self._poll_parent_children(location, deadline)
        if len(children) != len(chunk):
            raise SimulationError(
                f"multi-sim: số children ({len(children)}) khác số job đã gửi ({len(chunk)})"
            )

        results: list[SimulationResult] = []
        for expr, child_id in zip(exprs, children):
            child_location = (
                child_id if child_id.startswith(("http://", "https://", "/"))
                else f"/simulations/{child_id}"
            )
            try:
                progress = self._poll(child_location, deadline=deadline)
            except SimulationError as exc:
                logger.error("Multi-sim: con lỗi/timeout: {} | expr={}", exc, expr)
                if self.on_invalid_field is not None:
                    bad_field = extract_invalid_field(str(exc))
                    if bad_field:
                        self.on_invalid_field(bad_field)
                    for event_field in extract_event_fields(str(exc), expr):
                        self.on_invalid_field(event_field)
                results.append(SimulationResult(expression=expr, status="error", raw={"error": str(exc)}))
                continue
            alpha_id = progress.get("alpha")
            if not alpha_id:
                results.append(SimulationResult(expression=expr, status="error", raw=progress))
                continue
            results.append(self._fetch_metrics(expr, alpha_id))
        return results

    def _poll_parent_children(self, location: str, deadline: float) -> list[str]:
        """Poll simulation CHA tới khi JSON body xuất hiện `children` (list id con) khác rỗng —
        đúng cách wqb-mcp (`_wait_for_multisimulation_completion`) xác định cha đã sẵn sàng, vì
        tài liệu BRAIN API không đảm bảo field `status` của CHA phản ánh trạng thái CÁC CON (chỉ
        đảm bảo `children` xuất hiện "khi simulation cha hoàn tất"). Vẫn theo dõi `status` lỗi
        (ERROR/FAILED/CANCELLED/TIMEOUT) để không treo vô ích khi cả batch bị từ chối ở tầng cha
        (vd settings không tương thích)."""
        while True:
            resp = self.client.get(location)
            retry_after = resp.headers.get("Retry-After")
            if resp.status_code in (200, 201):
                payload = resp.json()
                children = payload.get("children")
                if children:
                    return children
                status = (payload.get("status") or "").upper()
                if status in ("ERROR", "FAILED", "CANCELLED", "TIMEOUT"):
                    detail = _error_detail(payload)
                    msg = f"multi-sim parent status={status}: {detail}" if detail else f"multi-sim parent status={status}"
                    raise SimulationError(msg)
            elif resp.status_code >= 400:
                raise SimulationError(f"poll multi-sim parent HTTP {resp.status_code}")

            if self._time() >= deadline:
                raise SimulationError("timeout khi poll multi-sim parent")

            delay = float(retry_after) if retry_after else self.POLL_INTERVAL
            self._sleep(delay)

    def _poll(self, location: str, deadline: float | None = None) -> dict:
        if deadline is None:
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
