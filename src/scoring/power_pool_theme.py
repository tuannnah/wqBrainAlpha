"""Lịch Power Pool Theme — dữ liệu THỦ CÔNG lấy từ bài 'Current month Power Pool Themes'
(https://support.worldquantbrain.com/hc/en-us/articles/38927747787031), WQ Brain cập nhật
HÀNG THÁNG/HÀNG TUẦN. Đây KHÔNG phải dữ liệu tự fetch được (chưa tìm ra endpoint/trang đọc
được tự động qua tool hiện có — đã thử WebFetch (403) và read_forum_post qua wqb-mcp (timeout))
— CẦN CẬP NHẬT THỦ CÔNG mỗi khi có tháng/theme mới, bằng cách thêm `PowerPoolThemeWeek` mới vào
CALENDAR bên dưới (copy nguyên văn từ bài viết, KHÔNG suy diễn field còn thiếu).

Nguồn xác nhận 2026-07-02 (tài khoản tuananhpo13@gmail.com, GOLD Genius, CONSULTANT_APPROVED):
2 theme đầu tháng 6 ("USA D1 Fast Datasets", "GLB D1 Datasets") chỉ có TÊN, không có filter chi
tiết trong bản đã nhận từ người dùng — để None/rỗng thay vì đoán. Theme tuần 29/6-5/7 có filter
đầy đủ. Cụm "neutralization in (slow, fast, slow and fast, ram, statistical, crowding)" KHÔNG
khớp VALID_NEUTRALIZATIONS chuẩn của SimConfig — giữ nguyên văn trong `unparsed_constraints`,
KHÔNG dùng để pass/fail trong `matches_theme()`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

# Map token tự do trong "neutralization in (...)" của theme -> enum API BRAIN (verbatim từ
# docs/worldquantbrain/docs/advanced-topics/*-risk-neutralized-alphas.md). Token lạ -> bỏ qua.
_NEUT_TOKEN_TO_API: dict[str, str] = {
    "slow": "SLOW",
    "fast": "FAST",
    "slow and fast": "SLOW_AND_FAST",
    "ram": "REVERSION_AND_MOMENTUM",
    "statistical": "STATISTICAL",
    "crowding": "CROWDING",
}


def parse_allowed_neutralizations(raw: str | None) -> frozenset[str]:
    """Từ cụm 'neutralization in (a, b, ...)' trả tập enum API cho phép. Token lạ bị bỏ (không
    đoán). Không có cụm / raw None -> frozenset() (nghĩa: theme không ràng buộc neutralization)."""
    if not raw:
        return frozenset()
    m = re.search(r"neutralization\s+in\s*\(([^)]*)\)", raw)
    if not m:
        return frozenset()
    out: set[str] = set()
    for tok in m.group(1).split(","):
        key = tok.strip().lower()
        if key in _NEUT_TOKEN_TO_API:
            out.add(_NEUT_TOKEN_TO_API[key])
    return frozenset(out)


@dataclass(frozen=True)
class PowerPoolThemeWeek:
    start_date: date
    end_date: date
    name: str | None = None
    region: str | None = None
    delay: int | None = None
    universe: str | None = None
    datasets_excluded: tuple[str, ...] = ()
    unparsed_constraints: str | None = None
    allowed_neutralizations: frozenset[str] = frozenset()

    def contains(self, d: date) -> bool:
        return self.start_date <= d <= self.end_date


def parse_theme_filter(raw: str) -> dict:
    """Trích region/delay/universe/datasets_excluded từ chuỗi filter dạng
    "region=USA & delay=1 & universe=TOP1000 and neutralization in (...) and datasets not in
    ['pv1']". Phần "neutralization in (...)" giữ nguyên văn trong `unparsed_constraints`."""
    result: dict = {
        "region": None, "delay": None, "universe": None,
        "datasets_excluded": (), "unparsed_constraints": None,
    }
    m = re.search(r"region\s*=\s*([A-Za-z0-9_]+)", raw)
    if m:
        result["region"] = m.group(1)
    m = re.search(r"delay\s*=\s*(\d+)", raw)
    if m:
        result["delay"] = int(m.group(1))
    m = re.search(r"universe\s*=\s*([A-Za-z0-9_]+)", raw)
    if m:
        result["universe"] = m.group(1)
    m = re.search(r"datasets\s+not\s+in\s*\[([^\]]*)\]", raw)
    if m:
        items = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
        result["datasets_excluded"] = tuple(items)
    m = re.search(r"(neutralization\s+in\s*\([^)]*\))", raw)
    if m:
        result["unparsed_constraints"] = m.group(1)
    return result


# CALENDAR — dữ liệu THỦ CÔNG, xem docstring module để biết cách cập nhật.
JUNE_JULY_2026_CALENDAR: list[PowerPoolThemeWeek] = [
    PowerPoolThemeWeek(date(2026, 6, 1), date(2026, 6, 7), name="USA D1 Fast Datasets"),
    PowerPoolThemeWeek(date(2026, 6, 8), date(2026, 6, 14), name="USA D1 Fast Datasets"),
    PowerPoolThemeWeek(date(2026, 6, 15), date(2026, 6, 21), name="GLB D1 Datasets"),
    PowerPoolThemeWeek(date(2026, 6, 22), date(2026, 6, 28), name="GLB D1 Datasets"),
    PowerPoolThemeWeek(
        date(2026, 6, 29), date(2026, 7, 5),
        region="USA", delay=1, universe="TOP1000",
        datasets_excluded=("pv1",),
        unparsed_constraints="neutralization in (slow, fast, slow and fast, ram, statistical, crowding)",
    ),
    PowerPoolThemeWeek(
        date(2026, 7, 6), date(2026, 7, 12),
        region="USA", delay=1, universe="TOP1000",
        datasets_excluded=("pv1",),
        unparsed_constraints="neutralization in (slow, fast, slow and fast, ram, statistical, crowding)",
        allowed_neutralizations=parse_allowed_neutralizations(
            "neutralization in (slow, fast, slow and fast, ram, statistical, crowding)"
        ),
    ),
]


def theme_for_date(
    d: date, calendar: list[PowerPoolThemeWeek] | None = None
) -> PowerPoolThemeWeek | None:
    """Trả theme đang áp dụng cho ngày `d`, hoặc None nếu ngoài lịch đã biết (lịch cần cập nhật
    thủ công khi sang tháng mới/theme mới — xem docstring module)."""
    cal = calendar if calendar is not None else JUNE_JULY_2026_CALENDAR
    for week in cal:
        if week.contains(d):
            return week
    return None


def matches_theme(
    week: PowerPoolThemeWeek, *, region: str, delay: int, universe: str, datasets_used: set[str],
    neutralization: str | None = None,
) -> tuple[bool, list[str]]:
    """Kiểm 1 alpha có khớp `week` không — CHỈ kiểm phần đã parse chắc chắn (region/delay/
    universe/datasets_excluded/allowed_neutralizations). Field nào của `week` là None thì KHÔNG
    chặn (chưa biết để so, không phải "match tất cả"). `week.unparsed_constraints` (nếu có) KHÔNG
    ảnh hưởng kết quả ở đây — người gọi tự đọc `week.unparsed_constraints` để xem lại thủ công.
    `neutralization=None` -> KHÔNG xét neut (giữ tương thích ngược cho nơi gọi cũ chưa truyền)."""
    reasons: list[str] = []
    if week.region is not None and region.upper() != week.region.upper():
        reasons.append(f"region {region} != theme yêu cầu {week.region}")
    if week.delay is not None and delay != week.delay:
        reasons.append(f"delay {delay} != theme yêu cầu {week.delay}")
    if week.universe is not None and universe.upper() != week.universe.upper():
        reasons.append(f"universe {universe} != theme yêu cầu {week.universe}")
    excluded_used = datasets_used & set(week.datasets_excluded)
    if excluded_used:
        reasons.append(f"dùng dataset bị loại trừ theo theme: {sorted(excluded_used)}")
    if (
        neutralization is not None
        and week.allowed_neutralizations
        and neutralization.upper() not in week.allowed_neutralizations
    ):
        reasons.append(
            f"neutralization {neutralization} không thuộc tập theme cho phép "
            f"{sorted(week.allowed_neutralizations)}"
        )
    return (not reasons, reasons)


def check_theme_compliance(
    *, region: str, delay: int, universe: str, neutralization: str,
    datasets_used: set[str], on_date: date,
    calendar: list[PowerPoolThemeWeek] | None = None,
) -> tuple[bool, list[str]]:
    """Gate trước khi nộp Pure Power Pool: alpha (region/delay/universe/neutralization/datasets)
    có khớp theme của `on_date` không. Không có theme cho ngày đó -> (True, []) (không chặn ở
    đây; việc có nộp Pure Power Pool hay không do nơi gọi quyết). Lệch -> (False, reasons) để
    log rõ (tránh để Brain trả 'does not match any Power Pool Theme').

    LƯU Ý TRẠNG THÁI (2026-07-09): đây là utility gọi THỦ CÔNG — người dùng tự gọi hàm này
    trước khi nộp một alpha Pure Power Pool (qua menu/wqb-mcp). Hàm này HIỆN CHƯA được gắn
    (wire) vào bất kỳ đường tự động nộp nào trong `main.py`/closed-loop, vì repo CHƯA có luồng
    "auto nộp Pure Power Pool" — mọi lần nộp Pure Power Pool hiện tại đều là nộp thủ công. Khi
    nào có đường auto-submit cho Pure Power Pool, cần wire gate này vào đó trước khi gọi
    submit_alpha; đừng nhầm là gate đã tự động chặn ở phiên hiện tại."""
    week = theme_for_date(on_date, calendar)
    if week is None:
        return (True, [])
    return matches_theme(
        week, region=region, delay=delay, universe=universe,
        datasets_used=datasets_used, neutralization=neutralization,
    )
