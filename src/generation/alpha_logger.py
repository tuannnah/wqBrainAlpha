"""Định dạng log chi tiết từng alpha ra text để user mang lên WQ Brain mô phỏng.

Mỗi alpha in đầy đủ: mã số, họ, giả thuyết, lý giải kinh tế, biểu thức FASTEXPR,
setting đạt chuẩn (khớp scope USA/TOP3000/delay=1), điểm local + lý do.
Neutralization được suy ra từ chính biểu thức (nếu bọc group_neutralize) để
setting nhất quán với cách viết alpha.
"""

from __future__ import annotations

import re
from datetime import date

# Setting mặc định, khớp config/sim_defaults.yaml (USA/TOP3000/delay=1).
DEFAULT_SETTINGS = {
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
}

# Thứ tự in setting cho dễ đọc.
_SETTING_ORDER = (
    "instrumentType",
    "region",
    "universe",
    "delay",
    "decay",
    "neutralization",
    "truncation",
    "pasteurization",
    "unitHandling",
    "nanHandling",
    "language",
    "visualization",
)

# group hợp lệ -> giá trị neutralization tương ứng trên WQ Brain.
_GROUP_TO_NEUTRALIZATION = {
    "market": "MARKET",
    "sector": "SECTOR",
    "industry": "INDUSTRY",
    "subindustry": "SUBINDUSTRY",
    "country": "COUNTRY",
    "exchange": "EXCHANGE",
}

_NEUTRALIZE_RE = re.compile(r"group_neutralize\(.+,\s*([A-Za-z_]\w*)\s*\)\s*$")


def _infer_neutralization(expression: str) -> str:
    """Suy neutralization từ biểu thức: nếu bọc group_neutralize(..., G) thì lấy G."""
    m = _NEUTRALIZE_RE.search(expression.strip())
    if m:
        group = m.group(1).lower()
        return _GROUP_TO_NEUTRALIZATION.get(group, DEFAULT_SETTINGS["neutralization"])
    return DEFAULT_SETTINGS["neutralization"]


def settings_for(expression: str) -> dict:
    """Setting đầy đủ cho một biểu thức (suy neutralization từ biểu thức)."""
    s = dict(DEFAULT_SETTINGS)
    s["neutralization"] = _infer_neutralization(expression)
    return s


def format_alpha(candidate, index: int) -> str:
    """In một alpha thành khối text chi tiết."""
    s = settings_for(candidate.expression)
    setting_lines = "\n".join(f"    {k} = {s[k]}" for k in _SETTING_ORDER)
    reasons = "; ".join(candidate.reasons) if candidate.reasons else "—"
    orig = "—" if candidate.originality is None else f"{candidate.originality:.2f}"
    comp = "—" if candidate.complexity is None else f"{candidate.complexity:.2f}"
    return (
        f"#{index}  [{candidate.family}]\n"
        f"  Giả thuyết : {candidate.hypothesis}\n"
        f"  Lý giải    : {candidate.rationale}\n"
        f"  FASTEXPR   : {candidate.expression}\n"
        f"  Điểm local : {candidate.score:.2f}  (originality={orig}, complexity={comp})\n"
        f"  Lý do đạt  : {reasons}\n"
        f"  Setting:\n{setting_lines}\n"
    )


def format_report(candidates) -> str:
    """In toàn bộ report: tiêu đề + thống kê họ + từng alpha đánh số."""
    total = len(candidates)
    by_family: dict[str, int] = {}
    for c in candidates:
        by_family[c.family] = by_family.get(c.family, 0) + 1
    family_summary = ", ".join(f"{k}={v}" for k, v in sorted(by_family.items())) or "—"

    header = (
        "=" * 70 + "\n"
        f"ALPHA ĐẠT CHUẨN — {date.today().isoformat()}\n"
        f"Tổng số: {total} alpha  |  Theo họ: {family_summary}\n"
        "Scope: USA / TOP3000 / delay=1 (FASTEXPR)\n"
        "Hướng dẫn: copy biểu thức FASTEXPR + setting tương ứng lên WQ Brain để mô phỏng.\n"
        + "=" * 70 + "\n\n"
    )
    blocks = [format_alpha(c, i + 1) for i, c in enumerate(candidates)]
    return header + "\n".join(blocks)
