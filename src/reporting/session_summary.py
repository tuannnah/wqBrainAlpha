"""Báo cáo cuối phiên closed-loop (IMPROVEMENT_SPEC §3 Pha 0).

Thu thập outcome mỗi ứng viên trong phiên -> funnel theo stage_reached, phân bố fail_check
và family, median thời gian mỗi stage, số ứng viên trùng bị chặn. Trả lời "chết ở đâu, vì
sao, tốn bao lâu" mà không phải parse CSV thủ công. In ra + ghi logs/session_summary_*.md.

Thuần logic, không mạng, không phụ thuộc IdeaOutcome cụ thể (dùng getattr) để test/consumer
dựng fake dễ dàng."""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


def summary_path(now: datetime | None = None, log_dir: str | Path = "logs") -> Path:
    ts = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    return Path(log_dir) / f"session_summary_{ts}.md"


class SessionSummary:
    """Gom thống kê funnel một phiên. `record(outcome)` mỗi ứng viên đã refine;
    `record_dup_blocked()` mỗi ứng viên trùng bị chặn TRƯỚC refine (không tính vào total)."""

    def __init__(self) -> None:
        self._by_stage: Counter[str] = Counter()
        self._by_fail: Counter[str] = Counter()
        self._by_family: Counter[str] = Counter()
        self._gen_ms: list[float] = []
        self._backtest_ms: list[float] = []
        self._sim_ms: list[float] = []
        self._total = 0
        self._passed = 0
        self._sims = 0
        self._dup_blocked = 0

    def record(self, outcome) -> None:
        self._total += 1
        g = lambda n: getattr(outcome, n, None)  # noqa: E731
        stage = g("stage_reached") or "?"
        self._by_stage[stage] += 1
        fc = g("fail_check")
        if fc:
            self._by_fail[fc] += 1
        fam = g("family")
        if fam:
            self._by_family[fam] += 1
        if getattr(outcome, "passed", False):
            self._passed += 1
        self._sims += int(getattr(outcome, "sims_used", 0) or 0)
        for name, bucket in (
            ("gen_ms", self._gen_ms), ("backtest_ms", self._backtest_ms), ("sim_ms", self._sim_ms),
        ):
            v = g(name)
            if v is not None:
                bucket.append(float(v))

    def record_dup_blocked(self) -> None:
        self._dup_blocked += 1

    @staticmethod
    def _median(xs: list[float]) -> float | None:
        return statistics.median(xs) if xs else None

    def as_dict(self) -> dict:
        return {
            "total": self._total,
            "passed": self._passed,
            "sims": self._sims,
            "dup_blocked": self._dup_blocked,
            "by_stage": dict(self._by_stage),
            "by_fail_check": dict(self._by_fail),
            "by_family": dict(self._by_family),
            "median_ms": {
                "gen": self._median(self._gen_ms),
                "backtest": self._median(self._backtest_ms),
                "sim": self._median(self._sim_ms),
            },
        }

    @staticmethod
    def _fmt_counter(c: dict, empty: str = "  (không có)") -> str:
        if not c:
            return empty
        items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        return "\n".join(f"  - {k}: {v}" for k, v in items)

    @staticmethod
    def _fmt_ms(v: float | None) -> str:
        return "—" if v is None else f"{v:.0f}ms"

    def render_markdown(self) -> str:
        d = self.as_dict()
        m = d["median_ms"]
        lines = [
            "# Tóm tắt phiên closed-loop",
            "",
            f"- Tổng ứng viên đã refine: **{d['total']}** "
            f"(đạt: {d['passed']}, sim đã dùng: {d['sims']})",
            f"- Ứng viên trùng bị chặn trước refine: **{d['dup_blocked']}**",
            "",
            "## Funnel theo stage_reached (chết ở đâu)",
            self._fmt_counter(d["by_stage"]),
            "",
            "## Phân bố fail_check (vì sao)",
            self._fmt_counter(d["by_fail_check"]),
            "",
            "## Phân bố family (họ nhân tố)",
            self._fmt_counter(d["by_family"]),
            "",
            "## Thời gian trung vị mỗi stage (tốn bao lâu)",
            f"  - gen: {self._fmt_ms(m['gen'])}",
            f"  - backtest: {self._fmt_ms(m['backtest'])}",
            f"  - sim: {self._fmt_ms(m['sim'])}",
            "",
        ]
        return "\n".join(lines)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render_markdown(), encoding="utf-8")
        return p
