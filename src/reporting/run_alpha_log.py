"""Ghi log mỗi phiên Auto SIM ra CSV: một dòng/ý tưởng (công thức + setting; Sharpe/fitness chỉ
cho ý tưởng ĐẠT) để người dùng soi độ lặp công thức giữa các lần chạy. Mỗi phiên một file
`logs/alphas_<timestamp>.csv`; append + flush từng dòng để Ctrl+C/hết quota vẫn giữ dữ liệu."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

COLUMNS = [
    "#", "status", "source", "expression", "region", "universe", "delay",
    "neutralization", "decay", "truncation", "sharpe", "fitness", "turnover",
    "self_corr", "power_pool", "wq_alpha_id", "sims", "stop_reason",
]


def run_log_path(now: datetime | None = None, log_dir: str | Path = "logs") -> Path:
    """Đường dẫn file log per-run: <log_dir>/alphas_<YYYY-MM-DD_HHMMSS>.csv.

    Phân giải đến GIÂY (không chỉ phút) để tránh trùng tên file khi người dùng
    Ctrl+C rồi khởi động lại phiên mới trong cùng phút -> tránh việc mở file mode
    "w" (truncate) ghi đè mất dữ liệu phiên trước.
    """
    ts = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    return Path(log_dir) / f"alphas_{ts}.csv"


def _s(v) -> str:
    """None -> ô trống; còn lại -> str."""
    return "" if v is None else str(v)


class RunAlphaLogger:
    """Mở CSV per-run + ghi header ngay; `.log(index, outcome)` append 1 dòng và flush."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8-sig")
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNS)
        self._f.flush()

    def log(self, index: int, outcome) -> None:
        s = outcome.sim_settings or {}
        passed = bool(outcome.passed)
        status = "passed" if passed else ("error" if outcome.stop_reason == "error" else "failed")
        # passed -> điền Sharpe/fitness; không đạt -> để trống theo yêu cầu người dùng.
        sharpe = _s(outcome.sharpe) if passed else ""
        fitness = _s(outcome.fitness) if passed else ""
        self._w.writerow([
            index, status, _s(outcome.source), _s(outcome.expr),
            _s(s.get("region")), _s(s.get("universe")), _s(s.get("delay")),
            _s(s.get("neutralization")), _s(s.get("decay")), _s(s.get("truncation")),
            sharpe, fitness, _s(outcome.turnover), _s(outcome.self_corr),
            _s(getattr(outcome, "power_pool_eligible", False)),
            _s(outcome.wq_alpha_id), _s(outcome.sims_used), _s(outcome.stop_reason),
        ])
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()
