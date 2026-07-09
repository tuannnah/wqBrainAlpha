"""Ghi log mỗi phiên Auto SIM ra CSV: một dòng/ý tưởng với schema CỐ ĐỊNH, LUÔN điền đủ cột
(IMPROVEMENT_SPEC §3 Pha 0). Tách local_sharpe/brain_sharpe/brain_fitness (đừng gộp local với
Brain), thêm trường chẩn đoán funnel: stage_reached, fail_check, family, expr_depth, *_ms,
dedup_key. Mỗi phiên một file `logs/alphas_<timestamp>.csv`; append + flush từng dòng để
Ctrl+C/hết quota vẫn giữ dữ liệu.

Khác bản cũ: brain_sharpe/brain_fitness ghi BẤT KỂ passed (sim đã chạy thì luôn có số) — để
phân biệt "sim rồi trượt" với "chưa sim"; log cũ nuốt sharpe khi failed nên không phân tích
tự động được (spec §1)."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

COLUMNS = [
    "#", "status", "stage_reached", "fail_check", "family", "source",
    "expression", "expr_depth", "dedup_key",
    "region", "universe", "delay", "neutralization", "decay", "truncation",
    "local_sharpe", "brain_sharpe", "brain_fitness", "turnover", "self_corr",
    "power_pool", "wq_alpha_id", "sims",
    "gen_ms", "backtest_ms", "sim_ms", "stop_reason",
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
    """Mở CSV per-run + ghi header ngay; `.log(index, outcome)` append 1 dòng và flush.

    Mọi dòng có đúng len(COLUMNS) field (không lệch cột). Trường thiếu -> ô trống,
    KHÔNG bỏ qua metric có sẵn."""

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
        g = lambda name: getattr(outcome, name, None)  # noqa: E731 - đọc trường Pha 0 an toàn
        self._w.writerow([
            index, status, _s(g("stage_reached")), _s(g("fail_check")), _s(g("family")),
            _s(outcome.source), _s(outcome.expr), _s(g("expr_depth")), _s(g("dedup_key")),
            _s(s.get("region")), _s(s.get("universe")), _s(s.get("delay")),
            _s(s.get("neutralization")), _s(s.get("decay")), _s(s.get("truncation")),
            _s(g("local_sharpe")), _s(outcome.sharpe), _s(outcome.fitness),
            _s(outcome.turnover), _s(outcome.self_corr),
            _s(getattr(outcome, "power_pool_eligible", False)),
            _s(outcome.wq_alpha_id), _s(outcome.sims_used),
            _s(g("gen_ms")), _s(g("backtest_ms")), _s(g("sim_ms")), _s(outcome.stop_reason),
        ])
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()
