# src/backtest/gates.py
"""GateVerdict + GateEvaluator — hard gates (chặn) tách bạch khỏi soft scores (xếp hạng).

Ngưỡng GATE (MAX_DEPTH, SELF_CORR_MAX, WEIGHT_CONCENTRATION_CAP, TURNOVER_BAND) CHỈ đọc từ
config/thresholds.py (Gap #7/R9 master spec) — không hardcode số ngưỡng ở đây. Riêng một
epsilon dung sai số học nhỏ (`1e-9`, để hấp thụ trôi số dấu phẩy động sau water-filling +
_scale ở portfolio.py) KHÔNG phải là ngưỡng gate nên được đặt inline tại nơi dùng — cùng quy
ước với portfolio.py::_truncate (`cap_abs + 1e-15`) và llm/mcts.py (`+ 1e-9`).
Hard gates: depth<=MAX_DEPTH, fields_ok, self_corr<SELF_CORR_MAX (strict), weight_
concentration<=WEIGHT_CONCENTRATION_CAP. Soft scores (B8: "tradable in search", không chặn
passed): sharpe, fitness, turnover-band, per_year_min — caller (filter.evaluate_local,
GP fitness Phase 7) tự quyết định ngưỡng thêm trên các điểm này.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.thresholds import MAX_DEPTH, SELF_CORR_MAX, TURNOVER_BAND, WEIGHT_CONCENTRATION_CAP
from src.backtest.metrics_local import AlphaMetrics


@dataclass(frozen=True, slots=True)
class GateVerdict:
    passed: bool
    hard_failures: list[str] = field(default_factory=list)
    soft_scores: dict[str, float] = field(default_factory=dict)


class GateEvaluator:
    """Đánh giá AlphaMetrics đã tính sẵn (không tự parse/eval/backtest — đó là việc của
    src/backtest/gate.py ở tầng wrapper end-to-end, Task 4.6 sẽ nối hai lớp này)."""

    def evaluate(
        self, m: AlphaMetrics, self_corr: float, depth: int, fields_ok: bool
    ) -> GateVerdict:
        hard_failures: list[str] = []

        if depth > MAX_DEPTH:
            hard_failures.append(f"depth {depth} > MAX_DEPTH {MAX_DEPTH}")
        if not fields_ok:
            hard_failures.append("fields_ok=False (field không hợp lệ)")
        if abs(self_corr) >= SELF_CORR_MAX:
            hard_failures.append(
                f"self_corr {self_corr:.3f} >= SELF_CORR_MAX {SELF_CORR_MAX}"
            )
        # +1e-9: dung sai FP cho cap đúng-bằng (vd PortfolioConfig.truncation=0.10 trùng
        # WEIGHT_CONCENTRATION_CAP=0.10) — water-filling (_truncate) + chia/nhân (_scale)
        # tích lũy sai số ~1e-17, không phải vi phạm concentration thật. Cùng kiểu dung sai
        # `cap_abs + 1e-15` đã dùng trong portfolio.py::_truncate, chỉ lớn hơn vì đây là
        # sau _scale (chia rồi nhân lại) nên trôi nhiều hơn vòng lặp gốc.
        if m.weight_concentration > WEIGHT_CONCENTRATION_CAP + 1e-9:
            hard_failures.append(
                f"weight_concentration {m.weight_concentration:.3f} > "
                f"WEIGHT_CONCENTRATION_CAP {WEIGHT_CONCENTRATION_CAP}"
            )

        soft_scores = {
            "sharpe": m.sharpe,
            "fitness": m.fitness,
            "turnover_band": self._turnover_band_score(m.turnover),
            "per_year_min": min(m.per_year_sharpe.values()) if m.per_year_sharpe else 0.0,
        }

        return GateVerdict(
            passed=len(hard_failures) == 0,
            hard_failures=hard_failures,
            soft_scores=soft_scores,
        )

    def _turnover_band_score(self, turnover: float) -> float:
        lo, hi = TURNOVER_BAND
        if lo <= turnover <= hi:
            return 1.0
        if turnover < lo:
            return -(lo - turnover)
        return -(turnover - hi)
