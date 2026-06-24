# src/calibration/report.py
"""CalibrationReport — kết quả của CalibrationHarness (B10 master spec).

spearman_sharpe là **chỉ số đầu tàu**: gate cả công cụ trên CALIBRATION_RHO_BAR
(config/thresholds.py). Không tin ranking local cho tới khi ρ >= bar trên tập >= ~50 alpha.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    n: int                              # số BrainRecord dùng để tính report
    spearman_sharpe: float              # local vs Brain Sharpe rank-corr (đầu tàu)
    spearman_fitness: float             # local vs Brain Fitness rank-corr
    self_corr_agreement: float          # tỉ lệ local/Brain đồng ý gate self_corr<0.70
    decile_hit_rate: float              # của top-decile Brain, bao nhiêu local cũng top-decile
    by_year: dict[int, float]           # spearman_sharpe theo năm (regime robustness)
