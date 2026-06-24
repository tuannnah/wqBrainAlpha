"""Mọi ngưỡng gate/submission của MiniBrain ở MỘT nơi (Gap #7/R9 master spec).

Không hardcode các số này ở call site — đổi theo cuộc thi thì sửa tại đây.
"""

from __future__ import annotations

# --- Cấu trúc biểu thức ---
MAX_DEPTH: int = 7  # trần độ sâu AST — đếm cây biểu thức TRẦN (nhất quán pre_filter, không thêm node wrapper config)

# --- Self-correlation (cổng chặn submission thật sự) ---
SELF_CORR_MAX: float = 0.70  # PnL self-corr >= ngưỡng này -> hard fail

# --- Metrics ---
TURNOVER_FLOOR: float = 0.125  # sàn turnover trong công thức fitness
WEIGHT_CONCENTRATION_CAP: float = 0.10  # |weight| 1 mã tối đa (gate tập trung)
SHARPE_MIN: float = 1.0  # sàn Sharpe (soft score)
PER_YEAR_SHARPE_MIN: float = 0.0  # sàn Sharpe năm tệ nhất (regime robustness)
TURNOVER_BAND: tuple[float, float] = (0.01, 0.70)  # dải turnover hợp lệ (soft)

# --- Calibration (tin cậy cả tool) ---
CALIBRATION_RHO_BAR: float = 0.5  # Spearman ρ tối thiểu để tin ranking local
