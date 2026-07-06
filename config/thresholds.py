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

# --- IS-Ladder Sharpe (robustness theo cửa sổ trượt, docs consultant-submission-tests) ---
# Xét Sharpe N=2..10 năm GẦN NHẤT: FAIL nếu tụt dưới sàn (alpha suy thoái gần đây),
# PASS nếu vượt ngưỡng leo thang. Delay-1 USA. Đổi theo cuộc thi thì sửa tại đây.
IS_LADDER_FAIL: float = 1.58  # Sharpe cửa sổ nào < ngưỡng này -> FAIL ngay
IS_LADDER_PASS: dict[int, float] = {
    2: 2.38, 3: 2.38, 4: 2.38, 5: 2.38,  # năm 2-5 chặt nhất
    6: 2.22, 7: 2.06, 8: 1.90, 9: 1.74, 10: 1.59,  # nới dần tới toàn kỳ
}
IS_LADDER_LOW_TURNOVER_CUTOFF: float = 0.30  # turnover < mức này -> nhân PASS ×mult
IS_LADDER_LOW_TURNOVER_MULT: float = 0.85  # chỉ nhân lên ngưỡng PASS, KHÔNG nhân FAIL

# --- Calibration (tin cậy cả tool) ---
CALIBRATION_RHO_BAR: float = 0.5  # Spearman ρ tối thiểu để tin ranking local
