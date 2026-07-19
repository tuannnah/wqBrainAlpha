# src/calibration/report.py
"""CalibrationReport — kết quả của CalibrationHarness (B10 master spec).

spearman_sharpe là **chỉ số đầu tàu**: gate cả công cụ trên CALIBRATION_RHO_BAR
(config/thresholds.py). Không tin ranking local cho tới khi ρ >= bar trên tập >= ~50 alpha.

(T4.1) spearman_submit_score là trục đo THÊM (không thay thế spearman_sharpe làm đầu tàu):
đo ρ trên submit_score (`min(sharpe/SUBMIT_SHARPE_REF, fitness/SUBMIT_FITNESS_REF)`) — Sharpe
thô có thể xếp hạng khớp Brain trong khi điểm-nộp thực tế (thứ quyết định alpha có NỘP được
hay không) lại lệch vì trục fitness rớt riêng. KHÔNG đổi hành vi gate nào — chỉ thêm trục báo
cáo.

(T4.2) by_family: breakdown theo họ nhân tố (`classify_family`) — mỗi họ có ρ riêng + hệ số
local->Brain ước lượng riêng (`FamilyCalibration`). CHỈ để báo cáo/lưu, KHÔNG tự động wire vào
gate pre-sim production (xem `config.thresholds.calibrated_floor`, tham số `family` tùy chọn)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FamilyCalibration:
    """Calibration riêng MỘT họ nhân tố (T4.2). CHỈ để báo cáo — không tự wire vào gate."""

    family: str                    # nhãn họ (classify_family)
    n: int                         # số BrainRecord re-score được local trong họ này
    spearman_sharpe: float         # ρ Sharpe riêng họ (NaN nếu <2 cặp hợp lệ hoặc hằng số)
    local_to_brain_ratio: float    # median(brain_sharpe/local_sharpe) riêng họ; NaN nếu không đủ dữ liệu


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    n: int                              # số BrainRecord dùng để tính report
    spearman_sharpe: float              # local vs Brain Sharpe rank-corr (đầu tàu)
    spearman_fitness: float             # local vs Brain Fitness rank-corr
    spearman_submit_score: float        # local vs Brain điểm-nộp rank-corr (T4.1, trục thêm)
    self_corr_agreement: float          # tỉ lệ local/Brain đồng ý gate self_corr<0.70
    decile_hit_rate: float              # của top-decile Brain, bao nhiêu local cũng top-decile
    by_year: dict[int, float]           # spearman_sharpe theo năm (regime robustness)
    by_family: dict[str, FamilyCalibration] = field(default_factory=dict)  # T4.2, mặc định rỗng
