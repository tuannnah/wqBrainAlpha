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

import math
from dataclasses import dataclass, field

from config.thresholds import CALIBRATION_MIN_SAMPLE_N, CALIBRATION_RHO_BAR


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


def _rho_warning(label: str, rho: float) -> str | None:
    """1 dòng cảnh báo cho một trục ρ (KHÔNG áp n nhỏ — chỗ gọi tự kiểm n riêng trước)."""
    if math.isnan(rho):
        return f"{label} không xác định (NaN) — thiếu dữ liệu để tính ρ."
    if rho < CALIBRATION_RHO_BAR:
        return (
            f"{label}={rho:.3f} < CALIBRATION_RHO_BAR={CALIBRATION_RHO_BAR} — "
            "KHÔNG hạ floor/ngưỡng, chỉ báo cáo."
        )
    return None


def calibration_warnings(report: CalibrationReport) -> list[str]:
    """Sinh danh sách cảnh báo TIẾNG VIỆT cho lệnh `calibrate` (T4.3): n nhỏ (<
    CALIBRATION_MIN_SAMPLE_N -> "ρ chưa đáng tin") và ρ thấp (< CALIBRATION_RHO_BAR ->
    "KHÔNG hạ floor/ngưỡng, chỉ báo cáo"), cả tổng lẫn theo từng họ (`by_family`). Rỗng khi mọi
    số liệu khoẻ (n đủ lớn + mọi ρ >= bar).

    KỶ LUẬT (task-4-brief.md): hàm này CHỈ LIỆT KÊ — không tự động hạ/siết bất kỳ ngưỡng/gate
    nào (CALIBRATION_LOCAL_TO_BRAIN, PRE_SIM_TARGET_BRAIN_SHARPE, CALIBRATION_RHO_BAR...).
    Quyết định đổi ngưỡng khi có đủ số liệu là của USER, không phải của tool."""
    warnings: list[str] = []

    if report.n < CALIBRATION_MIN_SAMPLE_N:
        warnings.append(
            f"n={report.n} < {CALIBRATION_MIN_SAMPLE_N} — ρ chưa đáng tin (mẫu quá nhỏ)."
        )

    for label, rho in (
        ("spearman_sharpe", report.spearman_sharpe),
        ("spearman_submit_score", report.spearman_submit_score),
    ):
        w = _rho_warning(label, rho)
        if w is not None:
            warnings.append(w)

    for family in sorted(report.by_family):
        fc = report.by_family[family]
        if fc.n < CALIBRATION_MIN_SAMPLE_N:
            warnings.append(
                f"family={family}: n={fc.n} < {CALIBRATION_MIN_SAMPLE_N} — ρ chưa đáng tin."
            )
            continue
        w = _rho_warning(f"family={family}: ρ", fc.spearman_sharpe)
        if w is not None:
            warnings.append(w)

    return warnings
