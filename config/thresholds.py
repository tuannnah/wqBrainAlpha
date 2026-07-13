"""Mọi ngưỡng gate/submission của MiniBrain ở MỘT nơi (Gap #7/R9 master spec).

Không hardcode các số này ở call site — đổi theo cuộc thi thì sửa tại đây.
"""

from __future__ import annotations

# --- Cấu trúc biểu thức ---
MAX_DEPTH: int = 7  # trần độ sâu AST — đếm cây biểu thức TRẦN (nhất quán pre_filter, không thêm node wrapper config)
# Trần số node (leaf+Call) — PHẢI khớp `PreFilter.max_nodes` mặc định (src/simulation/pre_filter.py)
# vì đây là ngưỡng thật sự pre_filter dùng để reject. (RC5) GP sinh cây phải tự ràng buộc
# ngay lúc sinh theo hằng số này để tránh generate-rồi-reject lãng phí gen_ms.
MAX_NODES: int = 30

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
# Pre-sim floor cho closed-loop: local Sharpe < mức này -> KHÔNG đốt sim Brain (chắc chắn
# rác). Calibrate local≈Brain/1.28 (đo trực tiếp: winner local 1.23 vs Brain 1.57). Floor
# 0.5 -> Brain ~0.64: chỉ loại rác (đã quan sát loop sim Sharpe 0.39/-0.02/-0.13), KHÔNG
# đụng ứng viên tốt (local ~1.2+). Bảo thủ để không đói loop; nới lên nếu muốn siết quota.
PRE_SIM_LOCAL_SHARPE_FLOOR: float = 0.5

# Tỉ lệ hiệu chỉnh local->Brain đo trực tiếp: Brain sharpe ≈ local × 1.28 (winner local 1.23
# -> Brain 1.57). Dùng để suy floor từ MỤC TIÊU Brain thay vì hằng số cứng (Pha 4).
CALIBRATION_LOCAL_TO_BRAIN: float = 1.28
# Mục tiêu Brain sharpe mặc định để suy pre-sim floor: 0.64 -> floor local 0.5 (khớp floor cũ,
# nhưng nay DERIVED, chỉnh 1 chỗ). Nâng target -> siết quota (loại nhiều ứng viên yếu hơn).
PRE_SIM_TARGET_BRAIN_SHARPE: float = 0.64


# --- Degenerate position (Task 4 — chặn trước sim, xem src/lang/meaningfulness.py cho rule
# structural AST; đây là backtest-cheap rule, chạy SAU backtest local trong LocalTunerRefiner) ---
# Bằng chứng thật (log 07-12): vị thế gần hằng số/vô hướng vẫn có thể lọt qua rule AST (base
# không phải sign(...) hoặc field không thuần volume) mà local backtest vẫn cho turnover≈0
# VÀ |sharpe|≈0 đồng thời -- đó chính là dấu hiệu vị thế suy biến (không đổi/không tương quan
# gì với lợi suất) -> không đáng đốt sim Brain. Ngưỡng CHẶT (turnover<0.005, |sharpe|<0.05)
# để không chặn oan alpha turnover thấp NHƯNG sharpe có ý nghĩa thật.
DEGENERATE_TURNOVER: float = 0.005
DEGENERATE_SHARPE: float = 0.05


# --- Combiner (Task 2 — sửa 0-combo, xem logs/diag_combiner_20260712.md) ---
# Sàn sharpe BRAIN THẬT để một expr được coi là "component quý" cho combiner (Fix 1):
# KHÔNG lọc theo status — alpha 'failed' vì LOW_SHARPE (vd sharpe 1.04 < IS_LADDER_FAIL 1.58)
# vẫn qua sàn này, vì Grinold-Kahn √N có thể đẩy nó lên ngưỡng nộp khi ghép với thành phần
# ít tương quan khác.
COMBINER_MIN_BRAIN_SHARPE: float = 0.8
# Trần độ sâu MỘT tín hiệu con được phép làm component combo (Fix 3): MAX_DEPTH(7) trừ 3
# tầng wrapper build_combined_expression luôn thêm (rank chuẩn hoá + 2 tầng add cân bằng cho
# N=4). Component sâu hơn mức này chắc chắn vượt trần sau khi bọc -> loại NGAY trước greedy
# thay vì phát hiện muộn (đo được 3/5 rồi 2/5 combo chết vì depth ở diag 20260712/20260713).
COMBINER_MAX_COMPONENT_DEPTH: int = 4

# --- Điểm-nộp (submission score, Task 2 Fix 4) ---
# combine_stage so combo với thành phần mạnh nhất bằng điểm-nộp
# min(sharpe/SUBMIT_SHARPE_REF, fitness/SUBMIT_FITNESS_REF) thay vì so fitness thô — buộc
# combo phải tiến GẦN NGƯỠNG NỘP thật trên CẢ HAI trục mới được coi 'vượt trội' đáng giữ.
SUBMIT_SHARPE_REF: float = 1.25  # tham chiếu Sharpe local ứng với ngưỡng nộp Brain (calibrate)
SUBMIT_FITNESS_REF: float = 1.0  # tham chiếu fitness — khớp docs consultant (fitness > 1)


# --- Mini-sweep alt-data (Task 5 — cứu hypothesis thay vì vứt sau 1 sim) ---
# Ngưỡng |sharpe| dùng để quyết định kiểu sweep khi sim CORE (nhánh alt-data đi thẳng Brain,
# `LocalTunerRefiner._sim_direct`) CHƯA pass:
#   sharpe <= -ngưỡng      -> thử FLIP DẤU (bằng chứng: seed social từng SAI DẤU, Sharpe -0.48
#                              lẽ ra +0.48 nếu được flip thay vì vứt thẳng hypothesis).
#   sharpe >= +ngưỡng      -> thử DECAY KHÁC quanh best-so-far (bằng chứng: analyst revision
#                              sim 1 phát ra 0.64 rồi vứt — đáng thử thêm 1-2 lần có kỷ luật).
#   |sharpe| < ngưỡng      -> chưa đủ tín hiệu để biết nên flip hay đổi decay -> KHÔNG sweep.
ALT_SWEEP_MIN_ABS_SHARPE: float = 0.5


def calibrated_floor(target_brain_sharpe: float = PRE_SIM_TARGET_BRAIN_SHARPE) -> float:
    """Floor local sharpe suy từ mục tiêu Brain: local >= target/1.28 thì Brain kỳ vọng >=
    target. Thay ngưỡng cứng 0.5 (Pha 4) — chỉnh mục tiêu Brain, floor tự suy theo hiệu chỉnh."""
    return target_brain_sharpe / CALIBRATION_LOCAL_TO_BRAIN
