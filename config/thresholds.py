"""Mọi ngưỡng gate/submission của MiniBrain ở MỘT nơi (Gap #7/R9 master spec).

Không hardcode các số này ở call site — đổi theo cuộc thi thì sửa tại đây.
"""

from __future__ import annotations

import math

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

# --- Ngưỡng NỘP THẬT (submit, khác hẳn "sim status=passed" — Bug 2 fix-submit-async) ---
# `failed_checks == []` lúc SIM KHÔNG đủ để coi alpha "sẵn sàng nộp": bằng chứng thật
# 2026-07-14, alpha KP9nwpEg (Sharpe 1.41, fitness 0.99, self-corr 0.4265, failed_checks=[]
# lúc sim) vẫn bị Brain trả 403 REJECTED lúc `POST /alphas/{id}/submit` với body
# `is.checks = [{"name":"LOW_SHARPE","result":"FAIL","limit":1.58,"value":1.41},
# {"name":"LOW_FITNESS","result":"FAIL","limit":1.0,"value":0.99}, ...]` — hai số dưới đây
# đo TRỰC TIẾP từ `limit` trong response 403 đó, khớp `IS_LADDER_FAIL` (docs IS-Ladder) cho
# Sharpe nhưng là gate NỘP riêng (đơn giản, không theo cửa sổ năm như IS-Ladder).
SUBMIT_MIN_SHARPE: float = 1.58
SUBMIT_MIN_FITNESS: float = 1.0

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

# Số mẫu tối thiểu để một số liệu ρ/hệ số hiệu chỉnh ĐÁNG TIN (T4.2/T4.3, WS4 calibration).
# Dùng ở HAI chỗ: (1) `calibrated_floor(family=...)` — họ có n < ngưỡng này -> fallback hệ số
# chung CALIBRATION_LOCAL_TO_BRAIN thay vì hệ số riêng họ (mẫu quá nhỏ để tin ước lượng
# riêng); (2) cảnh báo report calibrate (T4.3) — n < ngưỡng này -> in "ρ chưa đáng tin". 30 là
# đề xuất (chưa có cơ sở thống kê chặt — số nhỏ nhất để Spearman không dao động quá lớn do
# nhiễu mẫu vài chục điểm); CHỈNH SỬA CẦN SỐ LIỆU THẬT, không phải đoán.
CALIBRATION_MIN_SAMPLE_N: int = 30


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
# Trần độ sâu MỘT tín hiệu con được phép làm component combo (Fix 3, giá trị gốc cho N=4):
# MAX_DEPTH(7) trừ 3 tầng wrapper build_combined_expression luôn thêm (rank chuẩn hoá + 2
# tầng add cân bằng cho N=4). Component sâu hơn mức này chắc chắn vượt trần sau khi bọc ->
# loại NGAY trước greedy thay vì phát hiện muộn (đo được 3/5 rồi 2/5 combo chết vì depth ở
# diag 20260712/20260713).
# CẬP NHẬT (Task 1, T1.2 — src/generation/combiner.py:component_depth_cap): `combine_stage`
# KHÔNG còn dùng hằng số CỐ ĐỊNH này để lọc pool trước greedy nữa — mỗi lần thử n_max
# (4 -> 3 -> 2) tự suy trần ĐỘNG qua `component_depth_cap(attempt_n_max)` (công thức
# MAX_DEPTH - 1 - ceil(log2(n_max)), cho đúng N=4 ra lại giá trị 4 dưới đây). Hằng số này VẪN
# giữ lại làm default của `select_decorrelated_combos(max_component_depth=...)` khi caller
# không truyền cap tường minh (tương thích ngược, standalone/unit test) — không phải nơi
# "loại NGAY trước greedy" của combine_stage thật nữa.
COMBINER_MAX_COMPONENT_DEPTH: int = 4

# --- GP core depth budget (WS2 Task 2 — GP đừng đẻ core sâu/overfit, xem
# .superpowers/sdd/20260719/task-2-brief.md) ---
# Số cá thể (xếp theo sharpe_deflated giảm dần) GPEngine xét khi chọn "best-cho-combiner"
# (T2.1, ``src/gp/engine.py:_select_best_combinable``): trong top-K này, ưu tiên cá thể
# ĐẦU TIÊN có depth <= COMBINER_MAX_COMPONENT_DEPTH (combinable) thay vì luôn lấy cá thể
# sharpe cao nhất tuyệt đối (`best_by_sharpe`, vẫn giữ nguyên cho báo cáo — thường là cây
# sâu nhất/overfit nhất, xem bối cảnh brief). K nhỏ (không quét cả quần thể — mất ý nghĩa
# "best") nhưng đủ lớn để có cơ hội gặp cá thể nông trong nhóm sharpe cao của quần thể
# ~50 cá thể mặc định (`GPEngine.pop_size`).
GP_BEST_COMBINABLE_TOP_K: int = 10

# Trần độ sâu CORE lúc GP SINH/BIẾN DỊ (T2.2): MAX_DEPTH(7) trừ 3 tầng wrapper
# `scale(ts_decay(group_neut(...)))` LUÔN cộng thêm ở tầng cấu hình alpha cuối (stage
# separation B5, ngoài phạm vi tìm kiếm GP) — khớp trần combiner mặc định
# COMBINER_MAX_COMPONENT_DEPTH ở trên để core GP sinh ra LUÔN có cơ hội được combiner ghép,
# thay vì chỉ bị `GateEvaluator` (đã có từ trước, kiểm depth<=MAX_DEPTH=7 SAU KHI backtest —
# xem `src/backtest/gates.py`) phát hiện muộn, tốn cả 1 lượt backtest cho cây chắc chắn quá
# sâu để combiner dùng được. Là TRẦN MẶC ĐỊNH mới của `GPEngine.max_depth`/
# `src.gp.variation.crossover`/`subtree_mutation` (trước Task 2 mặc định = MAX_DEPTH = 7,
# đúng bằng trần gate bare-core nên GP tự do sinh cây sâu tới tận biên gate, không còn dư
# địa cho wrapper — nguồn gốc chính khiến combiner ra ~0 combo, xem bối cảnh task-2-brief.md).
# Vẫn có thể override tường minh (vd test cũ dùng max_depth=7) khi cố ý cần cây sâu hơn.
#
# RANH GIỚI QUAN TRỌNG (Fix review T2.2, Important của reviewer): hằng số này CHỈ áp cho
# cây SINH ngẫu nhiên (filler `ramped_half_and_half`/`random_tree`) và BIẾN DỊ (`crossover`/
# `subtree_mutation`) — KHÔNG áp cho SEED nạp vào quần thể khởi tạo (`init_population`'s
# `valid_seeds` filter, xem `src/gp/init.py`). Seed (frontier/alt-data, tri thức người viết
# đã qua kiểm định kinh tế, phổ biến depth 5-6) dùng ngân sách RIÊNG — tham số
# `seed_max_depth` của `init_population` (mặc định `MAX_DEPTH`=7, ngân sách RỘNG như trước
# Task 2). Lý do: seed không phải cây sinh ngẫu nhiên cần ép nông để tránh overfit — chọn
# lọc "seed sâu nào đáng giữ tới cuối" là việc của NSGA-II (T2.3, depth đã vào parsimony
# penalty) + `_select_best_combinable` (T2.1), KHÔNG phải việc của bộ lọc lúc khởi tạo. Bug
# đã fix: trước đây `init_population` dùng CHUNG `max_depth` cho cả seed lẫn filler, nên khi
# GPEngine đổi default sang GP_MAX_CORE_DEPTH=4 (T2.2), ~38% seed thủ công (đa số seed
# frontier/alt-data depth 5-6 — nguồn đa dạng chủ lực) bị lọc rớt OAN ngay từ khởi tạo, đồng
# thời làm best_combinable ≡ best_by_sharpe khi mọi cá thể đều ≤4 (mâu thuẫn T2.1).
GP_MAX_CORE_DEPTH: int = MAX_DEPTH - 3

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


# --- WS3 T3.1/T3.2 — sàn quota đa dạng + xoay seed theo bão hoà pool (phá family lock-in,
# xem .superpowers/sdd/20260719/task-3-brief.md) ---
# Panel local chỉ có 6 field PV -> GP hội tụ về pv_reversal bão hoà; seed frontier/alt-data/
# fundamental/hypothesis (đi thẳng `_sim_direct`) trước đây bị dồn hết vào 1 batch đầu rồi
# cạn, mọi batch SAU đó toàn PV suốt phần còn lại phiên (PROGRESS Session 16: "GP local không
# dùng được field alt-data -> hội tụ pv_reversal bão hoà"). Sàn 30%: đủ để mỗi batch còn cửa
# sổ orthogonal (giảm self-corr hàng loạt với pv_reversal), không quá cao để chặn batch khi
# nguồn non-PV cạn tự nhiên (T3.1: "không chặn batch" khi hàng đợi non-PV không đủ).
FRONTIER_MIN_FRACTION: float = 0.3
# T3.2: family chiếm > mức này trong pool ĐÃ PASS (đo qua classify_family) -> seed cùng family
# bị đẩy xuống CUỐI hàng đợi reserve (không xoá) ở batch kế — tránh tiếp tục đào family đã
# bão hoà. 0.5: quá nửa pool cùng 1 family là dấu hiệu rõ đã khai thác đủ, chưa tới mức cực
# đoan để loại hẳn seed đó (vẫn còn cơ hội thử lại khi các family khác cạn trước).
FRONTIER_SATURATION_K: float = 0.5


def calibrated_floor(
    target_brain_sharpe: float = PRE_SIM_TARGET_BRAIN_SHARPE,
    family: str | None = None,
    family_coefficients: dict[str, tuple[float, int]] | None = None,
) -> float:
    """Floor local sharpe suy từ mục tiêu Brain: local >= target/hệ_số thì Brain kỳ vọng >=
    target. Thay ngưỡng cứng 0.5 (Pha 4) — chỉnh mục tiêu Brain, floor tự suy theo hiệu chỉnh.

    (T4.2) `family` + `family_coefficients` TÙY CHỌN: hệ số local->Brain per-family (thay hệ
    số chung 1.28) khi họ đó đã đo đủ mẫu. `family_coefficients` là dict {family: (hệ_số, n)}
    — LỰA CHỌN ít xâm lấn: caller tự trích `CalibrationHarness.run().by_family` (mỗi
    `FamilyCalibration` có `.local_to_brain_ratio`/`.n`) rồi truyền dict vào, KHÔNG đọc DB/
    import ngược `src.calibration` ở module config này (tránh vòng import ngược + tránh
    thresholds.py phụ thuộc I/O). Chỉ dùng hệ số riêng họ khi n >= CALIBRATION_MIN_SAMPLE_N VÀ
    hệ số hữu hạn (không NaN — họ có thể chưa đủ dữ liệu để tính ratio dù n >= ngưỡng, vd toàn
    bộ record thiếu brain_sharpe). Thiếu family, family không có trong bảng, hoặc không đủ điều
    kiện trên -> fallback CALIBRATION_LOCAL_TO_BRAIN chung — ĐÂY LÀ HÀNH VI CŨ, không đổi khi
    gọi không truyền 2 tham số mới (mọi call site production hiện tại không truyền chúng).
    KHÔNG wire floor per-family vào gate pre-sim production trong task này — hàm chỉ SẴN SÀNG,
    quyết định dùng khi nào là của user sau khi có đủ số liệu (xem task-4-brief.md)."""
    coefficient = CALIBRATION_LOCAL_TO_BRAIN
    if family is not None and family_coefficients is not None:
        entry = family_coefficients.get(family)
        if entry is not None:
            family_coefficient, n = entry
            if n >= CALIBRATION_MIN_SAMPLE_N and math.isfinite(family_coefficient):
                coefficient = family_coefficient
    return target_brain_sharpe / coefficient
