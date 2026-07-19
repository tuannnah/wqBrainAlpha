"""Chuẩn hóa metrics từ simulation result về dict số thực."""

from __future__ import annotations

from config.thresholds import SUBMIT_FITNESS_REF, SUBMIT_SHARPE_REF

METRIC_KEYS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin")

# Giá trị mặc định khi metric thiếu (an toàn cho scoring/filter).
_DEFAULTS = {
    "sharpe": 0.0,
    "fitness": 0.0,
    "turnover": 0.5,
    "returns": 0.0,
    "drawdown": 1.0,
    "margin": 0.0,
}


def normalize(source) -> dict[str, float]:
    """Nhận dict hoặc object có thuộc tính metric → dict float đầy đủ."""
    if hasattr(source, "metrics") and callable(source.metrics):
        raw = source.metrics()
    elif isinstance(source, dict):
        raw = source
    else:
        raw = {k: getattr(source, k, None) for k in METRIC_KEYS}

    result = {}
    for key in METRIC_KEYS:
        value = raw.get(key)
        result[key] = _DEFAULTS[key] if value is None else float(value)
    return result


def submit_score(sharpe: float, fitness: float) -> float:
    """Công thức điểm-nộp DÙNG CHUNG (T4.1, gộp 2 bản chép trước đó — combine_stage Task 2
    Fix 4 và closed_loop_adapters): min(sharpe/SUBMIT_SHARPE_REF, fitness/SUBMIT_FITNESS_REF)
    — đo tiến GẦN NGƯỠNG NỘP thật (Sharpe~1.58, fitness~1) trên CẢ HAI trục, không phải fitness
    thô. Đây là công thức THUẦN (sharpe/fitness là float thật, không None/NaN) — caller tự
    quyết định fallback khi thiếu dữ liệu tùy mục đích dùng: ranking "chọn tốt nhất" (vd
    closed_loop_adapters) muốn None -> -inf để không bao giờ được chọn; đo tương quan
    (calibration harness T4.1) muốn None/thiếu -> NaN để cặp đó bị loại khỏi rho thay vì lẫn
    vào một số bịa (min() với NaN phụ thuộc thứ tự tham số — xem `_submit_score_or_nan` trong
    `src/calibration/harness.py`)."""
    return min(sharpe / SUBMIT_SHARPE_REF, fitness / SUBMIT_FITNESS_REF)
