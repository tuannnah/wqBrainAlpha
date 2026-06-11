"""Chuẩn hóa metrics từ simulation result về dict số thực."""

from __future__ import annotations

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
