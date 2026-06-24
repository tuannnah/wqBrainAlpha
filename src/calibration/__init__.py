"""Tầng validate: đo độ tin cậy ranking local so với Brain (B10 master spec).

Khác `src/lang/operators_local/engine/backtest` (tầng lõi, KHÔNG được biết gì về storage),
`src/calibration` ĐƯỢC PHÉP import `src.storage` — nhiệm vụ duy nhất của nó là đọc lịch sử
simulation thật từ DB và so sánh với re-score local. KHÔNG import `src.gp`, `src.llm`.
"""
