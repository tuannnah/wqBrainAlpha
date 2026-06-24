"""Tầng backtest local MiniBrain: config/portfolio/backtester (+ metrics/pool_corr ở
Phase 4/6). Dependency rule (master plan B1): src/backtest KHÔNG import src/gp,
src/storage, src/llm. `gate.py` (Task 3.5) là điểm duy nhất src/llm được phép gọi vào.
"""
