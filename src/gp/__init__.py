"""Tầng Genetic Programming MiniBrain (Phase 7, B13 master design): sinh, lai ghép, đột
biến, chọn lọc quần thể signal-core AST, correlation-aware (pool + population) từ ngày
đầu. Dependency rule B1: src/gp được phép import src.lang/src.engine/src.backtest/
src.storage/src.generation/src.llm (tầng "app", cao nhất trừ pipeline/Phase 8) — ngược lại
các tầng đó KHÔNG được import src.gp.
"""
