"""Quét không gian cấu hình cho một alpha tốt (T5.3), OOS làm trọng tài (T5.6).

Một alpha = (biểu thức + cấu hình). Cố định biểu thức, quét grid trên các chiều
cấu hình (decay, truncation, neutralization...). Chỉ giữ cấu hình tốt CẢ IS lẫn
OS — chọn cấu hình có IS sharpe cao nhất trong số đã qua kiểm chứng OOS.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from loguru import logger

from src.simulation.config import SimConfig
from src.simulation.oos import oos_passes


@dataclass
class SweepResult:
    best_config: SimConfig | None
    best_result: object | None
    trials: list = field(default_factory=list)  # [{config, sharpe, os_sharpe, oos_ok}]


class ConfigSweeper:
    def __init__(self, simulator):
        self.simulator = simulator

    def sweep(
        self,
        expression: str,
        base_config: SimConfig,
        grid: dict[str, list],
        oos_min_ratio: float = 0.5,
    ) -> SweepResult:
        """Quét tích Descartes các chiều trong `grid`, mỗi tổ hợp ghi đè lên
        `base_config`. Chọn cấu hình IS sharpe cao nhất trong số qua OOS."""
        dims = list(grid.keys())
        combos = list(itertools.product(*(grid[d] for d in dims))) if dims else [()]

        best_config = None
        best_result = None
        best_sharpe = float("-inf")
        trials: list = []

        for combo in combos:
            overrides = dict(zip(dims, combo))
            config = base_config.with_overrides(**overrides)
            result = self.simulator.simulate(expression, settings=config.to_settings())

            oos_ok = result.status != "error" and oos_passes(result, min_ratio=oos_min_ratio)
            trials.append({
                "config": config,
                "sharpe": result.sharpe,
                "os_sharpe": result.os_sharpe,
                "oos_ok": oos_ok,
            })
            if oos_ok and result.sharpe is not None and result.sharpe > best_sharpe:
                best_sharpe = result.sharpe
                best_config, best_result = config, result

        logger.info(
            "Sweep {} tổ hợp; chọn {}",
            len(combos), best_config.key() if best_config else "không có (không qua OOS)",
        )
        return SweepResult(best_config, best_result, trials)
