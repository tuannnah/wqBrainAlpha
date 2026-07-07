"""Proxy robustness sub-universe (local): kiểm alpha còn giữ Sharpe khi giới hạn về nhóm mã
thanh khoản nhất — bắt chước sub-universe test của Brain (sub_sharpe ≥ 0.75·√(sub/univ)·sharpe).
Local không có nhiều universe; xấp xỉ bằng top `frac` mã theo thanh khoản (mean(volume*close))."""

from __future__ import annotations

import dataclasses

import numpy as np

from src.backtest.config import PortfolioConfig
from src.lang.ast import Node
from src.lang.registry import OperatorRegistry
from src.local_types import Panel


def _sub_universe_mask(data, frac: float) -> Panel:
    """Mask (T,N) bool: giữ top `frac` mã theo thanh khoản trung bình (volume*close)."""
    close = data.field("close")
    volume = data.field("volume")
    with np.errstate(invalid="ignore"):
        liq = np.nanmean(volume * close, axis=0)  # (N,)
    n = liq.shape[0]
    keep = max(1, int(round(n * frac)))
    order = np.argsort(-np.nan_to_num(liq, nan=-np.inf))
    top = set(order[:keep].tolist())
    col = np.array([i in top for i in range(n)], dtype=bool)
    return data.universe & col[None, :]


def sub_universe_ok(
    node: Node, config: PortfolioConfig, data, registry: OperatorRegistry, *,
    full_sharpe: float, frac: float = 0.5,
) -> bool:
    """True nếu Sharpe trên sub-universe đạt ngưỡng 0.75·√frac·full_sharpe. full_sharpe ≤ 0 -> True."""
    if full_sharpe is None or full_sharpe <= 0:
        return True
    from src.backtest.backtester import Backtester
    from src.backtest.metrics_local import MetricsCalculator
    from src.backtest.portfolio import PortfolioBuilder
    from src.engine.evaluator import EvalContext, Evaluator

    try:
        sub_data = dataclasses.replace(data, universe=_sub_universe_mask(data, frac))
        signal = Evaluator(EvalContext(data=sub_data, registry=registry, cache=None)).evaluate(node)
        if np.all(np.isnan(signal)):
            return False
        weights = PortfolioBuilder().build(signal, config, sub_data)
        result = Backtester().run(weights, sub_data)
        if not np.isfinite(result.daily_pnl).any():
            return False
        sub_sharpe = MetricsCalculator().compute(result, sub_data).sharpe
    except (KeyError, ValueError, ZeroDivisionError):
        return False
    if sub_sharpe is None or not np.isfinite(sub_sharpe):
        return False
    return sub_sharpe >= 0.75 * (frac ** 0.5) * full_sharpe
