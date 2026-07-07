"""Test gate turnover trong `tune`: Brain đòi turnover 1%-70%; config vượt trần 0.70 phải
bị loại (điểm −inf) dù Sharpe local cao hơn, để winner luôn nằm trong dải nộp được."""

from __future__ import annotations

import src.operators_local  # noqa: F401
from src.backtest import local_tuner
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.local_tuner import tune
from src.backtest.metrics_local import AlphaMetrics


def _metrics(sharpe, turnover):
    return AlphaMetrics(
        sharpe=sharpe, annual_return=0.2, turnover=turnover, max_drawdown=0.1,
        fitness=1.2, per_year_sharpe={2020: sharpe}, weight_concentration=0.05,
    )


def test_tune_loai_config_turnover_qua_cao(monkeypatch):
    # decay=2 -> TO 0.9 (quá cao) Sharpe 3.0; các config khác -> TO 0.3 Sharpe 1.5.
    # Dù Sharpe 3.0 cao hơn, config TO>0.70 phải bị loại -> winner có TO<=0.70.
    def fake_metrics(node, config, data, registry):
        if config.decay == 2:
            return _metrics(3.0, 0.9)
        return _metrics(1.5, 0.3)

    monkeypatch.setattr(local_tuner, "local_metrics", fake_metrics)
    base = PortfolioConfig(neutralization=Neutralization.MARKET, decay=4, truncation=0.08)
    res = tune("rank(close)", base, data=object(), budget=80)
    assert res.local_sharpe == 1.5          # không phải 3.0 (config TO cao bị loại)
    assert res.best_config.decay != 2
    assert res.local_metrics.turnover <= 0.70
