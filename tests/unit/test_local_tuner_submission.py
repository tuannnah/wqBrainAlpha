"""Tuner xếp hạng theo ĐIỂM NỘP (min(Sharpe/1.25, Fitness/1.0)) — không phải Sharpe trần.
Bằng chứng live: core đạt Sharpe 1.45 nhưng FAIL nộp vì fitness 0.80; tuner cũ đuổi Sharpe
nên chọn config turnover cao (fitness thấp). Tuner mới ưu tiên config qua CẢ hai cổng."""

from __future__ import annotations

import src.operators_local  # noqa: F401
from src.backtest import local_tuner
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.local_tuner import tune
from src.backtest.metrics_local import AlphaMetrics


def _m(sharpe, fitness, turnover=0.3):
    return AlphaMetrics(
        sharpe=sharpe, annual_return=0.2, turnover=turnover, max_drawdown=0.1,
        fitness=fitness, per_year_sharpe={2020: sharpe}, weight_concentration=0.05,
    )


def test_tuner_chon_config_qua_ca_hai_cong_thay_vi_sharpe_tran(monkeypatch):
    # decay=2: Sharpe 3.0 nhưng fitness 0.6 (fail fitness) -> điểm nộp min(2.4,0.6)=0.6.
    # decay=6: Sharpe 1.4, fitness 1.2 -> điểm nộp min(1.12,1.2)=1.12 (qua CẢ hai cổng) -> thắng.
    def fake_metrics(node, config, data, registry):
        if config.decay == 2:
            return _m(3.0, 0.6)
        if config.decay == 6:
            return _m(1.4, 1.2)
        return _m(1.3, 1.05)   # base decay=4

    monkeypatch.setattr(local_tuner, "local_metrics", fake_metrics)
    base = PortfolioConfig(neutralization=Neutralization.MARKET, decay=4, truncation=0.08)
    res = tune("rank(close)", base, data=object(), budget=80)
    assert res.best_config.decay == 6            # KHÔNG phải decay=2 (Sharpe 3.0 nhưng fail fitness)
    assert res.local_metrics.fitness >= 1.0      # winner qua cổng fitness
    assert res.local_sharpe == 1.4               # báo cáo Sharpe THẬT của winner (không phải điểm nộp)


def test_tuner_thuong_nhe_turnover_thap_khi_diem_nop_hoa(monkeypatch):
    # A (decay=6): Sharpe 1.4/fit 1.1/turnover 0.25 (<0.30) -> điểm nộp gốc min(1.12,1.1)=1.10.
    # B (decay=2): Sharpe 1.4/fit 1.1/turnover 0.50 (>=0.30) -> điểm nộp gốc min(1.12,1.1)=1.10.
    # Hai điểm nộp GỐC hoà nhau; B gặp trước (decay=2 trước decay=6 trong _DECAYS) nên nếu
    # KHÔNG có thưởng turnover thấp, tuner giữ B (so sánh `>` nghiêm ngặt, hoà thì không cập nhật).
    # Có thưởng (turnover<0.30 nhân 1.10) -> điểm A = 1.21 > điểm B = 1.10 -> tuner chọn A.
    def fake_metrics(node, config, data, registry):
        if config.decay == 6:
            return _m(1.4, 1.1, turnover=0.25)
        if config.decay == 2:
            return _m(1.4, 1.1, turnover=0.50)
        return _m(1.0, 1.0, turnover=0.50)   # base decay=4 -> điểm thấp, không ảnh hưởng

    monkeypatch.setattr(local_tuner, "local_metrics", fake_metrics)
    base = PortfolioConfig(neutralization=Neutralization.MARKET, decay=4, truncation=0.08)
    res = tune("rank(close)", base, data=object(), budget=80)
    assert res.best_config.decay == 6             # A (turnover thấp) thắng nhờ thưởng
    assert res.local_metrics.turnover == 0.25
