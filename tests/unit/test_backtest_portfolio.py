"""Test PortfolioBuilder.build: decay -> neutralize -> truncate -> scale -> delay."""

from __future__ import annotations

import numpy as np
import pytest

from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData


def _tiny_panel() -> MarketData:
    """4 ngày x 4 mã, universe đủ, 2 sector {0,0,1,1} — dễ tính tay."""
    t, n = 4, 4
    dates = (np.datetime64("2021-01-01") + np.arange(t)).astype("datetime64[D]")
    assets = np.array(["A", "B", "C", "D"], dtype=np.str_)
    universe = np.ones((t, n), dtype=bool)
    returns = np.full((t, n), 0.01)
    sector = np.tile(np.array([0, 0, 1, 1]), (t, 1)).astype(np.int64)
    return MarketData(
        dates=dates, assets=assets, fields={}, universe=universe,
        returns=returns, groups={"sector": sector},
    )


def test_neutralize_none_keeps_signal_then_scale_dollar_neutral():
    data = _tiny_panel()
    signal = np.array([[1.0, -1.0, 2.0, -2.0]] * 4)
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, truncation=1.0, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data)
    # scale: w /= sum(|w|) -> [1,-1,2,-2]/6 ; sum(|w|) per day == scale_book(1.0)
    row0 = w[0]
    assert np.isclose(np.nansum(np.abs(row0)), 1.0)
    assert np.isclose(row0[0] / row0[2], 0.5)  # tỉ lệ tương đối giữ nguyên


def test_neutralize_market_demeans_cross_sectionally():
    data = _tiny_panel()
    signal = np.array([[1.0, 2.0, 3.0, 4.0]] * 4)
    cfg = PortfolioConfig(neutralization=Neutralization.MARKET, truncation=1.0, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data)
    # sau demean mean=0 -> cross-section sum trước scale = 0 -> vẫn 0 sau scale (chia hằng số)
    pre_scale_proxy = w[0] * np.nansum(np.abs(w[0]))  # phục hồi tỉ lệ tương đối
    assert np.isclose(np.nansum(pre_scale_proxy), 0.0, atol=1e-9)


def test_neutralize_sector_demeans_within_group():
    data = _tiny_panel()
    signal = np.array([[1.0, 3.0, 10.0, 20.0]] * 4)  # sector0={1,3} sector1={10,20}
    cfg = PortfolioConfig(neutralization=Neutralization.SECTOR, truncation=1.0, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data)
    ratio = w[0]
    # sau demean trong sector: sector0 -> [-1,1]; sector1 -> [-5,5] -> tỉ lệ A:B = -1:1
    assert np.isclose(ratio[0] / ratio[1], -1.0)
    assert np.isclose(ratio[2] / ratio[3], -1.0)


def test_truncate_caps_per_name_weight_and_renormalizes():
    # Panel 4 mã -> cap PHẢI >= 1/4=0.25 để khả thi cùng lúc với tổng |w|=1.0
    # (cap 0.10 cũ bất khả thi: 4*0.10=0.40<1.0 — đã sửa, xem _truncate water-filling).
    # cap 0.40 với signal cực đoan [100,-1,1,-100] hội tụ về [0.40,-0.10,0.10,-0.40].
    data = _tiny_panel()
    signal = np.array([[100.0, -1.0, 1.0, -100.0]] * 4)
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, truncation=0.40, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data)
    assert np.all(np.abs(w[0]) <= 0.40 + 1e-6)  # mọi vị thế bị cap đúng tỉ lệ sau scale
    assert np.isclose(np.nansum(np.abs(w[0])), 1.0)  # scale giữ tổng gross = scale_book
    # 2 mã lớn cap ở 0.40, 2 mã nhỏ chia phần còn lại theo tỉ lệ -> ±0.10
    np.testing.assert_allclose(np.sort(np.abs(w[0])), [0.10, 0.10, 0.40, 0.40], atol=1e-6)


def test_scale_book_sets_total_gross_exposure():
    data = _tiny_panel()
    signal = np.array([[1.0, -1.0, 2.0, -2.0]] * 4)
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, truncation=1.0,
                          scale_book=2.0, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data)
    assert np.isclose(np.nansum(np.abs(w[0])), 2.0)


def test_delay_shifts_weights_down_by_delay_rows():
    data = _tiny_panel()
    signal = np.array([[1.0, -1.0, 2.0, -2.0]] * 4)
    cfg_no_delay = PortfolioConfig(neutralization=Neutralization.NONE, truncation=1.0, delay=0)
    cfg_delay1 = PortfolioConfig(neutralization=Neutralization.NONE, truncation=1.0, delay=1)
    w0 = PortfolioBuilder().build(signal, cfg_no_delay, data)
    w1 = PortfolioBuilder().build(signal, cfg_delay1, data)
    assert np.all(np.isnan(w1[0]))  # ngày đầu chưa có weight để delay vào
    np.testing.assert_allclose(w1[1], w0[0])
    np.testing.assert_allclose(w1[2], w0[1])


def test_out_of_universe_cells_stay_nan():
    data = _tiny_panel()
    universe = data.universe.copy()
    universe[:, 3] = False  # mã D ngoài universe toàn bộ
    data2 = MarketData(dates=data.dates, assets=data.assets, fields=data.fields,
                        universe=universe, returns=data.returns, groups=data.groups)
    signal = np.array([[1.0, -1.0, 2.0, -2.0]] * 4)
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, truncation=1.0, delay=0)
    w = PortfolioBuilder().build(signal, cfg, data2)
    assert np.all(np.isnan(w[:, 3]))


def test_unknown_group_key_raises_keyerror():
    data = _tiny_panel()  # chỉ có groups["sector"], không có "industry"
    signal = np.ones((4, 4))
    cfg = PortfolioConfig(neutralization=Neutralization.INDUSTRY, delay=0)
    with pytest.raises(KeyError):
        PortfolioBuilder().build(signal, cfg, data)
