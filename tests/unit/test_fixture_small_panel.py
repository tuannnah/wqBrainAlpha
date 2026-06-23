import numpy as np

from src.data.market_panel import MarketData


def test_small_panel_shape_and_determinism(small_panel: MarketData):
    assert isinstance(small_panel, MarketData)
    assert small_panel.field("close").shape == (120, 30)
    assert small_panel.universe.dtype == np.bool_
    # ngày 0 của returns là NaN (không look-ahead)
    assert np.isnan(small_panel.returns[0]).all()


def test_small_panel_has_out_of_universe_nan(small_panel: MarketData):
    # có ít nhất 1 cell ngoài universe để các phase sau test NaN-propagation
    assert (~small_panel.universe).any()
