import numpy as np

from src.data.adapters.parquet_source import ParquetSource, save
from src.data.market_panel import MarketData


def _toy() -> MarketData:
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    assets = np.array(["AAA", "BBB"], dtype=np.str_)
    close = np.array([[1.0, 2.0], [1.1, np.nan]], dtype=np.float64)
    return MarketData(dates=dates, assets=assets, fields={"close": close},
                      universe=np.array([[True, True], [True, False]]),
                      returns=np.array([[np.nan, np.nan], [0.1, np.nan]]),
                      groups={"sector": np.array([[0, 1], [0, 1]])})


def test_round_trip(tmp_path):
    md = _toy()
    save(md, str(tmp_path))
    src = ParquetSource(str(tmp_path))
    out = src.load("2020-01-01", "2020-01-02")
    np.testing.assert_array_equal(out.dates, md.dates)
    np.testing.assert_array_equal(out.assets, md.assets)
    np.testing.assert_allclose(out.field("close"), md.field("close"), equal_nan=True)
    assert out.universe.tolist() == md.universe.tolist()
    assert "close" in src.available_fields()
