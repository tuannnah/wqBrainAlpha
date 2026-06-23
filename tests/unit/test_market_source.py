import numpy as np

from src.data.market_panel import MarketData
from src.data.market_source import MarketDataSource


class _Fake:
    def load(self, start: str, end: str, universe: str = "TOP3000") -> MarketData:
        d = np.array(["2020-01-01"], dtype="datetime64[D]")
        a = np.array(["AAA"], dtype=np.str_)
        z = np.zeros((1, 1), dtype=np.float64)
        return MarketData(dates=d, assets=a, fields={"close": z},
                          universe=np.ones((1, 1), dtype=np.bool_), returns=z, groups={})

    def available_fields(self) -> list[str]:
        return ["close"]


def test_fake_satisfies_protocol():
    src: MarketDataSource = _Fake()
    md = src.load("2020-01-01", "2020-01-02")
    assert md.field("close").shape == (1, 1)
    assert src.available_fields() == ["close"]
