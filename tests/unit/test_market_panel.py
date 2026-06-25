import numpy as np
import pytest

from src.data.market_panel import MarketData


def _toy() -> MarketData:
    dates = np.array(["2020-01-01", "2020-01-02", "2021-01-04"], dtype="datetime64[D]")
    assets = np.array(["AAA", "BBB"], dtype=np.str_)
    close = np.array([[1.0, 2.0], [1.1, np.nan], [1.2, 2.2]], dtype=np.float64)
    universe = np.array([[True, True], [True, False], [True, True]], dtype=np.bool_)
    returns = np.array([[np.nan, np.nan], [0.1, np.nan], [0.0909, 0.0]], dtype=np.float64)
    groups = {"sector": np.array([[0, 1], [0, 1], [0, 1]])}
    return MarketData(dates=dates, assets=assets, fields={"close": close},
                      universe=universe, returns=returns, groups=groups)


def test_field_returns_panel():
    md = _toy()
    assert md.field("close").shape == (3, 2)


def test_field_unknown_raises():
    md = _toy()
    with pytest.raises(KeyError):
        md.field("nope")


def test_field_resolves_returns_as_virtual_field():
    # `returns` là field WQ hợp lệ; MarketData lưu nó ở .returns (cho backtester), không trong
    # .fields. field("returns") phải resolve về .returns để expr tham chiếu `returns` eval được.
    md = _toy()
    assert np.array_equal(md.field("returns"), md.returns, equal_nan=True)


def test_field_explicit_returns_in_fields_takes_precedence():
    # Nếu caller cố tình đưa "returns" vào fields, dùng bản đó (không ghi đè bằng .returns).
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    assets = np.array(["AAA"], dtype=np.str_)
    explicit = np.array([[9.0], [9.0]], dtype=np.float64)
    md = MarketData(dates=dates, assets=assets,
                    fields={"close": np.zeros((2, 1)), "returns": explicit},
                    universe=np.ones((2, 1), dtype=np.bool_),
                    returns=np.zeros((2, 1), dtype=np.float64), groups={})
    assert np.array_equal(md.field("returns"), explicit)


def test_years_slices_by_calendar_year():
    md = _toy()
    years = md.years()
    assert set(years) == {2020, 2021}
    assert md.dates[years[2020]].tolist() == np.array(
        ["2020-01-01", "2020-01-02"], dtype="datetime64[D]").tolist()


def test_post_init_rejects_unsorted_dates():
    dates = np.array(["2021-01-01", "2020-01-01"], dtype="datetime64[D]")
    assets = np.array(["AAA"], dtype=np.str_)
    close = np.zeros((2, 1), dtype=np.float64)
    with pytest.raises(ValueError):
        MarketData(dates=dates, assets=assets, fields={"close": close},
                   universe=np.ones((2, 1), dtype=np.bool_),
                   returns=np.zeros((2, 1), dtype=np.float64), groups={})


def test_post_init_rejects_shape_mismatch():
    dates = np.array(["2020-01-01"], dtype="datetime64[D]")
    assets = np.array(["AAA", "BBB"], dtype=np.str_)
    bad = np.zeros((2, 2), dtype=np.float64)  # T=2 nhưng dates T=1
    with pytest.raises(ValueError):
        MarketData(dates=dates, assets=assets, fields={"close": bad},
                   universe=np.ones((1, 2), dtype=np.bool_),
                   returns=np.zeros((1, 2), dtype=np.float64), groups={})


def test_post_init_rejects_non_bool_universe():
    dates = np.array(["2020-01-01"], dtype="datetime64[D]")
    assets = np.array(["AAA"], dtype=np.str_)
    close = np.zeros((1, 1), dtype=np.float64)
    with pytest.raises(ValueError):
        MarketData(dates=dates, assets=assets, fields={"close": close},
                   universe=np.ones((1, 1), dtype=np.float64),
                   returns=np.zeros((1, 1), dtype=np.float64), groups={})


def test_post_init_rejects_non_datetime_dates():
    dates = np.array([20200101], dtype=np.int64)
    assets = np.array(["AAA"], dtype=np.str_)
    close = np.zeros((1, 1), dtype=np.float64)
    with pytest.raises(ValueError):
        MarketData(dates=dates, assets=assets, fields={"close": close},
                   universe=np.ones((1, 1), dtype=np.bool_),
                   returns=np.zeros((1, 1), dtype=np.float64), groups={})
