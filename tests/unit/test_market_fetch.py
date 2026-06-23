import numpy as np

from src.data.market_fetch import _assemble_panel


def test_assemble_panel_builds_aligned_marketdata():
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    assets = np.array(["AAA", "BBB"], dtype=np.str_)
    raw = {
        "close": (dates, assets, np.array([[10.0, 20.0], [11.0, np.nan]])),
        "volume": (dates, assets, np.array([[100.0, 50.0], [0.0, 5.0]])),
    }
    sector_raw = np.array([["10", "20"], ["10", "20"]], dtype=object)
    md = _assemble_panel(raw, sector_raw, tradable_field="volume")
    assert md.field("close").shape == (2, 2)
    # universe = volume hữu hạn & >0
    assert md.universe.tolist() == [[True, True], [False, True]]
    # returns ngày 0 = NaN (không look-ahead)
    assert np.isnan(md.returns[0]).all()
    assert "sector" in md.groups
