import numpy as np

from src.local_types import Assets, Dates, Mask, Panel


def test_aliases_usable_as_annotations():
    p: Panel = np.zeros((2, 3), dtype=np.float64)
    m: Mask = np.ones((2, 3), dtype=np.bool_)
    d: Dates = np.array(["2020-01-01"], dtype="datetime64[D]")
    a: Assets = np.array(["AAPL"], dtype=np.str_)
    assert p.shape == (2, 3) and m.dtype == np.bool_ and d.dtype.kind == "M" and a.dtype.kind == "U"
