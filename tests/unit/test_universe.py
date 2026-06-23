import numpy as np

from src.data.universe import build_universe_mask, sector_codes


def test_mask_true_only_for_finite_positive():
    vol = np.array([[10.0, 0.0], [np.nan, 5.0]], dtype=np.float64)
    mask = build_universe_mask(vol)
    assert mask.tolist() == [[True, False], [False, True]]


def test_mask_changes_per_day_no_survivorship():
    vol = np.array([[1.0, 1.0], [1.0, np.nan]], dtype=np.float64)
    mask = build_universe_mask(vol)
    # mã thứ 2 rời universe ngày 2 -> mask per-day khác nhau
    assert mask[0].tolist() == [True, True]
    assert mask[1].tolist() == [True, False]


def test_sector_codes_dense_ints():
    raw = np.array([["10", "20"], ["10", "30"]], dtype=object)
    codes = sector_codes(raw)
    assert codes.shape == (2, 2)
    assert codes.dtype.kind in ("i", "u")
    # mã cùng sector "10" -> cùng code
    assert codes[0, 0] == codes[1, 0]
