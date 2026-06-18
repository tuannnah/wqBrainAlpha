"""Test ma trận tổ hợp WQB và iter_scopes."""

from __future__ import annotations

from src.data.universe_matrix import WQB_MATRIX, iter_scopes


def test_iter_scopes_tat_ca_khong_rong():
    scopes = list(iter_scopes())
    assert len(scopes) > 0
    # Mỗi phần tử là bộ ba (region, universe, delay).
    region, universe, delay = scopes[0]
    assert isinstance(region, str) and isinstance(universe, str)
    assert isinstance(delay, int)


def test_iter_scopes_loc_theo_region():
    usa = list(iter_scopes(regions=["USA"]))
    assert usa, "USA phải có trong WQB_MATRIX"
    assert all(s[0] == "USA" for s in usa)
    # Bằng số universe * số delay của USA.
    cfg = WQB_MATRIX["USA"]
    assert len(usa) == len(cfg["universes"]) * len(cfg["delays"])


def test_iter_scopes_loc_theo_delay():
    d1 = list(iter_scopes(regions=["USA"], delays=[1]))
    assert d1
    assert all(s[2] == 1 for s in d1)
    assert len(d1) == len(WQB_MATRIX["USA"]["universes"])


def test_iter_scopes_region_khong_phan_biet_hoa_thuong():
    assert list(iter_scopes(regions=["usa"])) == list(iter_scopes(regions=["USA"]))
