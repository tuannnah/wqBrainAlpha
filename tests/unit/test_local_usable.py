from __future__ import annotations

import numpy as np

from src.backtest.gate import local_usable
from src.data.market_panel import MarketData
from src.gp.seeds import all_seed_cores
from src.lang.visitors import FieldCollector, Serializer
from src.lang.registry import default_registry


def _panel() -> MarketData:
    dates = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    assets = np.array(["A", "B"])
    shape = (3, 2)
    fields = {n: np.ones(shape) for n in ("close", "open", "high", "low", "volume", "vwap")}
    return MarketData(
        dates=dates, assets=assets, fields=fields,
        universe=np.ones(shape, dtype=bool), returns=np.zeros(shape),
        groups={"sector": np.zeros(shape, dtype=int)},
    )


def test_local_usable_chi_field_panel_thi_true():
    assert local_usable("rank(ts_mean(close, 5))", _panel()) is True
    # group_neutralize dùng 'sector' ở vị trí GROUP -> KHÔNG bị coi là field thiếu
    assert local_usable("group_neutralize(rank(close), sector)", _panel()) is True


def test_local_usable_field_altdata_thi_false():
    assert local_usable("rank(ts_mean(anl4_afv4_eps_mean, 5))", _panel()) is False


def test_local_usable_parse_loi_thi_false():
    assert local_usable("rank(", _panel()) is False


def test_all_seed_cores_loc_bo_core_altdata():
    panel = _panel()
    field_names = set(panel.field_names())
    cores = all_seed_cores(field_names=field_names)
    reg = default_registry()
    for node in cores:
        used = FieldCollector(reg).visit(node)
        assert used.issubset(field_names), f"core lọt field ngoài panel: {Serializer().visit(node)}"
    # không lọc (field_names=None) -> có core alt-data (nhiều hơn)
    assert len(all_seed_cores(field_names=None)) > len(cores)
