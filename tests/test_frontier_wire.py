"""Wire kho frontier vào build_closed_loop: cores frontier phải nằm trong direct_cores
(đường sim-thẳng AltDataIdeaSource) và tắt được qua cờ include_frontier."""

from __future__ import annotations

import inspect

from src.app.closed_loop_adapters import _gather_direct_cores, build_closed_loop
from src.generation.alt_data_seeds import ALT_DATA_CORES
from src.generation.frontier_seeds import FRONTIER_CORES


def test_gather_gom_du_cac_kho_theo_co() -> None:
    all_on = _gather_direct_cores(True, True, True, True)
    assert set(FRONTIER_CORES) <= set(all_on)
    assert set(ALT_DATA_CORES) <= set(all_on)
    # Tắt frontier -> không còn core frontier, kho cũ giữ nguyên.
    no_frontier = _gather_direct_cores(True, True, True, False)
    assert not (set(FRONTIER_CORES) & set(no_frontier))
    assert set(ALT_DATA_CORES) <= set(no_frontier)


def test_build_closed_loop_co_tham_so_include_frontier_mac_dinh_bat() -> None:
    sig = inspect.signature(build_closed_loop)
    assert "include_frontier" in sig.parameters
    assert sig.parameters["include_frontier"].default is True
