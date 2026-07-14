"""Wire kho frontier vào build_closed_loop: cores frontier phải nằm trong direct_cores
(đường sim-thẳng AltDataIdeaSource) và tắt được qua cờ include_frontier."""

from __future__ import annotations

import inspect

from src.app.closed_loop_adapters import (
    AltDataIdeaSource,
    _gather_direct_cores,
    build_closed_loop,
)
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


def test_altdata_idea_source_phuc_vu_core_frontier_qua_wiring_that() -> None:
    """F3: dựng AltDataIdeaSource TRỰC TIẾP (không simulator/sim_config -> _presim_batch
    no-op, nhẹ) với `cores=_gather_direct_cores(True, True, True, True)` — đúng lời gọi
    build_closed_loop dùng thật (dòng 1185-1187) — rồi khẳng định batch đầu tiên (đi thẳng
    Brain, KHÔNG qua fallback) chứa core frontier thật, không chỉ core alt-data/fundamental/
    hypothesis cũ."""

    class _EmptyFallback:
        def next_batch(self):  # không được gọi tới nếu wiring đúng (direct_cores khác rỗng)
            raise AssertionError("fallback không được gọi khi direct_cores có core")

    cores = _gather_direct_cores(True, True, True, True)
    source = AltDataIdeaSource(fallback=_EmptyFallback(), cores=cores)
    batch = source.next_batch()
    served = {c.expr for c in batch}
    frontier_served = served & set(FRONTIER_CORES)
    assert frontier_served, "batch đầu AltDataIdeaSource không phục vụ core frontier nào"
    # Vài core frontier tiêu biểu (insider + call_filing, xem finding F1) phải nằm trong batch —
    # không chỉ 1-2 core lẻ tẻ sống sót qua field-validity guard.
    assert "ts_backfill(directional_indicator_score, 66)" in served
    assert any("count_positive_profitability_answer" in e for e in served)
