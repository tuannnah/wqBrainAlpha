"""AltDataIdeaSource: yield các core alt-data (đi thẳng Brain) ở batch ĐẦU rồi ủy quyền
fallback (giống CuratedIdeaSource nhưng cho core alt-data)."""

from __future__ import annotations

import src.operators_local  # noqa: F401
from src.app.closed_loop_adapters import AltDataIdeaSource
from src.generation.alt_data_seeds import ALT_DATA_CORES


class _FakeFallback:
    def __init__(self):
        self.calls = 0

    def next_batch(self):
        self.calls += 1
        return [f"gp-{self.calls}"]


def test_batch_dau_yield_core_alt_data():
    fb = _FakeFallback()
    src = AltDataIdeaSource(fallback=fb)
    batch = src.next_batch()
    assert [c.expr for c in batch] == list(ALT_DATA_CORES)
    assert all(c.metrics is None for c in batch)  # không chấm local được → metrics=None
    assert fb.calls == 0


def test_batch_sau_uy_quyen_fallback():
    fb = _FakeFallback()
    src = AltDataIdeaSource(fallback=fb)
    src.next_batch()  # nuốt batch alt-data
    assert src.next_batch() == ["gp-1"]
    assert fb.calls == 1
