"""CuratedIdeaSource: seed các core price/volume ĐÃ KIỂM CHỨNG (Brain Sharpe ~1.5+) ở batch
ĐẦU TIÊN — thay vì để GP random pha loãng chúng — rồi mới ủy quyền cho nguồn GP fallback.
Mục tiêu: 1 phiên auto-search chạm được alpha đạt chuẩn nộp bằng cách thử core mạnh trước."""

from __future__ import annotations

import src.operators_local  # noqa: F401  # đăng ký operator (parse ts_mean/subtract…)
from src.app.closed_loop_adapters import VERIFIED_CORES, CuratedIdeaSource
from src.lang.parser import parse


class _FakeFallback:
    def __init__(self):
        self.calls = 0

    def next_batch(self):
        self.calls += 1
        return [f"gp-{self.calls}"]


def test_batch_dau_yield_core_kiem_chung():
    fb = _FakeFallback()
    src = CuratedIdeaSource(fallback=fb)
    batch = src.next_batch()
    exprs = [c.expr for c in batch]
    assert exprs == list(VERIFIED_CORES)   # batch đầu = core đã kiểm chứng
    assert fb.calls == 0                    # chưa đụng GP


def test_batch_sau_uy_quyen_gp():
    fb = _FakeFallback()
    src = CuratedIdeaSource(fallback=fb)
    src.next_batch()                        # nuốt batch curated
    assert src.next_batch() == ["gp-1"]     # từ batch 2 -> GP fallback
    assert src.next_batch() == ["gp-2"]
    assert fb.calls == 2


def test_core_kiem_chung_parse_duoc_va_dung_field_local():
    # Mọi core phải parse được và chỉ dùng close/open/vwap (có trong panel local).
    allowed = {"close", "open", "vwap", "volume", "high", "low", "returns"}
    from src.lang.registry import default_registry
    from src.lang.visitors import FieldCollector

    reg = default_registry()
    for expr in VERIFIED_CORES:
        node = parse(expr)
        assert FieldCollector(reg).visit(node).issubset(allowed), expr
