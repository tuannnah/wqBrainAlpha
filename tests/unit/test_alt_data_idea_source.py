"""AltDataIdeaSource: yield các core alt-data (đi thẳng Brain) ở batch ĐẦU rồi ủy quyền
fallback (giống CuratedIdeaSource nhưng cho core alt-data)."""

from __future__ import annotations

import src.operators_local  # noqa: F401
from src.app.closed_loop_adapters import AltDataIdeaSource
from src.generation.alt_data_seeds import ALT_DATA_CORES
from src.lang.registry import default_registry
from src.simulation.config import SimConfig
from src.simulation.simulator import SimulationResult


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


# --- Task 6: batch core alt-data sim CẢ NHÓM 1 lần qua simulate_many (thay vì mỗi core đợi
# refiner tự sim tuần tự sau này) — kết quả nạp vào presim_cache dùng chung với refiner. --------


class _FakeMultiSimulator:
    def __init__(self, results=None):
        self.calls: list[list[tuple[str, dict]]] = []
        self._results = results

    def simulate_many(self, jobs):
        self.calls.append(list(jobs))
        if self._results is not None:
            return self._results
        return [
            SimulationResult(expression=e, alpha_id=f"wq-{i}", status="passed", sharpe=1.0)
            for i, (e, _s) in enumerate(jobs)
        ]


def test_batch_multi_sim_khi_du_cau_hinh():
    """simulator+sim_config+presim_cache đủ, ≥2 core -> simulate_many gọi ĐÚNG 1 lần với đủ
    job cho toàn bộ core, kết quả nạp vào presim_cache theo đúng expr."""
    sim = _FakeMultiSimulator()
    cache: dict[str, object] = {}
    src = AltDataIdeaSource(
        fallback=_FakeFallback(), cores=ALT_DATA_CORES[:3],
        simulator=sim, sim_config=SimConfig.default(), presim_cache=cache,
    )
    batch = src.next_batch()
    assert len(sim.calls) == 1
    assert len(sim.calls[0]) == 3
    assert set(cache.keys()) == set(ALT_DATA_CORES[:3])
    assert all(r.status == "passed" for r in cache.values())
    assert [c.expr for c in batch] == list(ALT_DATA_CORES[:3])


def test_khong_batch_khi_thieu_simulator():
    """Mặc định (không truyền simulator/sim_config/presim_cache) -> KHÔNG gọi simulate_many gì
    cả — tương thích ngược hoàn toàn với hành vi trước Task 6."""
    src = AltDataIdeaSource(fallback=_FakeFallback(), cores=ALT_DATA_CORES[:3])
    batch = src.next_batch()
    assert [c.expr for c in batch] == list(ALT_DATA_CORES[:3])  # vẫn yield bình thường


def test_khong_batch_khi_chi_1_core():
    """Chỉ còn 1 core (sau lọc) -> KHÔNG gọi simulate_many (mảng multi-sim cần ≥2 phần tử) —
    để _sim_direct tự sim đường đơn như cũ, khỏi vòng round-trip thừa."""
    sim = _FakeMultiSimulator()
    cache: dict[str, object] = {}
    src = AltDataIdeaSource(
        fallback=_FakeFallback(), cores=ALT_DATA_CORES[:1],
        simulator=sim, sim_config=SimConfig.default(), presim_cache=cache,
    )
    src.next_batch()
    assert sim.calls == []
    assert cache == {}


def test_multi_sim_loi_khong_lam_hong_batch():
    """simulate_many raise lỗi bất kỳ -> next_batch() vẫn trả batch bình thường (KHÔNG crash
    phiên), presim_cache rỗng -> _sim_direct sẽ tự sim tuần tự (fallback đúng yêu cầu)."""
    class _BoomSimulator:
        def simulate_many(self, jobs):
            raise RuntimeError("multi-sim hỏng")

    cache: dict[str, object] = {}
    src = AltDataIdeaSource(
        fallback=_FakeFallback(), cores=ALT_DATA_CORES[:3],
        simulator=_BoomSimulator(), sim_config=SimConfig.default(), presim_cache=cache,
    )
    batch = src.next_batch()
    assert [c.expr for c in batch] == list(ALT_DATA_CORES[:3])
    assert cache == {}


def test_avoided_hashes_loc_truoc_khi_batch_multi_sim():
    """Core đã Brain-sim & lưu tried_hashes phiên trước (avoided_hashes) -> lọc TRƯỚC khi vào
    batch multi-sim (giống CuratedIdeaSource) — tránh gửi core trùng vào payload, vừa lãng phí
    slot mảng (tối đa 10) vừa tốn quota vô ích."""
    reg = default_registry()

    def dedup_key_fn(expr):
        return expr  # đơn giản hoá cho test: khoá = chuỗi thô

    cores = ALT_DATA_CORES[:3]
    sim = _FakeMultiSimulator()
    cache: dict[str, object] = {}
    src = AltDataIdeaSource(
        fallback=_FakeFallback(), cores=cores, registry=reg,
        avoided_hashes={cores[0]}, dedup_key_fn=dedup_key_fn,
        simulator=sim, sim_config=SimConfig.default(), presim_cache=cache,
    )
    batch = src.next_batch()
    served = [c.expr for c in batch]
    assert cores[0] not in served
    assert len(sim.calls) == 1
    assert len(sim.calls[0]) == 2  # đúng 2 core còn lại được gửi multi-sim, không lẫn core cũ
    assert cores[0] not in cache
