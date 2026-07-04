"""Test init.py: random_tree full/grow đúng depth, ramped_half_and_half đa dạng độ sâu,
init_population ưu tiên seed + lấp đầy random, tất cả <= max_depth, deterministic theo rng
inject."""

from __future__ import annotations

import numpy as np

from src.gp.individual import Individual
from src.gp.init import init_population, ramped_half_and_half, random_tree
from src.lang.ast import Call, Field
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor

_FIELDS = ("close", "volume", "returns")


def test_random_tree_full_reaches_exact_depth():
    rng = np.random.default_rng(0)
    registry = default_registry()
    tree = random_tree(registry, rng, depth=3, fields=_FIELDS, full=True)
    assert DepthVisitor().visit(tree) == 3


def test_random_tree_grow_does_not_exceed_depth():
    rng = np.random.default_rng(1)
    registry = default_registry()
    tree = random_tree(registry, rng, depth=4, fields=_FIELDS, full=False)
    assert DepthVisitor().visit(tree) <= 4


def test_random_tree_depth_one_is_a_leaf():
    rng = np.random.default_rng(2)
    registry = default_registry()
    tree = random_tree(registry, rng, depth=1, fields=_FIELDS, full=True)
    assert DepthVisitor().visit(tree) == 1


def test_random_tree_is_deterministic_for_same_seed():
    registry = default_registry()
    tree_a = random_tree(registry, np.random.default_rng(42), depth=3, fields=_FIELDS, full=False)
    tree_b = random_tree(registry, np.random.default_rng(42), depth=3, fields=_FIELDS, full=False)
    from src.lang.visitors import Serializer
    assert Serializer().visit(tree_a) == Serializer().visit(tree_b)


def test_ramped_half_and_half_spans_multiple_depths():
    rng = np.random.default_rng(3)
    registry = default_registry()
    trees = ramped_half_and_half(registry, rng, n=20, min_depth=2, max_depth=5, fields=_FIELDS)
    depths = {DepthVisitor().visit(t) for t in trees}
    assert len(trees) == 20
    assert min(depths) >= 2
    assert max(depths) <= 5
    assert len(depths) > 1  # đa dạng độ sâu, không dồn một mức


def test_init_population_uses_all_seeds_when_fewer_than_population_size():
    seed = Call(op="rank", args=(Field("close"),))
    rng = np.random.default_rng(4)
    registry = default_registry()
    pop = init_population(
        registry, rng, population_size=10, seed_cores=[seed], fields=_FIELDS, max_depth=5,
    )
    assert len(pop) == 10
    assert all(isinstance(ind, Individual) for ind in pop)
    assert any(ind.expr == seed for ind in pop)  # seed nguyên bản có mặt


def test_init_population_caps_seeds_when_more_than_population_size():
    seeds = [Call(op="rank", args=(Field(f),)) for f in _FIELDS]
    rng = np.random.default_rng(5)
    registry = default_registry()
    pop = init_population(
        registry, rng, population_size=2, seed_cores=seeds, fields=_FIELDS, max_depth=5,
    )
    assert len(pop) == 2


def test_init_population_all_individuals_within_max_depth():
    rng = np.random.default_rng(6)
    registry = default_registry()
    pop = init_population(
        registry, rng, population_size=15, seed_cores=[], fields=_FIELDS, max_depth=4,
    )
    assert all(ind.depth() <= 4 for ind in pop)


def test_rotating_slice_offset_0_giu_nguyen_lat_cat_thuong():
    from src.gp.init import _rotating_slice
    items = [1, 2, 3, 4, 5]
    assert _rotating_slice(items, offset=0, count=3) == [1, 2, 3]


def test_rotating_slice_offset_giua_danh_sach_khong_tran():
    from src.gp.init import _rotating_slice
    items = [1, 2, 3, 4, 5]
    assert _rotating_slice(items, offset=1, count=3) == [2, 3, 4]


def test_rotating_slice_offset_gay_wrap_around():
    from src.gp.init import _rotating_slice
    items = [1, 2, 3, 4, 5]
    assert _rotating_slice(items, offset=4, count=3) == [5, 1, 2]


def test_rotating_slice_offset_boi_so_do_dai_quay_lai_offset_0():
    from src.gp.init import _rotating_slice
    items = [1, 2, 3, 4, 5]
    assert _rotating_slice(items, offset=10, count=3) == [1, 2, 3]


def test_rotating_slice_danh_sach_rong_tra_rong():
    from src.gp.init import _rotating_slice
    assert _rotating_slice([], offset=5, count=3) == []


def test_init_population_seed_offset_mac_dinh_0_giu_nguyen_hanh_vi_cu():
    seeds = [Call(op="rank", args=(Field(f),)) for f in _FIELDS]
    rng = np.random.default_rng(5)
    registry = default_registry()
    pop = init_population(
        registry, rng, population_size=2, seed_cores=seeds, fields=_FIELDS, max_depth=5,
    )
    assert [ind.expr for ind in pop] == seeds[:2]


def test_init_population_seed_offset_xoay_sang_lo_ke_tiep():
    seeds = [Call(op="rank", args=(Field(f),)) for f in _FIELDS]  # 3 seed (close/volume/returns)
    rng = np.random.default_rng(5)
    registry = default_registry()
    pop = init_population(
        registry, rng, population_size=2, seed_cores=seeds, fields=_FIELDS, max_depth=5,
        seed_offset=2,
    )
    # offset=2, count=2, len=3 -> wrap: seeds[2:3] + seeds[0:1]
    assert [ind.expr for ind in pop] == [seeds[2], seeds[0]]
