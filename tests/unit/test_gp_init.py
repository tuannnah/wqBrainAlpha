"""Test init.py: random_tree full/grow đúng depth, ramped_half_and_half đa dạng độ sâu,
init_population ưu tiên seed + lấp đầy random, tất cả <= max_depth, deterministic theo rng
inject."""

from __future__ import annotations

import numpy as np

from src.gp.individual import Individual
from src.gp.init import init_population, ramped_half_and_half, random_tree
from src.lang.ast import Call, Field
from src.lang.registry import ArgKind, OpCategory, OperatorRegistry, OperatorSpec, default_registry
from src.lang.visitors import ComplexityVisitor, DepthVisitor

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


# --- RC5: ràng buộc node-count NGAY khi sinh (tránh generate-then-reject ở pre_filter) ---


def _binary_only_registry() -> OperatorRegistry:
    """Registry "suy biến" CHỈ có 1 operator nhị phân (PANEL, PANEL) — mọi cây full=True
    không phải leaf sẽ luôn nổ theo cấp số nhân (2^depth - 1 node), không có lối thoát về
    cây nhỏ hơn ở CÙNG depth. Dùng để kiểm resample bị chặn không lặp vô hạn."""
    registry = OperatorRegistry()
    registry.register(OperatorSpec(
        name="_only_binary", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=lambda *_: None, bounded=False,
    ))
    return registry


def test_ramped_half_and_half_never_exceeds_max_nodes():
    """(a) Với max_nodes nhỏ, MỌI cây trả về phải nằm trong ngân sách — kể cả sau khi hết
    lượt resample (fallback phải co dần depth chứ không được trả cây vượt ngân sách)."""
    rng = np.random.default_rng(11)
    registry = default_registry()
    small_max_nodes = 8
    trees = ramped_half_and_half(
        registry, rng, n=30, min_depth=2, max_depth=7, fields=_FIELDS,
        max_nodes=small_max_nodes,
    )
    assert len(trees) == 30
    assert all(ComplexityVisitor().visit(t) <= small_max_nodes for t in trees)


def test_ramped_half_and_half_default_budget_matches_pre_filter():
    """Không truyền max_nodes -> dùng mặc định MAX_NODES=30 (khớp PreFilter.max_nodes)."""
    from config.thresholds import MAX_NODES

    rng = np.random.default_rng(12)
    registry = default_registry()
    trees = ramped_half_and_half(
        registry, rng, n=20, min_depth=2, max_depth=7, fields=_FIELDS,
    )
    assert all(ComplexityVisitor().visit(t) <= MAX_NODES for t in trees)


def test_bounded_resample_terminates_on_degenerate_registry_that_only_makes_big_trees():
    """(b) Registry suy biến chỉ có operator nhị phân -> cây full=True depth cao LUÔN vượt
    ngân sách rất nhỏ (max_nodes=1). Vẫn phải kết thúc (không vòng lặp vô hạn) và trả về 1
    cây hợp lệ (leaf, 1 node) do fallback co depth."""
    from src.gp.init import _bounded_random_tree

    registry = _binary_only_registry()
    rng = np.random.default_rng(13)
    tree = _bounded_random_tree(
        registry, rng, depth=6, fields=_FIELDS, full=True, min_depth=1, max_nodes=1,
    )
    assert ComplexityVisitor().visit(tree) <= 1


def test_bounded_resample_returns_smallest_seen_when_shrink_impossible():
    """Ngân sách hợp lý (không cực đoan như =1) vẫn phải tôn trọng ngay cả với registry chỉ
    sinh cây nhị phân bùng nổ ở depth cao."""
    from src.gp.init import _bounded_random_tree

    registry = _binary_only_registry()
    rng = np.random.default_rng(14)
    tree = _bounded_random_tree(
        registry, rng, depth=6, fields=_FIELDS, full=True, min_depth=1, max_nodes=15,
    )
    assert ComplexityVisitor().visit(tree) <= 15


# --- B2: cân bằng dataset two-stage khi sinh leaf ngẫu nhiên ---


def test_random_leaf_two_stage_can_bang_nhom():
    """1000 leaf với nhóm (1 field pv) vs (1 field alt): tỉ lệ mỗi nhóm ~50% (±10 điểm %),
    dù nhóm pv có 99 field và alt chỉ 1 field thì mỗi NHÓM vẫn 50%."""
    from src.gp.init import _random_leaf

    rng = np.random.default_rng(7)
    pv = tuple(f"pv_{i}" for i in range(99))
    groups = (pv, ("alt_duy_nhat",))
    fields = pv + ("alt_duy_nhat",)
    dem_alt = sum(
        1 for _ in range(1000)
        if _random_leaf(rng, fields, field_groups=groups).name == "alt_duy_nhat"
    )
    assert 400 <= dem_alt <= 600  # uniform phẳng chỉ cho ~10/1000


def test_field_groups_none_giu_hanh_vi_cu():
    """field_groups mặc định None -> hành vi cũ (uniform phẳng) nguyên vẹn từng bit, số lần
    gọi rng giữ nguyên thứ tự (cùng seed cho ra cùng chuỗi field)."""
    from src.gp.init import _random_leaf

    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
    f = ("a", "b", "c")
    cu = [_random_leaf(rng1, f).name for _ in range(50)]
    moi = [_random_leaf(rng2, f, field_groups=None).name for _ in range(50)]
    assert cu == moi


# --- T2.2: khóa hành vi với GP_MAX_CORE_DEPTH (trần core GP mới, xem config/thresholds.py) ---
#
# XÁC MINH (không phải RED->GREEN mới): random_tree/_bounded_random_tree/ramped_half_and_half
# CẤU TRÚC ĐÃ đảm bảo depth <= tham số truyền vào (đệ quy giảm dần đúng 1 mỗi tầng, không
# bao giờ vượt) từ trước Task 2 -- không có lỗi nào ở init.py cần sửa cho T2.2. Trần bị lỏng
# TRƯỚC Task 2 nằm ở GIÁ TRỊ mặc định truyền VÀO các hàm này (GPEngine.max_depth mặc định = 7
# = MAX_DEPTH, xem T2.2 ở engine.py/variation.py), KHÔNG phải ở init.py. Test dưới đây khóa
# (regression-lock) hành vi ĐÚNG đã có, buộc tường minh vào hằng số GP_MAX_CORE_DEPTH thay vì
# số 4 rời rạc — nếu ai đó nới GP_MAX_CORE_DEPTH mà quên nối dây, test này vẫn phản ánh đúng
# giá trị mới (không phải hardcode).
def test_ramped_half_and_half_ton_trong_gp_max_core_depth_khi_duoc_truyen():
    from config.thresholds import GP_MAX_CORE_DEPTH

    rng = np.random.default_rng(15)
    registry = default_registry()
    trees = ramped_half_and_half(
        registry, rng, n=20, min_depth=1, max_depth=GP_MAX_CORE_DEPTH, fields=_FIELDS,
    )
    assert all(DepthVisitor().visit(t) <= GP_MAX_CORE_DEPTH for t in trees)


def test_init_population_ton_trong_gp_max_core_depth_khi_duoc_truyen():
    from config.thresholds import GP_MAX_CORE_DEPTH

    rng = np.random.default_rng(16)
    registry = default_registry()
    pop = init_population(
        registry, rng, population_size=15, seed_cores=[], fields=_FIELDS,
        max_depth=GP_MAX_CORE_DEPTH,
    )
    assert all(ind.depth() <= GP_MAX_CORE_DEPTH for ind in pop)
