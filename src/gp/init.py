"""Ramped half-and-half + seeding (B13): khởi tạo quần thể GP ban đầu. Function set CHỈ
từ ``registry.gp_function_set()`` (stage separation -- loại config wrapper). Mọi randomness
qua ``rng`` inject (Determinism, Global Constraints) -- không tự gọi ``np.random.default_rng()``
nội bộ.
"""

from __future__ import annotations

import logging

import numpy as np

from config.thresholds import MAX_NODES
from src.gp.individual import Individual
from src.lang.ast import Call, Constant, Field, Node
from src.lang.registry import ArgKind, OperatorRegistry
from src.lang.visitors import ComplexityVisitor, DepthVisitor

logger = logging.getLogger(__name__)

# RC5: số lần resample tối đa khi cây sinh ra vượt ngân sách node (tránh generate-rồi-bị
# pre_filter reject "Số node > 30" -- tốn gen_ms vô ích). Hết lượt vẫn vượt -> co dần depth
# (xem `_bounded_random_tree`), KHÔNG bao giờ vòng lặp vô hạn.
_MAX_RESAMPLE = 8

_SCALAR_RANGE = (-3.0, 3.0)  # biên độ hợp lý cho threshold/hệ số trong cây seed ngẫu nhiên
# Tập hằng số RỜI RẠC (IMPROVEMENT_SPEC §3 Pha 1.3): thay float uniform ngẫu nhiên bằng bộ
# hệ số/threshold có ý nghĩa kinh tế. Tránh winsorize(open, -1.9423623924877862) vô nghĩa vừa
# tốn backtest vừa khó dedup. Không chứa 0 (nhân 0 = triệt tín hiệu).
DISCRETE_SCALARS: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def _random_scalar(rng: np.random.Generator) -> float:
    """Một hằng số rời rạc từ DISCRETE_SCALARS (thay rng.uniform liên tục)."""
    return float(DISCRETE_SCALARS[rng.integers(0, len(DISCRETE_SCALARS))])


def random_tree(
    registry: OperatorRegistry,
    rng: np.random.Generator,
    depth: int,
    fields: tuple[str, ...],
    full: bool,
    min_depth: int = 1,
    *,
    kind: ArgKind = ArgKind.PANEL,
) -> Node:
    """Sinh 1 cây ngẫu nhiên sâu tối đa ``depth``, sâu tối thiểu ``min_depth``. full=True:
    mọi nhánh đi tới đúng depth (cây "full"). full=False ("grow"): dừng sớm ngẫu nhiên ở
    mỗi tầng, NHƯNG không dừng trước khi đạt ``min_depth`` (đảm bảo cây grow ở tầng thấp
    nhất của ramped half-and-half vẫn đạt sàn độ sâu, không co thành leaf đơn).

    ``kind`` là vai trò mà cây sinh ra sẽ đóng trong cây cha: mặc định ``ArgKind.PANEL``
    (cây phải là tín hiệu — leaf chỉ được là Field, KHÔNG Constant)."""
    if depth <= 1:
        return _random_leaf(rng, fields, kind=kind)

    must_expand = min_depth > 1  # còn phải mở rộng để chạm sàn min_depth
    stop_early = (not full) and (not must_expand) and rng.random() < (1.0 / depth)
    if stop_early:
        return _random_leaf(rng, fields, kind=kind)

    ops = registry.gp_function_set()
    if not ops:
        return _random_leaf(rng, fields, kind=kind)
    spec = ops[rng.integers(0, len(ops))]

    args: list[Node] = []
    for arg_kind in spec.signature:
        match arg_kind:
            case ArgKind.PANEL:
                args.append(random_tree(
                    registry, rng, depth - 1, fields, full, min_depth - 1, kind=ArgKind.PANEL,
                ))
            case ArgKind.WINDOW:
                choice = spec.window_choices[rng.integers(0, len(spec.window_choices))]
                args.append(Constant(float(choice)))
            case ArgKind.SCALAR:
                args.append(Constant(_random_scalar(rng)))
            case ArgKind.GROUP:
                raise NotImplementedError(
                    f"operator core '{spec.name}' dùng ArgKind.GROUP nhưng init.py chưa "
                    "hỗ trợ sinh GROUP cho function set tự do (chỉ config wrapper như "
                    "group_neutralize mới dùng GROUP và có gp_usable=False)"
                )
    return Call(op=spec.name, args=tuple(args))


def _bounded_random_tree(
    registry: OperatorRegistry,
    rng: np.random.Generator,
    depth: int,
    fields: tuple[str, ...],
    full: bool,
    min_depth: int = 1,
    *,
    kind: ArgKind = ArgKind.PANEL,
    max_nodes: int = MAX_NODES,
) -> Node:
    """``random_tree`` nhưng RÀNG BUỘC THÊM số node (RC5): cây vượt ``max_nodes`` sau khi
    sinh sẽ bị resample (tối đa ``_MAX_RESAMPLE`` lần) thay vì được trả về để rồi bị
    ``PreFilter`` reject ("Số node > 30") -- generate-then-reject lãng phí gen_ms.

    Hết lượt resample vẫn vượt ngân sách -> co dần ``depth`` (giữ ``min_depth`` không vượt
    quá depth mới) cho tới khi vừa ngân sách; ``depth=1`` LUÔN là leaf đơn (1 node) nên vòng
    lặp co depth CHẮC CHẮN kết thúc (không bao giờ vô hạn) và luôn trả một cây hợp lệ."""
    best: Node | None = None
    best_size: int | None = None
    for _ in range(_MAX_RESAMPLE):
        tree = random_tree(registry, rng, depth, fields, full, min_depth, kind=kind)
        size = ComplexityVisitor().visit(tree)
        if size <= max_nodes:
            return tree
        if best_size is None or size < best_size:
            best, best_size = tree, size

    shrink_depth = depth - 1
    while shrink_depth >= 1:
        tree = random_tree(
            registry, rng, shrink_depth, fields, full, min(min_depth, shrink_depth), kind=kind,
        )
        size = ComplexityVisitor().visit(tree)
        if size <= max_nodes:
            return tree
        if best_size is None or size < best_size:
            best, best_size = tree, size
        shrink_depth -= 1
    assert best is not None  # vòng for ở trên luôn chạy >=1 lần nên best luôn được gán
    return best


def _random_leaf(
    rng: np.random.Generator, fields: tuple[str, ...], *, kind: ArgKind = ArgKind.PANEL,
) -> Node:
    """Leaf cho cây ngẫu nhiên. Ở slot PANEL CHỈ trả ``Field`` (tín hiệu thật — Constant là
    literal số, không phải PANEL signal). Ở slot SCALAR mới được trả ``Constant`` float."""
    if kind is ArgKind.SCALAR:
        return Constant(_random_scalar(rng))
    return Field(fields[rng.integers(0, len(fields))])


def ramped_half_and_half(
    registry: OperatorRegistry,
    rng: np.random.Generator,
    n: int,
    min_depth: int,
    max_depth: int,
    fields: tuple[str, ...],
    max_nodes: int = MAX_NODES,
) -> list[Node]:
    """Chia n cây đều cho mỗi độ sâu trong [min_depth, max_depth], nửa full nửa grow mỗi
    độ sâu (Koza). Phần dư dồn vào độ sâu lớn nhất.

    (RC5) Mỗi cây sinh ra qua ``_bounded_random_tree`` -- ràng buộc số node <= ``max_nodes``
    NGAY khi sinh (reject-and-resample nội bộ), thay vì để ``PreFilter`` reject sau này."""
    depths = list(range(min_depth, max_depth + 1))
    per_depth = n // len(depths)
    remainder = n - per_depth * len(depths)

    trees: list[Node] = []
    for i, depth in enumerate(depths):
        count = per_depth + (remainder if i == len(depths) - 1 else 0)
        half = count // 2
        for j in range(count):
            full = j < half
            trees.append(_bounded_random_tree(
                registry, rng, depth, fields, full, min_depth=min_depth, max_nodes=max_nodes,
            ))
    return trees


def _rotating_slice(items: list[Node], offset: int, count: int) -> list[Node]:
    """Lát cắt xoay vòng bắt đầu tại offset % len(items), nối vòng lại đầu danh sách nếu
    tràn cuối. Dùng để mỗi batch GP (xem GPIdeaSource) dùng một lô seed khác nhau thay vì
    luôn cố định items[:count] — qua nhiều batch, toàn bộ seed lần lượt được dùng."""
    n = len(items)
    if n == 0:
        return []
    start = offset % n
    end = start + count
    if end <= n:
        return items[start:end]
    return items[start:] + items[: end - n]


def init_population(
    registry: OperatorRegistry,
    rng: np.random.Generator,
    population_size: int,
    seed_cores: list[Node],
    fields: tuple[str, ...],
    max_depth: int,
    seed_offset: int = 0,
) -> list[Individual]:
    """Quần thể ban đầu: ưu tiên seed kinh nghiệm, lấp đầy phần còn lại bằng ramped
    half-and-half. Seed/cây vượt max_depth bị loại + log warning (không crash).

    ``seed_offset`` chọn lô seed nào được dùng khi số seed hợp lệ nhiều hơn
    ``population_size`` (xoay vòng qua ``_rotating_slice`` thay vì luôn cố định
    seed_cores[:population_size]) — caller (GPEngine/GPIdeaSource) tăng dần offset mỗi
    batch để qua nhiều lần gọi, toàn bộ seed hợp lệ đều được dùng."""
    valid_seeds = [t for t in seed_cores if DepthVisitor().visit(t) <= max_depth]
    dropped = len(seed_cores) - len(valid_seeds)
    if dropped:
        logger.warning("init_population: bỏ qua %d seed vượt max_depth=%d", dropped, max_depth)

    if len(valid_seeds) >= population_size:
        chosen = _rotating_slice(valid_seeds, seed_offset, population_size)
        return [Individual(expr=t) for t in chosen]

    remaining = population_size - len(valid_seeds)
    filler = ramped_half_and_half(registry, rng, remaining, min_depth=2, max_depth=max_depth, fields=fields)
    return [Individual(expr=t) for t in valid_seeds + filler]
