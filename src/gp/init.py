"""Ramped half-and-half + seeding (B13): khởi tạo quần thể GP ban đầu. Function set CHỈ
từ ``registry.gp_function_set()`` (stage separation -- loại config wrapper). Mọi randomness
qua ``rng`` inject (Determinism, Global Constraints) -- không tự gọi ``np.random.default_rng()``
nội bộ.
"""

from __future__ import annotations

import logging

import numpy as np

from src.gp.individual import Individual
from src.lang.ast import Call, Constant, Field, Node
from src.lang.registry import ArgKind, OperatorRegistry
from src.lang.visitors import DepthVisitor

logger = logging.getLogger(__name__)

_SCALAR_RANGE = (-3.0, 3.0)  # biên độ hợp lý cho threshold/hệ số trong cây seed ngẫu nhiên


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
                args.append(Constant(float(rng.uniform(*_SCALAR_RANGE))))
            case ArgKind.GROUP:
                raise NotImplementedError(
                    f"operator core '{spec.name}' dùng ArgKind.GROUP nhưng init.py chưa "
                    "hỗ trợ sinh GROUP cho function set tự do (chỉ config wrapper như "
                    "group_neutralize mới dùng GROUP và có gp_usable=False)"
                )
    return Call(op=spec.name, args=tuple(args))


def _random_leaf(
    rng: np.random.Generator, fields: tuple[str, ...], *, kind: ArgKind = ArgKind.PANEL,
) -> Node:
    """Leaf cho cây ngẫu nhiên. Ở slot PANEL CHỈ trả ``Field`` (tín hiệu thật — Constant là
    literal số, không phải PANEL signal). Ở slot SCALAR mới được trả ``Constant`` float."""
    if kind is ArgKind.SCALAR:
        return Constant(float(rng.uniform(*_SCALAR_RANGE)))
    return Field(fields[rng.integers(0, len(fields))])


def ramped_half_and_half(
    registry: OperatorRegistry,
    rng: np.random.Generator,
    n: int,
    min_depth: int,
    max_depth: int,
    fields: tuple[str, ...],
) -> list[Node]:
    """Chia n cây đều cho mỗi độ sâu trong [min_depth, max_depth], nửa full nửa grow mỗi
    độ sâu (Koza). Phần dư dồn vào độ sâu lớn nhất."""
    depths = list(range(min_depth, max_depth + 1))
    per_depth = n // len(depths)
    remainder = n - per_depth * len(depths)

    trees: list[Node] = []
    for i, depth in enumerate(depths):
        count = per_depth + (remainder if i == len(depths) - 1 else 0)
        half = count // 2
        for j in range(count):
            full = j < half
            trees.append(random_tree(registry, rng, depth, fields, full, min_depth=min_depth))
    return trees


def init_population(
    registry: OperatorRegistry,
    rng: np.random.Generator,
    population_size: int,
    seed_cores: list[Node],
    fields: tuple[str, ...],
    max_depth: int,
) -> list[Individual]:
    """Quần thể ban đầu: ưu tiên seed kinh nghiệm, lấp đầy phần còn lại bằng ramped
    half-and-half. Seed/cây vượt max_depth bị loại + log warning (không crash)."""
    valid_seeds = [t for t in seed_cores if DepthVisitor().visit(t) <= max_depth]
    dropped = len(seed_cores) - len(valid_seeds)
    if dropped:
        logger.warning("init_population: bỏ qua %d seed vượt max_depth=%d", dropped, max_depth)

    if len(valid_seeds) >= population_size:
        chosen = valid_seeds[:population_size]
        return [Individual(expr=t) for t in chosen]

    remaining = population_size - len(valid_seeds)
    filler = ramped_half_and_half(registry, rng, remaining, min_depth=2, max_depth=max_depth, fields=fields)
    return [Individual(expr=t) for t in valid_seeds + filler]
