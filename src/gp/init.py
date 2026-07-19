"""Ramped half-and-half + seeding (B13): khởi tạo quần thể GP ban đầu. Function set CHỈ
từ ``registry.gp_function_set()`` (stage separation -- loại config wrapper). Mọi randomness
qua ``rng`` inject (Determinism, Global Constraints) -- không tự gọi ``np.random.default_rng()``
nội bộ.
"""

from __future__ import annotations

import logging

import numpy as np

from config.thresholds import MAX_DEPTH, MAX_NODES
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
    field_groups: "tuple[tuple[str, ...], ...] | None" = None,
) -> Node:
    """Sinh 1 cây ngẫu nhiên sâu tối đa ``depth``, sâu tối thiểu ``min_depth``. full=True:
    mọi nhánh đi tới đúng depth (cây "full"). full=False ("grow"): dừng sớm ngẫu nhiên ở
    mỗi tầng, NHƯNG không dừng trước khi đạt ``min_depth`` (đảm bảo cây grow ở tầng thấp
    nhất của ramped half-and-half vẫn đạt sàn độ sâu, không co thành leaf đơn).

    ``kind`` là vai trò mà cây sinh ra sẽ đóng trong cây cha: mặc định ``ArgKind.PANEL``
    (cây phải là tín hiệu — leaf chỉ được là Field, KHÔNG Constant).

    ``field_groups`` (B2, keyword-only, mặc định None): nhóm field theo dataset để leaf
    chọn field two-stage (nhóm uniform trước, field trong nhóm uniform sau) — xem
    ``_random_leaf``. None = hành vi cũ (uniform phẳng trên ``fields``)."""
    if depth <= 1:
        return _random_leaf(rng, fields, kind=kind, field_groups=field_groups)

    must_expand = min_depth > 1  # còn phải mở rộng để chạm sàn min_depth
    stop_early = (not full) and (not must_expand) and rng.random() < (1.0 / depth)
    if stop_early:
        return _random_leaf(rng, fields, kind=kind, field_groups=field_groups)

    ops = registry.gp_function_set()
    if not ops:
        return _random_leaf(rng, fields, kind=kind, field_groups=field_groups)
    spec = ops[rng.integers(0, len(ops))]

    args: list[Node] = []
    for arg_kind in spec.signature:
        match arg_kind:
            case ArgKind.PANEL:
                args.append(random_tree(
                    registry, rng, depth - 1, fields, full, min_depth - 1, kind=ArgKind.PANEL,
                    field_groups=field_groups,
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
    field_groups: "tuple[tuple[str, ...], ...] | None" = None,
) -> Node:
    """``random_tree`` nhưng RÀNG BUỘC THÊM số node (RC5): cây vượt ``max_nodes`` sau khi
    sinh sẽ bị resample (tối đa ``_MAX_RESAMPLE`` lần) thay vì được trả về để rồi bị
    ``PreFilter`` reject ("Số node > 30") -- generate-then-reject lãng phí gen_ms.

    Hết lượt resample vẫn vượt ngân sách -> co dần ``depth`` (giữ ``min_depth`` không vượt
    quá depth mới) cho tới khi vừa ngân sách; ``depth=1`` LUÔN là leaf đơn (1 node) nên vòng
    lặp co depth CHẮC CHẮN kết thúc (không bao giờ vô hạn) và luôn trả một cây hợp lệ.

    ``field_groups`` (B2): xem ``random_tree``/``_random_leaf`` — chỉ ảnh hưởng cách chọn
    field ở leaf, không ảnh hưởng logic ràng buộc số node."""
    best: Node | None = None
    best_size: int | None = None
    for _ in range(_MAX_RESAMPLE):
        tree = random_tree(
            registry, rng, depth, fields, full, min_depth, kind=kind, field_groups=field_groups,
        )
        size = ComplexityVisitor().visit(tree)
        if size <= max_nodes:
            return tree
        if best_size is None or size < best_size:
            best, best_size = tree, size

    shrink_depth = depth - 1
    while shrink_depth >= 1:
        tree = random_tree(
            registry, rng, shrink_depth, fields, full, min(min_depth, shrink_depth), kind=kind,
            field_groups=field_groups,
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
    field_groups: "tuple[tuple[str, ...], ...] | None" = None,
) -> Node:
    """Leaf cho cây ngẫu nhiên. Ở slot PANEL CHỈ trả ``Field`` (tín hiệu thật — Constant là
    literal số, không phải PANEL signal). Ở slot SCALAR mới được trả ``Constant`` float.

    ``field_groups`` (B2, mặc định None): nếu có, chọn field theo HAI TẦNG — chọn NHÓM
    dataset uniform trước, rồi field uniform TRONG nhóm — để dataset ít field (vd một
    alt-data hiếm) không bị dataset đông field (vd price/volume hàng trăm field) áp đảo
    xác suất xuất hiện trong quần thể khởi tạo. None (mặc định) giữ nguyên hành vi cũ:
    uniform phẳng trên toàn bộ ``fields``."""
    if kind is ArgKind.SCALAR:
        return Constant(_random_scalar(rng))
    if field_groups:
        nhom = field_groups[rng.integers(0, len(field_groups))]
        return Field(nhom[rng.integers(0, len(nhom))])
    return Field(fields[rng.integers(0, len(fields))])


def ramped_half_and_half(
    registry: OperatorRegistry,
    rng: np.random.Generator,
    n: int,
    min_depth: int,
    max_depth: int,
    fields: tuple[str, ...],
    max_nodes: int = MAX_NODES,
    *,
    field_groups: "tuple[tuple[str, ...], ...] | None" = None,
) -> list[Node]:
    """Chia n cây đều cho mỗi độ sâu trong [min_depth, max_depth], nửa full nửa grow mỗi
    độ sâu (Koza). Phần dư dồn vào độ sâu lớn nhất.

    (RC5) Mỗi cây sinh ra qua ``_bounded_random_tree`` -- ràng buộc số node <= ``max_nodes``
    NGAY khi sinh (reject-and-resample nội bộ), thay vì để ``PreFilter`` reject sau này.

    ``field_groups`` (B2, keyword-only, mặc định None): xem ``_random_leaf`` — two-stage
    sampling khi có, None giữ nguyên hành vi cũ."""
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
                field_groups=field_groups,
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
    *,
    seed_max_depth: int = MAX_DEPTH,
    field_groups: "tuple[tuple[str, ...], ...] | None" = None,
) -> list[Individual]:
    """Quần thể ban đầu: ưu tiên seed kinh nghiệm, lấp đầy phần còn lại bằng ramped
    half-and-half. Seed/cây vượt ngân sách bị loại + log warning (không crash).

    (Fix review T2.2, WS2 task-2-brief.md) HAI TRẦN ĐỘ SÂU TÁCH BIỆT, KHÔNG dùng chung
    ``max_depth``:

    - ``max_depth``: trần cây SINH ngẫu nhiên (filler qua ``ramped_half_and_half`` bên
      dưới khi seed hợp lệ không đủ lấp ``population_size``) — mặc định caller thật
      (``GPEngine``) truyền ``GP_MAX_CORE_DEPTH`` (4) để filler luôn combinable.
    - ``seed_max_depth`` (keyword-only, mặc định ``MAX_DEPTH`` = 7): trần lọc
      ``valid_seeds`` — ngân sách RỘNG như TRƯỚC Task 2, vì seed là tri thức người viết
      (seed thủ công/frontier/alt-data), ĐÃ qua kiểm định kinh tế, không phải cây sinh
      ngẫu nhiên cần ép nông để tránh overfit. Seed sâu (5-7) vẫn được nạp vào quần thể;
      việc chọn lọc "seed nào đáng giữ tới cuối" giao cho NSGA-II (T2.3, depth đã vào
      parsimony) + ``_select_best_combinable`` (T2.1) ở tầng engine, KHÔNG chặn cứng
      ngay từ lúc khởi tạo. Trước fix này, ``valid_seeds`` dùng chung ``max_depth`` với
      filler nên khi ``GPEngine`` đổi default sang ``GP_MAX_CORE_DEPTH=4`` (T2.2), ~38%
      seed thủ công (đa số seed frontier/alt-data depth 5-6) bị lọc rớt OAN ngay từ vòng
      khởi tạo — ngoài phạm vi T2.2 gốc (chỉ scope SINH/BIẾN DỊ, không phải seed nạp vào).

    ``seed_offset`` chọn lô seed nào được dùng khi số seed hợp lệ nhiều hơn
    ``population_size`` (xoay vòng qua ``_rotating_slice`` thay vì luôn cố định
    seed_cores[:population_size]) — caller (GPEngine/GPIdeaSource) tăng dần offset mỗi
    batch để qua nhiều lần gọi, toàn bộ seed hợp lệ đều được dùng.

    ``field_groups`` (B2, keyword-only, mặc định None): truyền xuống ``ramped_half_and_half``
    cho phần filler ngẫu nhiên — two-stage sampling khi có, None giữ nguyên hành vi cũ."""
    valid_seeds = [t for t in seed_cores if DepthVisitor().visit(t) <= seed_max_depth]
    dropped = len(seed_cores) - len(valid_seeds)
    if dropped:
        logger.warning(
            "init_population: bỏ qua %d seed vượt seed_max_depth=%d", dropped, seed_max_depth,
        )

    if len(valid_seeds) >= population_size:
        chosen = _rotating_slice(valid_seeds, seed_offset, population_size)
        return [Individual(expr=t) for t in chosen]

    remaining = population_size - len(valid_seeds)
    filler = ramped_half_and_half(
        registry, rng, remaining, min_depth=2, max_depth=max_depth, fields=fields,
        field_groups=field_groups,
    )
    return [Individual(expr=t) for t in valid_seeds + filler]
