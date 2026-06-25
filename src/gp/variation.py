"""Typed crossover + point/subtree/hoist mutation (B13) trên AST Phase 1. "Typed" = chỉ
tráo subtree đóng vai PANEL trong cây cha (không tráo nhầm vào vị trí WINDOW/SCALAR của
một Constant), chỉ đổi operator cùng signature, chỉ đổi window theo window_choices của
chính operator đó. Validity repair tối giản: hết lượt retry vẫn vượt max_depth -> giữ
nguyên cây gốc (an toàn hơn cắt cây tùy tiện). Dedup qua CanonicalHasher (Task 7.1).
"""

from __future__ import annotations

import numpy as np

from config.thresholds import MAX_DEPTH
from src.gp.individual import Individual
from src.gp.init import random_tree
from src.lang.ast import Call, Constant, Field, Node
from src.lang.registry import ArgKind, OperatorRegistry, OperatorSpec, default_registry
from src.lang.visitors import DepthVisitor, all_subtrees

_MAX_CROSSOVER_RETRIES = 10


def _panel_compatible_subtrees(root: Node, registry: OperatorRegistry) -> list[Node]:
    """Subtree mà vai trò của nó trong cây cha (nếu có) là ArgKind.PANEL. Root luôn hợp lệ
    (không phải tham số của ai)."""
    result: list[Node] = [root]

    def _walk(node: Node) -> None:
        if not isinstance(node, Call):
            return
        try:
            spec = registry.get(node.op)
        except KeyError:
            return
        for child, kind in zip(node.args, spec.signature):
            if kind is ArgKind.PANEL:
                result.append(child)
            _walk(child)

    _walk(root)
    return result


def _replace_subtree(root: Node, target: Node, replacement: Node) -> Node:
    """Trả cây mới với ``target`` (theo identity) được thay bằng ``replacement``; nếu
    root is target, trả replacement luôn."""
    if root is target:
        return replacement
    if not isinstance(root, Call):
        return root
    new_args = tuple(_replace_subtree(c, target, replacement) for c in root.args)
    return Call(op=root.op, args=new_args)


def crossover(
    a: Node, b: Node, rng: np.random.Generator, max_depth: int = MAX_DEPTH,
) -> tuple[Node, Node]:
    """Tráo 1 subtree PANEL-compatible của ``a`` với 1 của ``b`` (typed). Cả hai cây kết
    quả phải <= max_depth; hết ``_MAX_CROSSOVER_RETRIES`` lượt vẫn vượt -> trả (a, b)
    nguyên bản (validity repair tối giản: lùi về không đổi gì)."""
    registry = default_registry()
    for _ in range(_MAX_CROSSOVER_RETRIES):
        points_a = _panel_compatible_subtrees(a, registry)
        points_b = _panel_compatible_subtrees(b, registry)
        pa = points_a[rng.integers(0, len(points_a))]
        pb = points_b[rng.integers(0, len(points_b))]

        new_a = _replace_subtree(a, pa, pb)
        new_b = _replace_subtree(b, pb, pa)
        if DepthVisitor().visit(new_a) <= max_depth and DepthVisitor().visit(new_b) <= max_depth:
            return new_a, new_b
    return a, b


def _constant_arg_kind(root: Node, target: Constant, registry: OperatorRegistry) -> ArgKind | None:
    """Trả ArgKind của ``target`` xét theo vị trí trong cây cha; None nếu target là root
    (Constant đứng trần — không có cha) hoặc parent op không có trong registry."""

    def _walk(node: Node) -> ArgKind | None:
        if not isinstance(node, Call):
            return None
        try:
            spec = registry.get(node.op)
        except KeyError:
            return None
        for child, kind in zip(node.args, spec.signature):
            if child is target:
                return kind
            found = _walk(child)
            if found is not None:
                return found
        return None

    return _walk(root)


def point_mutation(
    node: Node, registry: OperatorRegistry, rng: np.random.Generator, fields: tuple[str, ...],
) -> Node:
    """Đổi tại CHỖ 1 node, type-aware theo vai trò trong cây cha:

    - Field -> field khác.
    - Constant ở vị trí WINDOW của cha -> resample từ ``parent_spec.window_choices`` (giữ
      đúng kiểu int — KHÔNG perturb Gaussian, vì float lẻ ở window là type-invalid theo WQ).
    - Constant ở vị trí SCALAR (hoặc Constant đứng trần không có cha) -> perturb Gaussian σ=0.5.
    - Constant ở vị trí khác (vd GROUP literal) -> giữ nguyên.
    - Call -> đổi op sang operator khác CÙNG signature (không có thì giữ nguyên).

    Trả cây mới (AST bất biến — không sửa ``node`` gốc).
    """
    targets = all_subtrees(node)
    target = targets[rng.integers(0, len(targets))]

    if isinstance(target, Field):
        replacement: Node = Field(fields[rng.integers(0, len(fields))])
        return _replace_subtree(node, target, replacement)

    if isinstance(target, Constant):
        kind = _constant_arg_kind(node, target, registry)
        if kind is ArgKind.WINDOW:
            # tìm parent spec để lấy window_choices (đúng theo CHÍNH operator cha — không
            # default; brand-mới nếu cha có choices riêng)
            parent_spec = _find_parent_spec(node, target, registry)
            choices = parent_spec.window_choices if parent_spec is not None else (5, 10, 20, 60, 120)
            replacement = Constant(float(choices[rng.integers(0, len(choices))]))
            return _replace_subtree(node, target, replacement)
        if kind is ArgKind.SCALAR or kind is None:
            replacement = Constant(float(target.value) + float(rng.normal(0, 0.5)))
            return _replace_subtree(node, target, replacement)
        # Constant ở GROUP/PANEL slot (hiếm — vd group literal): không perturb
        return node

    # target là Call: đổi op sang operator khác cùng signature
    assert isinstance(target, Call)  # narrowing — leaf đã xử lý ở hai nhánh trên
    spec = registry.get(target.op)
    candidates = [
        s for s in registry.gp_function_set()
        if s.signature == spec.signature and s.name != spec.name
    ]
    if not candidates:
        return node
    new_op = candidates[rng.integers(0, len(candidates))]
    replacement = Call(op=new_op.name, args=target.args)
    return _replace_subtree(node, target, replacement)


def _find_parent_spec(
    root: Node, target: Constant, registry: OperatorRegistry,
) -> OperatorSpec | None:
    """Tìm OperatorSpec của Call CHA trực tiếp của ``target``; None nếu không tìm thấy."""
    if not isinstance(root, Call):
        return None
    try:
        spec: OperatorSpec | None = registry.get(root.op)
    except KeyError:
        spec = None
    for child in root.args:
        if child is target:
            return spec
        found = _find_parent_spec(child, target, registry)
        if found is not None:
            return found
    return None


def subtree_mutation(
    node: Node, registry: OperatorRegistry, rng: np.random.Generator,
    fields: tuple[str, ...], max_depth: int = MAX_DEPTH,
) -> Node:
    """Thay 1 subtree ngẫu nhiên bằng cây ngẫu nhiên mới, depth giới hạn theo ``remaining``
    (đảm bảo cây kết quả không vượt ``max_depth``). Trả cây mới."""
    targets = all_subtrees(node)
    target = targets[rng.integers(0, len(targets))]
    target_depth = DepthVisitor().visit(target)
    full_depth = DepthVisitor().visit(node)
    remaining = max(1, max_depth - (full_depth - target_depth))
    new_subtree = random_tree(
        registry, rng, depth=int(rng.integers(1, remaining + 1)), fields=fields,
        full=bool(rng.integers(0, 2)),
    )
    return _replace_subtree(node, target, new_subtree)


def hoist_mutation(node: Node, rng: np.random.Generator) -> Node:
    """Nâng 1 subtree KHÔNG PHẢI root lên làm cây toàn bộ — chống bloat (B13/R6) bằng cách
    rút ngắn cây. Cây chỉ 1 node (leaf đơn) -> trả nguyên ``node`` (không có gì để hoist)."""
    candidates = [s for s in all_subtrees(node) if s is not node]
    if not candidates:
        return node
    return candidates[rng.integers(0, len(candidates))]


def dedup_population(
    individuals: list[Individual], registry: OperatorRegistry | None = None,
) -> list[Individual]:
    """Khử trùng lặp cấu trúc theo ``ind.canonical_hash()`` (Task 7.1): giữ cá thể ĐẦU
    TIÊN mỗi nhóm hash, loại phần còn lại — giữ thứ tự xuất hiện (ổn định cho test)."""
    seen: set[str] = set()
    result: list[Individual] = []
    for ind in individuals:
        h = ind.canonical_hash()
        if h in seen:
            continue
        seen.add(h)
        result.append(ind)
    return result
