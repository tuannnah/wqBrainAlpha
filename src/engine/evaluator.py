"""Evaluator: duyệt AST (NodeVisitor[Panel]) -> (T,N) Panel. Dispatch qua OperatorRegistry,
áp universe mask (NaN ngoài universe) sau mỗi Call, cache theo canonical hash (B6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.market_panel import MarketData
from src.engine.subexpr_cache import SubexprCache
from src.lang.ast import Call, Constant, Field, Node, NodeVisitor
from src.lang.registry import ArgKind, OperatorRegistry
from src.lang.visitors import CanonicalHasher
from src.local_types import Mask, Panel


@dataclass(frozen=True, slots=True)
class EvalContext:
    data: MarketData
    registry: OperatorRegistry
    cache: SubexprCache | None = None


def _apply_universe_mask(panel: Panel, universe: Mask) -> Panel:
    """NaN hóa mọi cell ngoài universe — bất biến B6: out-of-universe luôn NaN."""
    out = panel.copy()
    out[~universe] = np.nan
    return out


def _literal(node: Node) -> float | str:
    """Đọc giá trị literal của arg WINDOW/SCALAR/GROUP — không eval thành Panel.
    WINDOW/SCALAR đọc từ Constant.value; GROUP đọc tên group từ Field.name (group key
    được biểu diễn như Field trong AST vì cũng là một identifier, vd `sector`)."""
    if isinstance(node, Constant):
        return node.value
    if isinstance(node, Field):
        return node.name
    raise TypeError(
        f"arg literal (WINDOW/SCALAR/GROUP) phải là Constant hoặc Field, nhận {type(node)!r}"
    )


class Evaluator(NodeVisitor[Panel]):
    """Duyệt AST sinh Panel (T,N). Dùng `evaluate()` làm điểm vào (quản lý cache);
    `visit_*` không tự gọi lại `evaluate` của con qua `accept` mà qua `self.evaluate`
    để mọi sub-node cũng đi qua cache."""

    def __init__(self, ctx: EvalContext) -> None:
        self._ctx = ctx
        self._hasher = CanonicalHasher(ctx.registry)

    def evaluate(self, node: Node) -> Panel:
        if self._ctx.cache is not None:
            key = self._hasher.visit(node)
            cached = self._ctx.cache.get(key)
            if cached is not None:
                return cached
            result = node.accept(self)
            self._ctx.cache.put(key, result)
            return result
        return node.accept(self)

    def visit_constant(self, node: Constant) -> Panel:
        shape = self._ctx.data.universe.shape
        return np.full(shape, float(node.value), dtype=np.float64)

    def visit_field(self, node: Field) -> Panel:
        return _apply_universe_mask(self._ctx.data.field(node.name), self._ctx.data.universe)

    def visit_call(self, node: Call) -> Panel:
        spec = self._ctx.registry.get(node.op)
        eval_args: list[Panel | float | str] = []
        for arg, kind in zip(node.args, spec.signature, strict=True):
            if kind is ArgKind.PANEL:
                eval_args.append(self.evaluate(arg))
            else:  # WINDOW, SCALAR, GROUP
                eval_args.append(_literal(arg))
        out = spec.impl(self._ctx, *eval_args)
        return _apply_universe_mask(out, self._ctx.data.universe)
