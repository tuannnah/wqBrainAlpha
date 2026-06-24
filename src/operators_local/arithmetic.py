"""Operator số học: + - * / log abs sign power max min. Tất cả NaN-propagate tự nhiên
qua NumPy (NaN op x = NaN); không cần xử lý universe ở đây — Evaluator áp mask sau impl."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


@register(name="add", category=OpCategory.ARITHMETIC,
          signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=True)
def add(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    return x + y


@register(name="subtract", category=OpCategory.ARITHMETIC,
          signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=False)
def subtract(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    return x - y


@register(name="multiply", category=OpCategory.ARITHMETIC,
          signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=True)
def multiply(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    return x * y


@register(name="divide", category=OpCategory.ARITHMETIC,
          signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=False)
def divide(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    with np.errstate(divide="ignore", invalid="ignore"):
        return x / y


@register(name="log", category=OpCategory.ARITHMETIC,
          signature=(ArgKind.PANEL,), bounded=False, commutative=False)
def log(ctx: EvalContext, x: Panel) -> Panel:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x)


@register(name="abs", category=OpCategory.ARITHMETIC,
          signature=(ArgKind.PANEL,), bounded=False, commutative=False)
def abs_(ctx: EvalContext, x: Panel) -> Panel:
    return np.abs(x)


@register(name="sign", category=OpCategory.ARITHMETIC,
          signature=(ArgKind.PANEL,), bounded=True, commutative=False)
def sign(ctx: EvalContext, x: Panel) -> Panel:
    return np.sign(x)


@register(name="power", category=OpCategory.ARITHMETIC,
          signature=(ArgKind.PANEL, ArgKind.SCALAR), bounded=False, commutative=False)
def power(ctx: EvalContext, x: Panel, p: float) -> Panel:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.power(x, p)


@register(name="max", category=OpCategory.ARITHMETIC,
          signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=True)
def max_(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    return np.maximum(x, y)


@register(name="min", category=OpCategory.ARITHMETIC,
          signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=True)
def min_(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    return np.minimum(x, y)
