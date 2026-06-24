"""group_neutralize: trừ mean theo group mỗi ngày (wrapper config, gp_usable=False, B5)."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


@register(name="group_neutralize", category=OpCategory.GROUP,
          signature=(ArgKind.PANEL, ArgKind.GROUP), bounded=False, gp_usable=False,
          commutative=False)
def group_neutralize(ctx: EvalContext, x: Panel, group_name: str) -> Panel:
    groups = ctx.data.groups[group_name]
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row, grp_row = x[t], groups[t]
        valid = ~np.isnan(row)
        if not np.any(valid):
            continue
        for g in np.unique(grp_row[valid]):
            mask = valid & (grp_row == g)
            if not np.any(mask):
                continue
            out[t][mask] = row[mask] - float(np.mean(row[mask]))
    return out
