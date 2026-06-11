"""Sinh alpha từ template + param hợp lệ, lọc qua PreFilter."""

from __future__ import annotations

import random

from src.simulation.pre_filter import PreFilter

TEMPLATES = [
    "rank(ts_delta({field}, {d}))",
    "-rank(ts_delta({field}, {d}))",
    "rank(ts_mean({field}, {d1}) - ts_mean({field}, {d2}))",
    "-rank(ts_zscore({field}, {d}))",
    "rank(ts_zscore({field}, {d}))",
    "group_neutralize(rank({field}), {group})",
    "rank(ts_corr({f1}, {f2}, {d}))",
    "rank(ts_std_dev({field}, {d}))",
    "ts_rank({field}, {d})",
    "rank({field} / ts_mean({field}, {d}))",
    "rank(ts_sum({field}, {d}))",
    "group_rank({field}, {group})",
    "rank(ts_delta({field}, {d1}) - ts_delta({field}, {d2}))",
    "-ts_rank(ts_delta({field}, {d}), {d2})",
    "rank(ts_decay_linear({field}, {d}))",
    "group_neutralize(ts_zscore({field}, {d}), {group})",
    "rank(ts_max({field}, {d}) - ts_min({field}, {d}))",
    "rank(ts_product({field}, {d}))",
    "rank(ts_corr(rank({f1}), rank({f2}), {d}))",
]

PARAM_RANGES = {
    "d": [5, 10, 20, 40, 60],
    "d1": [5, 10, 20],
    "d2": [20, 40, 60],
    "group": ["market", "sector", "industry", "subindustry"],
}


class TemplateGenerator:
    def __init__(
        self,
        fields: list[str],
        prefilter: PreFilter,
        templates: list[str] | None = None,
        param_ranges: dict | None = None,
        rng: random.Random | None = None,
    ):
        if not fields:
            raise ValueError("Cần ít nhất một field để sinh template")
        self.fields = fields
        self.prefilter = prefilter
        self.templates = templates or list(TEMPLATES)
        self.param_ranges = param_ranges or {k: list(v) for k, v in PARAM_RANGES.items()}
        self.rng = rng or random.Random()

    def _fill(self, template: str) -> str:
        params = {
            "field": self.rng.choice(self.fields),
            "f1": self.rng.choice(self.fields),
            "f2": self.rng.choice(self.fields),
            "d": self.rng.choice(self.param_ranges["d"]),
            "d1": self.rng.choice(self.param_ranges["d1"]),
            "d2": self.rng.choice(self.param_ranges["d2"]),
            "group": self.rng.choice(self.param_ranges["group"]),
        }
        return template.format(**params)

    def generate(self, count: int, max_attempts: int | None = None) -> list[str]:
        max_attempts = max_attempts or count * 20
        seen: set[str] = set()
        result: list[str] = []
        attempts = 0
        while len(result) < count and attempts < max_attempts:
            attempts += 1
            template = self.rng.choice(self.templates)
            expr = self._fill(template)
            if expr in seen:
                continue
            seen.add(expr)
            ok, _ = self.prefilter.check(expr)
            if ok:
                result.append(expr)
        return result
