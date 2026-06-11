"""Test TemplateGenerator."""

from __future__ import annotations

import random

from src.generation.template import TemplateGenerator
from src.simulation.pre_filter import PreFilter


def test_generate_alpha_hop_le():
    fields = ["close", "open", "volume", "high", "low", "vwap"]
    pf = PreFilter(known_fields=set(fields), known_operators=None)  # operators không check
    gen = TemplateGenerator(fields, pf, rng=random.Random(42))

    alphas = gen.generate(50)
    assert len(alphas) == 50
    # Tất cả phải qua pre-filter và dùng field hợp lệ.
    for expr in alphas:
        ok, reason = pf.check(expr)
        assert ok, f"{expr}: {reason}"


def test_can_field_de_khoi_tao():
    pf = PreFilter()
    try:
        TemplateGenerator([], pf)
        assert False, "phải raise"
    except ValueError:
        pass
