"""Pha 2.1: FUNDAMENTAL_CORES — seed đi thẳng Brain trên field fundamental ĐÃ VERIFY LIVE
(fundamental6, USA/TOP3000/delay1, 2026-07-10). Mọi field sparse (coverage 0.5) BẮT BUỘC
ts_backfill (cardinal rule #3). Neutralization fundamental -> INDUSTRY (docs WQ)."""

from __future__ import annotations

import src.operators_local  # noqa: F401  # đăng ký operator (ts_backfill/divide/ts_delta…)
from src.generation.fundamental_seeds import FUNDAMENTAL_CORES, FUNDAMENTAL_FIELDS
from src.generation.alt_data_seeds import neutralization_for_expr
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import FieldCollector


def test_cores_parse_duoc():
    reg = default_registry()
    for core in FUNDAMENTAL_CORES:
        parse(core)  # không ném ParseError


def test_moi_field_deu_da_verify_live():
    """Mọi field trong cores phải nằm trong tập đã verify LIVE (cardinal rule #1: không bịa)."""
    reg = default_registry()
    for core in FUNDAMENTAL_CORES:
        fields = FieldCollector(reg).visit(parse(core))
        for f in fields:
            assert f in FUNDAMENTAL_FIELDS, f"field chưa verify LIVE: {f} trong {core}"


def test_moi_field_fundamental_co_ts_backfill():
    """Field fundamental sparse -> phải bọc ts_backfill (nếu không alpha chết giữa các kỳ)."""
    for core in FUNDAMENTAL_CORES:
        for f in FUNDAMENTAL_FIELDS:
            if f in core:
                assert f"ts_backfill({f}" in core, f"{f} thiếu ts_backfill trong {core}"


def test_neutralization_fundamental_la_industry():
    """Field fundamental -> INDUSTRY (docs WQ neutralization theo category)."""
    for core in FUNDAMENTAL_CORES:
        assert neutralization_for_expr(core) == "INDUSTRY", core


def test_co_it_nhat_4_tin_hieu():
    """Spec liệt kê 4 họ: gross-profitability, cash-flow yield, asset growth, (revenue/quality)."""
    assert len(FUNDAMENTAL_CORES) >= 4
