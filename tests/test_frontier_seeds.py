"""Kho seed frontier: core từ 12 dataset ít người đào, field verify live 2026-07-14.

Bất biến: parse được, depth trần ≤ 5 (chừa 2 mức cho wrapper tuner), field VECTOR phải
qua vec_avg/vec_sum, field thưa phải ts_backfill, mọi field nằm trong FRONTIER_FIELDS,
không trùng core với kho seed cũ (alt_data/fundamental/hypothesis)."""

from __future__ import annotations

import src.operators_local  # noqa: F401  # đăng ký operator (ts_backfill/ts_rank/vec_avg…)
from src.generation.alt_data_seeds import ALT_DATA_CORES
from src.generation.frontier_seeds import (
    FRONTIER_CATEGORY_BY_FIELD,
    FRONTIER_CORES,
    FRONTIER_FIELDS,
    FRONTIER_VECTOR_FIELDS,
)
from src.generation.fundamental_seeds import FUNDAMENTAL_CORES
from src.generation.hypothesis_seeds import HYPOTHESIS_CORES
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor, FieldCollector


def test_moi_core_parse_va_depth_tran_toi_da_5() -> None:
    for core in FRONTIER_CORES:
        node = parse(core)  # không được ném
        assert DepthVisitor().visit(node) <= 5, core


def test_so_luong_va_khong_trung_kho_cu() -> None:
    assert len(FRONTIER_CORES) >= 35
    assert len(set(FRONTIER_CORES)) == len(FRONTIER_CORES)
    cu = set(ALT_DATA_CORES) | set(FUNDAMENTAL_CORES) | set(HYPOTHESIS_CORES)
    assert not (set(FRONTIER_CORES) & cu)


def test_moi_field_trong_core_da_verify() -> None:
    reg = default_registry()
    for core in FRONTIER_CORES:
        for f in FieldCollector(reg).visit(parse(core)):
            assert f in FRONTIER_FIELDS, f"field chưa verify: {f} trong {core}"


def test_field_vector_phai_boc_vec_avg_hoac_vec_sum() -> None:
    for core in FRONTIER_CORES:
        for f in FRONTIER_VECTOR_FIELDS:
            if f in core:
                assert f"vec_avg({f})" in core or f"vec_sum({f})" in core, core


def test_field_thua_phai_ts_backfill() -> None:
    # Field sự kiện/quý (insider, call, filing, 13F, short-interest kỳ, search VECTOR):
    # thiếu ts_backfill là tín hiệu gần như toàn NaN (cardinal rule #3).
    sparse_cat = {"insider", "call_filing", "ownership", "short_period"}
    for core in FRONTIER_CORES:
        for f, cat in FRONTIER_CATEGORY_BY_FIELD.items():
            if f in core and cat in sparse_cat:
                assert "ts_backfill(" in core, f"thiếu ts_backfill cho {f}: {core}"


def test_moi_field_co_category() -> None:
    assert FRONTIER_FIELDS == set(FRONTIER_CATEGORY_BY_FIELD)
