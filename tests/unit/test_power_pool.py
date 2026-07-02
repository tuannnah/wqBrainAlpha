"""Test điều kiện Power Pool Alphas (sub-project A) — chỉ phần tính được LOCAL, KHÔNG gồm
Power Pool Correlation/Theme (xem docstring plan/module)."""

from __future__ import annotations

from src.scoring.power_pool import count_operators_fields


def test_dem_operator_field_co_ban():
    n_op, n_field = count_operators_fields("rank(add(close, open))")
    assert n_op == 2  # rank, add
    assert n_field == 2  # close, open


def test_loai_tru_ts_backfill_group_backfill_khoi_dem_operator():
    n_op, _ = count_operators_fields("rank(ts_backfill(group_backfill(close, sector), 5))")
    assert n_op == 1  # chỉ 'rank' tính; ts_backfill/group_backfill không tính theo tài liệu


def test_loai_tru_grouping_field_khoi_dem_field():
    _, n_field = count_operators_fields("group_rank(close, sector)")
    assert n_field == 1  # 'sector' là grouping field, bị loại; chỉ 'close' được tính


def test_operator_field_unique_khong_dem_lap():
    n_op, n_field = count_operators_fields("add(rank(close), rank(close))")
    assert n_op == 2  # add, rank (không đếm rank 2 lần)
    assert n_field == 1  # close (không đếm 2 lần)
