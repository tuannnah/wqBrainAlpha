"""Test điều kiện Power Pool Alphas (sub-project A) — chỉ phần tính được LOCAL, KHÔNG gồm
Power Pool Correlation/Theme (xem docstring plan/module)."""

from __future__ import annotations

from src.llm.hypothesis import Hypothesis
from src.scoring.power_pool import (
    build_power_pool_description,
    check_power_pool_eligibility,
    count_operators_fields,
    is_valid_power_pool_description,
)


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


def test_du_dieu_kien_power_pool():
    result = check_power_pool_eligibility("rank(add(close, open))", sharpe=1.2)
    assert result.eligible is True
    assert result.reasons == []
    assert result.n_operators == 2
    assert result.n_fields == 2


def test_khong_du_vi_sharpe_thap():
    result = check_power_pool_eligibility("rank(close)", sharpe=0.5)
    assert result.eligible is False
    assert any("Sharpe" in r for r in result.reasons)


def test_khong_du_vi_sharpe_none():
    result = check_power_pool_eligibility("rank(close)", sharpe=None)
    assert result.eligible is False


def test_khong_du_vi_qua_nhieu_operator():
    expr = "close"
    for i in range(9):
        expr = f"op{i}({expr})"
    result = check_power_pool_eligibility(expr, sharpe=1.5)
    assert result.eligible is False
    assert any("operator" in r for r in result.reasons)


def test_khong_du_vi_qua_nhieu_field():
    result = check_power_pool_eligibility("add(add(add(f1, f2), f3), f4)", sharpe=1.5)
    assert result.eligible is False
    assert any("field" in r for r in result.reasons)


def test_build_description_ghep_dung_mau_wq():
    h = Hypothesis(
        observation="Gia co phieu co xu huong dao chieu sau chuoi giam manh trong ngan han.",
        background="Ly thuyet mean-reversion tren thi truong von ngan han.",
        economic_rationale="Nha dau tu phan ung thai qua roi dieu chinh lai theo thoi gian.",
        implementation_spec="Dung field close, cua so 5 ngay, chuan hoa bang rank.",
    )
    desc = build_power_pool_description(h)
    assert "Idea:" in desc
    assert "Rationale for data used:" in desc
    assert "Rationale for operators used:" in desc
    assert is_valid_power_pool_description(desc) is True


def test_is_valid_description_do_dai():
    assert is_valid_power_pool_description("a" * 99) is False
    assert is_valid_power_pool_description("a" * 100) is True
