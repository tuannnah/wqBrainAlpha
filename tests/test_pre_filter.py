"""Test PreFilter."""

from __future__ import annotations

from src.simulation.pre_filter import PreFilter


def test_expression_hop_le_pass():
    pf = PreFilter(
        known_operators={"rank", "ts_delta"},
        known_fields={"close"},
    )
    ok, reason = pf.check("rank(ts_delta(close, 5))")
    assert ok, reason


def test_ngoac_khong_can_bang():
    ok, reason = PreFilter().check("rank(ts_delta(close, 5)")
    assert not ok
    assert "ngoặc" in reason.lower()


def test_operator_khong_ton_tai():
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    ok, reason = pf.check("ts_unknown(close, 5)")
    assert not ok
    assert "operator" in reason.lower()


def test_field_khong_ton_tai():
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    ok, reason = pf.check("rank(nonexistent_field)")
    assert not ok
    assert "field" in reason.lower()


def test_group_constant_chap_nhan():
    pf = PreFilter(known_operators={"group_neutralize", "rank"}, known_fields={"close"})
    ok, reason = pf.check("group_neutralize(rank(close), sector)")
    assert ok, reason


def test_qua_sau():
    pf = PreFilter(max_depth=2)
    ok, reason = pf.check("rank(ts_delta(close, 5))")
    assert not ok
    assert "sâu" in reason.lower()


# Biểu thức residual-momentum thật (độ sâu 7) từng bị loại khi max_depth=6.
_DEPTH7 = (
    "group_neutralize(ts_delay(divide(rank(ts_delay(ts_sum(returns, 210), 21)), "
    "rank(ts_std_dev(returns, 60))), 1), subindustry)"
)
_DEPTH7_OPS = {"group_neutralize", "ts_delay", "divide", "rank", "ts_sum", "ts_std_dev"}


def test_default_cho_phep_do_sau_7():
    pf = PreFilter(known_operators=_DEPTH7_OPS, known_fields={"returns"})
    ok, reason = pf.check(_DEPTH7)
    assert ok, reason


def test_default_van_chan_do_sau_8():
    pf = PreFilter(known_operators=_DEPTH7_OPS, known_fields={"returns"})
    ok, reason = pf.check(f"rank({_DEPTH7})")  # bọc thêm 1 tầng -> độ sâu 8
    assert not ok
    assert "sâu" in reason.lower()
