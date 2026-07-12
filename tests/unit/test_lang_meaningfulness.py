"""Test bộ lọc structural `check_meaningful` (src/lang/meaningfulness.py): từ chối biểu
thức AST vô nghĩa kinh tế/degenerate (no-op, domain-invalid, tự lồng dư thừa) BẤT KỂ điểm
số local — và KHÔNG từ chối oan biểu thức hợp lệ."""

from __future__ import annotations

import src.operators_local  # noqa: F401  side-effect: đăng ký toàn bộ operator thật vào registry
from src.lang.meaningfulness import MAX_SAME_TS_NEST, check_meaningful
from src.lang.parser import parse


def test_rejects_min_of_identical_args():
    ok, reason = check_meaningful(parse("min(returns, returns)"))
    assert ok is False
    assert reason


def test_rejects_subtract_of_identical_args():
    ok, reason = check_meaningful(parse("subtract(close, close)"))
    assert ok is False
    assert reason


def test_rejects_negative_power_exponent():
    ok, reason = check_meaningful(parse("power(returns, -0.5)"))
    assert ok is False
    assert reason


def test_rejects_log_of_ts_corr():
    ok, reason = check_meaningful(parse("log(ts_corr(close, volume, 10))"))
    assert ok is False
    assert reason


def test_rejects_triple_nested_same_ts_op():
    expr = "ts_std_dev(ts_std_dev(ts_std_dev(close, 5), 5), 5)"
    ok, reason = check_meaningful(parse(expr))
    assert ok is False
    assert reason


def test_rejects_add_of_x_and_negated_x():
    # add(x, multiply(-1, x)) ~ 0 -- suy biến dù không dùng min/max/subtract trực tiếp.
    ok, reason = check_meaningful(parse("add(close, multiply(-1, close))"))
    assert ok is False
    assert reason


def test_accepts_ts_mean_of_subtract_close_vwap():
    ok, reason = check_meaningful(parse("ts_mean(subtract(close, vwap), 10)"))
    assert (ok, reason) == (True, "")


def test_accepts_min_of_different_args():
    ok, reason = check_meaningful(parse("min(close, open)"))
    assert (ok, reason) == (True, "")


def test_accepts_power_with_positive_exponent():
    ok, reason = check_meaningful(parse("power(close, 2)"))
    assert (ok, reason) == (True, "")


def test_accepts_log_of_close():
    ok, reason = check_meaningful(parse("log(close)"))
    assert (ok, reason) == (True, "")


def test_accepts_add_of_two_ranks():
    ok, reason = check_meaningful(parse("add(rank(close), rank(volume))"))
    assert (ok, reason) == (True, "")


def test_accepts_double_nested_same_ts_op_at_threshold():
    # Đúng MAX_SAME_TS_NEST tầng (2) -- KHÔNG bị chặn, chỉ chặn khi > ngưỡng.
    assert MAX_SAME_TS_NEST == 2
    ok, reason = check_meaningful(parse("ts_std_dev(ts_std_dev(close, 5), 5)"))
    assert (ok, reason) == (True, "")


# --- Regression: no-op check phải dùng equality CẤU TRÚC (Serializer), KHÔNG được bóc scale
# dương ở gốc (CanonicalHasher._fold_positive_scale_at_root) -- nếu không sẽ chặn OAN các tín
# hiệu HỢP LỆ như subtract(close, multiply(2, close)) (= -close) vì hasher coi `close` và
# `multiply(2, close)` là "giống nhau" (bug từ commit 8fd2353, xem meaningfulness.py::_check_noop).


def test_accepts_subtract_of_x_and_scaled_x_not_flagged_as_noop():
    # subtract(close, multiply(2, close)) = close - 2*close = -close -- tín hiệu HỢP LỆ, KHÔNG
    # phải no-op dù CanonicalHasher (bóc scale) coi 2 nhánh "bằng nhau".
    ok, reason = check_meaningful(parse("subtract(close, multiply(2, close))"))
    assert (ok, reason) == (True, "")


def test_accepts_min_of_x_and_scaled_x_not_flagged_as_noop():
    ok, reason = check_meaningful(parse("min(close, multiply(2, close))"))
    assert (ok, reason) == (True, "")


def test_accepts_max_of_x_and_scaled_x_not_flagged_as_noop():
    ok, reason = check_meaningful(parse("max(volume, multiply(4, volume))"))
    assert (ok, reason) == (True, "")


def test_rejects_divide_of_identical_args():
    # divide(x, x) vẫn phải bị chặn -- 2 nhánh giống hệt cấu trúc (≡ hằng số 1), không liên
    # quan gì tới việc bóc scale.
    ok, reason = check_meaningful(parse("divide(close, close)"))
    assert ok is False
    assert reason
