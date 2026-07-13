"""Test bộ lọc structural `check_meaningful` (src/lang/meaningfulness.py): từ chối biểu
thức AST vô nghĩa kinh tế/degenerate (no-op, domain-invalid, tự lồng dư thừa) BẤT KỂ điểm
số local — và KHÔNG từ chối oan biểu thức hợp lệ."""

from __future__ import annotations

import pytest

import src.operators_local  # noqa: F401  side-effect: đăng ký toàn bộ operator thật vào registry
from src.app.closed_loop_adapters import VERIFIED_CORES
from src.generation.alt_data_seeds import ALT_DATA_CORES
from src.generation.fundamental_seeds import FUNDAMENTAL_CORES
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
    # Đổi field mẫu từ "volume" -> "high" (Task 4): "volume" đơn thuần rơi vào rule MỚI
    # "không có hướng giá" (đúng ý -- max(volume, 4*volume) THẬT SỰ vô hướng giá) nên không
    # còn hợp để làm fixture riêng cho regression no-op/scale-fold ở test này -- ý định gốc
    # (KHÔNG liên quan hướng giá) vẫn giữ nguyên với field có hướng giá khác ("high").
    ok, reason = check_meaningful(parse("max(high, multiply(4, high))"))
    assert (ok, reason) == (True, "")


def test_rejects_divide_of_identical_args():
    # divide(x, x) vẫn phải bị chặn -- 2 nhánh giống hệt cấu trúc (≡ hằng số 1), không liên
    # quan gì tới việc bóc scale.
    ok, reason = check_meaningful(parse("divide(close, close)"))
    assert ok is False
    assert reason


# --- Task 4: 2 rule mới -- power(sign(x), k chẵn) suy biến hằng số; biểu thức chỉ dùng field
# khối lượng (không field hướng giá) -- bằng chứng từ log thật 07-12 (đốt sim Brain oan).


def test_rejects_power_of_sign_even_exponent_bang_chung_log_that():
    # Bằng chứng thật (07-12 ý tưởng #11): sim Brain ra Sharpe 0.00/TO 0.00 -- sign(...)**2
    # luôn suy biến thành hằng số (sign ∈ {-1,0,1} -> bình phương ∈ {0,1}).
    expr = (
        "power(sign(trade_when(multiply(volume, volume), divide(returns, open), "
        "sign(vwap))), 2)"
    )
    ok, reason = check_meaningful(parse(expr))
    assert ok is False
    assert reason


def test_rejects_power_of_sign_exponent_4():
    ok, reason = check_meaningful(parse("power(sign(close), 4)"))
    assert ok is False
    assert reason


def test_accepts_power_of_sign_odd_exponent():
    # Số mũ LẺ không suy biến hằng số (sign(x)**3 == sign(x)) -- KHÔNG bị chặn bởi rule này.
    ok, reason = check_meaningful(parse("power(sign(close), 3)"))
    assert (ok, reason) == (True, "")


def test_accepts_power_of_non_sign_base_even_exponent():
    # power(x, 2) với x KHÔNG phải sign(...) -- hợp lệ (đã có test_accepts_power_with_positive_exponent
    # ở trên cho power(close, 2); test này khẳng định rõ base khác sign() không kích hoạt rule).
    ok, reason = check_meaningful(parse("power(rank(close), 2)"))
    assert (ok, reason) == (True, "")


def test_rejects_volume_only_expression_bang_chung_log_that():
    # Bằng chứng thật (07-12 ý tưởng #7): sim Brain ra Sharpe -0.13 -- biểu thức chỉ dùng field
    # "volume" (không field hướng giá close/open/high/low/vwap/returns) -- vô hướng kinh tế.
    expr = "multiply(-1, multiply(ts_mean(ts_mean(volume, 120), 120), ts_mean(ts_delta(volume, 3), 10)))"
    ok, reason = check_meaningful(parse(expr))
    assert ok is False
    assert reason


def test_rejects_expression_with_only_adv20_field():
    ok, reason = check_meaningful(parse("rank(ts_mean(adv20, 20))"))
    assert ok is False
    assert reason


def test_accepts_volume_mixed_with_price_field():
    # volume KẾT HỢP với field giá (close) -- CÓ hướng giá -- KHÔNG bị chặn bởi rule volume-only.
    ok, reason = check_meaningful(parse("multiply(volume, close)"))
    assert (ok, reason) == (True, "")


def test_accepts_volume_only_when_combined_with_returns():
    ok, reason = check_meaningful(parse("multiply(ts_mean(volume, 20), returns)"))
    assert (ok, reason) == (True, "")


# --- DoD: seed đã kiểm chứng/curated KHÔNG bị chặn oan bởi 2 rule mới -----------------------


@pytest.mark.parametrize("expr", VERIFIED_CORES)
def test_verified_cores_khong_bi_chan_oan(expr):
    ok, reason = check_meaningful(parse(expr))
    assert (ok, reason) == (True, "")


@pytest.mark.parametrize("expr", ALT_DATA_CORES)
def test_alt_data_cores_khong_bi_chan_oan(expr):
    ok, reason = check_meaningful(parse(expr))
    assert (ok, reason) == (True, "")


@pytest.mark.parametrize("expr", FUNDAMENTAL_CORES)
def test_fundamental_cores_khong_bi_chan_oan(expr):
    ok, reason = check_meaningful(parse(expr))
    assert (ok, reason) == (True, "")
