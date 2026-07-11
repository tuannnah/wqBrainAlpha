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


# ----------------------- luật type: chặn operator MATRIX áp lên field VECTOR
def _pf_typed():
    return PreFilter(
        known_operators={"ts_zscore", "rank", "vec_avg", "ts_mean", "add"},
        known_fields={"close", "composite_sentiment_score_2"},
        field_types={"close": "MATRIX", "composite_sentiment_score_2": "VECTOR"},
        matrix_only_ops={"ts_zscore", "rank", "ts_mean"},  # Time Series + Cross Sectional
    )


def test_chan_operator_matrix_tren_field_vector():
    """ts_zscore (Time Series) áp trực tiếp lên field VECTOR -> chặn trước khi sim."""
    ok, reason = _pf_typed().check("ts_zscore(composite_sentiment_score_2, 20)")
    assert not ok
    assert "vector" in reason.lower()
    assert "composite_sentiment_score_2" in reason


def test_field_vector_qua_vec_op_thi_hop_le():
    """vec_avg (Vector) tiêu thụ VECTOR trước -> ts_mean nhận Node, không bị chặn."""
    ok, reason = _pf_typed().check("ts_mean(vec_avg(composite_sentiment_score_2), 20)")
    assert ok, reason


def test_field_matrix_tren_operator_matrix_hop_le():
    ok, reason = _pf_typed().check("ts_zscore(close, 20)")
    assert ok, reason


def test_khong_khai_bao_type_thi_bo_qua_kiem_type():
    """Tương thích ngược: không truyền field_types -> không kiểm type."""
    pf = PreFilter(known_operators={"ts_zscore"}, known_fields={"composite_sentiment_score_2"})
    ok, reason = pf.check("ts_zscore(composite_sentiment_score_2, 20)")
    assert ok, reason


# ----------------------- luật arity: chặn operator thừa input (tolerant)
def test_chan_arity_thua_input():
    """rank nhận đúng 1 input; `rank(close, volume, 5)` (3 input) -> chặn trước sim.

    Tái hiện lỗi WQ thật: `Invalid number of inputs : 3, should be exactly 1`."""
    pf = PreFilter(
        known_operators={"rank"},
        known_fields={"close", "volume"},
        operator_arity={"rank": 1},
    )
    ok, reason = pf.check("rank(close, volume, 5)")
    assert not ok
    assert "input" in reason.lower()
    assert "rank" in reason


def test_arity_tham_so_tuy_chon_khong_chan():
    """ts_decay_linear có chữ ký 3 (gồm tham số tùy chọn) nhưng dùng 2 input vẫn
    hợp lệ. Tolerant: chỉ chặn khi THỪA, không chặn khi THIẾU."""
    pf = PreFilter(
        known_operators={"ts_decay_linear"},
        known_fields={"close"},
        operator_arity={"ts_decay_linear": 3},
    )
    ok, reason = pf.check("ts_decay_linear(close, 5)")
    assert ok, reason


def test_variadic_khong_chan_du_thua_input():
    """Operator variadic (vd add) nhận số input bất kỳ -> không chặn dù > arity chữ ký."""
    pf = PreFilter(
        known_operators={"add"},
        known_fields={"close", "volume", "returns"},
        operator_arity={"add": 2},
        variadic_ops={"add"},
    )
    ok, reason = pf.check("add(close, volume, returns)")
    assert ok, reason


def test_chan_named_param_goi_positional():
    """winsorize/bucket chỉ nhận 1 positional (std/buckets là named) -> gọi 2 positional bị chặn.

    Tái hiện lỗi WQ: winsorize(x, 3) -> 'should be exactly 1 input'; bucket(x, 10) ->
    'buckets/range required'. arity positional (bỏ named '=') = 1 cho cả hai."""
    pf = PreFilter(
        known_operators={"winsorize", "bucket", "ts_zscore", "rank"},
        known_fields={"close"},
        operator_arity={"winsorize": 1, "bucket": 1, "ts_zscore": 2, "rank": 1},
    )
    ok, _ = pf.check("winsorize(ts_zscore(close, 20), 3)")
    assert not ok
    ok, _ = pf.check("bucket(rank(close), 10)")
    assert not ok
    # Gọi đúng (1 positional) vẫn hợp lệ.
    ok, reason = pf.check("winsorize(close)")
    assert ok, reason


def test_khong_khai_bao_arity_thi_bo_qua():
    """Tương thích ngược: không truyền operator_arity -> không kiểm arity."""
    pf = PreFilter(known_operators={"rank"}, known_fields={"close", "volume"})
    ok, reason = pf.check("rank(close, volume, 5)")
    assert ok, reason


# ----------------------- luật arity: nguồn LOCAL registry bổ sung (lấp lỗ hổng catalog)
def test_local_arity_chan_khi_catalog_rong():
    """Tái hiện đúng lỗ hổng: `sign` VẮNG khỏi catalog Brain (operator_arity={}) -> trước
    đây bị bỏ qua kiểm arity hoàn toàn. Nguồn `local_arity` (suy từ chữ ký OperatorRegistry
    cục bộ, sign là unary) phải LẤP lỗ này và chặn `sign(close, volume)` (2 input cho op
    1-arg) — đây chính là lỗi WQ thật đã đốt sim: 'Invalid number of inputs : 2, should be
    exactly 1'."""
    pf = PreFilter(
        known_operators={"sign"},
        known_fields={"close", "volume"},
        operator_arity={},  # catalog Brain không có entry cho "sign" -> mô phỏng lỗ hổng
        local_arity={"sign": 1},
    )
    ok, reason = pf.check("sign(close, volume)")
    assert not ok
    assert "input" in reason.lower()
    assert "sign" in reason


def test_local_arity_khong_chan_oan_optional_arg():
    """`rank(close, 2)` (rate tùy chọn) và `ts_backfill(close, 22)` vẫn hợp lệ: catalog_arity
    (mô phỏng catalog Brain thật, đã tính cả tham số tùy chọn) > local_arity (chữ ký cục bộ
    unary) -> cap hiệu lực = max(catalog, local) không siết oan các op có optional arg."""
    pf = PreFilter(
        known_operators={"rank", "ts_backfill"},
        known_fields={"close"},
        operator_arity={"rank": 2, "ts_backfill": 2},
        local_arity={"rank": 1, "ts_backfill": 1},
    )
    ok, reason = pf.check("rank(close, 2)")
    assert ok, reason
    ok, reason = pf.check("ts_backfill(close, 22)")
    assert ok, reason


def test_bare_prefilter_van_hoat_dong():
    """PreFilter() không truyền gì (kể cả local_arity) vẫn chạy được bình thường — nhiều
    test/call site dựng PreFilter trần không có nguồn arity nào."""
    ok, reason = PreFilter().check("rank(ts_delta(close, 5))")
    assert ok, reason


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
