"""Test module chung expr_synth: auto-wrap, dựng prompt, vòng repair."""

from __future__ import annotations

from src.llm import expr_synth


def test_autowrap_boc_vec_avg_leaf_vector_duoi_matrix_op():
    out = expr_synth.autowrap_vector_fields(
        "ts_zscore(svec, 20)",
        field_types={"svec": "VECTOR"},
        matrix_only_ops={"ts_zscore"},
    )
    assert out == "ts_zscore(vec_avg(svec), 20)"


def test_autowrap_khong_dung_field_matrix_hay_so():
    out = expr_synth.autowrap_vector_fields(
        "ts_zscore(close, 20)",
        field_types={"close": "MATRIX"},
        matrix_only_ops={"ts_zscore"},
    )
    assert out == "ts_zscore(close, 20)"


def test_autowrap_idempotent_khi_da_co_vec_avg():
    out = expr_synth.autowrap_vector_fields(
        "rank(vec_avg(svec))",
        field_types={"svec": "VECTOR"},
        matrix_only_ops={"rank"},
    )
    assert out == "rank(vec_avg(svec))"


def test_autowrap_boc_nhieu_leaf_vector():
    out = expr_synth.autowrap_vector_fields(
        "add(rank(v1), ts_delta(v2, 5))",
        field_types={"v1": "VECTOR", "v2": "VECTOR"},
        matrix_only_ops={"rank", "ts_delta"},
    )
    assert out == "add(rank(vec_avg(v1)), ts_delta(vec_avg(v2), 5))"


def test_autowrap_no_op_khi_thieu_du_lieu_kieu():
    assert expr_synth.autowrap_vector_fields("rank(svec)", None, None) == "rank(svec)"
    assert expr_synth.autowrap_vector_fields("rank(svec)", {}, set()) == "rank(svec)"


def test_autowrap_giu_nguyen_khi_khong_parse_duoc():
    bad = "rank(svec"  # ngoặc lệch
    assert expr_synth.autowrap_vector_fields(bad, {"svec": "VECTOR"}, {"rank"}) == bad
