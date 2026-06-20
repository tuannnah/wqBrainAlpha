"""Test module chung expr_synth: auto-wrap, dựng prompt, vòng repair."""

from __future__ import annotations

import re

from src.llm import expr_synth
from src.simulation.pre_filter import PreFilter
from tests.fakes import FakeSymbolRepo


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


class _Field:
    def __init__(self, id, type="MATRIX", description="", dataset_id=""):
        self.id = id
        self.type = type
        self.description = description
        self.dataset_id = dataset_id


class _FieldRepo:
    def __init__(self, fields):
        self._fields = fields

    def load_cached(self, region=None, universe=None, delay=None):
        return self._fields


def test_build_symbol_context_chen_quy_tac_vector():
    repo = _FieldRepo([_Field("close", "MATRIX"), _Field("svec", "VECTOR")])
    ops = FakeSymbolRepo(["rank", "ts_zscore", "vec_avg", "vec_sum"])
    pf = PreFilter(known_operators={"rank"}, known_fields={"close", "svec"})
    out = expr_synth.build_symbol_context(repo, ops, pf, scope=None, relevance_text="svec")
    assert "FIELD TYPES" in out
    assert "VECTOR" in out
    assert "vec_avg" in out
    assert "ts_zscore(vec_avg(svec)" in out


def test_build_symbol_context_khong_vector_thi_khong_chen_quy_tac():
    repo = _FieldRepo([_Field("close", "MATRIX")])
    ops = FakeSymbolRepo(["rank"])
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    out = expr_synth.build_symbol_context(repo, ops, pf, scope=None)
    assert "QUY TAC VECTOR" not in out


def test_build_syntax_constraints_lay_gioi_han_tu_prefilter():
    pf = PreFilter(known_operators={"rank"}, max_depth=6, max_nodes=30)
    out = expr_synth.build_syntax_constraints(pf)
    assert "6" in out and "30" in out
    low = out.lower()
    assert "vị trí" in low and ("key=value" in low or "std=" in low)


def test_suggest_fields_uu_tien_cung_tien_to_dataset():
    repo = _FieldRepo([
        _Field("opt6_1dorhv_real"), _Field("opt6_close"),
        _Field("news12_sent"), _Field("close"),
    ])
    out = expr_synth.suggest_fields(repo, scope=None, bad_field="opt6_1dorhv")
    assert "opt6_1dorhv_real" in out
    assert out[0].startswith("opt6_")
