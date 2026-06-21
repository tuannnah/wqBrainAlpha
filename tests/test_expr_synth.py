"""Test module chung expr_synth: auto-wrap, dựng prompt, vòng repair."""

from __future__ import annotations

import json
import re

from src.llm import expr_synth
from src.simulation.pre_filter import PreFilter
from tests.fakes import FakeDeepSeek, FakeSymbolRepo


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


def test_build_symbol_context_pinned_chi_liet_field_ghim():
    repo = _FieldRepo([_Field("pcr_oi_30"), _Field("close"), _Field("volume")])
    ops = FakeSymbolRepo(["rank"])
    pf = PreFilter(known_operators={"rank"}, known_fields={"pcr_oi_30", "close", "volume"})
    out = expr_synth.build_symbol_context(repo, ops, pf, None, "bất kỳ", pinned=["pcr_oi_30"])
    assert "pcr_oi_30" in out
    # field ngoài palette ghim không được liệt vào dòng FIELDS
    field_line = [ln for ln in out.splitlines() if ln.startswith("FIELDS khả dụng")][0]
    assert "volume" not in field_line
    assert "KHÔNG bịa" in out


def test_build_symbol_context_pinned_none_giu_hanh_vi_cu():
    repo = _FieldRepo([_Field("close", "MATRIX"), _Field("svec", "VECTOR")])
    ops = FakeSymbolRepo(["rank", "ts_zscore", "vec_avg", "vec_sum"])
    pf = PreFilter(known_operators={"rank"}, known_fields={"close", "svec"})
    out = expr_synth.build_symbol_context(repo, ops, pf, scope=None, relevance_text="svec")
    assert "KHÔNG bịa" not in out  # không có câu ghim khi pinned=None


def test_build_symbol_context_pinned_khong_co_trong_cache_khong_chen_cam_bia():
    repo = _FieldRepo([_Field("close")])
    ops = FakeSymbolRepo(["rank"])
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    out = expr_synth.build_symbol_context(repo, ops, pf, None, "bất kỳ", pinned=["khong_ton_tai"])
    # pinned không khớp field nào trong cache -> fields rỗng -> field_line rơi về default
    # chung -> KHÔNG được chèn câu cấm-bịa (vì nó sẽ mâu thuẫn với danh sách default đó)
    assert "KHÔNG bịa" not in out


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


def test_suggest_fields_fallback_khi_khong_khop():
    repo = _FieldRepo([_Field("pcr_oi_30"), _Field("close"), _Field("volume")])
    out = expr_synth.suggest_fields(repo, None, "asset_growth_rate")
    assert out  # không rỗng dù 'asset_growth_rate' không khớp field nào
    assert all(isinstance(x, str) for x in out)


def test_suggest_fields_fallback_uu_tien_pinned():
    repo = _FieldRepo([_Field("close")])
    out = expr_synth.suggest_fields(repo, None, "zzz_khong_khop", pinned=["pcr_oi_30", "scl12_buzz"])
    assert out[0] == "pcr_oi_30"


def test_repair_tra_expr_khi_pass_lan_dau():
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    ds = FakeDeepSeek([json.dumps({"expression": "rank(close)"})])
    out = expr_synth.repair_to_expression(
        ds, pf, _FieldRepo([_Field("close")]), None, "sys", "usr", task=None
    )
    assert out == "rank(close)"
    assert len(ds.calls) == 1


def test_repair_autowrap_pass_khong_goi_lai_llm():
    pf = PreFilter(
        known_operators={"ts_zscore", "vec_avg"},
        known_fields={"svec"},
        field_types={"svec": "VECTOR"},
        matrix_only_ops={"ts_zscore"},
    )
    ds = FakeDeepSeek([json.dumps({"expression": "ts_zscore(svec, 20)"})])
    out = expr_synth.repair_to_expression(
        ds, pf, _FieldRepo([_Field("svec", "VECTOR")]), None, "sys", "usr", task=None
    )
    assert out == "ts_zscore(vec_avg(svec), 20)"
    assert len(ds.calls) == 1  # auto-wrap sửa, không cần round-trip thêm


def test_repair_them_hint_field_khi_field_bia():
    pf = PreFilter(known_operators={"rank"}, known_fields={"opt6_real"})
    ds = FakeDeepSeek([
        json.dumps({"expression": "rank(opt6_bia)"}),   # field bịa -> fail
        json.dumps({"expression": "rank(opt6_real)"}),  # sửa lại hợp lệ
    ])
    out = expr_synth.repair_to_expression(
        ds, pf, _FieldRepo([_Field("opt6_real")]), None, "sys", "usr", task=None
    )
    assert out == "rank(opt6_real)"
    # lượt user thứ 2 phải chứa hint field thật gần nhất
    assert "opt6_real" in ds.calls[1][1]


def test_repair_tra_none_khi_het_retry():
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    ds = FakeDeepSeek([json.dumps({"expression": "bad_op(x)"})] * 5)
    out = expr_synth.repair_to_expression(
        ds, pf, _FieldRepo([_Field("close")]), None, "sys", "usr", task=None
    )
    assert out is None
    assert len(ds.calls) == expr_synth.MAX_REPAIR_ATTEMPTS


def test_repair_pinned_tai_tiem_palette_vao_prompt():
    pf = PreFilter(known_operators={"rank"}, known_fields={"pcr_oi_30"})
    ds = FakeDeepSeek([
        json.dumps({"expression": "rank(asset_growth_rate)"}),  # field bịa -> fail
        json.dumps({"expression": "rank(pcr_oi_30)"}),          # sửa hợp lệ
    ])
    out = expr_synth.repair_to_expression(
        ds, pf, _FieldRepo([_Field("pcr_oi_30")]), None, "sys", "usr",
        task=None, pinned=["pcr_oi_30"],
    )
    assert out == "rank(pcr_oi_30)"
    assert "pcr_oi_30" in ds.calls[1][1]
    assert "CHỈ được dùng" in ds.calls[1][1]


def test_retrieve_palette_field_lien_quan_dung_dau():
    repo = _FieldRepo([
        _Field("pcr_oi_30", description="put call ratio open interest"),
        _Field("close"), _Field("volume"),
    ])
    out = expr_synth.retrieve_field_palette(repo, None, "put call open interest", min_k=1)
    assert out[0].id == "pcr_oi_30"


def test_retrieve_palette_khong_khop_van_khong_rong():
    repo = _FieldRepo([_Field(f"f{i}") for i in range(10)])
    out = expr_synth.retrieve_field_palette(repo, None, "asset growth rate", min_k=8)
    assert len(out) >= 8
    assert all(getattr(f, "id", None) for f in out)


def test_retrieve_palette_cache_rong_tra_rong():
    assert expr_synth.retrieve_field_palette(_FieldRepo([]), None, "bất kỳ") == []
