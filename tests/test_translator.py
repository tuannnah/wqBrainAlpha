"""Test dịch giả thuyết -> mô tả -> FASTEXPR + repair cú pháp (GĐ2: T2.4, T2.5)."""

from __future__ import annotations

import json

from src.llm.hypothesis import Hypothesis
from src.llm.translator import AlphaCandidate, AlphaTranslator
from src.simulation.pre_filter import PreFilter
from tests.fakes import FakeDeepSeek, FakeSymbolRepo


def _translator(deepseek):
    pf = PreFilter(known_operators={"rank", "ts_delta", "ts_mean"}, known_fields={"close", "volume"})
    fields = FakeSymbolRepo(["close", "volume"])
    ops = FakeSymbolRepo(["rank", "ts_delta", "ts_mean"])
    return AlphaTranslator(deepseek, fields, ops, pf)


def _hyp():
    return Hypothesis("quan sát", "nền", "lý giải", "dùng close, cửa sổ 5")


def test_suggest_fields_tra_field_that_gan_nhat():
    """Field thật gần 'bad_field': cùng tiền tố dataset + trùng token đứng trước."""
    from src.llm import expr_synth
    fields = FakeSymbolRepo(["opt6_1dorhv_real", "opt6_close", "news12_sent", "close"])
    out = expr_synth.suggest_fields(fields, None, "opt6_1dorhv")
    assert "opt6_1dorhv_real" in out  # cùng tiền tố opt6_ + trùng token 1dorhv
    assert out[0].startswith("opt6_")  # field cùng dataset đứng đầu, không phải 'close'


def test_translate_qua_buoc_mo_ta_roi_bieu_thuc():
    ds = FakeDeepSeek(
        [
            json.dumps({"description": "đảo chiều giá ngắn hạn dùng close"}),
            json.dumps({"expression": "rank(ts_delta(close, 5))"}),
        ]
    )
    cand = _translator(ds).translate(_hyp())
    assert isinstance(cand, AlphaCandidate)
    assert cand.description.startswith("đảo chiều")
    assert cand.expression == "rank(ts_delta(close, 5))"
    # Bắt buộc qua bước mô tả: mô tả phải xuất hiện trong prompt sinh biểu thức.
    expr_call_user = ds.calls[1][1]
    assert "đảo chiều giá ngắn hạn" in expr_call_user


def test_translate_repair_cu_phap():
    ds = FakeDeepSeek(
        [
            json.dumps({"description": "mô tả"}),
            json.dumps({"expression": "bad_op(close)"}),  # operator lạ -> fail
            json.dumps({"expression": "rank(close)"}),  # sửa lại hợp lệ
        ]
    )
    cand = _translator(ds).translate(_hyp())
    assert cand.expression == "rank(close)"
    assert len(ds.calls) == 3  # mô tả + 2 lần thử biểu thức


def test_translate_tra_none_khi_het_retry():
    ds = FakeDeepSeek(
        [json.dumps({"description": "mô tả"})]
        + [json.dumps({"expression": "bad_op(x)"})] * 5
    )
    assert _translator(ds).translate(_hyp()) is None


def test_translate_giu_lai_hypothesis():
    ds = FakeDeepSeek(
        [json.dumps({"description": "d"}), json.dumps({"expression": "rank(close)"})]
    )
    h = _hyp()
    cand = _translator(ds).translate(h)
    assert cand.hypothesis is h


# ------------------------------------------------- T3.6 tránh nhánh con phổ biến
def test_avoid_subtrees_chen_vao_prompt_sinh_bieu_thuc():
    ds = FakeDeepSeek(
        [json.dumps({"description": "mô tả"}), json.dumps({"expression": "rank(close)"})]
    )
    tr = _translator(ds)
    tr.set_avoid_subtrees(["ts_mean(F,N)", "ts_corr(F,F,N)"])
    tr.translate(_hyp())
    expr_system = ds.calls[1][0]  # system prompt của bước sinh biểu thức
    assert "ts_mean(F,N)" in expr_system
    assert "ts_corr(F,F,N)" in expr_system


def test_khong_avoid_subtrees_thi_prompt_khong_co_muc_tranh():
    ds = FakeDeepSeek(
        [json.dumps({"description": "mô tả"}), json.dumps({"expression": "rank(close)"})]
    )
    tr = _translator(ds)
    tr.translate(_hyp())
    expr_system = ds.calls[1][0]
    assert "tránh" not in expr_system.lower() or "F,N" not in expr_system


# ----------------------------- ràng buộc cú pháp trong prompt (qua pre-filter)
def test_prompt_neu_gioi_han_depth_node_lay_tu_prefilter():
    ds = FakeDeepSeek(
        [json.dumps({"description": "mô tả"}), json.dumps({"expression": "rank(close)"})]
    )
    tr = _translator(ds)  # PreFilter mặc định: max_depth=6, max_nodes=30
    tr.translate(_hyp())
    expr_system = ds.calls[1][0]
    low = expr_system.lower()
    assert str(tr.prefilter.max_depth) in expr_system and "độ sâu" in low
    assert str(tr.prefilter.max_nodes) in expr_system


def test_prompt_cam_doi_so_co_ten_keyvalue():
    ds = FakeDeepSeek(
        [json.dumps({"description": "mô tả"}), json.dumps({"expression": "rank(close)"})]
    )
    tr = _translator(ds)
    tr.translate(_hyp())
    low = ds.calls[1][0].lower()
    assert "vị trí" in low  # đối số theo vị trí
    assert "key=value" in low or "std=" in low  # cảnh báo không dùng đối số có tên


def test_prompt_neu_field_type_va_cach_giam_vector_truoc_matrix_ops():
    """Log thực tế cho thấy LLM hay gọi ts_zscore/rank trực tiếp trên VECTOR field."""

    class _Field:
        def __init__(self, id, type):
            self.id = id
            self.type = type
            self.description = ""
            self.dataset_id = ""

    class _FieldRepo:
        def load_cached(self, region=None, universe=None, delay=None):
            return [
                _Field("close", "MATRIX"),
                _Field("composite_sentiment_score_2", "VECTOR"),
            ]

    pf = PreFilter(
        known_operators={"rank", "ts_zscore", "vec_avg", "vec_sum"},
        known_fields={"close", "composite_sentiment_score_2"},
        field_types={"close": "MATRIX", "composite_sentiment_score_2": "VECTOR"},
        matrix_only_ops={"rank", "ts_zscore"},
    )
    tr = AlphaTranslator(
        FakeDeepSeek([json.dumps({"description": "d"}), json.dumps({"expression": "rank(close)"})]),
        _FieldRepo(),
        FakeSymbolRepo(["rank", "ts_zscore", "vec_avg", "vec_sum"]),
        pf,
    )

    h = Hypothesis("qs composite sentiment", "nền", "lý giải", "dùng composite_sentiment_score_2")
    tr.translate(h)
    expr_system = tr.deepseek.calls[1][0]

    assert "VECTOR" in expr_system
    assert "composite_sentiment_score_2" in expr_system
    assert "vec_avg" in expr_system
    assert "vec_sum" in expr_system
    assert "ts_zscore(vec_avg(composite_sentiment_score_2)" in expr_system


# ------------------------- chọn fields theo độ liên quan với hypothesis/mô tả
def test_field_lien_quan_duoc_uu_tien_vao_prompt():
    """Field hướng cần (nêu trong hypothesis) phải vào prompt dù nằm ngoài top-40."""
    dummies = [f"dummy_field_{i}" for i in range(50)]
    fields = FakeSymbolRepo(dummies + ["call_breakeven_10"])
    pf = PreFilter(known_operators={"rank"}, known_fields=set(dummies) | {"call_breakeven_10", "close"})
    ops = FakeSymbolRepo(["rank"])
    ds = FakeDeepSeek(
        [json.dumps({"description": "đo độ dốc kỳ hạn call breakeven"}),
         json.dumps({"expression": "rank(close)"})]
    )
    tr = AlphaTranslator(ds, fields, ops, pf)
    # implementation_spec nêu đích danh field cần dùng
    h = Hypothesis("qs", "nền", "lý giải", "dùng call_breakeven_10 để tính kỳ vọng biến động")
    tr.translate(h)
    expr_system = ds.calls[1][0]
    assert "call_breakeven_10" in expr_system  # field liên quan có mặt
    # không thể nhồi hết: prompt bị cắt, không chứa toàn bộ 50 dummy
    assert sum(d in expr_system for d in dummies) < len(dummies)


# ------------------------------------------------- T6.4 lọc fields theo scope
def test_set_scope_chi_dung_fields_dung_region():
    """Đặt scope -> prompt chỉ chứa fields của region đó, không trộn region khác."""
    pf = PreFilter(known_operators={"rank"}, known_fields={"close", "eur_only", "volume"})
    fields = FakeSymbolRepo(by_scope={
        ("USA", "TOP3000", 1): ["close", "volume"],
        ("EUR", "TOP1000", 0): ["eur_only"],
    })
    ops = FakeSymbolRepo(["rank"])
    tr = AlphaTranslator(
        FakeDeepSeek([json.dumps({"description": "d"}), json.dumps({"expression": "rank(close)"})]),
        fields, ops, pf,
    )
    tr.set_scope(region="USA", universe="TOP3000", delay=1)
    tr.translate(_hyp())
    expr_system = tr.deepseek.calls[1][0]
    assert "close" in expr_system
    assert "eur_only" not in expr_system  # không trộn fields của EUR


def test_khong_set_scope_thi_load_tat_ca_tuong_thich_nguoc():
    """Không đặt scope -> load_cached() không tham số (tương thích ngược)."""
    fields = FakeSymbolRepo(["close", "volume"])
    tr = _translator(
        FakeDeepSeek([json.dumps({"description": "d"}), json.dumps({"expression": "rank(close)"})])
    )
    tr.field_repo = fields
    tr.translate(_hyp())
    assert fields.scope_calls == [(None, None, None)]
