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
