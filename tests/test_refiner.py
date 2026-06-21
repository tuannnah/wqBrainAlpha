"""Test tinh chỉnh alpha nhắm chiều yếu nhất (GĐ2: T2.12)."""

from __future__ import annotations

import json

from src.llm.hypothesis import Hypothesis
from src.llm.refiner import AlphaRefiner
from src.llm.translator import AlphaCandidate, AlphaTranslator
from src.simulation.pre_filter import PreFilter
from tests.fakes import FakeDeepSeek, FakeSymbolRepo


def _setup(ds):
    pf = PreFilter(known_operators={"rank", "ts_delta", "ts_mean", "ts_decay_linear"}, known_fields={"close", "volume"})
    translator = AlphaTranslator(ds, FakeSymbolRepo(["close", "volume"]), FakeSymbolRepo(["rank", "ts_delta", "ts_mean", "ts_decay_linear"]), pf)
    return AlphaRefiner(ds, translator)


def _candidate():
    return AlphaCandidate(Hypothesis("o", "b", "r", "s"), "mô tả gốc", "rank(ts_delta(close, 5))")


def test_refiner_turnover_hint_theo_toc_do_signal():
    """Hint turnover phải biết tốc độ signal: turnover cao (fast) cảnh báo không làm
    mượt mù; turnover thấp khuyên tăng độ nhạy. (Review 6.)"""
    from src.llm.refiner import DIMENSION_HINTS

    r = AlphaRefiner(deepseek=None, translator=None)
    hint_fast = r._dimension_hint("turnover_fit", {"turnover": 0.95})
    hint_slow = r._dimension_hint("turnover_fit", {"turnover": 0.005})
    assert hint_fast != hint_slow
    # fast signal: cảnh báo smoothing/decay phá returns
    assert "decay" in hint_fast.lower() or "mượt" in hint_fast.lower()
    # chiều khác giữ nguyên hint tĩnh
    assert r._dimension_hint("sharpe", {"turnover": 0.5}) == DIMENSION_HINTS["sharpe"]


def test_refine_prompt_chua_chieu_yeu_va_alpha_hien_tai():
    ds = FakeDeepSeek(
        [
            json.dumps({"description": "làm mượt tín hiệu để giảm turnover"}),
            json.dumps({"expression": "rank(ts_mean(ts_delta(close, 5), 10))"}),
        ]
    )
    refiner = _setup(ds)
    metrics = {"sharpe": 1.5, "fitness": 1.2, "turnover": 0.9, "drawdown": 0.1}
    cand = refiner.refine(_candidate(), metrics, "turnover_fit")

    refine_system, refine_user = ds.calls[0]
    blob = refine_system + refine_user
    assert "turnover" in blob.lower()
    assert "rank(ts_delta(close, 5))" in blob  # alpha hiện tại có trong prompt
    assert isinstance(cand, AlphaCandidate)
    assert cand.expression == "rank(ts_mean(ts_delta(close, 5), 10))"


def test_refine_mo_ta_truoc_roi_bieu_thuc():
    ds = FakeDeepSeek(
        [
            json.dumps({"description": "mô tả cải tiến X"}),
            json.dumps({"expression": "rank(close)"}),
        ]
    )
    refiner = _setup(ds)
    refiner.refine(_candidate(), {"sharpe": 0.1}, "sharpe")
    # Lời gọi sinh biểu thức (call thứ 2) phải chứa mô tả cải tiến.
    assert "mô tả cải tiến X" in ds.calls[1][1]


def test_refine_giu_hypothesis_goc():
    ds = FakeDeepSeek(
        [json.dumps({"description": "d"}), json.dumps({"expression": "rank(close)"})]
    )
    refiner = _setup(ds)
    c = _candidate()
    out = refiner.refine(c, {"sharpe": 1.0}, "fitness")
    assert out.hypothesis is c.hypothesis


def test_refine_tra_none_khi_khong_sinh_duoc_bieu_thuc():
    ds = FakeDeepSeek(
        [json.dumps({"description": "d"})] + [json.dumps({"expression": "bad_op(x)"})] * 5
    )
    refiner = _setup(ds)
    assert refiner.refine(_candidate(), {"sharpe": 1.0}, "sharpe") is None


def test_refine_khong_lap_bieu_thuc_da_thu():
    """Trí nhớ: refiner đã đề xuất một biểu thức thì lần sau KHÔNG lặp y hệt
    (gốc rễ ts_mean(volume,5) bị bơm lặp vô hạn trong log thật)."""
    ds = FakeDeepSeek([
        json.dumps({"description": "d1"}),
        json.dumps({"expression": "rank(ts_mean(volume, 5))"}),
        json.dumps({"description": "d2"}),
        json.dumps({"expression": "rank(ts_mean(volume, 5))"}),  # lặp y hệt
    ])
    refiner = _setup(ds)
    first = refiner.refine(_candidate(), {"sharpe": 1.0}, "sharpe")
    second = refiner.refine(_candidate(), {"sharpe": 1.0}, "sharpe")
    assert first is not None and first.expression == "rank(ts_mean(volume, 5))"
    assert second is None  # đã thử -> không lặp lại


def test_refine_nhoi_danh_sach_da_thu_vao_prompt():
    """Biểu thức đã thử được nhồi vào prompt _propose để LLM tránh lặp."""
    ds = FakeDeepSeek([
        json.dumps({"description": "d1"}),
        json.dumps({"expression": "rank(ts_mean(volume, 5))"}),
        json.dumps({"description": "d2"}),
        json.dumps({"expression": "rank(ts_delta(close, 10))"}),
    ])
    refiner = _setup(ds)
    refiner.refine(_candidate(), {"sharpe": 1.0}, "sharpe")
    refiner.refine(_candidate(), {"sharpe": 1.0}, "sharpe")
    second_propose_user = ds.calls[2][1]  # _propose của lần refine thứ 2
    assert "rank(ts_mean(volume, 5))" in second_propose_user
