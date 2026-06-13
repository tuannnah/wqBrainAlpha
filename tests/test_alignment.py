"""Test chấm nhất quán giả thuyết–mô tả–công thức bằng LLM phụ (GĐ4: T4.1)."""

from __future__ import annotations

import json

from src.llm.alignment import AlignmentScorer
from src.llm.hypothesis import Hypothesis
from src.llm.translator import AlphaCandidate
from tests.fakes import FakeDeepSeek


def _cand(expr="rank(ts_corr(close, volume, 20))", desc="đo đồng biến giá-khối lượng"):
    h = Hypothesis("thanh khoản tăng", "lý thuyết vi cấu trúc", "spread thu hẹp", "dùng volume")
    return AlphaCandidate(hypothesis=h, description=desc, expression=expr)


def test_score_tra_diem_trong_khoang_0_1():
    ds = FakeDeepSeek([json.dumps({"score": 0.9, "reason": "khớp tốt"})])
    score = AlignmentScorer(ds).score(_cand())
    assert 0.0 <= score.value <= 1.0
    assert score.value == 0.9


def test_score_giu_ly_do():
    ds = FakeDeepSeek([json.dumps({"score": 0.4, "reason": "thiếu thành phần volume"})])
    score = AlignmentScorer(ds).score(_cand())
    assert "volume" in score.reason


def test_score_clamp_gia_tri_ngoai_khoang():
    ds = FakeDeepSeek([json.dumps({"score": 1.5, "reason": "x"})])
    assert AlignmentScorer(ds).score(_cand()).value == 1.0


def test_score_json_loi_tra_diem_trung_lap():
    # LLM trả rác -> không chặn pipeline, điểm trung lập 0.5.
    ds = FakeDeepSeek(["không phải json"])
    assert AlignmentScorer(ds).score(_cand()).value == 0.5


def test_prompt_chua_gia_thuyet_mo_ta_bieu_thuc():
    ds = FakeDeepSeek([json.dumps({"score": 0.8, "reason": "ok"})])
    AlignmentScorer(ds).score(_cand(expr="rank(close)", desc="đảo chiều giá"))
    system, user = ds.calls[0]
    assert "rank(close)" in user        # biểu thức
    assert "đảo chiều giá" in user      # mô tả
    assert "thanh khoản tăng" in user   # giả thuyết (observation)
