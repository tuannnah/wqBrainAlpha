"""Test các module gọi LLM truyền đúng `task` để ModelRouter định tuyến (GĐ6: T6.3)."""

from __future__ import annotations

import json

from src.llm.alignment import AlignmentScorer
from src.llm.hypothesis import Hypothesis, HypothesisGenerator
from src.llm.refiner import AlphaRefiner
from src.llm.translator import AlphaCandidate, AlphaTranslator
from src.simulation.pre_filter import PreFilter
from tests.fakes import FakeDeepSeek, FakeSymbolRepo


def _hyp():
    return Hypothesis("thanh khoản tăng", "vi cấu trúc", "spread thu hẹp", "dùng volume")


def test_hypothesis_truyen_task_hypothesis():
    ds = FakeDeepSeek([json.dumps({"observation": "x"})])
    HypothesisGenerator(ds).generate("momentum")
    assert ds.tasks == ["hypothesis"]


def test_alignment_truyen_task_alignment():
    ds = FakeDeepSeek([json.dumps({"score": 0.8, "reason": "ok"})])
    cand = AlphaCandidate(_hyp(), "mô tả", "rank(close)")
    AlignmentScorer(ds).score(cand)
    assert ds.tasks == ["alignment"]


def test_translator_describe_dung_describe_to_expression_dung_translate():
    ds = FakeDeepSeek([
        json.dumps({"description": "đảo chiều giá"}),
        json.dumps({"expression": "rank(close)"}),
    ])
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    tr = AlphaTranslator(ds, FakeSymbolRepo(["close"]), FakeSymbolRepo(["rank"]), pf)
    tr.translate(_hyp())
    # bước mô tả = suy luận (describe -> model mạnh); bước viết biểu thức = translate (rẻ).
    assert ds.tasks == ["describe", "translate"]


def test_refiner_truyen_task_refine():
    ds = FakeDeepSeek([
        json.dumps({"description": "giảm turnover"}),
        json.dumps({"expression": "rank(close)"}),
    ])
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    tr = AlphaTranslator(ds, FakeSymbolRepo(["close"]), FakeSymbolRepo(["rank"]), pf)
    cand = AlphaCandidate(_hyp(), "mô tả gốc", "rank(close)")
    AlphaRefiner(ds, tr).refine(cand, {"turnover": 0.9}, "turnover_fit")
    # đề xuất cải tiến = suy luận (refine); viết biểu thức = translate.
    assert ds.tasks == ["refine", "translate"]
