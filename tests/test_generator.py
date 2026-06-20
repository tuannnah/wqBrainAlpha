"""Test LLMAlphaGenerator — phần ngữ cảnh prompt (blacklist field chết)."""

from __future__ import annotations

from src.llm.generator import LLMAlphaGenerator
from tests.fakes import FakeDeepSeek, FakeSymbolRepo


def _make_generator(blacklist=None):
    return LLMAlphaGenerator(
        FakeDeepSeek(),
        field_repo=FakeSymbolRepo(["close", "volume", "news12_sent"]),
        operator_repo=FakeSymbolRepo(["rank", "ts_mean"]),
        prefilter=None,
        blacklist=blacklist,
    )


def test_prompt_y_tuong_co_dong_cam_field():
    gen = _make_generator(blacklist={"opt6_1dorhv", "asset_growth_rate"})
    prompt = gen.build_ideas_system_prompt()
    assert "TUYỆT ĐỐI KHÔNG dùng field" in prompt
    assert "opt6_1dorhv" in prompt
    assert "asset_growth_rate" in prompt


def test_prompt_y_tuong_khong_co_dong_cam_khi_blacklist_rong():
    gen = _make_generator(blacklist=None)
    prompt = gen.build_ideas_system_prompt()
    assert "TUYỆT ĐỐI KHÔNG dùng field" not in prompt


def test_parse_ideas_loai_bo_metric_bia():
    """LLM hay nhét metric BỊA (sharpe=2.1, fitness=0.92) vào text hướng — đó là
    số tự bịa, không phải đo thật; phải bị tước để không nhiễm xuống downstream."""
    import json

    gen = _make_generator()
    content = json.dumps({"ideas": [
        "scale(ts_decay_linear(group_neutralize(x, sector), 5))  (sharpe=2.1, fitness=0.92) — option IV term structure spread",
        "News novelty reversal sau coverage dày  sharpe=1.3 fitness=0.45",
    ]})
    ideas = gen._parse_ideas(content)
    assert all("sharpe" not in i.lower() for i in ideas), ideas
    assert all("fitness" not in i.lower() for i in ideas), ideas
    # Nội dung ý nghĩa (nguồn dữ liệu/hiện tượng) phải được giữ lại.
    assert any("option IV term structure" in i for i in ideas)
    assert any("News novelty reversal" in i for i in ideas)
