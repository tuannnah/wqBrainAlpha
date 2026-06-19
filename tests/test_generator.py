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
