"""Test DeepSeekClient (usage) và LLMAlphaGenerator (validation loop)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.llm.deepseek_client import DeepSeekClient
from src.llm.generator import LLMAlphaGenerator
from src.simulation.pre_filter import PreFilter


# ---------------------------------------------------------------- fakes
@dataclass
class _Field:
    id: str


@dataclass
class _Op:
    name: str


class FakeRepo:
    def __init__(self, items):
        self._items = items

    def load_cached(self):
        return self._items


class FakeDeepSeek:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, system, user, json_mode=True):
        self.calls.append((system, user))
        return self._responses.pop(0)


# ----- fake openai client cho DeepSeekClient
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


class _Resp:
    def __init__(self, content, p, c):
        self.choices = [_Choice(content)]
        self.usage = _Usage(p, c)


class _FakeCompletions:
    def create(self, **kwargs):
        return _Resp('{"expression": "rank(close)"}', 100, 50)


class _FakeChat:
    completions = _FakeCompletions()


class _FakeOpenAI:
    chat = _FakeChat()


# ----------------------------------------------------------------- tests
def _generator(deepseek):
    pf = PreFilter(known_operators={"rank", "ts_delta"}, known_fields={"close", "volume"})
    field_repo = FakeRepo([_Field("close"), _Field("volume")])
    op_repo = FakeRepo([_Op("rank"), _Op("ts_delta")])
    return LLMAlphaGenerator(deepseek, field_repo, op_repo, pf)


def test_build_system_prompt_co_operators_va_json():
    gen = _generator(FakeDeepSeek([]))
    prompt = gen.build_system_prompt()
    assert "rank" in prompt
    assert "close" in prompt
    assert "JSON" in prompt


def test_generate_validation_loop_tu_sua():
    # Lần 1: expr sai (operator không tồn tại) -> repair; lần 2: hợp lệ.
    deepseek = FakeDeepSeek(
        [
            json.dumps({"expression": "bad_op(close)"}),
            json.dumps({"expression": "rank(ts_delta(close, 5))"}),
        ]
    )
    gen = _generator(deepseek)
    result = gen.generate("momentum", n=1)
    assert result == ["rank(ts_delta(close, 5))"]
    assert len(deepseek.calls) == 2  # đã retry một lần


def test_generate_bo_qua_khi_het_retry():
    deepseek = FakeDeepSeek([json.dumps({"expression": "bad_op(x)"})] * 3)
    gen = _generator(deepseek)
    assert gen.generate("x", n=1) == []


def test_generate_ideas_parse_json():
    deepseek = FakeDeepSeek([json.dumps({"ideas": ["momentum", "reversal", "volume"]})])
    gen = _generator(deepseek)
    ideas = gen.generate_ideas(3)
    assert ideas == ["momentum", "reversal", "volume"]


def test_deepseek_client_track_usage():
    client = DeepSeekClient(api_key="x", client=_FakeOpenAI())
    content = client.complete("sys", "user")
    assert "rank(close)" in content
    assert client.usage.prompt_tokens == 100
    assert client.usage.completion_tokens == 50
    assert client.usage.total_tokens == 150
    assert client.usage.estimated_cost() > 0
