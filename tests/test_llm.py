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
    description: str = ""
    dataset_id: str = ""


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

    def complete(self, system, user, json_mode=True, task=None):
        self.calls.append((system, user))
        return self._responses.pop(0)


# ----- fake HTTP client cho DeepSeekClient qua Anthropic-compatible API
class _FakeResponse:
    status_code = 200

    def __init__(self, text='{"expression": "rank(close)"}', prompt_tokens=100, completion_tokens=50):
        self._text = text
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.text = text

    def json(self):
        return {
            "content": [{"type": "text", "text": self._text}],
            "usage": {
                "input_tokens": self._prompt_tokens,
                "output_tokens": self._completion_tokens,
            },
        }


class _FakeHTTPClient:
    def __init__(self, responses=None):
        self._responses = list(responses or [_FakeResponse()])
        self.calls = []

    def post(self, path, *, json, headers):
        self.calls.append({"path": path, "json": json, "headers": headers})
        return self._responses.pop(0)


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
    deepseek = FakeDeepSeek([json.dumps({"ideas": ["option skew", "analyst revision", "news novelty"]})])
    gen = _generator(deepseek)
    ideas = gen.generate_ideas(3)
    assert ideas == ["option skew", "analyst revision", "news novelty"]


def test_ideas_prompt_huong_dataset_thay_the():
    # Prompt sinh ý tưởng phải dẫn LLM sang dataset ít khai thác, không phải PV thuần.
    gen = _generator(FakeDeepSeek([]))
    system = gen.build_ideas_system_prompt()
    low = system.lower()
    # Nhắc ít nhất vài chủ đề dữ liệu thay thế đặc trưng.
    themes = ["implied", "tin tức", "analyst", "chuỗi cung ứng", "mạng xã hội", "option"]
    assert sum(t in low for t in themes) >= 3
    assert "json" in low


def test_ideas_prompt_canh_bao_cong_thuc_kinh_dien():
    # Phải nêu rõ tránh công thức PV/fundamental kinh điển vì correlation cao.
    gen = _generator(FakeDeepSeek([]))
    low = gen.build_ideas_system_prompt().lower()
    assert "correlation" in low or "tương quan" in low or "trùng" in low
    assert "tránh" in low or "không" in low


def test_generate_ideas_dung_prompt_moi():
    # generate_ideas phải gửi đúng system prompt mới (không còn prompt cũ generic).
    deepseek = FakeDeepSeek([json.dumps({"ideas": ["a", "b"]})])
    gen = _generator(deepseek)
    gen.generate_ideas(2)
    sent_system, _ = deepseek.calls[0]
    assert sent_system == gen.build_ideas_system_prompt()


def test_generate_ideas_retry_va_loai_cliche():
    deepseek = FakeDeepSeek(
        [
            json.dumps(
                {
                    "ideas": [
                        "Long cac co phieu co dong luong 20 ngay tang manh va volume cao.",
                        "Short cac co phieu giam qua 10% trong 5 ngay, ky vong phuc hoi.",
                    ]
                }
            ),
            json.dumps(
                {
                    "ideas": [
                        "Option implied volatility skew divergence between put-call demand and realized volatility.",
                        "Analyst net earnings revision surprise with sector-neutral confirmation.",
                    ]
                }
            ),
        ]
    )
    gen = _generator(deepseek)

    ideas = gen.generate_ideas(2)

    assert ideas == [
        "Option implied volatility skew divergence between put-call demand and realized volatility.",
        "Analyst net earnings revision surprise with sector-neutral confirmation.",
    ]
    assert len(deepseek.calls) == 2
    assert "rejected" in deepseek.calls[1][1].lower()


def test_ideas_prompt_dung_dataset_field_that_tu_cache():
    pf = PreFilter(known_operators={"rank"}, known_fields={"close", "volume", "opt_iv_skew", "news_sentiment"})
    field_repo = FakeRepo(
        [
            _Field("close", dataset_id="pv1"),
            _Field("volume", dataset_id="pv1"),
            _Field("opt_iv_skew", description="put call implied volatility skew", dataset_id="option8"),
            _Field("news_sentiment", description="event sentiment and novelty", dataset_id="news18"),
        ]
    )
    op_repo = FakeRepo([_Op("rank")])
    gen = LLMAlphaGenerator(FakeDeepSeek([]), field_repo, op_repo, pf)

    prompt = gen.build_ideas_system_prompt()

    assert "option8" in prompt
    assert "opt_iv_skew" in prompt
    assert "news18" in prompt
    assert "news_sentiment" in prompt


def test_deepseek_client_track_usage():
    fake_http = _FakeHTTPClient()
    client = DeepSeekClient(api_key="x", client=fake_http)
    content = client.complete("sys", "user")
    assert "rank(close)" in content
    assert client.usage.prompt_tokens == 100
    assert client.usage.completion_tokens == 50
    assert client.usage.total_tokens == 150
    assert client.usage.estimated_cost() > 0


def test_deepseek_client_goi_anthropic_messages_api():
    fake_http = _FakeHTTPClient()
    client = DeepSeekClient(api_key="x", client=fake_http)

    client.complete("sys", "hello", json_mode=False)

    call = fake_http.calls[0]
    assert call["path"] == "/v1/messages"
    assert call["headers"]["x-api-key"] == "x"
    assert call["json"]["model"] == "deepseek-v4-pro"
    assert call["json"]["system"] == "sys"
    assert call["json"]["messages"] == [{"role": "user", "content": "hello"}]


def test_deepseek_client_json_mode_ep_prompt_chat_va_retry_khi_parse_loi():
    fake_http = _FakeHTTPClient(
        [
            _FakeResponse("khong phai json"),
            _FakeResponse('{"ok": true}'),
        ]
    )
    client = DeepSeekClient(api_key="x", client=fake_http)

    content = client.complete("sys", "hello", json_mode=True)

    assert content == '{"ok": true}'
    assert len(fake_http.calls) == 2
    first_payload = fake_http.calls[0]["json"]
    assert "Return ONLY valid JSON" in first_payload["system"]
    assert "no Markdown" in first_payload["system"]
    retry_payload = fake_http.calls[1]["json"]
    assert "not valid JSON" in retry_payload["messages"][0]["content"]
    assert "khong phai json" in retry_payload["messages"][0]["content"]


class _ThinkingOnlyResponse:
    """Mô phỏng reasoning model bị cắt: chỉ có khối thinking, CHƯA kịp sinh text."""

    status_code = 200
    text = ""

    def json(self):
        return {
            "content": [{"type": "thinking", "thinking": "đang suy nghĩ...", "signature": "x"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 300},
        }


def test_deepseek_client_thinking_an_het_token_thi_raise_ro_rang():
    # Bug gốc: text rỗng vì thinking ngốn hết max_tokens -> không được nuốt im lặng.
    fake_http = _FakeHTTPClient([_ThinkingOnlyResponse(), _ThinkingOnlyResponse()])
    client = DeepSeekClient(api_key="x", client=fake_http, max_json_retries=1)

    import pytest

    with pytest.raises(ValueError, match="max_tokens"):
        client.complete("sys", "hello", json_mode=True)


def test_deepseek_client_json_mode_raise_sau_khi_het_retry():
    fake_http = _FakeHTTPClient(
        [
            _FakeResponse("sai 1"),
            _FakeResponse("sai 2"),
        ]
    )
    client = DeepSeekClient(api_key="x", client=fake_http, max_json_retries=1)

    import pytest

    with pytest.raises(ValueError, match="valid JSON"):
        client.complete("sys", "hello", json_mode=True)

    assert len(fake_http.calls) == 2


# ------------------------------------------- feedback từ DB cho bộ sinh hướng
def test_common_fields_dem_field():
    from src.llm.generator import _common_fields

    exprs = ["rank(actual_eps_value)", "ts_mean(actual_eps_value, 5)", "rank(close)"]
    assert _common_fields(exprs, top=1) == ["actual_eps_value"]


def test_build_feedback_prompt_co_top_va_weak():
    from src.llm.generator import build_feedback_prompt

    s = build_feedback_prompt([("scale(ts_zscore(returns, 5))", 1.95, 0.87)], ["actual_eps_value"])
    assert "scale(ts_zscore(returns, 5))" in s
    assert "1.95" in s
    assert "actual_eps_value" in s


class _FbRepo:
    def top_simulated(self, limit=5):
        return [("scale(ts_zscore(returns, 5))", 1.95, 0.87)]

    def recent_failures(self, limit=200):
        from types import SimpleNamespace

        return [SimpleNamespace(category="low_score", expression="rank(actual_eps_value)")]


def test_generate_ideas_chen_feedback_tu_db():
    deepseek = FakeDeepSeek([json.dumps({"ideas": ["option skew", "news novelty"]})])
    pf = PreFilter(known_operators={"rank"}, known_fields={"close"})
    gen = LLMAlphaGenerator(
        deepseek, FakeRepo([_Field("close")]), FakeRepo([_Op("rank")]), pf, repo=_FbRepo()
    )
    gen.generate_ideas(2)
    user = deepseek.calls[0][1]
    assert "scale(ts_zscore(returns, 5))" in user  # exploit: top alpha
    assert "actual_eps_value" in user              # tránh: field yếu


def test_generate_ideas_khong_repo_thi_khong_feedback():
    """Không truyền repo -> prompt không có mục feedback (tương thích ngược)."""
    deepseek = FakeDeepSeek([json.dumps({"ideas": ["option skew", "news novelty"]})])
    gen = _generator(deepseek)
    gen.generate_ideas(2)
    user = deepseek.calls[0][1]
    assert "MÔ PHỎNG TỐT NHẤT" not in user
