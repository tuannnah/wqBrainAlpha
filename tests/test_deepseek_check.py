"""Test lệnh smoke check DeepSeek không gọi mạng thật."""

from __future__ import annotations

import pytest

from src.app.cli import llm as main
from config.settings import Settings


class _FakeDeepSeekClient:
    created = []

    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.calls = []
        _FakeDeepSeekClient.created.append(self)

    def complete(self, system, user, json_mode=True):
        self.calls.append((system, user, json_mode))
        return "Hello from DeepSeek"


def test_settings_default_deepseek_model_la_v4_pro(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    assert Settings(_env_file=None).deepseek_model == "deepseek-v4-pro"


def test_settings_default_deepseek_base_url_la_anthropic(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    assert Settings(_env_file=None).deepseek_base_url == "https://api.deepseek.com/anthropic"


def test_run_deepseek_smoke_uses_base_url_model_va_gui_hello():
    _FakeDeepSeekClient.created = []

    reply = main.run_deepseek_smoke(
        api_key="sk-test",
        base_url="https://api.deepseek.com/anthropic",
        model="deepseek-v4-pro",
        message="hello",
        client_cls=_FakeDeepSeekClient,
    )

    assert reply == "Hello from DeepSeek"
    client = _FakeDeepSeekClient.created[0]
    assert client.api_key == "sk-test"
    assert client.base_url == "https://api.deepseek.com/anthropic"
    assert client.model == "deepseek-v4-pro"
    assert client.calls == [("You are a concise API smoke-test assistant.", "hello", False)]


def test_run_deepseek_smoke_can_bao_thieu_api_key():
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        main.run_deepseek_smoke(
            api_key="",
            base_url="https://api.deepseek.com/anthropic",
            model="deepseek-v4-pro",
            message="hello",
            client_cls=_FakeDeepSeekClient,
        )


def test_describe_deepseek_smoke_error_nhan_dien_insufficient_balance():
    exc = RuntimeError("Error code: 402 - {'error': {'message': 'Insufficient Balance'}}")

    message = main.describe_deepseek_smoke_error(exc)

    assert "đã tới deepseek" in message.lower()
    assert "Insufficient Balance" in message
