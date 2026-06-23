"""Chọn hướng nghiên cứu: người dùng nhập -> dùng nguyên; để trống -> LLM tự đề
xuất (giống miner cũ tự seed). Có fallback chuỗi mặc định khi LLM không sinh được."""

from __future__ import annotations

import main

DEFAULT = "mean-reversion theo thanh khoản"


def test_keeps_user_direction_without_calling_llm():
    called = {"n": 0}

    def provider():
        called["n"] += 1
        return ["hướng từ LLM"]

    direction, auto = main.resolve_direction("momentum trên analyst", provider)
    assert direction == "momentum trên analyst"
    assert auto is False
    assert called["n"] == 0  # không gọi LLM khi đã có hướng


def test_auto_direction_when_blank():
    direction, auto = main.resolve_direction("  ", lambda: ["hướng từ LLM", "khác"])
    assert direction == "hướng từ LLM"
    assert auto is True


def test_fallback_when_llm_returns_nothing():
    direction, auto = main.resolve_direction("", lambda: [])
    assert direction == DEFAULT
    assert auto is True


def test_fallback_when_llm_returns_blank():
    direction, auto = main.resolve_direction("", lambda: ["   "])
    assert direction == DEFAULT
    assert auto is True
