"""DeepSeek client qua Anthropic-compatible API + theo doi token usage/chi phi."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
from loguru import logger

from src.llm.jsonutil import extract_json

# Gia tham khao USD / 1K token - chi de uoc luong, co the chinh.
PRICE_PER_1K_INPUT = 0.00027
PRICE_PER_1K_OUTPUT = 0.0011
# Nhac bang tieng Anh cho on dinh hon voi deepseek-v4-pro (reasoning model).
JSON_HINT = "\n\nReturn ONLY valid JSON, no Markdown, no explanation."


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def estimated_cost(self) -> float:
        return (
            self.prompt_tokens / 1000 * PRICE_PER_1K_INPUT
            + self.completion_tokens / 1000 * PRICE_PER_1K_OUTPUT
        )


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/anthropic",
        model: str = "deepseek-v4-pro",
        max_tokens: int = 4096,
        max_json_retries: int = 2,
        client=None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        # deepseek-v4-pro la reasoning model: no phat mot khoi `thinking` truoc khoi
        # `text`. max_tokens nho -> thinking an het budget, text bi cat -> rong. Vi
        # vay mac dinh phai du rong de con cho cho cau tra loi cuoi.
        self.max_tokens = max_tokens
        self.max_json_retries = max_json_retries
        self.client = client or httpx.Client(base_url=self.base_url, timeout=60.0)
        self.usage = Usage()

    def complete(self, system: str, user: str, json_mode: bool = True, task: str | None = None) -> str:
        # `task` chi phuc vu dinh tuyen o ModelRouter; client don le bo qua.
        system_prompt = system + (JSON_HINT if json_mode else "")
        if not json_mode:
            return self._request(system_prompt, [{"role": "user", "content": user}])

        messages = [{"role": "user", "content": user}]
        last_raw = ""
        for _ in range(self.max_json_retries + 1):
            raw = self._request(system_prompt, messages)
            data = extract_json(raw)
            if data is not None:
                return json.dumps(data, ensure_ascii=False)
            last_raw = raw
            # Gui lai output hong + yeu cau sua, nhac dung tu khoa "not valid JSON".
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON:\n"
                        f"{raw}\n\nReturn ONLY valid JSON, no Markdown."
                    ),
                }
            ]
        raise ValueError(
            "DeepSeek did not return valid JSON after "
            f"{self.max_json_retries + 1} attempts (raw cuoi rong co the do "
            "reasoning 'thinking' an het max_tokens — thu tang max_tokens). "
            f"Raw: {last_raw[:200]!r}"
        )

    def _request(self, system_prompt: str, messages: list[dict]) -> str:
        resp = self.client.post(
            "/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": messages,
                "temperature": 1.0,
            },
        )
        if getattr(resp, "status_code", 200) >= 400:
            raise RuntimeError(f"Error code: {resp.status_code} - {getattr(resp, 'text', '')}")

        data = resp.json()
        self._track_usage(data)
        return self._extract_text(data)

    def _track_usage(self, data: dict) -> None:
        usage = data.get("usage")
        if usage is None:
            return
        self.usage.prompt_tokens += usage.get("input_tokens", 0) or 0
        self.usage.completion_tokens += usage.get("output_tokens", 0) or 0
        logger.debug(
            "DeepSeek usage: +{}/{} tok (tong {} tok, ~${:.4f})",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            self.usage.total_tokens,
            self.usage.estimated_cost(),
        )

    def _extract_text(self, data: dict) -> str:
        content = data.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""
