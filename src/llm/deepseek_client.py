"""DeepSeek client (API tương thích OpenAI) + theo dõi token usage/chi phí."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

# Giá tham khảo USD / 1K token (deepseek-chat) — chỉ để ước lượng, có thể chỉnh.
PRICE_PER_1K_INPUT = 0.00027
PRICE_PER_1K_OUTPUT = 0.0011


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
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        client=None,
    ):
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.usage = Usage()

    def complete(self, system: str, user: str, json_mode: bool = True, task: str | None = None) -> str:
        # `task` chỉ phục vụ định tuyến ở ModelRouter; client đơn lẻ bỏ qua.
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"} if json_mode else None,
            temperature=1.0,
        )
        self._track_usage(resp)
        return resp.choices[0].message.content

    def _track_usage(self, resp) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        self.usage.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.usage.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        logger.debug(
            "DeepSeek usage: +{}/{} tok (tổng {} tok, ~${:.4f})",
            getattr(usage, "prompt_tokens", 0),
            getattr(usage, "completion_tokens", 0),
            self.usage.total_tokens,
            self.usage.estimated_cost(),
        )
