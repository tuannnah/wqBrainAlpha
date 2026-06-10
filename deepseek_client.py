"""Client DeepSeek dùng JSON Output, có retry, token usage và redaction."""

import json
import os
from dataclasses import dataclass
from time import sleep


class DeepSeekConfigurationError(RuntimeError):
    """Thiếu cấu hình bắt buộc (ví dụ DEEPSEEK_API_KEY)."""


class DeepSeekResponseError(RuntimeError):
    """DeepSeek không trả về JSON hợp lệ sau số lần thử cho phép."""


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0


@dataclass(frozen=True)
class DeepSeekResult:
    request_type: str
    model: str
    data: dict
    usage: TokenUsage
    raw_response: dict


class DeepSeekClient:
    def __init__(self, config, session, environ=None, sleep_func=sleep):
        self.config = config
        self.session = session
        self.sleep = sleep_func
        environ = environ if environ is not None else os.environ
        self.api_key = environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise DeepSeekConfigurationError(
                "Thiếu biến môi trường DEEPSEEK_API_KEY. "
                "Hãy đặt API key trước khi bắt đầu nghiên cứu."
            )

    @property
    def _url(self):
        return f"{self.config.deepseek_base_url.rstrip('/')}/chat/completions"

    def generate_json(self, request_type, system_prompt, user_payload):
        payload = {
            "model": self.config.deepseek_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.config.deepseek_max_output_tokens,
            "stream": False,
        }
        return self._post_with_retry(request_type, payload)

    def _post_with_retry(self, request_type, payload):
        attempts = self.config.deepseek_max_retries + 1
        backoff = self.config.rate_limit_backoff_seconds
        for attempt in range(attempts):
            last_attempt = attempt == attempts - 1
            response = self.session.post(
                self._url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.deepseek_timeout_seconds,
            )

            if response.status_code == 429:
                if last_attempt:
                    raise DeepSeekResponseError("DeepSeek giới hạn tần suất.")
                self.sleep(float(response.headers.get("Retry-After", backoff)))
                continue

            if response.status_code >= 400:
                if last_attempt:
                    raise DeepSeekResponseError(
                        f"DeepSeek trả về HTTP {response.status_code}."
                    )
                self.sleep(backoff)
                continue

            data = self._parse_content(response.json())
            if data is not None:
                usage = self._usage(response.json().get("usage", {}))
                return DeepSeekResult(
                    request_type=request_type,
                    model=self.config.deepseek_model,
                    data=data,
                    usage=usage,
                    raw_response=response.json(),
                )
            if not last_attempt:
                self.sleep(backoff)

        raise DeepSeekResponseError("DeepSeek không trả về JSON hợp lệ.")

    @staticmethod
    def _parse_content(body):
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _usage(usage):
        usage = usage or {}
        return TokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cache_hit_tokens=int(
                usage.get("prompt_cache_hit_tokens", usage.get("cache_hit_tokens", 0))
            ),
        )
