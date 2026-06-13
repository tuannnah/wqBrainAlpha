"""Cầu nối LM qua file: Claude agent đóng vai LLM thay DeepSeek.

Cùng hình dạng DeepSeekClient (complete + .usage). Ghi request ra file, in marker
ra stdout, poll chờ file response rồi trả về. Đường DeepSeek thật không liên quan.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.llm.deepseek_client import Usage


class AgentBridgeClient:
    def __init__(
        self,
        bridge_dir: str,
        model: str = "agent",
        timeout_s: float = 600.0,
        poll_interval_s: float = 1.0,
        clock=None,
        sleep=None,
    ):
        self.dir = Path(bridge_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._n = 0
        self.usage = Usage()

    def complete(self, system: str, user: str, json_mode: bool = True, task=None) -> str:
        self._n += 1
        n = f"{self._n:03d}"
        req_path = self.dir / f"req_{n}.json"
        resp_path = self.dir / f"resp_{n}.json"
        req_path.write_text(
            json.dumps(
                {"n": self._n, "system": system, "user": user,
                 "json_mode": json_mode, "task": task},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[[LLM_REQUEST {n}]] {req_path.name}", flush=True)

        start = self._clock()
        while True:
            if resp_path.exists():
                content = json.loads(resp_path.read_text(encoding="utf-8"))["content"]
                self.usage.prompt_tokens += len(system + user) // 4
                self.usage.completion_tokens += len(content) // 4
                return content
            if self._clock() - start >= self.timeout_s:
                raise TimeoutError(f"Không có {resp_path.name} sau {self.timeout_s}s")
            self._sleep(self.poll_interval_s)
