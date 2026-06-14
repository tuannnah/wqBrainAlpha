# Cầu nối LM qua file (AgentBridgeClient) — Plan triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cho phép chạy `python main.py research ...` với LM do chính Claude agent đóng vai (qua file in/out), không sửa module nghiệp vụ; `simulate` vẫn gọi WQ Brain thật.

**Architecture:** Thêm `AgentBridgeClient` cùng hình dạng `DeepSeekClient` (`complete(...)` + `.usage`). Client ghi request ra `llm_bridge/req_NN.json`, in marker ra stdout, poll chờ `resp_NN.json`. Đấu nối qua `settings.llm_backend == "agent"` trong `_make_deepseek` của `main.py`. Đường DeepSeek thật không đổi.

**Tech Stack:** Python, pytest, json, pathlib. Tái dùng `Usage` từ `src/llm/deepseek_client.py`.

---

## File Structure

- Create: `src/llm/agent_bridge.py` — `AgentBridgeClient`, một trách nhiệm: cầu nối LM qua file.
- Create: `tests/test_agent_bridge.py` — test đọc response, timeout, marker.
- Modify: `main.py` — `_make_deepseek` rẽ nhánh theo `settings.llm_backend`.
- Đã xong: `config/settings.py` (`llm_backend`, `llm_bridge_dir`).

---

### Task 1: AgentBridgeClient — đọc response có sẵn

**Files:**
- Create: `src/llm/agent_bridge.py`
- Test: `tests/test_agent_bridge.py`

- [ ] **Step 1: Viết test thất bại**

```python
import json
from pathlib import Path

from src.llm.agent_bridge import AgentBridgeClient


def test_complete_doc_response_co_san(tmp_path):
    # Có sẵn resp_001.json -> complete() đọc và trả về content
    (tmp_path / "resp_001.json").write_text(
        json.dumps({"content": '{"hypothesis": "x"}'}), encoding="utf-8"
    )
    client = AgentBridgeClient(str(tmp_path), poll_interval_s=0.01, timeout_s=2)
    out = client.complete("sys", "usr", json_mode=True, task="translate")

    assert out == '{"hypothesis": "x"}'
    # request được ghi ra để agent đọc
    req = json.loads((tmp_path / "req_001.json").read_text(encoding="utf-8"))
    assert req["system"] == "sys"
    assert req["user"] == "usr"
    assert req["task"] == "translate"
    # usage tăng (thô)
    assert client.usage.total_tokens > 0
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `venv/Scripts/python -m pytest tests/test_agent_bridge.py::test_complete_doc_response_co_san -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.agent_bridge'`

- [ ] **Step 3: Viết implementation tối thiểu**

```python
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
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `venv/Scripts/python -m pytest tests/test_agent_bridge.py::test_complete_doc_response_co_san -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/agent_bridge.py tests/test_agent_bridge.py
git commit -m "feat(cầu nối LM): AgentBridgeClient đọc response qua file"
```

---

### Task 2: Timeout khi thiếu response

**Files:**
- Modify: `tests/test_agent_bridge.py`

- [ ] **Step 1: Viết test thất bại**

```python
import pytest


def test_complete_timeout_khi_thieu_response(tmp_path):
    # clock giả nhảy vọt -> không cần chờ thật
    ticks = iter([0.0, 0.0, 100.0])
    client = AgentBridgeClient(
        str(tmp_path), timeout_s=10, poll_interval_s=0.01,
        clock=lambda: next(ticks), sleep=lambda s: None,
    )
    with pytest.raises(TimeoutError):
        client.complete("sys", "usr")
    # vẫn ghi request ra để agent có thể trả lời ở lần sau
    assert (tmp_path / "req_001.json").exists()
```

- [ ] **Step 2: Chạy test để xác nhận pass (logic đã có ở Task 1)**

Run: `venv/Scripts/python -m pytest tests/test_agent_bridge.py::test_complete_timeout_khi_thieu_response -v`
Expected: PASS (clock thứ 3 = 100.0 ≥ timeout_s=10 → raise). Nếu FAIL, sửa điều kiện timeout trong `complete`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent_bridge.py
git commit -m "test(cầu nối LM): timeout khi thiếu file response"
```

---

### Task 3: Marker in ra stdout

**Files:**
- Modify: `tests/test_agent_bridge.py`

- [ ] **Step 1: Viết test thất bại**

```python
def test_complete_in_marker_stdout(tmp_path, capsys):
    (tmp_path / "resp_001.json").write_text(
        json.dumps({"content": "ok"}), encoding="utf-8"
    )
    client = AgentBridgeClient(str(tmp_path), poll_interval_s=0.01, timeout_s=2)
    client.complete("sys", "usr")
    out = capsys.readouterr().out
    assert "[[LLM_REQUEST 001]]" in out
```

- [ ] **Step 2: Chạy test để xác nhận pass (marker đã in ở Task 1)**

Run: `venv/Scripts/python -m pytest tests/test_agent_bridge.py::test_complete_in_marker_stdout -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent_bridge.py
git commit -m "test(cầu nối LM): marker LLM_REQUEST in ra stdout"
```

---

### Task 4: Đấu nối backend=agent vào main.py

**Files:**
- Modify: `main.py` — hàm `_make_deepseek` (khoảng dòng 394-403)

- [ ] **Step 1: Đọc lại `_make_deepseek` hiện tại**

Run: xem `main.py:394-403`. Hiện trả `DeepSeekClient(...)` luôn.

- [ ] **Step 2: Sửa `_make_deepseek` rẽ nhánh theo backend**

Thay thân hàm `_make_deepseek`:

```python
def _make_deepseek(model: str | None = None):
    if settings.llm_backend == "agent":
        from src.llm.agent_bridge import AgentBridgeClient

        return AgentBridgeClient(settings.llm_bridge_dir)

    from src.llm.deepseek_client import DeepSeekClient

    if not settings.deepseek_api_key:
        console.print("[red]Thiếu DEEPSEEK_API_KEY trong .env[/red]")
        raise typer.Exit(code=1)
    return DeepSeekClient(
        settings.deepseek_api_key, settings.deepseek_base_url,
        model=model or settings.deepseek_model,
    )
```

- [ ] **Step 3: Kiểm tra không vỡ — chạy toàn bộ test**

Run: `venv/Scripts/python -m pytest -q`
Expected: PASS toàn bộ (các test cũ dùng mock, không đụng nhánh agent).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(cầu nối LM): _make_deepseek rẽ nhánh backend=agent"
```

---

### Task 5: Chạy thử research 1 lần qua cầu nối

Không phải task code — thao tác chạy. Làm sau khi Task 1-4 pass.

- [ ] **Step 1:** Đặt `WQ_LLM_BACKEND=agent` (qua môi trường khi chạy lệnh).
- [ ] **Step 2:** Chạy nền: `python main.py research --direction "mean-reversion theo thanh khoản" --max-sims 1 --no-align`. Đăng nhập WQ Brain khi được hỏi (có thể quét QR).
- [ ] **Step 3:** Dùng Monitor canh marker `[[LLM_REQUEST` trong log/stdout.
- [ ] **Step 4:** Mỗi request: Read `llm_bridge/req_NN.json`, soạn JSON đóng vai LLM, Write `llm_bridge/resp_NN.json`.
- [ ] **Step 5:** Đọc kết quả cuối (`_render_research_result`): giả thuyết, biểu thức, điểm, token.

---

## Self-Review

- **Spec coverage:** `AgentBridgeClient` (Task 1-3), đấu nối backend (Task 4), chạy thử (Task 5) — khớp 3 thành phần trong spec.
- **Placeholder scan:** không có TBD/TODO; mọi step có code/lệnh cụ thể.
- **Type consistency:** `complete(system, user, json_mode, task)` và `.usage` (kiểu `Usage`) khớp `DeepSeekClient`. Tên file `req_{n:03d}.json`/`resp_{n:03d}.json` nhất quán giữa client và bước chạy thử.
