# Thiết kế: Cầu nối LM qua file (AgentBridgeClient)

**Ngày:** 2026-06-14
**Bối cảnh:** Không có Anthropic API key. Tạm thay backend LLM (DeepSeek) bằng
chính Claude agent này, để chạy thử vòng `research` mà không tốn quota LLM thật.

## Mục tiêu

Cho phép chạy `python main.py research ...` với LM do agent đóng vai, **không sửa**
các module nghiệp vụ (hypothesis/translator/refiner/alignment/router). Bước
`simulate` vẫn gọi WorldQuant BRAIN thật.

## Nguyên tắc

Mọi module gọi LM qua đúng một interface: `complete(system, user, json_mode, task) -> str`
(xem `src/llm/deepseek_client.py`). Vì vậy chỉ cần một client mới **cùng hình dạng**
là đủ — không đụng nghiệp vụ.

## Thành phần

### 1. `AgentBridgeClient` (`src/llm/agent_bridge.py`)

Cùng hình dạng `DeepSeekClient`:

- `__init__(bridge_dir, model="agent", timeout_s=600, poll_interval_s=1.0, clock=None, sleep=None)`
  - `clock`/`sleep` tiêm được để test timeout không phải chờ thật.
- `complete(system, user, json_mode=True, task=None) -> str`:
  1. Tăng bộ đếm `n` (001, 002, ...).
  2. Ghi `bridge_dir/req_{n}.json` gồm `{"n", "system", "user", "json_mode", "task"}`.
  3. In marker ra **stdout**: `[[LLM_REQUEST {n}]] req_{n}.json` (để Monitor bắt).
  4. Poll: cứ `poll_interval_s` kiểm tra `bridge_dir/resp_{n}.json`. Khi có → đọc
     trường `content` (string), xoá/giữ file tuỳ, cập nhật `.usage`, trả về `content`.
  5. Quá `timeout_s` mà không có file → `raise TimeoutError`.
- `.usage`: tái dùng `Usage` của `deepseek_client`. Token đếm thô:
  `prompt_tokens += len(system+user)//4`, `completion_tokens += len(content)//4`
  (chỉ để render `_render_research_result` không vỡ; không cần chính xác).

Định dạng file (UTF-8, JSON):
- request `req_001.json`: `{"n": 1, "system": "...", "user": "...", "json_mode": true, "task": "translate"}`
- response `resp_001.json`: `{"content": "{...}"}` — `content` là **chuỗi** (thường là JSON string khi `json_mode`).

### 2. Đấu nối (`config/settings.py` + `main.py`)

- `settings.llm_backend` (mặc định `"deepseek"`), `settings.llm_bridge_dir` (mặc định `"llm_bridge"`).
- Sửa `_make_deepseek(model=None)` trong `main.py`: nếu `settings.llm_backend == "agent"`
  → trả `AgentBridgeClient(settings.llm_bridge_dir)`; ngược lại giữ nguyên đường DeepSeek.
- Bật qua `.env`: `WQ_LLM_BACKEND=agent`. Không đặt thì hành vi cũ y nguyên.

### 3. Chạy thử

`python main.py research --direction "..." --max-sims 1 --no-align` chạy nền.
Agent canh marker `[[LLM_REQUEST` bằng Monitor, mỗi request: đọc `req_{n}.json`,
soạn JSON đóng vai LLM (giả thuyết → công thức), ghi `resp_{n}.json`. Tiến trình
tự chạy tiếp. `simulate` gọi WQ Brain thật (cần đăng nhập, có thể quét QR).

## TDD

- Đặt sẵn `resp_001.json` trong thư mục tạm → `complete()` trả đúng `content`, `.usage` tăng.
- Tiêm `clock`/`sleep` giả: không có file response → `complete()` raise `TimeoutError` sau ngưỡng.
- Marker được in ra stdout (capture qua `capsys`).

## YAGNI

Một chiều file in/out, không retry, không stream, không xoá thư mục tự động.
Đủ để chạy thử 1 lần. Đường DeepSeek thật không thay đổi.
