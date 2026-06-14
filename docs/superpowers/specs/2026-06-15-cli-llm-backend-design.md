# Thiết kế: Backend LLM qua CLI (Claude / Codex)

**Ngày:** 2026-06-15
**Bối cảnh:** Không có (hoặc không muốn tốn) DeepSeek API key. Dùng chính CLI
`claude` (Claude Code headless) hoặc `codex` (Codex exec) làm "LLM" để chạy
research/auto/llm-* **tự động, không cần trực tay**. Bước `simulate` vẫn gọi
WorldQuant BRAIN thật.

Khác với `agent` file-bridge (đứng chờ người ghi `resp_NNN.json`), backend này
**tự gọi CLI như tiến trình con** và trả kết quả ngay.

## Mục tiêu

Cho `LLM_BACKEND=claude-cli` (hoặc `codex-cli`) trong `.env` → mọi module gọi LM
(hypothesis/translator/refiner/alignment/router/generator) chạy qua CLI mà
**không sửa nghiệp vụ**. Chọn engine bằng biến môi trường.

## Nguyên tắc

Mọi module gọi LM qua đúng một interface
`complete(system, user, json_mode=True, task=None) -> str` (xem
`src/llm/deepseek_client.py`). Chỉ cần một client mới **cùng hình dạng**.

## Thành phần

### 1. `CliLLMClient` (`src/llm/cli_client.py`)

Cùng hình dạng `DeepSeekClient`:

- `__init__(self, argv_builder, *, cwd=None, timeout_s=180, model="cli", runner=None)`
  - `argv_builder(system, user, json_mode) -> (argv: list[str], stdin: str | None)`
    — tách phần "dựng lệnh" khỏi phần "chạy" để test riêng từng cái.
  - `runner(argv, stdin, cwd, timeout_s) -> str` — chạy subprocess, trả stdout;
    raise nếu exit ≠ 0 hoặc quá hạn. Inject được để test không spawn thật.
    Mặc định = runner dựa trên `subprocess.run`.
  - `cwd`: thư mục trung lập (mặc định = thư mục tạm) để CLI **không** nuốt
    `CLAUDE.md`/hooks/auto-memory của chính dự án này.
- `complete(system, user, json_mode=True, task=None) -> str`:
  1. `argv, stdin = argv_builder(system, user, json_mode)`.
  2. `out = runner(argv, stdin, cwd, timeout_s)`.
  3. Nếu `json_mode`: `data = extract_json(out)`; có dict/list → trả
     `json.dumps(data, ensure_ascii=False)` (sạch như DeepSeek); không thì trả
     `out` thô (để caller tự `extract_json`/repair).
  4. `_track_usage(system, user, out)`: tái dùng `Usage`, đếm thô
     `prompt += len(system+user)//4`, `completion += len(out)//4`. Backend coi
     như miễn phí; `estimated_cost()` không có ý nghĩa thật nhưng không vỡ render.
- Lỗi: `runner` raise → để propagate kèm thông điệp rõ (CLI nào, exit code,
  stderr tóm tắt). Không tự retry (vòng repair của caller đã lo).

### 2. Hai argv_builder (cùng file)

- `build_claude_argv(system, user, json_mode, *, bin="claude")`:
  `[bin, "-p", user, "--append-system-prompt", system + JSON_HINT,
   "--disable-slash-commands"]`, `stdin=None`.
  - **Không** dùng `--bare` (ép `ANTHROPIC_API_KEY`; ta dùng auth subscription).
- `build_codex_argv(system, user, json_mode, *, bin="codex")`:
  Codex không có cờ system riêng → ghép: `prompt = system + "\n\n" + user + JSON_HINT`.
  `[bin, "exec", prompt]`, `stdin=None`.
- `JSON_HINT` (khi `json_mode`): câu nhắc "Chỉ in DUY NHẤT một khối JSON hợp lệ,
  không kèm giải thích/markdown."

### 3. Factory `make_cli_client(backend, settings) -> CliLLMClient`

Map `backend ∈ {"claude-cli", "codex-cli"}` → chọn argv_builder + bin tương ứng,
lấy `timeout_s` từ settings, `cwd` = thư mục tạm.

### 4. Đấu nối (`config/settings.py` + `main.py`)

- `config/settings.py`: `llm_backend` nhận thêm `"claude-cli" | "codex-cli"`.
  Thêm: `llm_cli_timeout_s: int = 180`, `claude_bin: str = "claude"`,
  `codex_bin: str = "codex"`. (Biến .env tương ứng: `LLM_CLI_TIMEOUT_S`,
  `CLAUDE_BIN`, `CODEX_BIN` — tên = field viết hoa, **không** tiền tố WQ_.)
- `main.py` `_make_deepseek(model=None)`: thêm nhánh — nếu
  `settings.llm_backend in {"claude-cli", "codex-cli"}` → trả
  `make_cli_client(settings.llm_backend, settings)`. Giữ nguyên `deepseek` và
  `agent`.

## TDD (`tests/test_cli_client.py`)

- runner giả trả stdout có fences/preamble → `complete(json_mode=True)` trả JSON
  **sạch** (parse lại được), `.usage` tăng.
- `build_claude_argv` chứa `-p`, `--append-system-prompt`; `build_codex_argv`
  chứa `exec` và prompt ghép system+user.
- runner raise (exit≠0) → `complete()` propagate lỗi.
- `json_mode=False` → trả stdout thô, không cắt.
- Không spawn tiến trình thật (luôn inject runner giả).

## YAGNI

Một lượt gọi/đáp, không stream, không retry, không song song. Giữ `agent`
file-bridge cũ nguyên vẹn (vẫn dùng được khi cần can thiệp tay). Đường DeepSeek
thật không đổi.
