"""Backend LLM qua CLI: gọi `claude -p` hoặc `codex exec` như tiến trình con.

Cùng hình dạng DeepSeekClient (complete + .usage) nên mọi module nghiệp vụ dùng
được mà không cần sửa. Bước simulate vẫn gọi WorldQuant BRAIN thật.
"""

from __future__ import annotations

import json
import subprocess
import tempfile

from loguru import logger

from src.llm.deepseek_client import Usage
from src.llm.errors import QuotaExhaustedError, is_quota_error
from src.llm.jsonutil import extract_json

# Nhắc model chỉ in JSON khi json_mode (agent CLI hay chèn lời mở đầu/markdown).
JSON_HINT = (
    "\n\nQUAN TRỌNG: Chỉ in DUY NHẤT một khối JSON hợp lệ, KHÔNG kèm giải thích, "
    "KHÔNG khối mã markdown."
)


def build_claude_argv(
    system: str, user: str, json_mode: bool, *, bin: str = "claude",
    model: str | None = None, effort: str | None = None,
):
    """Argv cho Claude Code headless. Không dùng --bare (nó ép ANTHROPIC_API_KEY).

    model/effort: tùy chọn — `--model` nhận alias ('opus'/'sonnet'/'fable') hoặc tên
    đầy đủ; `--effort` chọn mức suy luận ('high'...). Rỗng/None -> dùng mặc định CLI."""
    sys_prompt = system + (JSON_HINT if json_mode else "")
    argv = [
        bin,
        "-p",
        user,
        "--append-system-prompt",
        sys_prompt,
        "--disable-slash-commands",
    ]
    if model:
        argv += ["--model", model]
    if effort:
        argv += ["--effort", effort]
    return argv, None


def build_codex_argv(
    system: str, user: str, json_mode: bool, *, bin: str = "codex", model: str | None = None,
):
    """Argv cho Codex exec. Codex không có cờ system riêng -> ghép system+user.

    model: tùy chọn -> `--model`; rỗng/None dùng mặc định CLI. (Codex không có effort.)"""
    prompt = system + "\n\n" + user + (JSON_HINT if json_mode else "")
    argv = [bin, "exec"]
    if model:
        argv += ["--model", model]
    argv.append(prompt)
    return argv, None


def _subprocess_runner(argv, stdin, cwd, timeout_s) -> str:
    """Chạy CLI thật, trả stdout. Raise kèm thông điệp rõ nếu exit != 0/timeout."""
    try:
        proc = subprocess.run(
            argv,
            input=stdin,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"CLI '{argv[0]}' quá hạn sau {timeout_s}s") from exc
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()[:500]
        # Hết quota -> QuotaExhaustedError để marathon DỪNG hẳn; lỗi khác (mạng,
        # cú pháp...) -> RuntimeError để marathon retry rồi bỏ hướng.
        if is_quota_error(proc.stderr or ""):
            raise QuotaExhaustedError(f"CLI '{argv[0]}' báo hết quota: {err}")
        raise RuntimeError(f"CLI '{argv[0]}' exit {proc.returncode}: {err}")
    return proc.stdout or ""


class CliLLMClient:
    def __init__(
        self,
        argv_builder,
        *,
        cwd: str | None = None,
        timeout_s: float = 180.0,
        model: str = "cli",
        runner=None,
    ):
        self.argv_builder = argv_builder
        # cwd trung lập -> CLI không nuốt CLAUDE.md/hooks/auto-memory của dự án này.
        self.cwd = cwd or tempfile.gettempdir()
        self.timeout_s = timeout_s
        self.model = model
        self._runner = runner or _subprocess_runner
        self.usage = Usage()

    def complete(self, system: str, user: str, json_mode: bool = True, task=None) -> str:
        argv, stdin = self.argv_builder(system, user, json_mode)
        out = self._runner(argv, stdin, self.cwd, self.timeout_s)
        self._track_usage(system, user, out)
        if json_mode:
            data = extract_json(out)
            if data is not None:
                return json.dumps(data, ensure_ascii=False)
        return out

    def _track_usage(self, system: str, user: str, out: str) -> None:
        self.usage.prompt_tokens += len(system + user) // 4
        self.usage.completion_tokens += len(out) // 4
        logger.debug(
            "CLI usage: +{}/{} tok (tổng {} tok)",
            len(system + user) // 4,
            len(out) // 4,
            self.usage.total_tokens,
        )


def make_cli_client(backend: str, settings, *, runner=None) -> CliLLMClient:
    """Dựng CliLLMClient theo backend ('claude-cli' | 'codex-cli')."""
    timeout_s = getattr(settings, "llm_cli_timeout_s", 300)
    if backend == "claude-cli":
        bin_ = getattr(settings, "claude_bin", "claude")
        cli_model = getattr(settings, "claude_cli_model", "") or None
        effort = getattr(settings, "claude_cli_effort", "") or None
        builder = lambda s, u, j: build_claude_argv(  # noqa: E731
            s, u, j, bin=bin_, model=cli_model, effort=effort
        )
        model = "claude-cli"
    elif backend == "codex-cli":
        bin_ = getattr(settings, "codex_bin", "codex")
        cli_model = getattr(settings, "codex_cli_model", "") or None
        builder = lambda s, u, j: build_codex_argv(s, u, j, bin=bin_, model=cli_model)  # noqa: E731
        model = "codex-cli"
    else:
        raise ValueError(f"backend CLI không hợp lệ: {backend!r}")
    return CliLLMClient(builder, timeout_s=timeout_s, model=model, runner=runner)
