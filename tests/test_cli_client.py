"""Test CliLLMClient: gọi CLI (claude/codex) làm LLM, cùng hình dạng DeepSeekClient."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.llm.cli_client import (
    CliLLMClient,
    build_claude_argv,
    build_codex_argv,
    make_cli_client,
)


class _FakeRunner:
    """Ghi lại lời gọi và trả stdout đã định sẵn (không spawn tiến trình thật)."""

    def __init__(self, stdout="", exc=None):
        self.stdout = stdout
        self.exc = exc
        self.calls = []

    def __call__(self, argv, stdin, cwd, timeout_s):
        self.calls.append(SimpleNamespace(argv=argv, stdin=stdin, cwd=cwd, timeout_s=timeout_s))
        if self.exc is not None:
            raise self.exc
        return self.stdout


def _claude_builder(system, user, json_mode):
    return build_claude_argv(system, user, json_mode, bin="claude")


# ----------------------------------------------------------------- complete()
def test_complete_json_mode_tra_json_sach_tu_stdout_lon_xon():
    # stdout có preamble + fences -> complete phải trả JSON sạch parse lại được.
    messy = 'Đây là kết quả:\n```json\n{"ideas": ["a", "b"]}\n```\n'
    runner = _FakeRunner(stdout=messy)
    client = CliLLMClient(_claude_builder, runner=runner)

    out = client.complete("sys", "user", json_mode=True)

    assert json.loads(out) == {"ideas": ["a", "b"]}


def test_complete_json_mode_tang_usage():
    runner = _FakeRunner(stdout='{"expression": "rank(close)"}')
    client = CliLLMClient(_claude_builder, runner=runner)

    client.complete("system dài", "user dài hơn", json_mode=True)

    assert client.usage.prompt_tokens > 0
    assert client.usage.completion_tokens > 0


def test_complete_json_mode_false_tra_stdout_tho():
    runner = _FakeRunner(stdout="văn bản thô không phải JSON")
    client = CliLLMClient(_claude_builder, runner=runner)

    out = client.complete("sys", "user", json_mode=False)

    assert out == "văn bản thô không phải JSON"


def test_complete_truyen_argv_tu_builder_sang_runner():
    runner = _FakeRunner(stdout="{}")
    client = CliLLMClient(_claude_builder, runner=runner)

    client.complete("HỆ THỐNG", "NGƯỜI DÙNG", json_mode=True)

    assert len(runner.calls) == 1
    argv = runner.calls[0].argv
    assert argv[0] == "claude"
    assert "NGƯỜI DÙNG" in argv  # user prompt đi qua -p


def test_complete_loi_runner_propagate():
    runner = _FakeRunner(exc=RuntimeError("CLI exit 1: boom"))
    client = CliLLMClient(_claude_builder, runner=runner)

    with pytest.raises(RuntimeError, match="boom"):
        client.complete("sys", "user")


# ----------------------------------------------------------------- argv builders
def test_build_claude_argv_co_print_va_system():
    argv, stdin = build_claude_argv("HỆ THỐNG", "CÂU HỎI", json_mode=True, bin="claude")

    assert argv[0] == "claude"
    assert "-p" in argv
    assert "CÂU HỎI" in argv
    assert "--append-system-prompt" in argv
    # nội dung system phải nằm trong tham số ngay sau --append-system-prompt
    sys_arg = argv[argv.index("--append-system-prompt") + 1]
    assert "HỆ THỐNG" in sys_arg
    assert stdin is None


def test_build_codex_argv_co_exec_va_ghep_system_user():
    argv, stdin = build_codex_argv("HỆ THỐNG", "CÂU HỎI", json_mode=True, bin="codex")

    assert argv[0] == "codex"
    assert "exec" in argv
    prompt = argv[-1]
    assert "HỆ THỐNG" in prompt and "CÂU HỎI" in prompt


# ----------------------------------------------------------------- factory
def test_make_cli_client_claude_dung_claude_bin():
    settings = SimpleNamespace(llm_cli_timeout_s=99, claude_bin="claude", codex_bin="codex")
    runner = _FakeRunner(stdout="{}")
    client = make_cli_client("claude-cli", settings, runner=runner)

    client.complete("s", "u", json_mode=True)

    assert runner.calls[0].argv[0] == "claude"
    assert runner.calls[0].timeout_s == 99


def test_make_cli_client_codex_dung_codex_bin():
    settings = SimpleNamespace(llm_cli_timeout_s=120, claude_bin="claude", codex_bin="codex")
    runner = _FakeRunner(stdout="{}")
    client = make_cli_client("codex-cli", settings, runner=runner)

    client.complete("s", "u", json_mode=True)

    assert runner.calls[0].argv[0] == "codex"
    assert "exec" in runner.calls[0].argv


def test_make_cli_client_backend_la_raise():
    settings = SimpleNamespace(llm_cli_timeout_s=120, claude_bin="claude", codex_bin="codex")
    with pytest.raises(ValueError):
        make_cli_client("khong-ton-tai", settings)
