"""Phát hiện hết quota từ output CLI: phân biệt 'hết quota' (dừng marathon) với
lỗi tạm thời (timeout/mạng -> retry)."""

from __future__ import annotations

import pytest

from src.llm.errors import QuotaExhaustedError, is_quota_error


# ----------------------------------------------------------------- is_quota_error
@pytest.mark.parametrize(
    "text",
    [
        "Claude usage limit reached. Try again later.",
        "Error: rate limit exceeded",
        "You have exceeded your quota",
        "HTTP 429 Too Many Requests",
        "too many requests, slow down",
        "5-hour limit reached",
    ],
)
def test_is_quota_error_nhan_dien_mau_quota(text):
    assert is_quota_error(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Connection timed out",
        "network unreachable",
        "TypeError: bad operand",
        "",
        "some random failure",
    ],
)
def test_is_quota_error_bo_qua_loi_thuong(text):
    assert is_quota_error(text) is False


def test_is_quota_error_khong_phan_biet_hoa_thuong():
    assert is_quota_error("USAGE LIMIT REACHED") is True


# ----------------------------------------------------- _subprocess_runner phân loại
class _FakeProc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_subprocess_runner_loi_quota_nem_quota_exhausted(monkeypatch):
    import subprocess

    from src.llm import cli_client

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _FakeProc(1, stderr="Claude usage limit reached"),
    )
    with pytest.raises(QuotaExhaustedError):
        cli_client._subprocess_runner(["claude", "-p", "x"], None, ".", 5)


def test_subprocess_runner_loi_thuong_van_nem_runtimeerror(monkeypatch):
    import subprocess

    from src.llm import cli_client

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _FakeProc(1, stderr="connection reset by peer"),
    )
    with pytest.raises(RuntimeError) as exc:
        cli_client._subprocess_runner(["claude", "-p", "x"], None, ".", 5)
    assert not isinstance(exc.value, QuotaExhaustedError)


def test_subprocess_runner_thanh_cong_tra_stdout(monkeypatch):
    import subprocess

    from src.llm import cli_client

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _FakeProc(0, stdout='{"ok": true}'),
    )
    out = cli_client._subprocess_runner(["claude"], None, ".", 5)
    assert out == '{"ok": true}'


def test_quota_exhausted_la_runtimeerror_con():
    """QuotaExhaustedError kế thừa RuntimeError -> code cũ bắt RuntimeError vẫn chạy."""
    assert issubclass(QuotaExhaustedError, RuntimeError)
