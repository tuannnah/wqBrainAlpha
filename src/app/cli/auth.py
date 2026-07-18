"""Lệnh đăng nhập WQ Brain."""

from __future__ import annotations

import typer
from rich.console import Console

from src.app.cli.common import _make_client

console = Console()


def prompt_credentials(input_func=input, password_func=None):
    """Nhập email/mật khẩu trực tiếp trong console (mật khẩu ẩn)."""
    import getpass

    password_func = password_func or getpass.getpass
    while True:
        email = input_func("\nEmail WorldQuant BRAIN: ").strip()
        password = password_func("Mật khẩu (ẩn): ")
        if email and password:
            return email, password
        console.print("[red]❌ Email và mật khẩu không được để trống[/red]")


def login(force: bool = typer.Option(False, help="Đăng nhập lại dù session còn hạn")) -> None:
    """Đăng nhập (dùng session cũ nếu còn hạn)."""
    # Nhập trễ: _setup_logging còn ở main.py (chưa tách riêng, dùng chung cho mọi lệnh
    # CLI) — import trễ trong thân hàm để tránh vòng import main<->auth (main.py import
    # module auth ở đầu file, trước cả khi _setup_logging được định nghĩa).
    from main import _setup_logging

    _setup_logging()
    from src.storage.db import write_active_account

    client = _make_client()
    client.authenticate(force=force)
    # Ghi email tài khoản -> các lệnh sau chọn đúng DB theo email (mỗi tài khoản 1 DB).
    if client.email:
        write_active_account(client.email)
    console.print("[green]OK[/green]")
