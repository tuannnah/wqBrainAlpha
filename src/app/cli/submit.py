"""Lệnh submit alpha lên Brain."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from src.app.cli.common import _make_client
from src.storage.db import init_db, make_engine, make_session_factory

console = Console()


def submit(
    dry_run: bool = typer.Option(True, help="Chỉ liệt kê, không nộp thật"),
    diversify: bool = typer.Option(
        True, "--diversify/--no-diversify",
        help="Loại alpha trùng cấu trúc (AST) với alpha đã chọn trong tập nộp (T7.1)",
    ),
    power_pool: bool = typer.Option(
        False, "--power-pool",
        help="Đường nộp PURE Power Pool: alpha Sharpe>=1.0 không đạt Regular nhưng đạt "
        "cấu trúc PP + khớp theme tuần hiện tại (lịch src/scoring/power_pool_theme.py)",
    ),
) -> None:
    """Chọn và nộp alpha đạt ngưỡng (mặc định dry-run)."""
    # Nhập trễ: _setup_logging còn ở main.py (chưa tách riêng, dùng chung cho mọi lệnh
    # CLI) — import trễ trong thân hàm để tránh vòng import main<->submit.
    from main import _setup_logging

    _setup_logging()
    from src.submission.correlation import CorrelationChecker
    from src.submission.manager import SubmissionManager

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    client = _make_client()
    client.authenticate()

    manager = SubmissionManager(
        client, session_factory, CorrelationChecker(client), diversify=diversify
    )

    if power_pool:
        outcomes = manager.submit_power_pool(dry_run=dry_run)
        title = "Pure Power Pool — dry-run" if dry_run else "Pure Power Pool — đã xử lý"
        pp_table = Table(title=f"{title} ({len(outcomes)} ứng viên)")
        pp_table.add_column("WQ Alpha")
        pp_table.add_column("Sharpe", justify="right")
        pp_table.add_column("Theme")
        pp_table.add_column("Kết quả / lý do bỏ qua", overflow="fold")
        for cand, result in outcomes:
            pp_table.add_row(
                cand.wq_alpha_id,
                f"{cand.sharpe:.2f}" if cand.sharpe is not None else "—",
                "khớp" if cand.theme_ok else "lệch",
                (result.status + (f" ({result.detail})" if result.detail else ""))
                if result is not None
                else (cand.skip_reason or "sẵn sàng (dry-run — nộp bằng --no-dry-run)"),
            )
        console.print(pp_table)
        return

    selected = manager.run_daily(dry_run=dry_run)

    title = "Sẽ nộp (dry-run)" if dry_run else "Đã nộp"
    table = Table(title=f"{title} — {len(selected)} alpha")
    table.add_column("WQ Alpha")
    table.add_column("Expression", overflow="fold")
    table.add_column("Sharpe", justify="right")
    table.add_column("Score", justify="right")
    for c in selected:
        table.add_row(
            c.wq_alpha_id,
            c.expression,
            f"{c.sharpe:.3f}" if c.sharpe is not None else "—",
            f"{c.score:.3f}" if c.score is not None else "—",
        )
    console.print(table)
