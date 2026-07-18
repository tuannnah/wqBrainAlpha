"""Lệnh marathon (chạy nhiều hướng nghiên cứu liên tiếp)."""

from __future__ import annotations


import typer
from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.storage.db import init_db, make_engine, make_session_factory
from src.llm.marathon import MarathonReport, run_marathon
from src.app.cli.common import _cached_symbols, _local_operator_arity, _make_client
from src.app.cli.llm import _make_llm_generator
from src.app.cli.research import _make_research_loop, resolve_direction

console = Console()


def _render_marathon_report(report, deepseek) -> None:
    console.print("\n[bold green]=== Marathon kết thúc ===[/bold green]")
    table = Table(show_header=False)
    table.add_column("", style="cyan")
    table.add_column("", justify="right")
    table.add_row("Lý do dừng", report.stop_reason or "-")
    table.add_row("Hướng hoàn tất", str(report.directions_completed))
    table.add_row("Hướng bỏ qua", str(report.directions_skipped))
    table.add_row("Tổng số sim", str(report.total_sims))
    table.add_row("Alpha vào zoo", str(report.total_zoo_added))
    console.print(table)
    console.print(
        f"[dim]Token: {deepseek.usage.total_tokens} "
        f"(~${deepseek.usage.estimated_cost():.4f})[/dim]"
    )


def _marathon_direction_provider(session_factory):
    """Closure sinh hướng nghiên cứu mới mỗi vòng (LLM tự đề xuất)."""
    from src.simulation.pre_filter import PreFilter

    def _provider():
        f, o, ft, mo, oa = _cached_symbols(session_factory)
        pf = PreFilter(
            known_operators=o or None, known_fields=set(f) or None,
            field_types=ft, matrix_only_ops=mo, operator_arity=oa,
            local_arity=_local_operator_arity(),
        )
        ideas = _make_llm_generator(session_factory, pf).generate_ideas(1)
        direction = resolve_direction("", lambda: ideas)[0]
        console.print(f"\n[cyan]Hướng mới:[/cyan] {direction}")
        return direction

    return _provider


def _marathon_on_event(kind, direction, payload) -> None:
    if kind == "done":
        console.print(
            f"  [green]✓ xong[/green] ({payload.stop_reason}) "
            f"sim={payload.sims_used} zoo+{payload.zoo_added}"
        )
    elif kind == "retry":
        console.print(f"  [yellow]lỗi tạm, retry:[/yellow] {payload}")
    elif kind == "skip":
        console.print(f"  [red]bỏ hướng (lỗi tạm dai dẳng):[/red] {payload}")
    elif kind == "quota":
        console.print("[bold yellow]Hết quota — dừng marathon.[/bold yellow]")


def _run_marathon_session(
    session_factory, client, region, universe, delay,
    decay, truncation, neutralization, per_direction_sims, max_patience, retry,
) -> None:
    """Lõi marathon dùng chung cho lệnh `marathon` và mục menu: dựng loop có trọng
    tài LLM + ConfigTuner, chạy đến khi hết quota (Ctrl+C để dừng tay)."""
    from src.simulation.config import SimConfig

    sim_config = SimConfig(
        region=region, universe=universe, delay=delay,
        decay=decay, truncation=truncation, neutralization=neutralization,
    )
    loop, deepseek = _make_research_loop(
        session_factory, client, region, universe, delay,
        per_direction_sims, max_patience, sim_config=sim_config, marathon=True,
    )
    console.print(
        "[bold cyan]=== Marathon: chạy đến khi hết quota (Ctrl+C để dừng) ===[/bold cyan]"
    )
    try:
        report = run_marathon(
            _marathon_direction_provider(session_factory),
            lambda direction: loop.run(direction),
            max_retries=retry,
            on_event=_marathon_on_event,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Đã dừng marathon (Ctrl+C). Alpha đã sinh vẫn lưu trong DB.[/yellow]")
        report = MarathonReport(stop_reason="interrupted")
    _render_marathon_report(report, deepseek)


def marathon(
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    decay: int = typer.Option(4, "--decay", help="Decay khởi đầu (LLM có thể đổi qua tune_config)"),
    truncation: float = typer.Option(0.01, "--truncation", help="Truncation khởi đầu (LLM có thể đổi)"),
    neutralization: str = typer.Option("MARKET", "--neutralization", help="Neutralization khởi đầu (LLM có thể đổi)"),
    per_direction_sims: int = typer.Option(30, "--per-direction-sims", help="Trần số sim mỗi hướng"),
    max_patience: int = typer.Option(8, "--max-patience", help="Trần cứng số vòng không cải thiện mỗi hướng (an toàn)"),
    retry: int = typer.Option(2, "--retry", help="Số lần retry lỗi tạm (timeout/mạng) trước khi bỏ hướng"),
) -> None:
    """Mở kịch trần: chạy liên tục, LLM tự đổi hướng + tự quyết tinh chỉnh/đổi config,
    đến khi hết quota thì dừng (Ctrl+C để dừng tay). Config khởi đầu: decay=4,
    truncation=0.01, neutralization=MARKET."""
    from main import _setup_logging

    _setup_logging()
    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    if not _cached_symbols(session_factory)[0]:
        console.print("[red]Chưa có fields — chạy fetch-fields trước.[/red]")
        raise typer.Exit(code=1)
    client = _make_client()
    client.authenticate()
    _run_marathon_session(
        session_factory, client, region, universe, delay,
        decay, truncation, neutralization, per_direction_sims, max_patience, retry,
    )
