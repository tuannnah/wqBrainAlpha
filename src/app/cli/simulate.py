"""Lệnh simulate/sweep-config."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.app.cli.common import _make_client
from src.simulation.simulator import Simulator
from src.storage.db import init_db, make_engine, make_session_factory
from src.storage.repository import AlphaRepository

console = Console()


def simulate(
    expr: str = typer.Option(..., help="Biểu thức FASTEXPR"),
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    decay: int = typer.Option(0, "--decay", help="Decay simulation config"),
    truncation: float = typer.Option(0.08, "--truncation", help="Truncation simulation config"),
    neutralization: str = typer.Option("SUBINDUSTRY", "--neutralization", help="Neutralization simulation config"),
) -> None:
    """Chạy một simulation và lưu metrics."""
    from main import _setup_logging

    _setup_logging()
    from src.simulation.config import SimConfig

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    client = _make_client()
    client.authenticate()

    sim_config = SimConfig(
        region=region,
        universe=universe,
        delay=delay,
        decay=decay,
        truncation=truncation,
        neutralization=neutralization,
    )
    sim = Simulator(client)
    result = sim.simulate(expr, settings=sim_config.to_settings())

    repo = AlphaRepository(session_factory)
    repo.save_simulation(result, region=region, universe=universe, config_key=sim_config.key())

    table = Table(title=f"Simulation: {expr}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("status", result.status)
    for key, value in result.metrics().items():
        table.add_row(key, "—" if value is None else f"{value:.4f}")
    console.print(table)


def sweep_config(
    expr: str = typer.Option(..., help="Biểu thức FASTEXPR của alpha tốt cần quét cấu hình"),
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    decays: str = typer.Option("0,2,4,8", "--decays", help="Danh sách decay, phẩy ngăn"),
    truncations: str = typer.Option("0.05,0.08,0.1", "--truncations", help="Danh sách truncation"),
    neutralizations: str = typer.Option(
        "SUBINDUSTRY,INDUSTRY", "--neutralizations", help="Danh sách mức neutralization"
    ),
    oos_ratio: float = typer.Option(0.5, "--oos-ratio", help="Tỉ lệ OOS/IS sharpe tối thiểu"),
) -> None:
    """GĐ5 (T5.3): quét cấu hình (decay/truncation/neutralization) cho một alpha tốt,
    OOS làm trọng tài — chỉ giữ cấu hình tốt cả In-Sample lẫn Out-of-Sample."""
    from main import _setup_logging

    _setup_logging()
    from src.simulation.config import SimConfig
    from src.simulation.sweep import ConfigSweeper

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    client = _make_client()
    client.authenticate()

    grid = {
        "decay": [int(x) for x in decays.split(",") if x.strip()],
        "truncation": [float(x) for x in truncations.split(",") if x.strip()],
        "neutralization": [x.strip() for x in neutralizations.split(",") if x.strip()],
    }
    base = SimConfig.default(region=region, universe=universe, delay=delay)
    n_combos = len(grid["decay"]) * len(grid["truncation"]) * len(grid["neutralization"])
    console.print(f"[cyan]Quét {n_combos} tổ hợp cấu hình cho:[/cyan] {expr}")

    sweeper = ConfigSweeper(Simulator(client))
    res = sweeper.sweep(expr, base, grid, oos_min_ratio=oos_ratio)

    table = Table(title=f"Sweep cấu hình ({len(res.trials)} tổ hợp)")
    table.add_column("decay", justify="right")
    table.add_column("trunc", justify="right")
    table.add_column("neutralize")
    table.add_column("IS sharpe", justify="right")
    table.add_column("OS sharpe", justify="right")
    table.add_column("OOS", justify="center")
    for t in res.trials:
        c = t["config"]
        table.add_row(
            str(c.decay), f"{c.truncation}", c.neutralization,
            "—" if t["sharpe"] is None else f"{t['sharpe']:.3f}",
            "—" if t["os_sharpe"] is None else f"{t['os_sharpe']:.3f}",
            "[green]✓[/green]" if t["oos_ok"] else "[red]✗[/red]",
        )
    console.print(table)

    if res.best_config is None:
        console.print("[yellow]Không cấu hình nào qua kiểm chứng OOS.[/yellow]")
    else:
        c = res.best_config
        console.print(
            f"[bold green]Cấu hình tốt nhất:[/bold green] decay={c.decay}, "
            f"truncation={c.truncation}, neutralization={c.neutralization} "
            f"(IS sharpe={res.best_result.sharpe:.3f}, OS sharpe={res.best_result.os_sharpe:.3f})"
        )
