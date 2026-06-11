"""CLI entry cho WorldQuant Brain Auto-Alpha Tool."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

# Đảm bảo in được tiếng Việt trên console Windows (cp1252) khi output bị pipe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from config.settings import settings
from src.data.client import WQBrainClient
from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository
from src.simulation.simulator import Simulator
from src.storage.db import init_db, make_engine, make_session_factory
from src.storage.repository import AlphaRepository

app = typer.Typer(help="WorldQuant Brain Auto-Alpha Tool")
console = Console()

LOG_DIR = Path("logs")


def _setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(LOG_DIR / "wq_alpha_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="14 days")


def _make_client() -> WQBrainClient:
    if not settings.wq_email or not settings.wq_password:
        console.print("[red]Thiếu WQ_EMAIL / WQ_PASSWORD trong .env[/red]")
        raise typer.Exit(code=1)
    return WQBrainClient(settings.wq_email, settings.wq_password)


@app.command()
def login() -> None:
    """Kiểm tra đăng nhập WQ Brain."""
    _setup_logging()
    client = _make_client()
    client.authenticate()
    console.print("[green]OK[/green] - đăng nhập thành công")


@app.command("fetch-fields")
def fetch_fields(
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
) -> None:
    """Lấy & cache data fields."""
    _setup_logging()
    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    client = _make_client()
    client.authenticate()
    repo = FieldRepository(client, session_factory)
    fields = repo.fetch_all(region, universe, delay)
    console.print(f"[green]Đã lưu {len(fields)} fields[/green] ({region}/{universe}/delay={delay})")


@app.command("fetch-operators")
def fetch_operators() -> None:
    """Lấy & cache operators."""
    _setup_logging()
    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    client = _make_client()
    client.authenticate()
    repo = OperatorRepository(client, session_factory)
    operators = repo.fetch_all()
    console.print(f"[green]Đã lưu {len(operators)} operators[/green]")


@app.command()
def simulate(
    expr: str = typer.Option(..., help="Biểu thức FASTEXPR"),
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
) -> None:
    """Chạy một simulation và lưu metrics."""
    _setup_logging()
    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    client = _make_client()
    client.authenticate()

    sim = Simulator(client)
    result = sim.simulate(expr, settings={"region": region, "universe": universe, "delay": delay})

    repo = AlphaRepository(session_factory)
    repo.save_simulation(result, region=region, universe=universe)

    table = Table(title=f"Simulation: {expr}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("status", result.status)
    for key, value in result.metrics().items():
        table.add_row(key, "—" if value is None else f"{value:.4f}")
    console.print(table)


if __name__ == "__main__":
    app()
