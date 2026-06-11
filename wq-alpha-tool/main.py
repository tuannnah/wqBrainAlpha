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


def _cached_symbols(session_factory):
    """Trả (field_ids, operator_names) đã cache trong DB."""
    field_repo = FieldRepository(None, session_factory)
    op_repo = OperatorRepository(None, session_factory)
    fields = [f.id for f in field_repo.load_cached() if f.id]
    operators = {o.name for o in op_repo.load_cached() if o.name}
    return fields, operators


@app.command()
def generate(
    method: str = typer.Option("template", help="Phương pháp sinh (hiện hỗ trợ: template)"),
    count: int = typer.Option(100),
) -> None:
    """Sinh alpha hợp lệ qua pre-filter và lưu vào DB."""
    _setup_logging()
    from src.generation.template import TemplateGenerator
    from src.simulation.pre_filter import PreFilter

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    fields, operators = _cached_symbols(session_factory)
    if not fields:
        console.print("[red]Chưa có fields trong DB — chạy fetch-fields trước.[/red]")
        raise typer.Exit(code=1)

    pf = PreFilter(known_operators=operators or None, known_fields=set(fields))
    gen = TemplateGenerator(fields, pf)
    alphas = gen.generate(count)

    repo = AlphaRepository(session_factory)
    for expr in alphas:
        repo.save_alpha(expr, source=method)
    console.print(f"[green]Đã sinh {len(alphas)} alpha[/green] (method={method})")


@app.command("run-ga")
def run_ga(
    population: int = typer.Option(30),
    generations: int = typer.Option(10),
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
) -> None:
    """Chạy Genetic Algorithm tối ưu alpha."""
    _setup_logging()
    import random

    from src.generation.template import TemplateGenerator
    from src.optimization.evolution import GeneticOptimizer
    from src.simulation.pre_filter import PreFilter

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    fields, operators = _cached_symbols(session_factory)
    if not fields:
        console.print("[red]Chưa có fields — chạy fetch-fields trước.[/red]")
        raise typer.Exit(code=1)

    client = _make_client()
    client.authenticate()
    sim = Simulator(client)
    pf = PreFilter(known_operators=operators or None, known_fields=set(fields))
    tgen = TemplateGenerator(fields, pf, rng=random.Random())

    def seed_factory():
        exprs = tgen.generate(1)
        return GeneticOptimizer.expr_to_node(exprs[0] if exprs else f"rank({fields[0]})")

    opt = GeneticOptimizer(
        simulator=sim,
        prefilter=pf,
        seed_factory=seed_factory,
        fields=fields,
        population_size=population,
        generations=generations,
    )
    best = opt.run()

    repo = AlphaRepository(session_factory)
    from src.generation.ast_utils import to_expression

    for node in best[:10]:
        repo.save_alpha(to_expression(node), source="ga")
    console.print(f"[green]GA xong[/green] — best: {opt.history[-1].best_expression}")


@app.command()
def top(
    n: int = typer.Option(20),
    sort: str = typer.Option("score", help="score/sharpe/fitness"),
) -> None:
    """Hiển thị alpha tốt nhất theo simulation đã lưu."""
    _setup_logging()
    from src.storage.models import AlphaModel, SimulationModel

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    session = session_factory()
    try:
        column = {
            "score": SimulationModel.score,
            "sharpe": SimulationModel.sharpe,
            "fitness": SimulationModel.fitness,
        }.get(sort, SimulationModel.score)
        rows = (
            session.query(SimulationModel, AlphaModel)
            .join(AlphaModel, SimulationModel.alpha_id == AlphaModel.id)
            .order_by(column.desc())
            .limit(n)
            .all()
        )
    finally:
        session.close()

    table = Table(title=f"Top {n} alpha (sort={sort})")
    table.add_column("Expression", overflow="fold")
    table.add_column("Sharpe", justify="right")
    table.add_column("Fitness", justify="right")
    table.add_column("Score", justify="right")
    for sim_row, alpha_row in rows:
        table.add_row(
            alpha_row.expression,
            f"{sim_row.sharpe:.3f}" if sim_row.sharpe is not None else "—",
            f"{sim_row.fitness:.3f}" if sim_row.fitness is not None else "—",
            f"{sim_row.score:.3f}" if sim_row.score is not None else "—",
        )
    console.print(table)


if __name__ == "__main__":
    app()
