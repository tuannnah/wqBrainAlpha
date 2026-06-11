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
    seed_llm: bool = typer.Option(False, "--seed-llm", help="Trộn 50% seed từ DeepSeek"),
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

    llm_pool: list[str] = []
    if seed_llm:
        llm_gen = _make_llm_generator(session_factory, pf)
        ideas = llm_gen.generate_ideas(5)
        for idea in ideas:
            llm_pool.extend(llm_gen.generate(idea, n=2))
        console.print(f"[cyan]LLM seed pool: {len(llm_pool)} alpha[/cyan]")

    def seed_factory():
        if llm_pool and random.random() < 0.5:
            return GeneticOptimizer.expr_to_node(random.choice(llm_pool))
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


def _make_llm_generator(session_factory, prefilter):
    from src.llm.deepseek_client import DeepSeekClient
    from src.llm.generator import LLMAlphaGenerator

    if not settings.deepseek_api_key:
        console.print("[red]Thiếu DEEPSEEK_API_KEY trong .env[/red]")
        raise typer.Exit(code=1)
    deepseek = DeepSeekClient(settings.deepseek_api_key, settings.deepseek_base_url)
    field_repo = FieldRepository(None, session_factory)
    op_repo = OperatorRepository(None, session_factory)
    return LLMAlphaGenerator(deepseek, field_repo, op_repo, prefilter)


@app.command("llm-generate")
def llm_generate(
    idea: str = typer.Option(..., help="Ý tưởng alpha bằng ngôn ngữ tự nhiên"),
    count: int = typer.Option(5),
) -> None:
    """Sinh alpha từ một ý tưởng bằng DeepSeek."""
    _setup_logging()
    from src.simulation.pre_filter import PreFilter

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    fields, operators = _cached_symbols(session_factory)
    pf = PreFilter(known_operators=operators or None, known_fields=set(fields) or None)
    llm_gen = _make_llm_generator(session_factory, pf)

    alphas = llm_gen.generate(idea, n=count)
    repo = AlphaRepository(session_factory)
    for expr in alphas:
        repo.save_alpha(expr, source="llm")
    console.print(f"[green]Đã sinh {len(alphas)} alpha[/green] từ ý tưởng: {idea}")
    for expr in alphas:
        console.print(f"  • {expr}")
    console.print(f"[dim]Token usage: {llm_gen.deepseek.usage.total_tokens} "
                  f"(~${llm_gen.deepseek.usage.estimated_cost():.4f})[/dim]")


@app.command("llm-ideas")
def llm_ideas(count: int = typer.Option(10)) -> None:
    """Cho DeepSeek brainstorm các ý tưởng alpha."""
    _setup_logging()
    from src.simulation.pre_filter import PreFilter

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    pf = PreFilter()
    llm_gen = _make_llm_generator(session_factory, pf)
    ideas = llm_gen.generate_ideas(count)
    for i, idea in enumerate(ideas, 1):
        console.print(f"  {i}. {idea}")


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


@app.command()
def submit(
    dry_run: bool = typer.Option(True, help="Chỉ liệt kê, không nộp thật"),
) -> None:
    """Chọn và nộp alpha đạt ngưỡng (mặc định dry-run)."""
    _setup_logging()
    from src.submission.correlation import CorrelationChecker
    from src.submission.manager import SubmissionManager

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    client = _make_client()
    client.authenticate()

    manager = SubmissionManager(client, session_factory, CorrelationChecker(client))
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


if __name__ == "__main__":
    app()
