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


def _make_client() -> WQBrainClient:
    # Ưu tiên .env nếu đã điền; nếu trống thì nhập tương tác trong PowerShell.
    email = settings.wq_email
    password = settings.wq_password
    if not email or not password:
        email, password = prompt_credentials()
    return WQBrainClient(email, password)


@app.command()
def login(force: bool = typer.Option(False, help="Đăng nhập lại dù session còn hạn")) -> None:
    """Đăng nhập (dùng session cũ nếu còn hạn)."""
    _setup_logging()
    client = _make_client()
    client.authenticate(force=force)
    console.print("[green]OK[/green]")


@app.command("probe-fields")
def probe_fields(
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
) -> None:
    """Gọi /data-fields THẬT và in nguyên JSON 1 trang để kiểm tra format."""
    _setup_logging()
    client = _make_client()
    client.authenticate()
    resp = client.get(
        "/data-fields",
        params={
            "instrumentType": "EQUITY",
            "region": region,
            "universe": universe,
            "delay": delay,
            "limit": 5,
            "offset": 0,
        },
    )
    console.print(f"[dim]HTTP {resp.status_code}[/dim]")
    console.print_json(resp.text)


@app.command("fetch-fields")
def fetch_fields(
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    reload: bool = typer.Option(False, "--reload", help="Ép tải lại từ API (ghi đè cache)"),
) -> None:
    """Fetch một lần (bỏ qua nếu đã cache). --reload để ép tải lại."""
    _setup_logging()
    from src.data.fields import FieldFetchError

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    client = _make_client()
    client.authenticate()
    repo = FieldRepository(client, session_factory)
    try:
        fields = repo.get_fields(region, universe, delay, force_reload=reload)
    except FieldFetchError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]{len(fields)} fields[/green] cho {region}/{universe}/delay={delay}")


@app.command("cache-status")
def cache_status() -> None:
    """Xem trạng thái cache (các tổ hợp đã fetch)."""
    _setup_logging()
    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    states = FieldRepository(None, session_factory).all_states()
    table = Table(title="Trạng thái cache")
    table.add_column("Tổ hợp")
    table.add_column("Số field", justify="right")
    table.add_column("Cập nhật")
    table.add_column("Trạng thái")
    for s in states:
        table.add_row(
            f"{s.region}/{s.universe}/delay={s.delay}",
            str(s.total_count or 0),
            s.fetched_at.strftime("%Y-%m-%d %H:%M") if s.fetched_at else "-",
            s.status or "-",
        )
    console.print(table)


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


@app.command("list-fields")
def list_fields(
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    dataset: str = typer.Option(None, help="Lọc theo dataset id"),
    search: str = typer.Option(None, help="Tìm trong id/mô tả"),
    limit: int = typer.Option(50, help="Số dòng hiển thị"),
) -> None:
    """Xem các data field đã tải về (trong DB), có lọc/tìm kiếm."""
    _setup_logging()
    from src.storage.models import DataFieldModel

    engine = init_db(make_engine())
    session = make_session_factory(engine)()
    try:
        query = session.query(DataFieldModel).filter_by(
            region=region, universe=universe, delay=delay
        )
        if dataset:
            query = query.filter(DataFieldModel.dataset_id == dataset)
        if search:
            like = f"%{search}%"
            query = query.filter(
                DataFieldModel.id.like(like) | DataFieldModel.description.like(like)
            )
        total = query.count()
        rows = query.order_by(DataFieldModel.id).limit(limit).all()
    finally:
        session.close()

    table = Table(title=f"Fields {region}/{universe}/delay={delay} — {total} field (hiện {len(rows)})")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Dataset")
    table.add_column("Mô tả", overflow="fold")
    for r in rows:
        table.add_row(r.id, r.type or "-", r.dataset_id or "-", (r.description or "")[:90])
    console.print(table)
    if total > len(rows):
        console.print(f"[dim]... còn {total - len(rows)} field. Dùng --limit/--search/--dataset để lọc.[/dim]")


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


class _WizardState:
    """Giữ phiên đăng nhập + DB cho luồng wizard chạy theo từng bước."""

    def __init__(self):
        engine = init_db(make_engine())
        self.session_factory = make_session_factory(engine)
        self.client: WQBrainClient | None = None
        self.region = settings.default_region
        self.universe = settings.default_universe
        self.delay = settings.default_delay

    @property
    def logged_in(self) -> bool:
        return self.client is not None and self.client.authenticated

    def fields_count(self) -> int:
        return FieldRepository(None, self.session_factory).cached_count(
            self.region, self.universe, self.delay
        )

    def operators_count(self) -> int:
        return OperatorRepository(None, self.session_factory).cached_count()


def _ask(prompt: str, default: str = "") -> str:
    raw = input(prompt).strip()
    return raw or default


def _wizard_login(state: _WizardState) -> None:
    client = _make_client()  # tự nhập email/mật khẩu nếu .env trống
    client.authenticate()  # tự xử lý QR nếu cần
    state.client = client


def _wizard_fields(state: _WizardState) -> None:
    from src.data.fields import FieldFetchError

    repo = FieldRepository(state.client, state.session_factory)
    scope = f"{state.region}/{state.universe}/delay={state.delay}"
    cached = repo.cached_count(state.region, state.universe, state.delay)
    force = False
    if cached > 0:
        choice = _ask(f"Đã có {cached} field ({scope}). [1] Dùng lại  [2] Tải mới (Enter=1): ", "1")
        if choice != "2":
            console.print(f"[green]Dùng lại {cached} field đã lưu.[/green]")
            return
        force = True
    try:
        fields, fetched = repo.ensure(state.region, state.universe, state.delay, force=force)
    except FieldFetchError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    console.print(f"[green]{'Đã tải mới' if fetched else 'Dùng cache'}: {len(fields)} field[/green]")


def _wizard_operators(state: _WizardState) -> None:
    from src.data.operators import OperatorFetchError

    repo = OperatorRepository(state.client, state.session_factory)
    cached = repo.cached_count()
    force = False
    if cached > 0:
        choice = _ask(f"Đã có {cached} operator. [1] Dùng lại  [2] Tải mới (Enter=1): ", "1")
        if choice != "2":
            console.print(f"[green]Dùng lại {cached} operator đã lưu.[/green]")
            return
        force = True
    try:
        operators, fetched = repo.ensure(force=force)
    except OperatorFetchError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    console.print(
        f"[green]{'Đã tải mới' if fetched else 'Dùng cache'}: {len(operators)} operator[/green]"
    )


def _wizard_simulate(state: _WizardState) -> None:
    expr = _ask("Biểu thức FASTEXPR (vd rank(close)): ")
    if not expr:
        return
    sim = Simulator(state.client)
    result = sim.simulate(
        expr, settings={"region": state.region, "universe": state.universe, "delay": state.delay}
    )
    AlphaRepository(state.session_factory).save_simulation(
        result, region=state.region, universe=state.universe
    )
    table = Table(title=f"Simulation: {expr}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("status", result.status)
    for key, value in result.metrics().items():
        table.add_row(key, "—" if value is None else f"{value:.4f}")
    console.print(table)


def _wizard_generate(state: _WizardState, count: int) -> None:
    from src.generation.template import TemplateGenerator
    from src.simulation.pre_filter import PreFilter

    fields, operators = _cached_symbols(state.session_factory)
    pf = PreFilter(known_operators=operators or None, known_fields=set(fields))
    gen = TemplateGenerator(fields, pf)
    alphas = gen.generate(count)
    repo = AlphaRepository(state.session_factory)
    for expr in alphas:
        repo.save_alpha(expr, source="template")
    console.print(f"[green]Đã sinh {len(alphas)} alpha.[/green]")


def _wizard_run_ga(state: _WizardState) -> None:
    import random

    from src.generation.ast_utils import to_expression
    from src.generation.template import TemplateGenerator
    from src.optimization.evolution import GeneticOptimizer
    from src.simulation.pre_filter import PreFilter

    pop = _ask("Population (Enter=30): ", "30")
    gens = _ask("Generations (Enter=10): ", "10")
    fields, operators = _cached_symbols(state.session_factory)
    pf = PreFilter(known_operators=operators or None, known_fields=set(fields))
    tgen = TemplateGenerator(fields, pf, rng=random.Random())
    sim = Simulator(state.client)

    def seed_factory():
        exprs = tgen.generate(1)
        return GeneticOptimizer.expr_to_node(exprs[0] if exprs else f"rank({fields[0]})")

    opt = GeneticOptimizer(
        simulator=sim,
        prefilter=pf,
        seed_factory=seed_factory,
        fields=fields,
        population_size=int(pop) if pop.isdigit() else 30,
        generations=int(gens) if gens.isdigit() else 10,
    )
    best = opt.run()
    repo = AlphaRepository(state.session_factory)
    for node in best[:10]:
        repo.save_alpha(to_expression(node), source="ga")
    console.print(f"[green]GA xong — best: {opt.history[-1].best_expression}[/green]")


def _wizard_list_fields(state: _WizardState) -> None:
    from src.storage.models import DataFieldModel

    search = _ask("Tìm (Enter=tất cả): ")
    session = state.session_factory()
    try:
        query = session.query(DataFieldModel).filter_by(
            region=state.region, universe=state.universe, delay=state.delay
        )
        if search:
            like = f"%{search}%"
            query = query.filter(
                DataFieldModel.id.like(like) | DataFieldModel.description.like(like)
            )
        total = query.count()
        rows = query.order_by(DataFieldModel.id).limit(50).all()
    finally:
        session.close()

    table = Table(title=f"Fields {state.region}/{state.universe}/delay={state.delay} — {total} (hiện {len(rows)})")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Dataset")
    table.add_column("Mô tả", overflow="fold")
    for r in rows:
        table.add_row(r.id, r.type or "-", r.dataset_id or "-", (r.description or "")[:90])
    console.print(table)
    if total > len(rows):
        console.print(f"[dim]... còn {total - len(rows)} field, gõ từ khóa để lọc.[/dim]")


def _wizard_scope(state: _WizardState) -> None:
    state.region = _ask(f"Region (Enter={state.region}): ", state.region)
    state.universe = _ask(f"Universe (Enter={state.universe}): ", state.universe)
    delay_raw = _ask(f"Delay (Enter={state.delay}): ", str(state.delay))
    try:
        state.delay = int(delay_raw)
    except ValueError:
        console.print("[yellow]Delay không hợp lệ, giữ nguyên.[/yellow]")
    console.print(f"[cyan]Scope: {state.region}/{state.universe}/delay={state.delay}[/cyan]")


def _wizard_menu(state: _WizardState) -> None:
    def lock(ok: bool) -> str:
        return "" if ok else " [dim](khóa: cần bước trước)[/dim]"

    fields_ok = state.fields_count() > 0
    console.print("\n[bold cyan]=== WQ Auto-Alpha — Chạy theo từng bước ===[/bold cyan]")
    console.print(f"Scope: [cyan]{state.region}/{state.universe}/delay={state.delay}[/cyan]")
    status = "[green]✓ đã đăng nhập[/green]" if state.logged_in else "[red]✗ chưa đăng nhập[/red]"
    console.print(
        f"Trạng thái: {status} | fields: {state.fields_count()} | "
        f"operators: {state.operators_count()}"
    )
    console.print(" 1) Đăng nhập")
    console.print(f" 2) Tải data fields{lock(state.logged_in)}")
    console.print(f" 3) Tải operators{lock(state.logged_in)}")
    console.print(f" 4) Mô phỏng một biểu thức{lock(state.logged_in)}")
    console.print(f" 5) Sinh alpha (template){lock(fields_ok)}")
    console.print(f" 6) Chạy Genetic Algorithm{lock(state.logged_in and fields_ok)}")
    console.print(f" 7) Xem fields đã tải{lock(fields_ok)}")
    console.print(" 8) Đổi scope (region/universe/delay)")
    console.print(" 0) Thoát")


@app.command()
def start() -> None:
    """Chạy theo từng bước (wizard): đăng nhập → tải fields → ... giữ phiên đăng nhập."""
    _setup_logging()
    from src.data.client import AuthError

    state = _WizardState()
    while True:
        _wizard_menu(state)
        choice = _ask("\nChọn: ")
        try:
            if choice == "1":
                _wizard_login(state)
            elif choice == "0":
                break
            elif choice == "8":
                _wizard_scope(state)
            elif choice == "7":
                if state.fields_count() == 0:
                    console.print("[yellow]Chưa có field nào — tải ở bước 2 trước.[/yellow]")
                else:
                    _wizard_list_fields(state)
            elif choice in {"2", "3", "4"} and not state.logged_in:
                console.print("[yellow]Hãy đăng nhập (1) trước.[/yellow]")
            elif choice == "2":
                _wizard_fields(state)
            elif choice == "3":
                _wizard_operators(state)
            elif choice == "4":
                _wizard_simulate(state)
            elif choice == "5":
                if state.fields_count() == 0:
                    console.print("[yellow]Cần tải data fields (2) trước khi sinh alpha.[/yellow]")
                else:
                    count = _ask("Số lượng (Enter=100): ", "100")
                    _wizard_generate(state, int(count) if count.isdigit() else 100)
            elif choice == "6":
                if not (state.logged_in and state.fields_count() > 0):
                    console.print("[yellow]Cần đăng nhập và tải fields trước.[/yellow]")
                else:
                    _wizard_run_ga(state)
            else:
                console.print("[red]Lựa chọn không hợp lệ.[/red]")
        except AuthError as exc:
            console.print(f"[red]Lỗi đăng nhập: {exc}[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Đã hủy bước hiện tại.[/yellow]")
    console.print("[cyan]Kết thúc.[/cyan]")


if __name__ == "__main__":
    app()
