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
from src.pipeline.auto import (
    AutoEvent,
    AutoPipeline,
    DirectionOutcome,
    PassedAlpha,
    PrepareInfo,
    passed_from_ga,
)

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


@app.command("sweep-config")
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
    max_sims: int = typer.Option(0, "--max-sims", help="Trần số alpha mô phỏng (0 = không giới hạn, để test pipeline)"),
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
        max_simulations=max_sims or None,
    )
    best = opt.run()

    repo = AlphaRepository(session_factory)
    from src.generation.ast_utils import to_expression

    for node in best[:10]:
        repo.save_alpha(to_expression(node), source="ga")
    console.print(
        f"[green]GA xong[/green] — {opt.simulations_used} lần mô phỏng — "
        f"best: {opt.history[-1].best_expression}"
    )


def _make_deepseek(model: str | None = None):
    if settings.llm_backend == "agent":
        from src.llm.agent_bridge import AgentBridgeClient

        return AgentBridgeClient(settings.llm_bridge_dir)

    from src.llm.deepseek_client import DeepSeekClient

    if not settings.deepseek_api_key:
        console.print("[red]Thiếu DEEPSEEK_API_KEY trong .env[/red]")
        raise typer.Exit(code=1)
    return DeepSeekClient(
        settings.deepseek_api_key, settings.deepseek_base_url,
        model=model or settings.deepseek_model,
    )


def _make_router():
    """LLM client cho vòng nghiên cứu. Có model mạnh riêng -> ModelRouter định tuyến
    tác vụ khó sang model mạnh (T6.3); không -> dùng một DeepSeekClient."""
    cheap = _make_deepseek()
    if not settings.deepseek_model_strong:
        return cheap
    from src.llm.router import ModelRouter

    strong = _make_deepseek(settings.deepseek_model_strong)
    return ModelRouter(cheap=cheap, strong=strong, default="strong")


def _make_llm_generator(session_factory, prefilter):
    from src.llm.generator import LLMAlphaGenerator

    deepseek = _make_deepseek()
    field_repo = FieldRepository(None, session_factory)
    op_repo = OperatorRepository(None, session_factory)
    return LLMAlphaGenerator(deepseek, field_repo, op_repo, prefilter)


def _make_research_loop(
    session_factory, client, region, universe, delay, max_sims, patience,
    align=True, regularize=False, penalty_lambda=0.3,
):
    """Lắp RefinementLoop GĐ2 với DeepSeek + Simulator thật. Trả (loop, deepseek)."""
    from src.decorrelation.similarity import common_subtrees
    from src.decorrelation.zoo import ReferenceZoo
    from src.llm.alignment import AlignmentScorer
    from src.llm.hypothesis import HypothesisGenerator
    from src.llm.loop import RefinementLoop
    from src.llm.refiner import AlphaRefiner
    from src.llm.translator import AlphaTranslator
    from src.simulation.pre_filter import PreFilter

    deepseek = _make_router()  # T6.3: routing tác vụ khó -> model mạnh (nếu cấu hình)
    fields, operators = _cached_symbols(session_factory)
    pf = PreFilter(known_operators=operators or None, known_fields=set(fields) or None)
    field_repo = FieldRepository(None, session_factory)
    op_repo = OperatorRepository(None, session_factory)
    translator = AlphaTranslator(deepseek, field_repo, op_repo, pf)
    # T6.4: giới hạn fields trong prompt theo đúng region/universe/delay (đa region).
    translator.set_scope(region=region, universe=universe, delay=delay)
    refiner = AlphaRefiner(deepseek, translator)
    repo = AlphaRepository(session_factory)
    # Zoo tham chiếu = Alpha101 + các alpha đã pass trong DB (T3.4/T3.5).
    passed_exprs = [a.expression for a in repo.zoo(200)]
    zoo = ReferenceZoo.default(extra=passed_exprs)
    # T3.6: nhánh con phổ biến trong alpha tốt -> yêu cầu LLM tránh dùng lại.
    translator.set_avoid_subtrees(
        c for c, _ in common_subtrees(passed_exprs, min_count=3, top_n=8)
    )
    # T4.2: bộ lọc nhất quán giả thuyết–công thức trước sim (bật/tắt qua --align).
    aligner = AlignmentScorer(deepseek) if align else None
    loop = RefinementLoop(
        hypothesis_gen=HypothesisGenerator(deepseek),
        translator=translator,
        refiner=refiner,
        simulator=Simulator(client),
        prefilter=pf,
        repo=repo,
        region=region,
        universe=universe,
        delay=delay,
        max_simulations=max_sims,
        no_improve_patience=patience,
        zoo=zoo,
        aligner=aligner,
        regularize=regularize,
        penalty_lambda=penalty_lambda,
    )
    return loop, deepseek


def _render_research_result(result, deepseek) -> None:
    """In kết quả vòng nghiên cứu: giả thuyết, mô tả, biểu thức, điểm, zoo."""
    if result.best_candidate is None:
        console.print("[red]Không sinh được alpha nào (xem failure/log).[/red]")
        return
    cand = result.best_candidate
    vec = result.best_vector
    h = cand.hypothesis

    console.print("\n[bold green]=== Alpha tốt nhất ===[/bold green]")
    htab = Table(show_header=False)
    htab.add_column("", style="cyan")
    htab.add_column("", overflow="fold")
    htab.add_row("Quan sát", h.observation)
    htab.add_row("Nền tảng", h.background)
    htab.add_row("Lý giải KT", h.economic_rationale)
    htab.add_row("Triển khai", h.implementation_spec)
    htab.add_row("Mô tả", cand.description)
    htab.add_row("Biểu thức", f"[bold]{cand.expression}[/bold]")
    console.print(htab)

    stab = Table(title="Điểm đa chiều (chuẩn hoá)")
    for name in ("sharpe", "fitness", "turnover_fit", "drawdown_fit"):
        stab.add_column(name, justify="right")
    stab.add_column("total", justify="right")
    d = vec.dimensions()
    stab.add_row(
        f"{d['sharpe']:.2f}", f"{d['fitness']:.2f}", f"{d['turnover_fit']:.2f}",
        f"{d['drawdown_fit']:.2f}", f"[bold]{vec.total:.3f}[/bold]",
    )
    console.print(stab)

    console.print(
        f"[cyan]{result.sims_used} lần mô phỏng[/cyan] | "
        f"[green]+{result.zoo_added} vào zoo[/green] | "
        f"[yellow]{len(result.failures)} thất bại ghi nhận[/yellow]"
    )
    console.print(
        f"[dim]Token: {deepseek.usage.total_tokens} "
        f"(~${deepseek.usage.estimated_cost():.4f})[/dim]"
    )


@app.command()
def research(
    direction: str = typer.Option(..., "--direction", help="Hướng nghiên cứu (ngôn ngữ tự nhiên)"),
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    max_sims: int = typer.Option(20, "--max-sims", help="Trần số simulation cho cả vòng"),
    no_improve: int = typer.Option(3, "--no-improve", help="Dừng sau N vòng không cải thiện"),
    align: bool = typer.Option(True, "--align/--no-align", help="Bật lọc nhất quán giả thuyết–công thức trước sim (T4.2)"),
    regularize: bool = typer.Option(False, "--regularize/--no-regularize", help="Chọn best theo điểm điều chuẩn (trừ phạt độc đáo/khớp/phức tạp) (T4.4)"),
    penalty_lambda: float = typer.Option(0.3, "--lambda", help="Hệ số λ cho số hạng phạt điều chuẩn"),
    mcts: bool = typer.Option(False, "--mcts/--greedy", help="Dùng MCTS (giữ nhiều nhánh, UCB) thay vòng greedy (T6.1)"),
) -> None:
    """GĐ2: vòng lặp AI — sinh giả thuyết → mô phỏng → tinh chỉnh tham lam."""
    _setup_logging()
    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    if not _cached_symbols(session_factory)[0]:
        console.print("[red]Chưa có fields — chạy fetch-fields trước.[/red]")
        raise typer.Exit(code=1)
    client = _make_client()
    client.authenticate()
    loop, deepseek = _make_research_loop(
        session_factory, client, region, universe, delay, max_sims, no_improve,
        align, regularize, penalty_lambda,
    )
    result = _run_research_with_progress(loop, direction, max_sims, mcts=mcts)
    _render_research_result(result, deepseek)


def _run_research_with_progress(loop, direction, max_sims, mcts=False):
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("sim {task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Sinh giả thuyết...", total=max_sims)

        def on_progress(ev):
            progress.update(
                task,
                completed=min(ev.sims_used, max_sims),
                description=f"[{ev.phase}] best={ev.best_total:.3f} {ev.detail}"[:70],
            )

        if mcts:
            return loop.run_mcts(direction, iterations=max_sims, on_progress=on_progress)
        return loop.run(direction, on_progress=on_progress)


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
def originality(
    expr: str = typer.Option(..., "--expr", help="Biểu thức FASTEXPR cần đo độ độc đáo"),
) -> None:
    """GĐ3: đo độ độc đáo của một alpha so với zoo tham chiếu (Alpha101 + alpha đã pass)."""
    _setup_logging()
    from src.decorrelation.zoo import ReferenceZoo
    from src.storage.repository import AlphaRepository

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    repo = AlphaRepository(session_factory)
    zoo = ReferenceZoo.default(extra=[a.expression for a in repo.zoo(200)])

    score = zoo.originality(expr)
    nearest, ratio = zoo.most_similar(expr)

    table = Table(title="Độ độc đáo (AST vs zoo)")
    table.add_column("", style="cyan")
    table.add_column("", overflow="fold")
    table.add_row("Biểu thức", f"[bold]{expr}[/bold]")
    table.add_row("Kích thước zoo", str(len(zoo)))
    table.add_row("Độ độc đáo", f"[bold]{score:.3f}[/bold]  (1.0 = hoàn toàn độc đáo)")
    table.add_row("Tương đồng cao nhất", f"{ratio:.3f}")
    table.add_row("Alpha gần nhất", nearest or "—")
    console.print(table)
    console.print(
        "[dim]Lưu ý: AST-similarity KHÁC return-correlation thật của WQ — đây chỉ là "
        "bộ lọc rẻ chạy local. Correlation thật kiểm ở bước nộp (GĐ7).[/dim]"
    )


@app.command()
def submit(
    dry_run: bool = typer.Option(True, help="Chỉ liệt kê, không nộp thật"),
    diversify: bool = typer.Option(
        True, "--diversify/--no-diversify",
        help="Loại alpha trùng cấu trúc (AST) với alpha đã chọn trong tập nộp (T7.1)",
    ),
) -> None:
    """Chọn và nộp alpha đạt ngưỡng (mặc định dry-run)."""
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
    if alphas:
        table = Table(title="Alpha vừa sinh")
        table.add_column("#", justify="right")
        table.add_column("Expression", overflow="fold")
        for i, expr in enumerate(alphas, 1):
            table.add_row(str(i), expr)
        console.print(table)
        console.print("[dim]Đã lưu vào DB — xem lại bằng lệnh 'top' hoặc chạy GA (6).[/dim]")


def _wizard_run_ga(state: _WizardState) -> None:
    import random

    from src.generation.ast_utils import to_expression
    from src.generation.template import TemplateGenerator
    from src.optimization.evolution import GeneticOptimizer
    from src.simulation.pre_filter import PreFilter

    mode = _ask(
        "Chế độ GA: [1] Auto (chạy đủ pop×gen)  [2] Test - giới hạn số alpha mô phỏng (Enter=1): ",
        "1",
    )
    max_sims = None
    if mode == "2":
        n = _ask("Số alpha tối đa được mô phỏng: ")
        max_sims = int(n) if n.isdigit() and int(n) > 0 else None
        if max_sims:
            console.print(f"[cyan]Sẽ dừng sau tối đa {max_sims} lần mô phỏng.[/cyan]")

    pop = _ask("Population (Enter=30): ", "30")
    gens = _ask("Generations (Enter=10): ", "10")
    fields, operators = _cached_symbols(state.session_factory)
    pf = PreFilter(known_operators=operators or None, known_fields=set(fields))
    tgen = TemplateGenerator(fields, pf, rng=random.Random())
    sim = Simulator(state.client)

    def seed_factory():
        exprs = tgen.generate(1)
        return GeneticOptimizer.expr_to_node(exprs[0] if exprs else f"rank({fields[0]})")

    gens_int = int(gens) if gens.isdigit() else 10
    opt = GeneticOptimizer(
        simulator=sim,
        prefilter=pf,
        seed_factory=seed_factory,
        fields=fields,
        population_size=int(pop) if pop.isdigit() else 30,
        generations=gens_int,
        max_simulations=max_sims,
    )

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    sim_budget = f"/{max_sims}" if max_sims else ""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("gen {task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Khởi tạo quần thể...", total=gens_int)

        def on_simulation(n, expr, score):
            progress.update(
                task, description=f"Mô phỏng #{n}{sim_budget} — điểm gần nhất {score:.3f}"
            )

        def on_generation(stats):
            progress.update(
                task,
                advance=1,
                description=f"Gen {stats.generation}: best={stats.best_score:.3f} avg={stats.avg_score:.3f}",
            )

        best = opt.run(on_generation=on_generation, on_simulation=on_simulation)

    repo = AlphaRepository(state.session_factory)
    for node in best[:10]:
        repo.save_alpha(to_expression(node), source="ga")
    console.print(
        f"[green]GA xong — {opt.simulations_used} lần mô phỏng — "
        f"best: {opt.history[-1].best_expression}[/green]"
    )
    if opt.history:
        table = Table(title="Tiến độ qua các thế hệ")
        table.add_column("Gen", justify="right")
        table.add_column("Best", justify="right")
        table.add_column("Avg", justify="right")
        table.add_column("Best expression", overflow="fold")
        for s in opt.history:
            avg = "—" if s.avg_score == float("-inf") else f"{s.avg_score:.3f}"
            best_v = "—" if s.best_score == float("-inf") else f"{s.best_score:.3f}"
            table.add_row(str(s.generation), best_v, avg, s.best_expression)
        console.print(table)


def _wizard_research(state: _WizardState) -> None:
    direction = _ask("Hướng nghiên cứu (vd: mean-reversion theo thanh khoản): ")
    if not direction:
        console.print("[yellow]Cần một hướng nghiên cứu.[/yellow]")
        return
    ms = _ask("Trần số mô phỏng (Enter=20): ", "20")
    ni = _ask("Dừng sau N vòng không cải thiện (Enter=3): ", "3")
    max_sims = int(ms) if ms.isdigit() and int(ms) > 0 else 20
    patience = int(ni) if ni.isdigit() and int(ni) > 0 else 3
    loop, deepseek = _make_research_loop(
        state.session_factory, state.client, state.region, state.universe,
        state.delay, max_sims, patience,
    )
    result = _run_research_with_progress(loop, direction, max_sims)
    _render_research_result(result, deepseek)


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
    console.print(f" 7) Nghiên cứu alpha bằng AI (giả thuyết + tinh chỉnh){lock(state.logged_in and fields_ok)}")
    console.print(f" 8) Xem fields đã tải{lock(fields_ok)}")
    console.print(" 9) Đổi scope (region/universe/delay)")
    console.print(" 0) Thoát")


def _auto_prepare(client_box: dict, session_factory, region, universe, delay) -> PrepareInfo:
    """Đăng nhập + ensure fields/operators (cache nếu có). Trả PrepareInfo."""
    client = _make_client()
    client.authenticate()
    client_box["client"] = client

    field_repo = FieldRepository(client, session_factory)
    fields, _ = field_repo.ensure(region, universe, delay)

    op_repo = OperatorRepository(client, session_factory)
    operators, _ = op_repo.ensure()

    return PrepareInfo(fields=len(fields), operators=len(operators))


def _auto_run_direction_ai(client_box, session_factory, region, universe, delay, per_direction_box):
    """Trả callback run_direction cho engine AI."""
    def run(direction: str) -> DirectionOutcome:
        loop, _deepseek = _make_research_loop(
            session_factory, client_box["client"], region, universe, delay,
            max_sims=per_direction_box["per_direction"], patience=3,
        )
        result = loop.run(direction)
        passed: list[PassedAlpha] = []
        cand = result.best_candidate
        if cand is not None and result.zoo_added > 0 and result.best_vector is not None:
            d = result.best_vector.dimensions()
            passed.append(
                PassedAlpha(
                    expression=cand.expression,
                    sharpe=d.get("sharpe"),
                    fitness=d.get("fitness"),
                    direction=direction,
                )
            )
        return DirectionOutcome(passed=passed, sims_used=result.sims_used)
    return run


def _auto_run_direction_ga(client_box, session_factory, region, universe, delay, per_direction_box):
    """Trả callback run_direction cho engine GA."""
    import random

    from src.generation.ast_utils import to_expression
    from src.generation.template import TemplateGenerator
    from src.optimization.evolution import GeneticOptimizer
    from src.simulation.pre_filter import PreFilter

    def run(direction: str) -> DirectionOutcome:
        fields, operators = _cached_symbols(session_factory)
        pf = PreFilter(known_operators=operators or None, known_fields=set(fields))
        tgen = TemplateGenerator(fields, pf, rng=random.Random())
        sim = Simulator(client_box["client"])

        def seed_factory():
            exprs = tgen.generate(1)
            return GeneticOptimizer.expr_to_node(exprs[0] if exprs else f"rank({fields[0]})")

        results: dict = {}
        original_simulate = sim.simulate

        def simulate_capture(expr, **kwargs):
            res = original_simulate(expr, **kwargs)
            results[expr] = res
            return res

        sim.simulate = simulate_capture
        opt = GeneticOptimizer(
            simulator=sim, prefilter=pf, seed_factory=seed_factory, fields=fields,
            population_size=30, generations=10,
            max_simulations=per_direction_box["per_direction"],
        )
        best_nodes = opt.run()
        best_exprs = [to_expression(n) for n in best_nodes]
        passed = passed_from_ga(best_exprs, results)
        return DirectionOutcome(passed=passed, sims_used=opt.simulations_used)
    return run


@app.command()
def auto(
    engine: str = typer.Option("ai", help="ai | ga"),
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    target_passes: int = typer.Option(3, "--target", help="Dừng khi đủ K alpha đạt ngưỡng"),
    max_sims: int = typer.Option(60, "--max-sims", help="Trần cứng tổng số simulation"),
    max_directions: int = typer.Option(5, "--directions", help="Số hướng nghiên cứu tối đa (engine ai)"),
) -> None:
    """Chạy toàn trình: login → cache → tìm/mô phỏng/cải thiện → log. KHÔNG nộp."""
    _setup_logging()
    engine = engine.lower().strip()
    if engine not in {"ai", "ga"}:
        console.print("[red]--engine chỉ nhận 'ai' hoặc 'ga'.[/red]")
        raise typer.Exit(code=1)

    engine_box = init_db(make_engine())
    session_factory = make_session_factory(engine_box)
    client_box: dict = {}
    per_direction_box = {"per_direction": max_sims}

    def prepare() -> PrepareInfo:
        return _auto_prepare(client_box, session_factory, region, universe, delay)

    def propose(n: int) -> list[str]:
        if engine == "ga":
            return [""]
        from src.simulation.pre_filter import PreFilter
        gen = _make_llm_generator(session_factory, PreFilter())
        return gen.generate_ideas(n)

    run_builder = _auto_run_direction_ai if engine == "ai" else _auto_run_direction_ga
    run_direction_raw = run_builder(
        client_box, session_factory, region, universe, delay, per_direction_box
    )

    state = {"sims_used": 0, "dirs_total": 1}

    def run_direction(direction: str) -> DirectionOutcome:
        # Chia trần sim: phần còn lại / số hướng còn lại (hướng đầu không ăn hết).
        remaining = max_sims - state["sims_used"]
        dirs_left = max(1, state["dirs_total"])
        per_direction_box["per_direction"] = max(1, remaining // dirs_left)
        outcome = run_direction_raw(direction)
        state["sims_used"] += outcome.sims_used
        state["dirs_total"] = max(1, state["dirs_total"] - 1)
        return outcome

    def on_event(ev: AutoEvent) -> None:
        if ev.kind == "directions":
            state["dirs_total"] = max(1, len(ev.data.get("directions", [])))
        logger.info("[auto:{}] {} | {}", ev.kind, ev.message, ev.data)
        style = {"stop": "bold green", "prepare": "cyan"}.get(ev.kind, "")
        console.print(f"[{style}]{ev.message}[/{style}]" if style else ev.message)

    pipe = AutoPipeline(
        prepare=prepare,
        propose_directions=propose,
        run_direction=run_direction,
        target_passes=target_passes,
        max_total_sims=max_sims,
        max_directions=max_directions if engine == "ai" else 1,
        on_event=on_event,
    )
    result = pipe.run()

    table = Table(title=f"Alpha đạt ngưỡng ({len(result.passed_alphas)}) — engine={engine}, dừng: {result.stop_reason}")
    table.add_column("Expression", overflow="fold")
    table.add_column("Sharpe", justify="right")
    table.add_column("Fitness", justify="right")
    table.add_column("Hướng nguồn", overflow="fold")
    for p in result.passed_alphas:
        table.add_row(
            p.expression,
            f"{p.sharpe:.3f}" if p.sharpe is not None else "—",
            f"{p.fitness:.3f}" if p.fitness is not None else "—",
            p.direction or "—",
        )
    console.print(table)
    console.print(
        "[dim]Đã lưu DB — xem bằng lệnh 'top'. CHƯA nộp; nộp bằng 'submit' khi muốn.[/dim]"
    )


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
            elif choice == "9":
                _wizard_scope(state)
            elif choice == "8":
                if state.fields_count() == 0:
                    console.print("[yellow]Chưa có field nào — tải ở bước 2 trước.[/yellow]")
                else:
                    _wizard_list_fields(state)
            elif choice == "7":
                if not (state.logged_in and state.fields_count() > 0):
                    console.print("[yellow]Cần đăng nhập và tải fields trước.[/yellow]")
                else:
                    _wizard_research(state)
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
