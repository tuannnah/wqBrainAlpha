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
from src.data.operators import OperatorRepository, count_positional_arity
from src.data.universe_matrix import iter_scopes
from src.data.warm_cache import warm_cache
from src.optimization.hybrid import HybridEngine
from src.simulation.simulator import Simulator
from src.storage.db import init_db, make_engine, make_session_factory
from src.storage.migrate import migrate_all, _same_database
from src.storage.repository import AlphaRepository, InvalidFieldRepository
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


@app.command("migrate-sqlite")
def migrate_sqlite(
    source: str = typer.Option("sqlite:///wq_alpha.db", help="URL DB nguồn (SQLite)"),
    dest: str = typer.Option("", help="URL DB đích; rỗng = dùng DATABASE_URL"),
) -> None:
    """Copy toàn bộ dữ liệu từ SQLite sang DB đích (Postgres), idempotent."""
    _setup_logging()
    dest_url = dest or settings.database_url
    if _same_database(source, dest_url):
        console.print("[red]❌ DB đích trùng DB nguồn — không có gì để migrate.[/red]")
        raise typer.Exit(code=1)
    counts = migrate_all(make_engine(source), make_engine(dest_url))
    table = Table(title="Đã migrate")
    table.add_column("Bảng")
    table.add_column("Số rows", justify="right")
    for name, n in counts.items():
        table.add_row(name, str(n))
    console.print(table)
    console.print(f"[green]OK[/green] {source} -> {dest_url}")


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


@app.command("warm-cache")
def warm_cache_cmd(
    regions: str = typer.Option("", help="CSV region cần tải; rỗng = tất cả trong WQB_MATRIX"),
    delays: str = typer.Option("0,1", help="CSV delay cần tải"),
    force: bool = typer.Option(False, help="Tải lại tất cả, bỏ qua cache"),
    sleep: float = typer.Option(2.0, help="Giây nghỉ giữa các scope có gọi API"),
) -> None:
    """Tải sẵn toàn bộ datafields + operators vào DB (resume được)."""
    _setup_logging()
    region_list = [r.strip() for r in regions.split(",") if r.strip()] or None
    delay_list = [int(d.strip()) for d in delays.split(",") if d.strip()]

    engine = init_db(make_engine())
    sf = make_session_factory(engine)
    client = _make_client()
    client.authenticate()
    field_repo = FieldRepository(client, sf)
    op_repo = OperatorRepository(client, sf)

    scopes = list(iter_scopes(regions=region_list, delays=delay_list))
    console.print(f"[cyan]Bắt đầu warm-cache {len(scopes)} tổ hợp...[/cyan]")

    def _on_event(kind: str, scope) -> None:
        console.print(f"  [{kind}] {scope[0]}/{scope[1]}/delay={scope[2]}")

    report = warm_cache(
        field_repo, op_repo, scopes, force=force, sleep_s=sleep, on_event=_on_event
    )

    table = Table(title="Kết quả warm-cache")
    table.add_column("Hạng mục")
    table.add_column("Số lượng", justify="right")
    table.add_row("Operators", str(report.operators))
    table.add_row("Fetch mới", str(report.fetched))
    table.add_row("Bỏ qua (đã cache)", str(report.skipped))
    table.add_row("Không quyền", str(report.no_access))
    table.add_row("Lỗi", str(len(report.errors)))
    console.print(table)
    for scope, msg in report.errors:
        console.print(f"[red]  lỗi {scope}: {msg}[/red]")


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
    decay: int = typer.Option(0, "--decay", help="Decay simulation config"),
    truncation: float = typer.Option(0.08, "--truncation", help="Truncation simulation config"),
    neutralization: str = typer.Option("SUBINDUSTRY", "--neutralization", help="Neutralization simulation config"),
) -> None:
    """Chạy một simulation và lưu metrics."""
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
    """Trả (field_ids, operator_names, field_types, matrix_only_ops, operator_arity).

    field_types: id->MATRIX/VECTOR/GROUP để prefilter chặn type mismatch.
    matrix_only_ops: operator Time Series/Cross Sectional đòi input MATRIX.
    operator_arity: name->arity (số input tối đa theo chữ ký) để prefilter chặn
    biểu thức thừa input (lỗi WQ "Invalid number of inputs")."""
    field_repo = FieldRepository(None, session_factory)
    op_repo = OperatorRepository(None, session_factory)
    cached_fields = field_repo.load_cached()
    cached_ops = op_repo.load_cached()
    # Loại field 'chết' (WQ từ chối khi simulate) khỏi nguồn sinh — vùng chết tự học.
    blacklist = InvalidFieldRepository(session_factory).blacklist()
    fields = [f.id for f in cached_fields if f.id and f.id not in blacklist]
    operators = {o.name for o in cached_ops if o.name}
    field_types = {f.id: f.type for f in cached_fields if f.id and getattr(f, "type", None)}
    matrix_only_ops = {
        o.name for o in cached_ops
        if o.name and getattr(o, "category", "") in ("Time Series", "Cross Sectional")
    }
    # Arity positional (bỏ tham số named-only có '=') tính lại từ definition đã lưu
    # -> chặn cả lỗi thừa input lẫn gọi named-param (winsorize/bucket) positional.
    operator_arity = {}
    for o in cached_ops:
        if not o.name:
            continue
        n = count_positional_arity(o.definition or "")
        if n:
            operator_arity[o.name] = n
    return fields, operators, field_types, matrix_only_ops, operator_arity


def _make_invalid_field_recorder(session_factory, region, universe):
    """Trả callback(field_id) ghi field 'chết' vào blacklist (tự học vùng chết)."""
    repo = InvalidFieldRepository(session_factory)

    def record(field_id: str) -> None:
        logger.warning("Field WQ từ chối (chết/event) -> blacklist: {}", field_id)
        repo.record(field_id, region=region, universe=universe, reason="WQ từ chối (chết/event)")

    return record


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
    fields, operators, field_types, matrix_only_ops, operator_arity = _cached_symbols(session_factory)
    if not fields:
        console.print("[red]Chưa có fields trong DB — chạy fetch-fields trước.[/red]")
        raise typer.Exit(code=1)

    pf = PreFilter(
        known_operators=operators or None, known_fields=set(fields),
        field_types=field_types, matrix_only_ops=matrix_only_ops,
        operator_arity=operator_arity,
    )
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
    delay: int = typer.Option(settings.default_delay),
    decay: int = typer.Option(0, "--decay", help="Fixed decay setting for GA simulations"),
    truncation: float = typer.Option(0.08, "--truncation", help="Fixed truncation setting for GA simulations"),
    neutralization: str = typer.Option("SUBINDUSTRY", "--neutralization", help="Fixed neutralization setting for GA simulations"),
    seed_llm: bool = typer.Option(False, "--seed-llm", help="Trộn 50% seed từ DeepSeek"),
    max_sims: int = typer.Option(0, "--max-sims", help="Trần số alpha mô phỏng (0 = không giới hạn, để test pipeline)"),
) -> None:
    """Chạy Genetic Algorithm tối ưu alpha."""
    _setup_logging()
    import random

    from src.generation.template import TemplateGenerator
    from src.optimization.evolution import GeneticOptimizer
    from src.simulation.config import SimConfig
    from src.simulation.pre_filter import PreFilter

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    fields, operators, field_types, matrix_only_ops, operator_arity = _cached_symbols(session_factory)
    if not fields:
        console.print("[red]Chưa có fields — chạy fetch-fields trước.[/red]")
        raise typer.Exit(code=1)

    client = _make_client()
    client.authenticate()
    sim = Simulator(client)
    sim_config = SimConfig(
        region=region,
        universe=universe,
        delay=delay,
        decay=decay,
        truncation=truncation,
        neutralization=neutralization,
    )
    pf = PreFilter(
        known_operators=operators or None, known_fields=set(fields),
        field_types=field_types, matrix_only_ops=matrix_only_ops,
        operator_arity=operator_arity,
    )
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
        simulation_settings=sim_config.to_settings(),
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

    if settings.llm_backend in ("claude-cli", "codex-cli"):
        from src.llm.cli_client import make_cli_client

        return make_cli_client(settings.llm_backend, settings)

    from src.llm.deepseek_client import DeepSeekClient

    if not settings.deepseek_api_key:
        console.print("[red]Thiếu DEEPSEEK_API_KEY trong .env[/red]")
        raise typer.Exit(code=1)
    return DeepSeekClient(
        settings.deepseek_api_key, settings.deepseek_base_url,
        model=model or settings.deepseek_model,
        max_tokens=settings.deepseek_max_tokens,
    )


def run_deepseek_smoke(
    *,
    api_key: str,
    base_url: str,
    model: str,
    message: str = "hello",
    client_cls=None,
) -> str:
    """Gọi chat completion rất ngắn để kiểm tra DeepSeek API."""
    if not api_key.strip():
        raise ValueError("Thiếu DEEPSEEK_API_KEY")
    if not base_url.strip():
        raise ValueError("Thiếu DEEPSEEK_BASE_URL")
    if not model.strip():
        raise ValueError("Thiếu DEEPSEEK_MODEL")

    from src.llm.deepseek_client import DeepSeekClient

    client_cls = client_cls or DeepSeekClient
    client = client_cls(api_key.strip(), base_url.strip().rstrip("/"), model=model.strip())
    return client.complete(
        "You are a concise API smoke-test assistant.",
        message,
        json_mode=False,
    )


def describe_deepseek_smoke_error(exc: Exception) -> str:
    """Diễn giải lỗi smoke check theo ngữ cảnh người dùng cần biết."""
    text = str(exc)
    if "Insufficient Balance" in text or "Error code: 402" in text:
        return (
            "Đã tới DeepSeek, nhưng chat completion bị từ chối vì "
            "Insufficient Balance. Hãy nạp balance hoặc kiểm tra quota của API key."
        )
    return text


@app.command("check-deepseek")
def check_deepseek(
    message: str = typer.Option("hello", "--message", "-m", help="Tin nhắn test gửi tới DeepSeek"),
    model: str = typer.Option("", "--model", help="Ghi đè DEEPSEEK_MODEL cho lần check này"),
) -> None:
    """Gọi DeepSeek chat thật bằng DEEPSEEK_API_KEY/BASE_URL/MODEL."""
    _setup_logging()
    selected_model = model or settings.deepseek_model
    base_url = settings.deepseek_base_url
    if not base_url.rstrip("/").endswith("/anthropic"):
        console.print(
            "[yellow]Cảnh báo:[/yellow] repo này dùng Anthropic-compatible API, "
            "DEEPSEEK_BASE_URL nên là https://api.deepseek.com/anthropic"
        )

    console.print(f"[dim]DEEPSEEK_BASE_URL={base_url}[/dim]")
    console.print(f"[dim]DEEPSEEK_MODEL={selected_model}[/dim]")
    try:
        reply = run_deepseek_smoke(
            api_key=settings.deepseek_api_key,
            base_url=base_url,
            model=selected_model,
            message=message,
        )
    except Exception as exc:
        console.print(f"[red]DeepSeek API check thất bại:[/red] {describe_deepseek_smoke_error(exc)}")
        raise typer.Exit(code=1)

    console.print("[green]DeepSeek API OK[/green]")
    console.print((reply or "").strip() or "[dim]<empty response>[/dim]")


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
    # repo -> bộ sinh hướng đọc phản hồi từ DB (top alpha để khai thác, field yếu tránh).
    return LLMAlphaGenerator(
        deepseek, field_repo, op_repo, prefilter, repo=AlphaRepository(session_factory)
    )


def _make_research_loop(
    session_factory, client, region, universe, delay, max_sims, patience,
    align=True, regularize=False, penalty_lambda=0.3, sim_config=None,
):
    """Lắp RefinementLoop GĐ2 với DeepSeek + Simulator thật. Trả (loop, deepseek)."""
    from src.decorrelation.similarity import avoid_subtree_canons
    from src.decorrelation.zoo import ReferenceZoo
    from src.llm.alignment import AlignmentScorer
    from src.llm.hypothesis import HypothesisGenerator
    from src.llm.loop import RefinementLoop
    from src.llm.refiner import AlphaRefiner
    from src.llm.translator import AlphaTranslator
    from src.simulation.pre_filter import PreFilter

    deepseek = _make_router()  # T6.3: routing tác vụ khó -> model mạnh (nếu cấu hình)
    fields, operators, field_types, matrix_only_ops, operator_arity = _cached_symbols(session_factory)
    pf = PreFilter(
        known_operators=operators or None, known_fields=set(fields) or None,
        field_types=field_types, matrix_only_ops=matrix_only_ops,
        operator_arity=operator_arity,
    )
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
    # T3.6: nhánh con phổ biến trong alpha tốt + bộ khung lặp lại trong các thất
    # bại (duplicate/low_score/sim_error) -> yêu cầu LLM tránh dùng lại để giữ đa
    # dạng và không đi lại vết xe đổ.
    failed_exprs = [f.expression for f in repo.recent_failures(200) if f.expression]
    translator.set_avoid_subtrees(avoid_subtree_canons(passed_exprs, failed_exprs))
    # T4.2: bộ lọc nhất quán giả thuyết–công thức trước sim (bật/tắt qua --align).
    aligner = AlignmentScorer(deepseek) if align else None
    loop = RefinementLoop(
        hypothesis_gen=HypothesisGenerator(deepseek),
        translator=translator,
        refiner=refiner,
        simulator=Simulator(
            client, on_invalid_field=_make_invalid_field_recorder(session_factory, region, universe)
        ),
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
        sim_config=sim_config,
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
    decay: int = typer.Option(0, "--decay", help="Fixed decay setting for research simulations"),
    truncation: float = typer.Option(0.08, "--truncation", help="Fixed truncation setting for research simulations"),
    neutralization: str = typer.Option("SUBINDUSTRY", "--neutralization", help="Fixed neutralization setting for research simulations"),
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
    from src.simulation.config import SimConfig

    sim_config = SimConfig(
        region=region,
        universe=universe,
        delay=delay,
        decay=decay,
        truncation=truncation,
        neutralization=neutralization,
    )
    loop, deepseek = _make_research_loop(
        session_factory, client, region, universe, delay, max_sims, no_improve,
        align, regularize, penalty_lambda, sim_config=sim_config,
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


def _run_ga_with_progress(opt, total):
    """Chạy GeneticOptimizer kèm thanh tiến trình (đếm sim + thế hệ)."""
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
        task = progress.add_task("GA: khởi tạo quần thể...", total=max(1, total))

        def on_sim(n, expr, score):
            progress.update(task, completed=min(n, total))

        def on_gen(stats):
            progress.update(
                task,
                description=f"GA gen {stats.generation} best={stats.best_score:.3f}"[:60],
            )

        return opt.run(on_generation=on_gen, on_simulation=on_sim)


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
    fields, operators, field_types, matrix_only_ops, operator_arity = _cached_symbols(session_factory)
    pf = PreFilter(
        known_operators=operators or None, known_fields=set(fields) or None,
        field_types=field_types, matrix_only_ops=matrix_only_ops,
        operator_arity=operator_arity,
    )
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


def _auto_prepare(client_box: dict, session_factory, region, universe, delay,
                  existing_client=None) -> PrepareInfo:
    """Đăng nhập + ensure fields/operators (cache nếu có). Trả PrepareInfo.

    existing_client: tái dùng phiên đã đăng nhập (vd từ menu) thay vì login lại.
    """
    client = existing_client or _make_client()
    client.authenticate()
    client_box["client"] = client

    field_repo = FieldRepository(client, session_factory)
    fields, _ = field_repo.ensure(region, universe, delay)

    op_repo = OperatorRepository(client, session_factory)
    operators, _ = op_repo.ensure()

    return PrepareInfo(fields=len(fields), operators=len(operators))


def _auto_run_direction_ai(client_box, session_factory, region, universe, delay, per_direction_box, sim_config):
    """Trả callback run_direction cho engine AI."""
    def run(direction: str) -> DirectionOutcome:
        per_direction = per_direction_box["per_direction"]
        loop, _deepseek = _make_research_loop(
            session_factory, client_box["client"], region, universe, delay,
            max_sims=per_direction, patience=3,
            sim_config=sim_config,
        )
        # Hiển thị tiến trình (spinner + đếm sim + pha) để người dùng biết đang làm gì.
        result = _run_research_with_progress(loop, direction, per_direction)
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


def _auto_run_direction_ga(client_box, session_factory, region, universe, delay, per_direction_box, sim_config):
    """Trả callback run_direction cho engine GA."""
    import random

    from src.generation.ast_utils import to_expression
    from src.generation.template import TemplateGenerator
    from src.optimization.evolution import GeneticOptimizer
    from src.simulation.pre_filter import PreFilter

    def run(direction: str) -> DirectionOutcome:
        fields, operators, field_types, matrix_only_ops, operator_arity = _cached_symbols(session_factory)
        pf = PreFilter(
            known_operators=operators or None, known_fields=set(fields),
            field_types=field_types, matrix_only_ops=matrix_only_ops,
            operator_arity=operator_arity,
        )
        tgen = TemplateGenerator(fields, pf, rng=random.Random())
        sim = Simulator(
            client_box["client"],
            on_invalid_field=_make_invalid_field_recorder(session_factory, region, universe),
        )

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
        per_direction = per_direction_box["per_direction"]
        opt = GeneticOptimizer(
            simulator=sim, prefilter=pf, seed_factory=seed_factory, fields=fields,
            population_size=30, generations=10,
            max_simulations=per_direction,
            simulation_settings=sim_config.to_settings() if sim_config is not None else None,
        )
        best_nodes = _run_ga_with_progress(opt, per_direction)
        best_exprs = [to_expression(n) for n in best_nodes]
        passed = passed_from_ga(best_exprs, results)
        return DirectionOutcome(passed=passed, sims_used=opt.simulations_used)
    return run


def _make_refiner(session_factory, prefilter, region, universe, delay):
    """Dựng AlphaRefiner (DeepSeek/router + AlphaTranslator có scope) cho LLM-in-loop."""
    from src.llm.refiner import AlphaRefiner
    from src.llm.translator import AlphaTranslator

    deepseek = _make_router()
    field_repo = FieldRepository(None, session_factory)
    op_repo = OperatorRepository(None, session_factory)
    translator = AlphaTranslator(deepseek, field_repo, op_repo, prefilter)
    translator.set_scope(region=region, universe=universe, delay=delay)
    return AlphaRefiner(deepseek, translator)


def _run_hybrid_with_progress(engine):
    """Chạy HybridEngine kèm thanh tiến trình (đếm sim + thế hệ)."""
    from rich.progress import (
        BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TimeElapsedColumn(), console=console, transient=True,
    ) as progress:
        task = progress.add_task("Hybrid: seed + tiến hóa...", total=None)

        def on_simulation(n, expr, score):
            progress.update(task, description=f"Hybrid: {n} sim, best gần nhất {score:.3f}"[:60])

        def on_generation(stats):
            progress.update(
                task,
                description=f"Hybrid gen {stats.generation} best={stats.best_score:.3f}"[:60],
            )

        return engine.run(on_generation=on_generation, on_simulation=on_simulation)


def _run_auto(region, universe, delay, max_sims=0, generations=0,
              existing_client=None, swallow_errors=False,
              decay=0, truncation=0.08, neutralization="SUBINDUSTRY"):
    """Toàn trình hybrid: login → cache → seed LLM → GA tiến hóa + LLM-in-loop → lưu DB.

    max_sims/generations = 0 nghĩa là VÔ HẠN (None). swallow_errors giữ để tương
    thích chữ ký gọi từ menu; HybridEngine tự nuốt lỗi LLM nên không cần dùng.
    Trả danh sách Node tốt nhất, hoặc None nếu thiếu điều kiện (chưa có fields).
    """
    import random as _random

    from src.generation.ast_utils import to_expression
    from src.generation.template import TemplateGenerator
    from src.decorrelation.zoo import ReferenceZoo
    from src.simulation.config import SimConfig
    from src.simulation.pre_filter import PreFilter

    engine_box = init_db(make_engine())
    session_factory = make_session_factory(engine_box)

    client = existing_client or _make_client()
    if not getattr(client, "authenticated", False):
        client.authenticate()

    fields, operators, field_types, matrix_only_ops, operator_arity = _cached_symbols(session_factory)
    if not fields:
        console.print("[red]Chưa có fields — tải fields (menu 2) trước.[/red]")
        return None

    sim_config = SimConfig(
        region=region, universe=universe, delay=delay,
        decay=decay, truncation=truncation, neutralization=neutralization,
    )
    pf = PreFilter(
        known_operators=operators or None, known_fields=set(fields),
        field_types=field_types, matrix_only_ops=matrix_only_ops,
        operator_arity=operator_arity,
    )
    sim = Simulator(
        client,
        on_invalid_field=_make_invalid_field_recorder(session_factory, region, universe),
    )
    repo = AlphaRepository(session_factory)
    zoo = ReferenceZoo.default(extra=[a.expression for a in repo.zoo(200)])
    tgen = TemplateGenerator(fields, pf, rng=_random.Random())

    engine = HybridEngine(
        simulator=sim, prefilter=pf, fields=fields,
        llm_generator=_make_llm_generator(session_factory, pf),
        refiner=_make_refiner(session_factory, pf, region, universe, delay),
        zoo=zoo, template_generator=tgen,
        max_simulations=max_sims or None, generations=generations or None,
        simulation_settings=sim_config.to_settings(),
    )

    best_nodes = _run_hybrid_with_progress(engine)
    best_exprs = [to_expression(n) for n in best_nodes[:10]]
    for expr in best_exprs:
        repo.save_alpha(expr, source="hybrid")

    table = Table(title=f"Top alpha hybrid ({len(best_exprs)}) — {engine.simulations_used} sim")
    table.add_column("Expression", overflow="fold")
    for expr in best_exprs:
        table.add_row(expr)
    console.print(table)
    console.print(
        "[dim]Đã lưu DB — xem bằng lệnh 'top'. CHƯA nộp; nộp bằng 'submit' khi muốn.[/dim]"
    )
    return best_nodes


@app.command()
def auto(
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    max_sims: int = typer.Option(0, "--max-sims", help="Trần tổng simulation (0 = vô hạn)"),
    generations: int = typer.Option(0, "--generations", help="Số thế hệ GA (0 = vô hạn)"),
    decay: int = typer.Option(0, "--decay", help="Decay simulation config"),
    truncation: float = typer.Option(0.08, "--truncation", help="Truncation simulation config"),
    neutralization: str = typer.Option("SUBINDUSTRY", "--neutralization", help="Neutralization simulation config"),
) -> None:
    """Chạy engine hybrid: login → cache → seed LLM → GA tiến hóa + LLM-in-loop. KHÔNG nộp."""
    _setup_logging()
    if _run_auto(
        region, universe, delay, max_sims=max_sims, generations=generations,
        decay=decay, truncation=truncation, neutralization=neutralization,
    ) is None:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------- menu (start)
class _MenuState:
    """Giữ phiên đăng nhập + DB + scope cho menu."""

    def __init__(self):
        engine = init_db(make_engine())
        self.session_factory = make_session_factory(engine)
        self.client = None
        self.region = settings.default_region
        self.universe = settings.default_universe
        self.delay = settings.default_delay

    @property
    def logged_in(self) -> bool:
        return self.client is not None and self.client.authenticated


def _menu_login(state: _MenuState) -> None:
    client = _make_client()
    client.authenticate()
    state.client = client
    console.print("[green]✓ Đăng nhập xong.[/green]")


def _menu_fields(state: _MenuState) -> None:
    from src.data.fields import FieldFetchError

    repo = FieldRepository(state.client, state.session_factory)
    try:
        fields, fetched = repo.ensure(state.region, state.universe, state.delay)
    except FieldFetchError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    console.print(
        f"[green]{'Đã tải mới' if fetched else 'Dùng cache'}: {len(fields)} field "
        f"({state.region}/{state.universe}/delay={state.delay})[/green]"
    )


def _menu_operators(state: _MenuState) -> None:
    from src.data.operators import OperatorFetchError

    repo = OperatorRepository(state.client, state.session_factory)
    if repo.cached_count() > 0:
        console.print(f"[green]Đã có {repo.cached_count()} operator (dùng cache).[/green]")
        return
    try:
        operators, _ = repo.ensure()
    except OperatorFetchError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    console.print(f"[green]Đã tải {len(operators)} operator.[/green]")


def _menu_ask_sim_settings() -> dict:
    decay_raw = input("Decay [Enter=0]: ").strip()
    truncation_raw = input("Truncation [Enter=0.08]: ").strip()
    neutralization_raw = input("Neutralization [Enter=SUBINDUSTRY]: ").strip()
    return {
        "decay": int(decay_raw or "0"),
        "truncation": float(truncation_raw or "0.08"),
        "neutralization": (neutralization_raw or "SUBINDUSTRY").upper(),
    }


def _print_menu(state: _MenuState) -> None:
    status = "[green]✓ đã đăng nhập[/green]" if state.logged_in else "[red]✗ chưa đăng nhập[/red]"
    console.print("\n[bold cyan]=== WQ Auto-Alpha ===[/bold cyan]")
    console.print(f"Scope: [cyan]{state.region}/{state.universe}/delay={state.delay}[/cyan] | {status}")
    console.print(" 1) Đăng nhập")
    console.print(" 2) Tải data fields (dùng cache nếu có)")
    console.print(" 3) Tải operators (nếu chưa có)")
    console.print(" 4) Chạy toàn trình auto")
    console.print(" 5) Chạy thử luồng (tìm + mô phỏng 1 alpha)")
    console.print(" 0) Thoát")


@app.command()
def start() -> None:
    """Menu đơn giản: đăng nhập → tải fields/operators → chạy auto / thử luồng."""
    _setup_logging()
    from src.data.client import AuthError

    state = _MenuState()
    while True:
        _print_menu(state)
        choice = input("\nChọn: ").strip()
        try:
            if choice == "0":
                break
            elif choice == "1":
                _menu_login(state)
            elif choice in {"2", "3", "4", "5"} and not state.logged_in:
                console.print("[yellow]Hãy đăng nhập (1) trước.[/yellow]")
            elif choice == "2":
                _menu_fields(state)
            elif choice == "3":
                _menu_operators(state)
            elif choice == "4":
                sim_settings = _menu_ask_sim_settings()
                # Hybrid chạy vô hạn, chỉ dừng khi LLM hết token / Ctrl+C.
                _run_auto(
                    state.region, state.universe, state.delay,
                    swallow_errors=True, existing_client=state.client,
                    **sim_settings,
                )
            elif choice == "5":
                sim_settings = _menu_ask_sim_settings()
                console.print("[cyan]Thử luồng: seed + tiến hóa ngắn (trần nhỏ)...[/cyan]")
                _run_auto(
                    state.region, state.universe, state.delay,
                    max_sims=5, generations=2,
                    existing_client=state.client, **sim_settings,
                )
            else:
                console.print("[red]Lựa chọn không hợp lệ.[/red]")
        except AuthError as exc:
            console.print(f"[red]Lỗi đăng nhập: {exc}[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Đã hủy bước hiện tại.[/yellow]")
    console.print("[cyan]Kết thúc.[/cyan]")


if __name__ == "__main__":
    app()
