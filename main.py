"""CLI entry cho WorldQuant Brain Auto-Alpha Tool."""

from __future__ import annotations

import math
import os
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
from src.simulation.simulator import Simulator
from src.storage.db import init_db, make_engine, make_session_factory
from src.storage.migrate import migrate_all, _same_database
from src.storage.repository import AlphaRepository, InvalidFieldRepository
from src.llm.marathon import MarathonReport, run_marathon
from src.pipeline.auto import PrepareInfo

app = typer.Typer(help="WorldQuant Brain Auto-Alpha Tool")
console = Console()

LOG_DIR = Path("logs")


def _setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    # WQ_NO_FILE_LOG: bỏ file sink (conftest đặt khi chạy test) để không ghi
    # đè log production bằng nhiễu fixture.
    if os.environ.get("WQ_NO_FILE_LOG"):
        return
    LOG_DIR.mkdir(exist_ok=True)
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
    from src.storage.db import write_active_account

    client = _make_client()
    client.authenticate(force=force)
    # Ghi email tài khoản -> các lệnh sau chọn đúng DB theo email (mỗi tài khoản 1 DB).
    if client.email:
        write_active_account(client.email)
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
    """Fetch một lần (bỏ qua nếu đã cache). --reload để ép tải lại (ghi đè)."""
    _setup_logging()
    from src.data.fields import FieldFetchError

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    repo = FieldRepository(None, session_factory)
    # Đã cache & không --reload: đọc thẳng từ DB, KHÔNG đăng nhập/gọi API.
    if not reload and repo._is_cached(region, universe, delay):
        fields = repo._load_from_db(region, universe, delay)
        console.print(
            f"[green]Data fields: {len(fields)}[/green] — dùng CACHE, không tải mới "
            f"({region}/{universe}/delay={delay})"
        )
        return
    client = _make_client()
    client.authenticate()
    repo.client = client
    try:
        fields = repo.get_fields(region, universe, delay, force_reload=reload)
    except FieldFetchError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"[green]Data fields: {len(fields)}[/green] — ĐÃ TẢI MỚI từ API "
        f"({region}/{universe}/delay={delay})"
    )


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
def fetch_operators(
    reload: bool = typer.Option(False, "--reload", help="Ép tải lại từ API (ghi đè cache)"),
) -> None:
    """Lấy & cache operators (bỏ qua nếu đã cache). --reload để ép tải lại (ghi đè)."""
    _setup_logging()
    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    repo = OperatorRepository(None, session_factory)
    # Đã cache & không --reload: đọc thẳng từ DB, KHÔNG đăng nhập/gọi API.
    if not reload and repo.cached_count() > 0:
        operators = repo.load_cached()
        console.print(f"[green]Operators: {len(operators)}[/green] — dùng CACHE, không tải mới")
        return
    client = _make_client()
    client.authenticate()
    repo.client = client
    operators = repo.fetch_all()
    console.print(f"[green]Operators: {len(operators)}[/green] — ĐÃ TẢI MỚI từ API")


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


def _make_validated_simulator(client, pf, session_factory, region, universe):
    """Dựng Simulator có cổng tiền-kiểm (pf.check) + recorder loại field chết khỏi
    pf.known_fields ngay trong phiên (không thử lại) và ghi blacklist bền vững."""
    record = _make_invalid_field_recorder(session_factory, region, universe)

    def on_invalid_field(field_id: str) -> None:
        if pf.known_fields is not None:
            pf.known_fields.discard(field_id)
        record(field_id)

    return Simulator(
        client, on_invalid_field=on_invalid_field, pre_sim_validator=pf.check
    )


def _portfolio_config_from_opts(
    neutralization: str, decay: int, truncation: float, delay: int,
):
    """Dựng PortfolioConfig từ option CLI; neutralization là tên enum không phân biệt hoa."""
    from src.backtest.config import Neutralization, PortfolioConfig

    try:
        neut = Neutralization[neutralization.upper()]
    except KeyError as exc:
        console.print(
            f"[red]neutralization '{neutralization}' không hợp lệ. Chọn: "
            f"{', '.join(n.name for n in Neutralization)}[/red]"
        )
        raise typer.Exit(code=1) from exc
    return PortfolioConfig(
        neutralization=neut, decay=decay, truncation=truncation, scale_book=1.0, delay=delay,
    )


@app.command()
def generate(
    method: str = typer.Option("gp", help="Phương pháp sinh (hỗ trợ: gp)"),
    count: int = typer.Option(50, help="Kích thước quần thể GP (số alpha quần thể cuối)"),
    n_generations: int = typer.Option(3, help="Số thế hệ tiến hóa GP"),
    seed: int = typer.Option(42, help="Seed master cho determinism (R8)"),
    market_data_dir: str = typer.Option(
        ..., help="Thư mục parquet MarketData (ParquetSource) để đánh giá thật"
    ),
    universe: str = typer.Option("TOP3000", help="Universe panel"),
    top_k: int = typer.Option(10, help="Số alpha giữ trong short-list cuối"),
    max_corr: float = typer.Option(0.70, help="Ngưỡng |rho| decorrelate short-list"),
    neutralization: str = typer.Option("NONE", help="NONE/MARKET/SECTOR/INDUSTRY/SUBINDUSTRY"),
    decay: int = typer.Option(0, help="Decay (ngày)"),
    truncation: float = typer.Option(0.10, help="Truncation trọng số"),
    delay: int = typer.Option(1, help="Delay (delay-1 chuẩn)"),
) -> None:
    """Sinh alpha qua GPEngine (Phase 7): seed→biến đổi→đánh giá thật qua Phase 2/3/4/6
    →chọn lọc NSGA-II→persist mọi outcome (pass/fail/seed) vào DB MiniBrain.
    In short-list đã decorrelate qua generate_many."""
    _setup_logging()

    if method != "gp":
        console.print(f"[red]Method '{method}' không được hỗ trợ. Chỉ có: gp[/red]")
        raise typer.Exit(code=1)

    import src.operators_local  # noqa: F401  (side-effect: nạp 27 operator vào registry)
    from src.data.adapters.parquet_source import ParquetSource
    from src.gp.engine import GPEngine
    from src.lang.registry import default_registry
    from src.pipeline.runner import generate_many
    from src.storage.repository import MiniBrainRepository

    if not Path(market_data_dir).is_dir():
        console.print(f"[red]Không thấy thư mục MarketData: {market_data_dir}[/red]")
        raise typer.Exit(code=1)

    engine_db = init_db(make_engine())
    session_factory = make_session_factory(engine_db)

    panel_source = ParquetSource(market_data_dir)
    try:
        data = panel_source.load("1900-01-01", "2999-12-31", universe)
    except (FileNotFoundError, AssertionError, OSError) as exc:
        console.print(f"[red]Không load được MarketData từ {market_data_dir}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    repo = MiniBrainRepository(session_factory)
    cfg = _portfolio_config_from_opts(neutralization, decay, truncation, delay)
    gp_engine = GPEngine(
        data=data, repo=repo, config=cfg, registry=default_registry(),
        pop_size=count, n_generations=n_generations, seed=seed,
    )

    pool = repo.load_pool() or None
    shortlist = generate_many(
        gp_engine=gp_engine, cfg=cfg, data=data, top_k=top_k, max_corr=max_corr, pool=pool,
    )
    table = Table(title=f"Short-list ({len(shortlist)} alpha, max_corr={max_corr})")
    table.add_column("#")
    table.add_column("expr")
    table.add_column("sharpe")
    table.add_column("fitness")
    for i, c in enumerate(shortlist, 1):
        table.add_row(str(i), c.expr, f"{c.metrics.sharpe:.3f}", f"{c.metrics.fitness:.3f}")
    console.print(table)
    console.print(f"[green]GP done[/green]: short-list {len(shortlist)} alpha (đã decorrelate).")


@app.command("score-one")
def score_one_cmd(
    expr: str = typer.Argument(..., help="Biểu thức FASTEXPR cần chấm (signal core)"),
    market_data_dir: str = typer.Option(..., help="Thư mục parquet MarketData (ParquetSource)"),
    universe: str = typer.Option("TOP3000", help="Universe panel"),
    neutralization: str = typer.Option("NONE", help="NONE/MARKET/SECTOR/INDUSTRY/SUBINDUSTRY"),
    decay: int = typer.Option(0, help="Decay (ngày)"),
    truncation: float = typer.Option(0.10, help="Truncation trọng số"),
    delay: int = typer.Option(1, help="Delay (delay-1 chuẩn)"),
    no_pool: bool = typer.Option(False, "--no-pool", help="Bỏ qua pool DB (self_corr=0)"),
) -> None:
    """Chấm 1 expression local (parse→eval→backtest→metrics→gate), KHÔNG đốt sim Brain. Nạp
    pool PnL từ DB hiện hành để gate self-correlation có nghĩa (trừ khi --no-pool)."""
    _setup_logging()

    import src.operators_local  # noqa: F401  (nạp 27 operator vào registry)

    from src.data.adapters.parquet_source import ParquetSource
    from src.pipeline.runner import score_one
    from src.storage.repository import MiniBrainRepository

    if not Path(market_data_dir).is_dir():
        console.print(f"[red]Không thấy thư mục MarketData: {market_data_dir}[/red]")
        raise typer.Exit(code=1)

    try:
        data = ParquetSource(market_data_dir).load("1900-01-01", "2999-12-31", universe)
    except (FileNotFoundError, AssertionError, OSError) as exc:
        console.print(f"[red]Không load được MarketData: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    cfg = _portfolio_config_from_opts(neutralization, decay, truncation, delay)

    pool = None
    if not no_pool:
        repo = MiniBrainRepository(make_session_factory(init_db(make_engine())))
        pool = repo.load_pool() or None

    metrics, verdict = score_one(expr, cfg, data, pool=pool)
    table = Table(title=f"score-one: {expr}")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("sharpe", f"{metrics.sharpe:.4f}")
    table.add_row("fitness", f"{metrics.fitness:.4f}")
    table.add_row("turnover", f"{metrics.turnover:.4f}")
    table.add_row("max_drawdown", f"{metrics.max_drawdown:.4f}")
    table.add_row("passed", str(verdict.passed))
    if verdict.hard_failures:
        table.add_row("fail", "; ".join(verdict.hard_failures))
    console.print(table)


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
    # blacklist field chết -> cấm LLM nêu lại trong prompt sinh ý tưởng.
    blacklist = InvalidFieldRepository(session_factory).blacklist()
    # repo -> bộ sinh hướng đọc phản hồi từ DB (top alpha để khai thác, field yếu tránh).
    return LLMAlphaGenerator(
        deepseek, field_repo, op_repo, prefilter,
        repo=AlphaRepository(session_factory), blacklist=blacklist,
    )


def _make_pool_corr_fn(client):
    """Đóng gói CorrelationChecker của WQ thành callback (wq_alpha_id -> self-corr|None)
    cho RefinementLoop. Đây chính là con số chặn-nộp thật của nền tảng. Lỗi hạ tầng
    (exception) -> None để loop không chặn nhầm trên trục trặc tạm thời."""
    from src.submission.correlation import CorrelationChecker

    checker = CorrelationChecker(client)

    def _pool_corr(wq_alpha_id):
        try:
            return checker.max_self_correlation(wq_alpha_id)
        except Exception as exc:  # noqa: BLE001 — trục trặc mạng không nên giết vòng
            logger.warning("Không đo được self-corr cho {}: {}", wq_alpha_id, exc)
            return None

    return _pool_corr


def _make_pnl_fn(client):
    """Helper lấy daily PnL của một alpha từ WQ recordset (review 3). WQ trả PnL TÍCH
    LUỸ -> diff thành gia số ngày. Trả [(date, daily_pnl)] hoặc None khi lỗi/rỗng."""

    def _pnl(wq_alpha_id):
        try:
            resp = client.get(f"/alphas/{wq_alpha_id}/recordsets/pnl")
            if resp.status_code not in (200, 201):
                return None
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không lấy được PnL cho {}: {}", wq_alpha_id, exc)
            return None
        records = payload.get("records") or []
        props = (payload.get("schema") or {}).get("properties") or []
        date_i, pnl_i = 0, 1
        for i, p in enumerate(props):
            name = (p.get("name") or "").lower() if isinstance(p, dict) else ""
            if name == "date":
                date_i = i
            elif "pnl" in name:
                pnl_i = i
        out, prev = [], None
        for row in records:
            if not isinstance(row, (list, tuple)) or len(row) <= max(date_i, pnl_i):
                continue
            cum = float(row[pnl_i])
            if prev is not None:
                out.append((row[date_i], cum - prev))
            prev = cum
        return out or None

    return _pnl


def _make_research_loop(
    session_factory, client, region, universe, delay, max_sims, patience,
    align=True, regularize=False, penalty_lambda=0.3, sim_config=None,
    oos_min_ratio=None, deflate_haircut=0.0, regime_min=None, align_gate=True,
    improve_margin=0.0, reseed_every=0, marathon=False,
):
    """Lắp RefinementLoop GĐ2 với DeepSeek + Simulator thật. Trả (loop, deepseek).

    marathon=True: bật trọng tài LLM (Referee) + ConfigTuner -> sau mỗi sim LLM tự
    quyết refine_formula | tune_config | abandon (dùng cho lệnh `marathon`)."""
    from src.decorrelation.similarity import avoid_subtree_canons
    from src.decorrelation.zoo import ReferenceZoo
    from src.llm.alignment import AlignmentScorer
    from src.llm.hypothesis import HypothesisGenerator
    from src.llm.loop import RefinementLoop
    from src.llm.refiner import AlphaRefiner
    from src.llm.referee import ConfigTuner, Referee
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
    # Task 1b: re-seed diversity — chỉ dựng idea generator khi bật (tốn lượt LLM).
    idea_generator = _make_llm_generator(session_factory, pf) if reseed_every > 0 else None
    # Marathon: trọng tài LLM + bộ tinh chỉnh config (dùng chung deepseek/router).
    referee = Referee(deepseek) if marathon else None
    config_tuner = ConfigTuner(deepseek) if marathon else None
    loop = RefinementLoop(
        hypothesis_gen=HypothesisGenerator(deepseek),
        translator=translator,
        refiner=refiner,
        simulator=_make_validated_simulator(client, pf, session_factory, region, universe),
        prefilter=pf,
        repo=repo,
        region=region,
        universe=universe,
        delay=delay,
        max_simulations=max_sims,
        no_improve_patience=patience,
        zoo=zoo,
        aligner=aligner,
        align_gate=align_gate,
        regularize=regularize,
        penalty_lambda=penalty_lambda,
        sim_config=sim_config,
        # (1) Self-correlation với pool là ràng buộc hạng nhất: gate crowded sau sim
        # + đưa vào điểm để best né đỉnh đông (dùng đúng endpoint chặn-nộp của WQ).
        pool_corr_fn=_make_pool_corr_fn(client),
        # (4) OOS gate + deflated-sharpe chống overfit IS (None/0 = tắt).
        oos_min_ratio=oos_min_ratio,
        deflate_haircut=deflate_haircut,
        # (3) Regime gate: sàn Sharpe năm tệ nhất (None = tắt). Chỉ lắp pnl_fn khi bật.
        pnl_fn=(_make_pnl_fn(client) if regime_min else None),
        regime_min=regime_min,
        improve_margin=improve_margin,
        idea_generator=idea_generator,
        reseed_every=reseed_every,
        referee=referee,
        config_tuner=config_tuner,
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
    for name in ("sharpe", "fitness", "pool_fit", "turnover_fit", "drawdown_fit"):
        stab.add_column(name, justify="right")
    stab.add_column("total", justify="right")
    d = vec.dimensions()
    stab.add_row(
        f"{d['sharpe']:.2f}", f"{d['fitness']:.2f}", f"{d['pool_fit']:.2f}",
        f"{d['turnover_fit']:.2f}", f"{d['drawdown_fit']:.2f}",
        f"[bold]{vec.total:.3f}[/bold]",
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


def resolve_direction(direction: str, idea_provider) -> tuple[str, bool]:
    """Trả (hướng, đã_tự_sinh). Người dùng nhập -> dùng nguyên (không gọi LLM).
    Để trống -> lấy 1 hướng từ idea_provider() (LLM tự đề xuất, giống miner cũ).
    Fallback chuỗi mặc định nếu LLM không sinh được hướng hợp lệ."""
    direction = (direction or "").strip()
    if direction:
        return direction, False
    ideas = idea_provider() or []
    first = ideas[0].strip() if ideas and ideas[0] and ideas[0].strip() else ""
    return (first or "mean-reversion theo thanh khoản"), True


@app.command()
def research(
    direction: str = typer.Option("", "--direction", help="Hướng nghiên cứu (ngôn ngữ tự nhiên); để trống -> LLM tự đề xuất"),
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    decay: int = typer.Option(0, "--decay", help="Fixed decay setting for research simulations"),
    truncation: float = typer.Option(0.08, "--truncation", help="Fixed truncation setting for research simulations"),
    neutralization: str = typer.Option("SUBINDUSTRY", "--neutralization", help="Fixed neutralization setting for research simulations"),
    max_sims: int = typer.Option(20, "--max-sims", help="Trần số simulation cho cả vòng"),
    no_improve: int = typer.Option(3, "--no-improve", help="Dừng sau N vòng không cải thiện"),
    align: bool = typer.Option(True, "--align/--no-align", help="Bật lọc nhất quán giả thuyết–công thức trước sim (T4.2)"),
    align_soft: bool = typer.Option(False, "--align-soft", help="Alignment chỉ là tín hiệu mềm (không loại trước sim) — tránh giết edge (review 5)"),
    regularize: bool = typer.Option(False, "--regularize/--no-regularize", help="Chọn best theo điểm điều chuẩn (trừ phạt độc đáo/khớp/phức tạp) (T4.4)"),
    penalty_lambda: float = typer.Option(0.3, "--lambda", help="Hệ số λ cho số hạng phạt điều chuẩn"),
    mcts: bool = typer.Option(False, "--mcts/--greedy", help="Dùng MCTS (giữ nhiều nhánh, UCB) thay vòng greedy (T6.1)"),
    oos_ratio: float = typer.Option(0.0, "--oos-ratio", help="Tỉ lệ OOS/IS sharpe tối thiểu để gắn passed (0 = tắt) (review 4)"),
    deflate: float = typer.Option(0.0, "--deflate", help="Hệ số haircut điểm theo budget sim đã dùng — chống overfit IS (0 = tắt) (review 4b)"),
    min_annual_sharpe: float = typer.Option(0.0, "--min-annual-sharpe", help="Sàn Sharpe năm tệ nhất — loại alpha mỏng manh theo regime (0 = tắt) (review 3)"),
    improve_margin: float = typer.Option(0.0, "--improve-margin", help="Biên cải thiện tương đối tối thiểu để soán best (vd 0.1 = 10%; 0 = tắt)"),
    reseed_every: int = typer.Option(0, "--reseed-every", help="Re-seed direction mới khi nhánh kẹt N vòng không cải thiện (0 = tắt) — salvage diversity"),
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

    # Hướng để trống -> LLM tự đề xuất (giống miner cũ tự seed). Closure chỉ chạy
    # khi cần (không nhập hướng) để khỏi tốn lượt LLM khi đã có hướng.
    def _auto_direction():
        from src.simulation.pre_filter import PreFilter

        f, o, ft, mo, oa = _cached_symbols(session_factory)
        pf = PreFilter(
            known_operators=o or None, known_fields=set(f) or None,
            field_types=ft, matrix_only_ops=mo, operator_arity=oa,
        )
        return _make_llm_generator(session_factory, pf).generate_ideas(1)

    direction, auto_dir = resolve_direction(direction, _auto_direction)
    if auto_dir:
        console.print(f"[cyan]Hướng nghiên cứu (LLM tự đề xuất):[/cyan] {direction}")

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
        align_gate=not align_soft,
        oos_min_ratio=(oos_ratio if oos_ratio > 0 else None),
        deflate_haircut=deflate,
        regime_min=(min_annual_sharpe if min_annual_sharpe > 0 else None),
        improve_margin=improve_margin,
        reseed_every=reseed_every,
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


@app.command()
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


# ============================ Menu tương tác (start) ============================
# Khôi phục wizard cũ: đăng nhập 1 lần, giữ phiên + DB trong cùng tiến trình, hiện
# số fields/operators ngay sau đăng nhập để người dùng tự quyết có tải lại không.
# Engine sinh alpha = RefinementLoop (lệnh research) — thay cho HybridEngine cũ.


class _MenuState:
    """Giữ phiên đăng nhập + DB (mở sau khi biết email) + scope cho menu."""

    def __init__(self):
        self.client = None
        self.session_factory = None
        self.email = ""
        self.region = settings.default_region
        self.universe = settings.default_universe
        self.delay = settings.default_delay

    @property
    def logged_in(self) -> bool:
        return (
            self.client is not None
            and self.client.authenticated
            and self.session_factory is not None
        )


def _menu_counts(state: _MenuState) -> tuple[int, int]:
    """(số fields trong scope, số operators) trong DB hiện tại; (0,0) nếu chưa mở DB."""
    if state.session_factory is None:
        return 0, 0
    n_fields = FieldRepository(None, state.session_factory).cached_count(
        state.region, state.universe, state.delay
    )
    n_ops = OperatorRepository(None, state.session_factory).cached_count()
    return n_fields, n_ops


def _menu_login(state: _MenuState) -> None:
    from src.storage.db import active_database_url, write_active_account

    client = _make_client()
    client.authenticate()
    state.client = client
    state.email = client.email or ""
    # Biết email rồi mới mở DB (DB tách theo email), rồi hiện số liệu để quyết định.
    if state.email:
        write_active_account(state.email)
    engine = init_db(make_engine())
    state.session_factory = make_session_factory(engine)

    n_fields, n_ops = _menu_counts(state)
    console.print(f"[green]✓ Đăng nhập xong[/green] ({state.email})")
    console.print(f"[dim]DB: {active_database_url()}[/dim]")
    console.print(
        f"[bold]Data fields:[/bold] {n_fields}   [bold]Operators:[/bold] {n_ops}   "
        f"[dim]({state.region}/{state.universe}/delay={state.delay})[/dim]"
    )
    if n_fields == 0 or n_ops == 0:
        console.print("[yellow]Thiếu dữ liệu — chọn 2/3 để tải về.[/yellow]")
    else:
        console.print("[dim]Đủ dữ liệu. Bấm 4 để chạy engine, hoặc 2/3 để tải lại.[/dim]")


def _menu_fields(state: _MenuState) -> None:
    """Tải lại data fields từ API (ghi đè cache)."""
    from src.data.fields import FieldFetchError

    repo = FieldRepository(state.client, state.session_factory)
    try:
        fields = repo.get_fields(state.region, state.universe, state.delay, force_reload=True)
    except FieldFetchError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    console.print(
        f"[green]Đã tải mới {len(fields)} data fields[/green] "
        f"({state.region}/{state.universe}/delay={state.delay})"
    )


def _menu_operators(state: _MenuState) -> None:
    """Tải lại operators từ API (ghi đè cache)."""
    from src.data.operators import OperatorFetchError

    repo = OperatorRepository(state.client, state.session_factory)
    try:
        operators = repo.fetch_all()
    except OperatorFetchError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    console.print(f"[green]Đã tải mới {len(operators)} operators[/green]")


def _menu_research(state: _MenuState, max_sims: int, no_improve: int) -> None:
    """Chạy engine sinh alpha (RefinementLoop). Hỏi hướng — Enter = LLM tự đề xuất."""
    from src.simulation.config import SimConfig
    from src.simulation.pre_filter import PreFilter

    n_fields, _ = _menu_counts(state)
    if n_fields == 0:
        console.print("[red]Chưa có data fields — chọn 2 để tải trước.[/red]")
        return

    direction = input("\nHướng nghiên cứu [Enter = để LLM tự đề xuất]: ").strip()

    def _auto_direction():
        f, o, ft, mo, oa = _cached_symbols(state.session_factory)
        pf = PreFilter(
            known_operators=o or None, known_fields=set(f) or None,
            field_types=ft, matrix_only_ops=mo, operator_arity=oa,
        )
        return _make_llm_generator(state.session_factory, pf).generate_ideas(1)

    direction, auto = resolve_direction(direction, _auto_direction)
    if auto:
        console.print(f"[cyan]Hướng nghiên cứu (LLM tự đề xuất):[/cyan] {direction}")

    sim_config = SimConfig(region=state.region, universe=state.universe, delay=state.delay)
    loop, deepseek = _make_research_loop(
        state.session_factory, state.client, state.region, state.universe, state.delay,
        max_sims, no_improve, sim_config=sim_config,
    )
    result = _run_research_with_progress(loop, direction, max_sims)
    _render_research_result(result, deepseek)


def _menu_marathon(state: _MenuState) -> None:
    """Chạy marathon từ menu: config khởi đầu decay=4/truncation=0.01/MARKET, chạy
    đến khi hết quota (Ctrl+C để dừng tay)."""
    n_fields, _ = _menu_counts(state)
    if n_fields == 0:
        console.print("[red]Chưa có data fields — chọn 2 để tải trước.[/red]")
        return
    _run_marathon_session(
        state.session_factory, state.client, state.region, state.universe, state.delay,
        decay=4, truncation=0.01, neutralization="MARKET",
        per_direction_sims=30, max_patience=8, retry=2,
    )


def _print_menu(state: _MenuState) -> None:
    if state.logged_in:
        n_fields, n_ops = _menu_counts(state)
        status = f"[green]✓ đã đăng nhập[/green] ({state.email})"
        data = f"[bold]Fields:[/bold] {n_fields}   [bold]Operators:[/bold] {n_ops}"
    else:
        status = "[red]✗ chưa đăng nhập[/red]"
        data = "[dim](đăng nhập để xem số fields/operators)[/dim]"
    console.print("\n[bold cyan]=== WQ Auto-Alpha ===[/bold cyan]")
    console.print(f"Scope: [cyan]{state.region}/{state.universe}/delay={state.delay}[/cyan] | {status}")
    console.print(data)
    console.print(" 1) Đăng nhập")
    console.print(" 2) Tải lại data fields (ghi đè cache)")
    console.print(" 3) Tải lại operators (ghi đè cache)")
    console.print(" 4) Chạy engine sinh alpha")
    console.print(" 5) Chạy thử (ngắn)")
    console.print(" 6) Marathon (chạy đến khi hết quota)")
    console.print(" 0) Thoát")


@app.command()
def start(
    max_sims: int = typer.Option(20, "--max-sims", help="Trần số simulation mỗi lần chạy engine (mục 4)"),
    no_improve: int = typer.Option(3, "--no-improve", help="Dừng sau N vòng không cải thiện"),
) -> None:
    """Menu tương tác: đăng nhập → xem/tải fields-operators → chạy engine sinh alpha."""
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
            elif choice in {"2", "3", "4", "5", "6"} and not state.logged_in:
                console.print("[yellow]Hãy đăng nhập (1) trước.[/yellow]")
            elif choice == "2":
                _menu_fields(state)
            elif choice == "3":
                _menu_operators(state)
            elif choice == "4":
                _menu_research(state, max_sims, no_improve)
            elif choice == "5":
                _menu_research(state, max_sims=5, no_improve=2)
            elif choice == "6":
                _menu_marathon(state)
            else:
                console.print("[red]Lựa chọn không hợp lệ.[/red]")
        except AuthError as exc:
            console.print(f"[red]Lỗi đăng nhập: {exc}[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Đã hủy bước hiện tại.[/yellow]")
    console.print("[cyan]Kết thúc.[/cyan]")


@app.command("calibrate")
def calibrate(
    db_url: str = typer.Option("", help="URL DB nguồn alpha đã sim; rỗng = DATABASE_URL hiện tại"),
    market_data_dir: str = typer.Option(
        "", help="Thư mục parquet panel (ParquetSource) để re-score local"
    ),
    start: str = typer.Option("", help="Ngày đầu cửa sổ load (YYYY-MM-DD); rỗng = toàn bộ"),
    end: str = typer.Option("", help="Ngày cuối cửa sổ load (YYYY-MM-DD); rỗng = toàn bộ"),
    universe: str = typer.Option(settings.default_universe, help="Universe để load panel"),
    limit: int = typer.Option(0, help="Giới hạn số BrainRecord (0 = không giới hạn)"),
) -> None:
    """Đo Spearman ρ local-vs-Brain trên alpha đã mô phỏng thật (B10 calibration).

    Re-score MỖI alpha trong DB hoàn toàn local (không đốt sim) rồi so ranking local với
    ranking Brain. Cần nguồn MarketData thật (--market-data-dir parquet) — KHÔNG in báo cáo
    giả nếu thiếu data.

    ⚠️ CHỈ dùng DB GROUND-TRUTH chuyên dụng (mọi alpha sim với CÙNG config NONE/decay0/trunc0/
    delay1 mà local re-score). Trỏ vào wq_alpha_*.db thường (config lẫn lộn) -> ρ vô nghĩa.
    """
    from config.thresholds import CALIBRATION_RHO_BAR
    from src.calibration.harness import CalibrationHarness, make_local_scorer
    from src.calibration.loader import load_brain_records

    _setup_logging()
    url = db_url or settings.database_url
    engine = init_db(make_engine(url))  # tạo bảng nếu DB mới (đồng nhất các lệnh khác)
    session_factory = make_session_factory(engine)

    records = load_brain_records(session_factory, limit=limit or None)
    if not records:
        console.print(
            "[yellow]Không có BrainRecord nào trong DB — chưa có alpha đã mô phỏng dùng được.[/yellow]"
        )
        raise typer.Exit(code=0)

    if not market_data_dir:
        console.print(
            "[red]calibrate cần nguồn MarketData để re-score local: truyền --market-data-dir "
            "<thư mục parquet>. Chưa wire nguồn data thị trường nào (Gap#3 bulk OHLCV).[/red]"
        )
        raise typer.Exit(code=1)

    from src.data.adapters.parquet_source import ParquetSource

    panel_source = ParquetSource(market_data_dir)
    lo = start or "1900-01-01"
    hi = end or "2999-12-31"
    try:
        data = panel_source.load(lo, hi, universe)
    except (FileNotFoundError, AssertionError, OSError) as exc:
        console.print(f"[red]Không load được MarketData từ {market_data_dir}: {exc}[/red]")
        raise typer.Exit(code=1)

    report = CalibrationHarness(scorer=make_local_scorer(data)).run(records)

    if report.n == 0:
        console.print(
            f"[red]Không re-score local được alpha nào trong {len(records)} BrainRecord "
            "(field không có trong panel / parse lỗi) — không tính được ρ. Kiểm tra panel có "
            "đủ field mà alpha dùng không.[/red]"
        )
        raise typer.Exit(code=1)

    table = Table(title=f"Calibration report (n={report.n})")
    table.add_column("Chỉ số")
    table.add_column("Giá trị", justify="right")
    table.add_row("spearman_sharpe (ρ)", f"{report.spearman_sharpe:.4f}")
    table.add_row("spearman_fitness", f"{report.spearman_fitness:.4f}")
    table.add_row("self_corr_agreement", f"{report.self_corr_agreement:.4f}")
    table.add_row("decile_hit_rate", f"{report.decile_hit_rate:.4f}")
    console.print(table)
    console.print(
        "[dim]Lưu ý: ρ chỉ hợp lệ nếu mọi alpha trong DB được sim với cùng config "
        "(NONE/decay0/trunc0/delay1) mà local re-score.[/dim]"
    )

    rho = report.spearman_sharpe
    if math.isnan(rho):
        console.print(
            "[yellow]ρ không xác định (mẫu thiếu brain_sharpe hoặc hằng số) — không kết luận "
            "được độ tin cậy ranking local.[/yellow]"
        )
    elif rho < CALIBRATION_RHO_BAR:
        console.print(
            f"[red]CẢNH BÁO: ρ={rho:.3f} < CALIBRATION_RHO_BAR={CALIBRATION_RHO_BAR} — KHÔNG "
            "tin ranking local cho tới khi fix data/operator fidelity (B10 master spec).[/red]"
        )
    else:
        console.print(
            f"[green]ρ={rho:.3f} ≥ {CALIBRATION_RHO_BAR} — ranking local đáng tin cậy.[/green]"
        )


if __name__ == "__main__":
    app()
