"""CLI entry cho WorldQuant Brain Auto-Alpha Tool."""

from __future__ import annotations

import math
import os
import random
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
from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository
from src.simulation.simulator import Simulator
from src.storage.db import init_db, make_engine, make_session_factory
from src.storage.migrate import migrate_all, _same_database
from src.storage.repository import AlphaRepository, InvalidFieldRepository
from src.llm.marathon import MarathonReport, run_marathon
from src.app.cli import common as cli_common
from src.app.cli import auth as cli_auth
from src.app.cli import fields as cli_fields
from src.app.cli import simulate as cli_simulate

app = typer.Typer(help="WorldQuant Brain Auto-Alpha Tool")
app.command()(cli_auth.login)
app.command("probe-fields")(cli_fields.probe_fields)
app.command("warm-cache")(cli_fields.warm_cache_cmd)
app.command("fetch-fields")(cli_fields.fetch_fields)
app.command("cache-status")(cli_fields.cache_status)
app.command("fetch-operators")(cli_fields.fetch_operators)
app.command("list-fields")(cli_fields.list_fields)
app.command()(cli_simulate.simulate)
app.command("sweep-config")(cli_simulate.sweep_config)
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


@app.command("migrate-sqlite")
def migrate_sqlite(
    source: str = typer.Option("sqlite:///data/db/wq_alpha.db", help="URL DB nguồn (SQLite)"),
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
    from src.lang.registry import default_registry, enforce_gp_vocab_against_catalog
    from src.pipeline.runner import generate_many
    from src.storage.repository import MiniBrainRepository

    if not Path(market_data_dir).is_dir():
        console.print(f"[red]Không thấy thư mục MarketData: {market_data_dir}[/red]")
        raise typer.Exit(code=1)

    engine_db = init_db(make_engine())
    session_factory = make_session_factory(engine_db)
    # Guard tổng quát: loại khỏi vocab GP mọi operator KHÔNG có trong catalog Brain live
    # (vd bug ts_std/ts_std_dev đã gặp) TRƯỚC khi GPEngine dựng cây — tránh sinh biểu thức
    # Brain chắc chắn từ chối (tốn phí pre-sim). Catalog rỗng (chưa `wq load-operators`)
    # -> hàm tự bỏ qua, không crash.
    _, _catalog_ops, _, _, _ = cli_common._cached_symbols(session_factory)
    enforce_gp_vocab_against_catalog(default_registry(), _catalog_ops)

    panel_source = ParquetSource(market_data_dir)
    try:
        data = panel_source.load("1900-01-01", "2999-12-31", universe)
    except (FileNotFoundError, AssertionError, OSError) as exc:
        console.print(f"[red]Không load được MarketData từ {market_data_dir}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    repo = MiniBrainRepository(session_factory)
    cfg = cli_common._portfolio_config_from_opts(neutralization, decay, truncation, delay)
    gp_engine = GPEngine(
        data=data, repo=repo, config=cfg, registry=default_registry(),
        pop_size=count, n_generations=n_generations, seed=seed,
    )

    pool = repo.load_pool() or None
    # generate_many re-score qua score_one (một nguồn AlphaMetrics duy nhất) — chấp nhận backtest lại.
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

    cfg = cli_common._portfolio_config_from_opts(neutralization, decay, truncation, delay)

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


def _resolve_base_seed(base_seed: int | None) -> int:
    """Seed cho GP sinh ý tưởng của vòng kín. None hoặc 0 -> ngẫu nhiên (mỗi lần chạy một
    quần thể khác, tránh kẹt vào cùng quần thể -> no_more_ideas khi pool đã tích lũy). Số
    dương cụ thể -> giữ nguyên để tái lập được."""
    if base_seed:  # khác 0 và khác None
        return base_seed
    return random.randrange(1, 2**31 - 1)


def _run_reseed_until_quota(build_loop, first_seed: int, *, reseed_fn=_resolve_base_seed):
    """Chạy vòng kín LẶP LẠI cho chế độ không trần max_ideas (menu 5 / --max-ideas 0):
    `no_more_ideas` chỉ là cạn ý tưởng TẠM THỜI (batch GP rỗng sau dedup/family-closed) —
    reseed GP rồi dựng lại vòng và chạy tiếp thay vì kết thúc phiên. Chỉ trả về khi
    stop_reason khác (vd "quota"); QuotaExhausted/KeyboardInterrupt lan ra caller xử lý."""
    seed = first_seed
    while True:
        report = build_loop(seed).run()
        if report.stop_reason != "no_more_ideas":
            return report
        seed = reseed_fn(None)
        console.print(
            f"[yellow]Cạn ý tưởng tạm thời — reseed GP (seed={seed}) và chạy tiếp "
            "(hết quota hoặc Ctrl+C mới dừng)…[/yellow]"
        )


def _local_neutralization(neutralization: str, available_groups) -> str:
    """Hạ cấp neutralization cho LOCAL gate về nhóm mà panel cục bộ CÓ. Brain có đủ
    country/sector/industry/subindustry, nhưng panel local (vd market_yf) thường chỉ có
    'sector' — dùng SUBINDUSTRY sẽ KeyError làm local gate loại sạch ý tưởng. Thứ tự hạ:
    nhóm yêu cầu (nếu có) -> SECTOR (nếu có) -> NONE. NONE/MARKET không cần nhóm."""
    n = neutralization.strip().upper()
    if n in ("NONE", "MARKET"):
        return n
    group_key = {"SECTOR": "sector", "INDUSTRY": "industry", "SUBINDUSTRY": "subindustry"}.get(n)
    if group_key and group_key in available_groups:
        return n
    if "sector" in available_groups:
        return "SECTOR"
    return "NONE"


def _closed_loop_configs(
    neutralization: str, decay: int, truncation: float, delay: int,
    region: str, universe: str, available_groups,
):
    """Dựng cặp (PortfolioConfig local gate, SimConfig Brain sim). Brain sim dùng
    neutralization ĐẦY ĐỦ (vd SUBINDUSTRY); local gate hạ cấp về nhóm panel cục bộ CÓ
    (tránh KeyError loại sạch ý tưởng) nhưng GIỮ decay/truncation khớp Brain."""
    from src.simulation.config import SimConfig

    local_neut = _local_neutralization(neutralization, available_groups)
    cfg = cli_common._portfolio_config_from_opts(local_neut, decay, truncation, delay)
    sim_config = SimConfig.default(region=region, universe=universe, delay=delay).with_overrides(
        neutralization=neutralization, decay=decay, truncation=truncation,
    )
    return cfg, sim_config


def _run_closed_loop_session(
    session_factory, client, region, universe, delay, market_data_dir,
    *, pop_size: int = 30, n_generations: int = 3, top_k: int = 10, max_corr: float = 0.70,
    patience: int = 5, max_ideas: int | None = None,
    neutralization: str = "MARKET", decay: int = 4, truncation: float = 0.08,
    base_seed: int | None = None, refiner_kind: str = "local",
    include_alt_data: bool = True, include_combiner: bool = True,
    max_gp_sims: int | None = 3, alt_sweep_budget: int = 2,
) -> bool:
    """Dựng + chạy vòng kín AI+MiniBrain thật (dùng chung cho CLI `closed-loop` và menu mục 5).

    `refiner_kind`: "local" (mặc định) -> LocalTunerRefiner — tune tham số/config bằng eval
    local (Task 2/3), CHỈ sim Brain 1 lần cho cấu hình tốt nhất, KHÔNG dùng LLM refine (bỏ
    bước ~16 phút/ý tưởng). "llm" -> RefinementLoopRefiner cũ (bọc `loop` AI refine nhiều bước).

    Trả False nếu không load được MarketData (lỗi cấu hình, chưa kịp chạy); True nếu đã chạy
    xong (kể cả dừng do hết quota/Ctrl+C — kết quả vẫn lưu DB)."""
    import src.operators_local  # noqa: F401
    from datetime import date as _date

    from src.app.closed_loop_adapters import build_closed_loop
    from src.app.power_pool_config import resolve_theme_sim_config
    from src.data.adapters.parquet_source import ParquetSource
    from src.lang.registry import default_registry, enforce_gp_vocab_against_catalog
    from src.pipeline.closed_loop import QuotaExhausted
    from src.storage.repository import MiniBrainRepository

    repo = MiniBrainRepository(session_factory)
    # Guard tổng quát (Task 2): loại khỏi vocab GP mọi operator KHÔNG có trong catalog
    # Brain live TRƯỚC khi vòng kín sinh ý tưởng — né phí pre-sim vô ích khi GP emit
    # operator Brain chắc chắn từ chối (vd ts_std trước đây). Catalog rỗng -> bỏ qua.
    _catalog_fields, _catalog_ops, _, _, _ = cli_common._cached_symbols(session_factory)
    enforce_gp_vocab_against_catalog(default_registry(), _catalog_ops)
    # Field-validity guard (RC1/RC2 fix idea-generator, Task known_fields): core alt-data/
    # fundamental/hypothesis tham chiếu field KHÔNG có trong catalog cache thật bị lọc bỏ
    # TRƯỚC khi chạm Brain sim (cardinal rule #1 — đừng tin tên field, đừng đốt quota vì field
    # bịa). Catalog rỗng (chưa `wq load-fields`) -> None để KHÔNG lọc oan (an toàn hơn là lọc
    # sạch mọi core khi chưa có dữ liệu để so).
    _known_fields = set(_catalog_fields) if _catalog_fields else None
    try:
        data = ParquetSource(market_data_dir).load("1900-01-01", "2999-12-31", universe)
    except (FileNotFoundError, AssertionError, OSError) as exc:
        console.print(f"[red]Không load được MarketData: {exc}[/red]")
        return False

    cfg, sim_config = _closed_loop_configs(
        neutralization, decay, truncation, delay, region, universe,
        set(data.groups.keys()),
    )
    # MẶC ĐỊNH đọc Power Pool Theme hôm nay: có theme -> sim đúng region/universe/delay theme
    # (nộp được Pure Power Pool); không có -> giữ config Regular + cảnh báo.
    _res = resolve_theme_sim_config(sim_config, _date.today())
    pp_allowed = _res.allowed_neutralizations
    if _res.theme is not None:
        sim_config = _res.sim_config
        region, universe = _res.region, _res.universe
        delay = sim_config.delay  # đồng bộ delay CLI theo theme (mọi theme hiện tại delay=1,
        # cứng hoá để tránh lệch nếu sau này có theme delay khác, vì `delay` còn được đẩy vào
        # _make_research_loop bên dưới)
        console.print(
            f"[cyan]Power Pool Theme {_res.theme.start_date}..{_res.theme.end_date}: "
            f"sim {region}/{universe}/delay={sim_config.delay}, "
            f"neutralization ∈ {sorted(pp_allowed)}[/cyan]"
        )
    else:
        console.print(f"[yellow]{_res.warning}[/yellow]")
    loop, _deepseek = _make_research_loop(
        session_factory, client, region, universe, delay,
        max_sims=10**9, patience=patience, marathon=True, sim_config=sim_config,
    )
    loop.market_data = data          # bật local gate trước sim
    loop.local_gate_cfg = cfg
    # Pre-sim floor (calibrate local≈Brain/1.28): bỏ qua sim alpha local Sharpe quá thấp
    # (chắc chắn rác) -> tiết kiệm quota cho chạy dài. Bảo thủ nên không đói loop.
    import functools as _functools

    from config.thresholds import PRE_SIM_LOCAL_SHARPE_FLOOR
    from src.backtest.gate import score_local_gate as _score_local_gate

    loop.local_gate_fn = _functools.partial(
        _score_local_gate, min_sharpe=PRE_SIM_LOCAL_SHARPE_FLOOR
    )
    loop.max_simulations = 10**9     # không trần local; dừng theo quota Brain (QuotaExhausted)

    seed = _resolve_base_seed(base_seed)
    if refiner_kind == "local":
        from src.app.closed_loop_adapters import LocalTunerRefiner

        # loop.repo là AlphaRepository (save_alpha/save_simulation) — KHÁC `repo`
        # (MiniBrainRepository) mà ClosedLoop/GPIdeaSource dùng ở trên. LocalTunerRefiner
        # cần đúng AlphaRepository mà RefinementLoop đang dùng để lịch sử alpha/sim nhất
        # quán với các đường refine khác (marathon/research) trong cùng DB.
        refiner: object | None = LocalTunerRefiner(
            simulator=loop.simulator, repo=loop.repo, data=data,
            local_config=cfg, sim_config=sim_config,
            pool_corr_fn=loop.pool_corr_fn, region=region, universe=universe,
            # repo (MiniBrainRepository) là kho calibration: lưu local-eval expr đã tune để
            # CalibrationTracker đo ρ local↔Brain (join theo hash với record_brain_sim).
            calib_repo=repo,
            pp_allowed_neutralizations=pp_allowed,
            # Pha 3.1: tune thử bọc regression_neut(expr, rank(volume)) — trừ thành phần
            # crowded theo thanh khoản (factor phổ biến nhất gây self-corr cao) để hạ self-corr
            # Brain. rank(volume) có trong panel local market_yf nên tune chấm được thật.
            neut_risk_factors=["rank(volume)"],
            # Mini-sweep alt-data (Task 5): cứu hypothesis sai dấu/decay yếu bằng ≤
            # alt_sweep_budget sim thêm thay vì vứt sau 1 sim (bằng chứng: seed social từng
            # sai dấu, analyst revision 1-shot rồi bỏ).
            alt_sweep_budget=alt_sweep_budget,
        )
    else:
        refiner = None  # build_closed_loop mặc định RefinementLoopRefiner(loop) (đường LLM cũ)

    from src.reporting.run_alpha_log import RunAlphaLogger, run_log_path
    from src.reporting.session_summary import SessionSummary, summary_path

    _log_path = run_log_path()
    _alpha_logger = RunAlphaLogger(_log_path)
    _summary = SessionSummary()
    console.print(f"[cyan]📄 Log công thức alpha phiên này: {_log_path}[/cyan]")

    def _build_cl(_seed: int):
        # Dựng lại được NHIỀU LẦN với seed khác nhau: chế độ không trần max_ideas reseed
        # sau mỗi lần no_more_ideas (cạn ý tưởng tạm thời) thay vì kết thúc phiên.
        # _alpha_logger/_summary dùng chung qua các vòng -> log/tóm tắt gộp cả phiên.
        return build_closed_loop(
            data=data, repo=repo, config=cfg, registry=default_registry(), loop=loop,
            region=region, universe=universe, pop_size=pop_size, n_generations=n_generations,
            top_k=top_k, max_corr=max_corr, max_ideas=max_ideas, base_seed=_seed,
            refiner=refiner, include_alt_data=include_alt_data, alpha_logger=_alpha_logger,
            include_combiner=include_combiner, session_summary=_summary,
            known_fields=_known_fields, max_gp_sims=max_gp_sims,
        )

    console.print(f"[cyan]Bắt đầu vòng kín (base_seed={seed}, Ctrl+C để dừng)…[/cyan]")
    # `finally` bao trùm mọi đường ra (chạy xong bình thường/QuotaExhausted/Ctrl+C) để
    # RunAlphaLogger LUÔN được đóng tường minh, tránh rò rỉ file handle khi vòng kín dừng
    # giữa chừng.
    try:
        try:
            if max_ideas is None:
                # Menu 5 / --max-ideas 0: chạy tới hết quota Brain hoặc Ctrl+C — cạn ý
                # tưởng tạm thời thì reseed GP và tiếp tục, không kết thúc phiên.
                report = _run_reseed_until_quota(_build_cl, seed)
            else:
                report = _build_cl(seed).run()
        except QuotaExhausted:
            console.print(
                "[yellow]Hết quota Brain — vòng kín dừng tự động. Kết quả đã lưu DB.[/yellow]"
            )
            return True
        except KeyboardInterrupt:
            console.print("\n[yellow]Đã dừng tay (Ctrl+C). Kết quả đã lưu DB.[/yellow]")
            return True
        console.print(
            f"[green]Vòng kín xong[/green] ({report.stop_reason}): ý tưởng={report.ideas_tried} "
            f"sim={report.sims_used} pass={report.n_passed} bỏ={report.n_abandoned} "
            f"ρ={report.rho_sharpe}"
        )
        return True
    finally:
        _alpha_logger.close()
        # Báo cáo funnel cuối phiên (Pha 0): in + ghi file để trả lời "chết ở đâu, vì sao,
        # tốn bao lâu" mà không phải parse CSV thủ công.
        try:
            _sum_path = summary_path()
            _summary.write(_sum_path)
            console.print(_summary.render_markdown())
            console.print(f"[cyan]📊 Tóm tắt phiên: {_sum_path}[/cyan]")
        except Exception as _exc:  # noqa: BLE001 - báo cáo lỗi không được làm hỏng phiên
            console.print(f"[yellow]Không ghi được session_summary: {_exc}[/yellow]")


@app.command("closed-loop")
def closed_loop_cmd(
    market_data_dir: str = typer.Option(..., help="Thư mục parquet MarketData (gate local)"),
    region: str = typer.Option("USA"),
    universe: str = typer.Option("TOP3000"),
    delay: int = typer.Option(1),
    patience: int = typer.Option(5, help="Bỏ ý tưởng sau N lần refine không cải thiện"),
    pop_size: int = typer.Option(30, help="Kích thước quần thể GP mỗi batch ý tưởng"),
    n_generations: int = typer.Option(3),
    top_k: int = typer.Option(10, help="Số ý tưởng/batch sau decorrelate"),
    max_corr: float = typer.Option(0.70),
    max_ideas: int = typer.Option(0, help="0 = không trần (chạy đến hết quota)"),
    neutralization: str = typer.Option(
        "MARKET", help="neutralization khoi diem (sweep se chon MARKET/SECTOR)"
    ),
    decay: int = typer.Option(4),
    truncation: float = typer.Option(0.08),
    base_seed: int = typer.Option(
        0, help="Seed GP sinh ý tưởng; 0 = ngẫu nhiên mỗi lần (tránh no_more_ideas)"
    ),
    refiner: str = typer.Option(
        "local", help="local (LocalTuner, mặc định, không LLM) | llm (RefinementLoop cũ)"
    ),
    alt_data: bool = typer.Option(
        True, "--alt-data/--no-alt-data",
        help="Seed core alt-data (option8 IV / socialmedia8 sentiment) đi THẲNG Brain sim -> "
             "mở rộng khỏi họ price/volume bão hòa, giảm self-corr (đòn bẩy yield #1, IMPROVEMENT_"
             "SPEC §2.1). BẬT mặc định; --no-alt-data để tắt (so sánh A/B single-variable §6).",
    ),
    combine: bool = typer.Option(
        True, "--combine/--no-combine",
        help="Nối tiếp mỗi batch bằng ALPHA GHÉP: tổ hợp tín hiệu ít tương quan (batch + kho DB) "
             "thành add(rank(...)) -> Sharpe ~√N (Grinold–Kahn), dễ chạm ngưỡng nộp. Mặc định bật.",
    ),
    max_gp_sims: int = typer.Option(
        3, help="Trần sim Brain THẬT/phiên riêng cho ứng viên GP (nguồn nhiễu, calibration "
                 "ρ=0.308 không đủ tin để lọc trước) -> ưu tiên quota cho seed đã kiểm chứng/"
                 "alpha ghép. 0 = không cap.",
    ),
    alt_sweep_budget: int = typer.Option(
        2, help="Ngân sách mini-sweep cho đường alt-data đi thẳng Brain: sau sim core, tối đa "
                 "N sim THÊM (flip dấu nếu sharpe quá âm, đổi decay nếu sharpe dương nhưng "
                 "chưa pass) trước khi chọn kết quả điểm-nộp cao nhất. 0 = tắt sweep (đúng 1 "
                 "sim/ý tưởng như cũ).",
    ),
) -> None:
    """Vòng kín AI + MiniBrain: GP sinh ý tưởng → refine (LocalTuner local mặc định, hoặc AI
    refine ≤patience nếu --refiner llm) + gate local → SIM Brain → lưu DB + feedback → lặp
    đến khi hết quota (Ctrl+C để dừng tay). Cần đăng nhập (+ .env AI nếu --refiner llm)."""
    _setup_logging()

    if not Path(market_data_dir).is_dir():
        console.print(f"[red]Không thấy thư mục MarketData: {market_data_dir}[/red]")
        raise typer.Exit(code=1)

    client = cli_common._make_client()
    client.authenticate()

    engine_db = init_db(make_engine())
    session_factory = make_session_factory(engine_db)

    ok = _run_closed_loop_session(
        session_factory, client, region, universe, delay, market_data_dir,
        pop_size=pop_size, n_generations=n_generations, top_k=top_k, max_corr=max_corr,
        patience=patience, max_ideas=(max_ideas or None),
        neutralization=neutralization, decay=decay, truncation=truncation,
        base_seed=(base_seed or None), refiner_kind=refiner,
        include_alt_data=alt_data, include_combiner=combine,
        max_gp_sims=(max_gp_sims or None), alt_sweep_budget=alt_sweep_budget,
    )
    if not ok:
        raise typer.Exit(code=1)


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
    fields, operators, field_types, matrix_only_ops, operator_arity = cli_common._cached_symbols(session_factory)
    pf = PreFilter(
        known_operators=operators or None, known_fields=set(fields) or None,
        field_types=field_types, matrix_only_ops=matrix_only_ops,
        operator_arity=operator_arity, local_arity=cli_common._local_operator_arity(),
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
        simulator=cli_common._make_validated_simulator(client, pf, session_factory, region, universe),
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
        # Marathon: cho referee ít nhất 2 bước refine trước khi được bỏ hướng — tránh
        # abandon ngay sau seed (thủ phạm 'best total=0.4, 1 sim, abandon' trong log dài).
        min_refine_steps_before_abandon=2 if marathon else 0,
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
    if not cli_common._cached_symbols(session_factory)[0]:
        console.print("[red]Chưa có fields — chạy fetch-fields trước.[/red]")
        raise typer.Exit(code=1)
    client = cli_common._make_client()
    client.authenticate()

    # Hướng để trống -> LLM tự đề xuất (giống miner cũ tự seed). Closure chỉ chạy
    # khi cần (không nhập hướng) để khỏi tốn lượt LLM khi đã có hướng.
    def _auto_direction():
        from src.simulation.pre_filter import PreFilter

        f, o, ft, mo, oa = cli_common._cached_symbols(session_factory)
        pf = PreFilter(
            known_operators=o or None, known_fields=set(f) or None,
            field_types=ft, matrix_only_ops=mo, operator_arity=oa,
            local_arity=cli_common._local_operator_arity(),
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
        f, o, ft, mo, oa = cli_common._cached_symbols(session_factory)
        pf = PreFilter(
            known_operators=o or None, known_fields=set(f) or None,
            field_types=ft, matrix_only_ops=mo, operator_arity=oa,
            local_arity=cli_common._local_operator_arity(),
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
    if not cli_common._cached_symbols(session_factory)[0]:
        console.print("[red]Chưa có fields — chạy fetch-fields trước.[/red]")
        raise typer.Exit(code=1)
    client = cli_common._make_client()
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
    fields, operators, field_types, matrix_only_ops, operator_arity = cli_common._cached_symbols(session_factory)
    pf = PreFilter(
        known_operators=operators or None, known_fields=set(fields) or None,
        field_types=field_types, matrix_only_ops=matrix_only_ops,
        operator_arity=operator_arity, local_arity=cli_common._local_operator_arity(),
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
    power_pool: bool = typer.Option(
        False, "--power-pool",
        help="Đường nộp PURE Power Pool: alpha Sharpe>=1.0 không đạt Regular nhưng đạt "
        "cấu trúc PP + khớp theme tuần hiện tại (lịch src/scoring/power_pool_theme.py)",
    ),
) -> None:
    """Chọn và nộp alpha đạt ngưỡng (mặc định dry-run)."""
    _setup_logging()
    from src.submission.correlation import CorrelationChecker
    from src.submission.manager import SubmissionManager

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    client = cli_common._make_client()
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


@app.command("genius-report")
def genius_report_cmd() -> None:
    """Báo cáo tie-break BRAIN Genius tính được LOCAL (avg/total distinct operators/fields của
    alpha đã nộp) — CHỈ để tham khảo, KHÔNG phải gate (sub-project G)."""
    _setup_logging()
    from src.scoring.genius_report import (
        average_distinct_fields_per_alpha,
        average_distinct_operators_per_alpha,
        total_distinct_fields,
        total_distinct_operators,
    )

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)

    avg_ops = average_distinct_operators_per_alpha(session_factory)
    avg_fields = average_distinct_fields_per_alpha(session_factory)
    total_ops = total_distinct_operators(session_factory)
    total_fields = total_distinct_fields(session_factory)

    table = Table(title="BRAIN Genius — tie-break metrics (chỉ tham khảo, không phải gate)")
    table.add_column("Chỉ số")
    table.add_column("Giá trị", justify="right")
    table.add_row(
        "Avg distinct Operators/Alpha (thấp hơn tốt hơn)",
        "—" if avg_ops is None else f"{avg_ops:.2f}",
    )
    table.add_row(
        "Avg distinct Fields/Alpha (thấp hơn tốt hơn)",
        "—" if avg_fields is None else f"{avg_fields:.2f}",
    )
    table.add_row("Total distinct Operators (cao hơn tốt hơn)", str(total_ops))
    table.add_row("Total distinct Fields (cao hơn tốt hơn)", str(total_fields))
    console.print(table)
    if avg_ops is None:
        console.print("[dim]Chưa có alpha nào status='submitted' trong DB để tính.[/dim]")


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
    """Đăng nhập rồi TỰ ĐẢM BẢO có data fields + operators (dùng cache nếu có, tự tải
    nếu thiếu) — để mục 4/5 chạy được ngay mà không bắt người dùng tự bấm 2/3 lần đầu."""
    from src.storage.db import active_database_url, write_active_account

    client = cli_common._make_client()
    client.authenticate()
    state.client = client
    state.email = client.email or ""
    # Biết email rồi mới mở DB (DB tách theo email).
    if state.email:
        write_active_account(state.email)
    engine = init_db(make_engine())
    state.session_factory = make_session_factory(engine)

    console.print(f"[green]✓ Đăng nhập xong[/green] ({state.email})")
    console.print(f"[dim]DB: {active_database_url()}[/dim]")

    field_repo = FieldRepository(state.client, state.session_factory)
    fields, fields_fetched = field_repo.ensure(state.region, state.universe, state.delay)
    op_repo = OperatorRepository(state.client, state.session_factory)
    operators, ops_fetched = op_repo.ensure()

    tai_moi = [n for n, done in (("data fields", fields_fetched), ("operators", ops_fetched)) if done]
    if tai_moi:
        console.print(f"[cyan]Đã tự tải mới: {', '.join(tai_moi)}[/cyan]")
    console.print(
        f"[bold]Data fields:[/bold] {len(fields)}   [bold]Operators:[/bold] {len(operators)}   "
        f"[dim]({state.region}/{state.universe}/delay={state.delay})[/dim]"
    )


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


def _find_market_data_dir() -> str | None:
    """Ưu tiên `settings.market_data_dir`; nếu thiếu, quét `data/*/returns.parquet` (bắt
    các panel đã có sẵn như `data/market_yf`) và dùng thư mục đầu tiên tìm được."""
    default = Path(settings.market_data_dir)
    if (default / "returns.parquet").is_file():
        return str(default)
    for candidate in sorted(Path("data").glob("*/returns.parquet")):
        return str(candidate.parent)
    return None


def _menu_test_engine(state: _MenuState) -> None:
    """Mục 4: test 1 lượt engine HOÀN TOÀN LOCAL (GP sinh nhanh → LLM refine THẬT →
    re-score local) — KHÔNG cần đăng nhập, không đụng WQ Brain API/quota. Mục đích: tự bắt
    lỗi wiring (DB/GP/LLM/gate) trước khi chạy thật mục 5 (tốn sim quota Brain)."""
    from src.app.local_engine_test import run_local_engine_test
    from src.data.adapters.parquet_source import ParquetSource
    from src.lang.registry import default_registry
    from src.simulation.pre_filter import PreFilter
    from src.storage.db import active_database_url, read_active_account
    from src.storage.repository import MiniBrainRepository

    email = read_active_account()
    if not email:
        console.print(
            "[red]Chưa từng đăng nhập lần nào — chạy mục 1 ít nhất 1 lần trước (sau đó có "
            "thể dùng mục 4 mà không cần đăng nhập lại).[/red]"
        )
        return

    engine = init_db(make_engine())
    sf = make_session_factory(engine)
    console.print(f"[dim]Tài khoản: {email} | DB: {active_database_url()}[/dim]")

    field_repo = FieldRepository(None, sf)
    op_repo = OperatorRepository(None, sf)
    if field_repo.cached_count(state.region, state.universe, state.delay) == 0 or op_repo.cached_count() == 0:
        console.print(
            "[red]Chưa có data fields/operators trong DB — đăng nhập (mục 1) ít nhất 1 lần "
            "trước.[/red]"
        )
        return

    market_data_dir = _find_market_data_dir()
    if market_data_dir is None:
        console.print(
            f"[red]Không tìm thấy thư mục MarketData nào (đã thử {settings.market_data_dir} "
            "và quét data/*/returns.parquet).[/red]"
        )
        return
    console.print(f"[dim]MarketData: {market_data_dir}[/dim]")

    try:
        data = ParquetSource(market_data_dir).load("1900-01-01", "2999-12-31", state.universe)
    except (FileNotFoundError, AssertionError, OSError) as exc:
        console.print(f"[red]Không load được MarketData: {exc}[/red]")
        return

    try:
        deepseek = _make_router()
    except typer.Exit:
        console.print("[red]Chưa cấu hình LLM backend hợp lệ trong .env (LLM_BACKEND).[/red]")
        return

    import src.operators_local  # noqa: F401  (nạp 27 operator vào registry)

    f, o, ft, mo, oa = cli_common._cached_symbols(sf)
    prefilter = PreFilter(
        known_operators=o or None, known_fields=set(f) or None,
        field_types=ft, matrix_only_ops=mo, operator_arity=oa,
        local_arity=cli_common._local_operator_arity(),
    )
    cfg = cli_common._portfolio_config_from_opts("NONE", 0, 0.10, state.delay)

    console.print("[cyan]Đang chạy 1 lượt test engine cục bộ (GP → LLM refine → re-score)…[/cyan]")
    result = run_local_engine_test(
        data=data, repo=MiniBrainRepository(sf), config=cfg, registry=default_registry(),
        deepseek=deepseek, field_repo=field_repo, operator_repo=op_repo, prefilter=prefilter,
    )

    if not result.ok:
        console.print(f"[red]Test engine LỖI:[/red] {result.error}")
        return

    table = Table(title="Kết quả test engine (local, không tốn sim Brain)")
    table.add_column("")
    table.add_column("Trước refine", justify="right")
    table.add_column("Sau refine", justify="right")
    table.add_row("expression", result.idea_expr or "—", result.refined_expr or "—")
    table.add_row(
        "sharpe",
        "—" if result.sharpe_before is None else f"{result.sharpe_before:.3f}",
        "—" if result.sharpe_after is None else f"{result.sharpe_after:.3f}",
    )
    table.add_row(
        "fitness",
        "—" if result.fitness_before is None else f"{result.fitness_before:.3f}",
        "—" if result.fitness_after is None else f"{result.fitness_after:.3f}",
    )
    console.print(table)
    console.print(f"[bold]passed cục bộ:[/bold] {result.passed}")
    if result.hard_failures:
        console.print(f"[yellow]hard_failures:[/yellow] {'; '.join(result.hard_failures)}")
    console.print(
        "[green]✓ Pipeline chạy sạch[/green] (DB/GP/LLM/gate cục bộ wiring đúng) — mục 5 "
        "(Auto SIM) nhiều khả năng chạy được, rủi ro còn lại chỉ là mạng/quota WQ thật."
    )


def _menu_auto_sim(state: _MenuState) -> None:
    """Mục 5: vòng kín AI+MiniBrain thật — tự tìm thư mục MarketData (như mục 4, không hỏi
    gì thêm ngoài LLM đã cấu hình sẵn trong .env) rồi chạy đến khi hết quota."""
    n_fields, _ = _menu_counts(state)
    if n_fields == 0:
        console.print("[red]Chưa có data fields — chọn 1 để đăng nhập (tự tải) trước.[/red]")
        return

    market_data_dir = _find_market_data_dir()
    if market_data_dir is None:
        console.print(
            f"[red]Không tìm thấy thư mục MarketData nào (đã thử {settings.market_data_dir} "
            "và quét data/*/returns.parquet). Chạy scripts/fetch_yfinance_panel.py trước.[/red]"
        )
        return
    console.print(f"[dim]MarketData: {market_data_dir}[/dim]")

    _run_closed_loop_session(
        state.session_factory, state.client, state.region, state.universe, state.delay,
        market_data_dir, refiner_kind="local",
    )


def _menu_view_submit(state: _MenuState) -> None:
    """Mục 6: xem alpha đã mô phỏng đạt (status='passed', từ mục 5/CLI research/marathon) và
    tự chọn nộp THẬT hay chỉ xem trước — dry-run mặc định, hỏi xác nhận rõ ràng trước khi tốn
    quota nộp ngày thật. Alpha đạt điều kiện Power Pool sẽ tự được gắn tag (sub-project A/C,
    đã tích hợp sẵn trong SubmissionManager.submit())."""
    from src.submission.correlation import CorrelationChecker
    from src.submission.manager import SubmissionManager

    manager = SubmissionManager(state.client, state.session_factory, CorrelationChecker(state.client))
    preview = manager.run_daily(dry_run=True)
    if not preview:
        console.print(
            "[yellow]Chưa có alpha nào đạt điều kiện nộp (status='passed', chưa nộp, qua được "
            "lọc self-correlation/trùng cấu trúc).[/yellow]"
        )
        return

    table = Table(title=f"Sẽ nộp (dry-run) — {len(preview)} alpha, quota/ngày={manager.daily_quota}")
    table.add_column("#")
    table.add_column("WQ Alpha")
    table.add_column("Expression", overflow="fold")
    table.add_column("Sharpe", justify="right")
    table.add_column("Score", justify="right")
    for i, c in enumerate(preview, 1):
        table.add_row(
            str(i), c.wq_alpha_id, c.expression,
            f"{c.sharpe:.3f}" if c.sharpe is not None else "—",
            f"{c.score:.3f}" if c.score is not None else "—",
        )
    console.print(table)

    answer = input(
        f"\nNộp THẬT {len(preview)} alpha này lên WQ Brain (tốn quota nộp ngày thật)? "
        "Gõ 'yes' để xác nhận, Enter để bỏ qua: "
    ).strip().lower()
    if answer != "yes":
        console.print("[dim]Đã bỏ qua — chưa nộp gì, có thể chọn lại mục này sau.[/dim]")
        return

    submitted = manager.run_daily(dry_run=False)
    console.print(
        f"[green]Đã nộp {len(submitted)} alpha.[/green] Alpha đạt điều kiện Power Pool "
        "(Sharpe≥1.0, operator/field trong giới hạn, có mô tả) sẽ tự được gắn tag PowerPoolSelected."
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
    # Power Pool Theme hôm nay (nếu có) — Auto SIM (mục 5) sẽ TỰ override scope sim theo theme này
    # để nộp được Pure Power Pool; dòng "Scope" trên chỉ là mặc định menu, không phải config sim thật.
    from datetime import date as _date

    from src.scoring.power_pool_theme import theme_for_date as _theme_for_date

    _theme = _theme_for_date(_date.today())
    if _theme is not None:
        _neut = sorted(_theme.allowed_neutralizations) if _theme.allowed_neutralizations else []
        console.print(
            f"Power Pool Theme {_theme.start_date}..{_theme.end_date} → Auto SIM dùng "
            f"[green]{_theme.region or state.region}/{_theme.universe or state.universe}/"
            f"delay={_theme.delay if _theme.delay is not None else state.delay}[/green]"
            + (f", neutralization ∈ [green]{_neut}[/green]" if _neut else "")
        )
    else:
        console.print(
            f"[yellow]Không có Power Pool Theme cho {_date.today()} trong lịch[/yellow] — Auto SIM "
            "giữ scope Regular; cập nhật lịch trong src/scoring/power_pool_theme.py nếu muốn nộp Pure Power Pool."
        )
    console.print(data)
    console.print(" 1) Đăng nhập (tự tải data fields + operators)")
    console.print(" 2) Tải lại data fields (ghi đè cache)")
    console.print(" 3) Tải lại operators (ghi đè cache)")
    console.print(" 4) Test engine (không cần đăng nhập — kiểm tra luồng cục bộ)")
    console.print(" 5) Auto SIM (vòng kín AI+MiniBrain, cần đăng nhập)")
    console.print(" 6) Xem & nộp alpha đã tìm được (dry-run trước, hỏi xác nhận)")
    console.print(" 0) Thoát")


@app.command()
def start() -> None:
    """Menu tương tác: đăng nhập → tải/kiểm tra fields-operators → test engine → Auto SIM."""
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
            elif choice == "4":
                _menu_test_engine(state)
            elif choice in {"2", "3", "5", "6"} and not state.logged_in:
                console.print("[yellow]Hãy đăng nhập (1) trước.[/yellow]")
            elif choice == "2":
                _menu_fields(state)
            elif choice == "3":
                _menu_operators(state)
            elif choice == "5":
                _menu_auto_sim(state)
            elif choice == "6":
                _menu_view_submit(state)
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
