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
from src.llm.marathon import MarathonReport, run_marathon
from src.app.cli import common as cli_common
from src.app.cli import auth as cli_auth
from src.app.cli import fields as cli_fields
from src.app.cli import simulate as cli_simulate
from src.app.cli import generate as cli_generate
from src.app.cli import submit as cli_submit
from src.app.cli import report as cli_report
from src.app.cli import migrate as cli_migrate
from src.app.cli import llm as cli_llm
from src.app.cli import research as cli_research

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
app.command()(cli_generate.generate)
app.command("score-one")(cli_generate.score_one_cmd)
app.command()(cli_submit.submit)
app.command()(cli_report.top)
app.command()(cli_report.originality)
app.command("genius-report")(cli_report.genius_report_cmd)
app.command("migrate-sqlite")(cli_migrate.migrate_sqlite)
app.command("calibrate")(cli_migrate.calibrate)
app.command("check-deepseek")(cli_llm.check_deepseek)
app.command("llm-generate")(cli_llm.llm_generate)
app.command("llm-ideas")(cli_llm.llm_ideas)
app.command()(cli_research.research)
console = Console()

LOG_DIR = Path("logs")

# Cầu nối tạm (Task 14): `_make_research_loop`/`resolve_direction` đã chuyển sang
# src/app/cli/research.py, nhưng `_run_closed_loop_session` (Task 15) và
# `_run_marathon_session`/`_marathon_direction_provider` (Task 16) vẫn còn ở lại
# main.py và gọi tên trần bên dưới -> alias module-level để không vỡ code/test hiện
# có (vd tests/test_marathon_command.py monkeypatch `main._make_research_loop`).
# Dọn nốt khi Task 15/16 chuyển các hàm còn lại đó sang module riêng.
_make_research_loop = cli_research._make_research_loop
resolve_direction = cli_research.resolve_direction


def _setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    # WQ_NO_FILE_LOG: bỏ file sink (conftest đặt khi chạy test) để không ghi
    # đè log production bằng nhiễu fixture.
    if os.environ.get("WQ_NO_FILE_LOG"):
        return
    LOG_DIR.mkdir(exist_ok=True)
    logger.add(LOG_DIR / "wq_alpha_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="14 days")


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
        ideas = cli_llm._make_llm_generator(session_factory, pf).generate_ideas(1)
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
        deepseek = cli_llm._make_router()
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


if __name__ == "__main__":
    app()
