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
from src.app.cli import closed_loop as cli_closed_loop
from src.app.cli import marathon as cli_marathon

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
app.command("closed-loop")(cli_closed_loop.closed_loop_cmd)
app.command()(cli_marathon.marathon)
console = Console()

LOG_DIR = Path("logs")

# Cầu nối tạm (Task 15): `_run_closed_loop_session` đã chuyển sang
# src/app/cli/closed_loop.py, nhưng `_menu_auto_sim` (mục 5 menu, chưa thuộc task
# tách CLI nào) vẫn ở lại main.py và gọi tên trần bên dưới -> alias module-level để
# không vỡ tests/test_menu_counts.py (monkeypatch `main._run_closed_loop_session`).
# Dọn nốt khi menu functions được tách sang module riêng.
_run_closed_loop_session = cli_closed_loop._run_closed_loop_session


def _setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    # WQ_NO_FILE_LOG: bỏ file sink (conftest đặt khi chạy test) để không ghi
    # đè log production bằng nhiễu fixture.
    if os.environ.get("WQ_NO_FILE_LOG"):
        return
    LOG_DIR.mkdir(exist_ok=True)
    logger.add(LOG_DIR / "wq_alpha_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="14 days")


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
