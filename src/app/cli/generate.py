"""Lệnh generate/score-one."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.app.cli.common import _cached_symbols, _portfolio_config_from_opts
from src.storage.db import init_db, make_engine, make_session_factory

console = Console()


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
    from main import _setup_logging

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
    _, _catalog_ops, _, _, _ = _cached_symbols(session_factory)
    enforce_gp_vocab_against_catalog(default_registry(), _catalog_ops)

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
    from main import _setup_logging

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
