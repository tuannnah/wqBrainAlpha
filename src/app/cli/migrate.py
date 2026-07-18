"""Lệnh migrate DB và calibrate."""

from __future__ import annotations

import math

import typer
from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.storage.db import init_db, make_engine, make_session_factory
from src.storage.migrate import migrate_all, _same_database

console = Console()


def migrate_sqlite(
    source: str = typer.Option("sqlite:///data/db/wq_alpha.db", help="URL DB nguồn (SQLite)"),
    dest: str = typer.Option("", help="URL DB đích; rỗng = dùng DATABASE_URL"),
) -> None:
    """Copy toàn bộ dữ liệu từ SQLite sang DB đích (Postgres), idempotent."""
    # Nhập trễ: _setup_logging còn ở main.py (chưa tách riêng, dùng chung cho mọi lệnh
    # CLI) — import trễ trong thân hàm để tránh vòng import main<->migrate.
    from main import _setup_logging

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

    # Nhập trễ: _setup_logging còn ở main.py (chưa tách riêng, dùng chung cho mọi lệnh
    # CLI) — import trễ trong thân hàm để tránh vòng import main<->migrate.
    from main import _setup_logging

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
