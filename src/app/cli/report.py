"""Lệnh báo cáo: top alpha, originality, genius report."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from src.storage.db import init_db, make_engine, make_session_factory

console = Console()


def top(
    n: int = typer.Option(20),
    sort: str = typer.Option("score", help="score/sharpe/fitness"),
) -> None:
    """Hiển thị alpha tốt nhất theo simulation đã lưu."""
    # Nhập trễ: _setup_logging còn ở main.py (chưa tách riêng, dùng chung cho mọi lệnh
    # CLI) — import trễ trong thân hàm để tránh vòng import main<->report.
    from main import _setup_logging

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


def originality(
    expr: str = typer.Option(..., "--expr", help="Biểu thức FASTEXPR cần đo độ độc đáo"),
) -> None:
    """GĐ3: đo độ độc đáo của một alpha so với zoo tham chiếu (Alpha101 + alpha đã pass)."""
    from main import _setup_logging

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


def genius_report_cmd() -> None:
    """Báo cáo tie-break BRAIN Genius tính được LOCAL (avg/total distinct operators/fields của
    alpha đã nộp) — CHỈ để tham khảo, KHÔNG phải gate (sub-project G)."""
    from main import _setup_logging

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
