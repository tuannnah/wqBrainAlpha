"""Lệnh quản lý fields/operators (probe, fetch, cache, list)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.app.cli.common import _make_client
from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository
from src.data.universe_matrix import iter_scopes
from src.data.warm_cache import warm_cache
from src.storage.db import init_db, make_engine, make_session_factory

console = Console()


def probe_fields(
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
) -> None:
    """Gọi /data-fields THẬT và in nguyên JSON 1 trang để kiểm tra format."""
    # Nhập trễ: _setup_logging còn ở main.py (chưa tách riêng, dùng chung cho mọi lệnh
    # CLI) — import trễ trong thân hàm để tránh vòng import main<->fields (main.py
    # import module fields ở đầu file, trước cả khi _setup_logging được định nghĩa).
    from main import _setup_logging

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


def warm_cache_cmd(
    regions: str = typer.Option("", help="CSV region cần tải; rỗng = tất cả trong WQB_MATRIX"),
    delays: str = typer.Option("0,1", help="CSV delay cần tải"),
    force: bool = typer.Option(False, help="Tải lại tất cả, bỏ qua cache"),
    sleep: float = typer.Option(2.0, help="Giây nghỉ giữa các scope có gọi API"),
) -> None:
    """Tải sẵn toàn bộ datafields + operators vào DB (resume được)."""
    from main import _setup_logging

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


def fetch_fields(
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    reload: bool = typer.Option(False, "--reload", help="Ép tải lại từ API (ghi đè cache)"),
) -> None:
    """Fetch một lần (bỏ qua nếu đã cache). --reload để ép tải lại (ghi đè)."""
    from main import _setup_logging

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


def cache_status() -> None:
    """Xem trạng thái cache (các tổ hợp đã fetch)."""
    from main import _setup_logging

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


def fetch_operators(
    reload: bool = typer.Option(False, "--reload", help="Ép tải lại từ API (ghi đè cache)"),
) -> None:
    """Lấy & cache operators (bỏ qua nếu đã cache). --reload để ép tải lại (ghi đè)."""
    from main import _setup_logging

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


def list_fields(
    region: str = typer.Option(settings.default_region),
    universe: str = typer.Option(settings.default_universe),
    delay: int = typer.Option(settings.default_delay),
    dataset: str = typer.Option(None, help="Lọc theo dataset id"),
    search: str = typer.Option(None, help="Tìm trong id/mô tả"),
    limit: int = typer.Option(50, help="Số dòng hiển thị"),
) -> None:
    """Xem các data field đã tải về (trong DB), có lọc/tìm kiếm."""
    from main import _setup_logging

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
