"""Helper CLI dùng chung nhiều nhóm lệnh (client factory, cache field/operator, config từ option)."""

from __future__ import annotations

import typer
from loguru import logger
from rich.console import Console

from config.settings import settings
from src.data.client import WQBrainClient
from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository, count_max_arity
from src.simulation.simulator import Simulator
from src.storage.repository import InvalidFieldRepository

console = Console()


def _make_client() -> WQBrainClient:
    # Ưu tiên .env nếu đã điền; nếu trống thì nhập tương tác trong PowerShell.
    email = settings.wq_email
    password = settings.wq_password
    if not email or not password:
        # Nhập trễ: prompt_credentials nằm ở src/app/cli/auth.py, module đó lại import
        # _make_client từ common.py này ở đầu file -> nếu đưa import này lên đầu module
        # sẽ tạo vòng import module-level auth<->common. Import trễ trong thân hàm tránh
        # vòng lặp vì chỉ chạy khi hàm thực sự được gọi, lúc đó cả 2 module đã nạp xong.
        from src.app.cli.auth import prompt_credentials

        email, password = prompt_credentials()
    return WQBrainClient(email, password)


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
    # Arity TỐI ĐA (gồm cả param có default `=`) từ definition đã lưu -> chỉ chặn khi THỪA
    # input so với chữ ký (Brain cho truyền positional cả param default: ts_backfill/rank/…).
    operator_arity = {}
    for o in cached_ops:
        if not o.name:
            continue
        n = count_max_arity(o.definition or "")
        if n:
            operator_arity[o.name] = n
    return fields, operators, field_types, matrix_only_ops, operator_arity


def _local_operator_arity() -> dict[str, int]:
    """Cap arity LOCAL suy từ chữ ký `OperatorRegistry` (`len(spec.signature)`).

    Nguồn BỔ SUNG cho PreFilter, lấp lỗ hổng: op vắng mặt trong catalog Brain (hoặc
    definition không parse được -> `count_max_arity` trả 0) trước đây bị BỎ QUA kiểm arity
    hoàn toàn (3 sim đã chết vì lỗi WQ "Invalid number of inputs" do lỗ hổng này). Import
    `src.operators_local` trước để đảm bảo registry đã nạp đủ toàn bộ operator thật (không
    chỉ 6 op tối thiểu Phase 1) — idempotent, an toàn gọi lại nhiều lần."""
    import src.operators_local  # noqa: F401  (nạp operator thật vào registry)
    from src.lang.registry import default_registry

    return {name: len(spec.signature) for name, spec in default_registry().all_specs().items()}


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

    # auto_tag "wqtool": mọi alpha do vòng kín sim đều gắn tag để lọc được trên web Brain
    # (tab Alphas -> filter tag) — yêu cầu người dùng 2026-07-18.
    return Simulator(
        client, on_invalid_field=on_invalid_field, pre_sim_validator=pf.check,
        auto_tag="wqtool",
    )


def _portfolio_config_from_opts(
    neutralization: str, decay: int, truncation: float, delay: int,
) -> "PortfolioConfig":  # noqa: F821
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
