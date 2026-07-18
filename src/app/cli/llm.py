"""Lệnh LLM: deepseek smoke-check, llm-generate/ideas."""

from __future__ import annotations

import typer
from rich.console import Console

from config.settings import settings
from src.app.cli.common import _cached_symbols, _local_operator_arity
from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository
from src.storage.db import init_db, make_engine, make_session_factory
from src.storage.repository import AlphaRepository, InvalidFieldRepository

console = Console()


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


def check_deepseek(
    message: str = typer.Option("hello", "--message", "-m", help="Tin nhắn test gửi tới DeepSeek"),
    model: str = typer.Option("", "--model", help="Ghi đè DEEPSEEK_MODEL cho lần check này"),
) -> None:
    """Gọi DeepSeek chat thật bằng DEEPSEEK_API_KEY/BASE_URL/MODEL."""
    from main import _setup_logging

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


def llm_generate(
    idea: str = typer.Option(..., help="Ý tưởng alpha bằng ngôn ngữ tự nhiên"),
    count: int = typer.Option(5),
) -> None:
    """Sinh alpha từ một ý tưởng bằng DeepSeek."""
    from main import _setup_logging

    _setup_logging()
    from src.simulation.pre_filter import PreFilter

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    fields, operators, field_types, matrix_only_ops, operator_arity = _cached_symbols(session_factory)
    pf = PreFilter(
        known_operators=operators or None, known_fields=set(fields) or None,
        field_types=field_types, matrix_only_ops=matrix_only_ops,
        operator_arity=operator_arity, local_arity=_local_operator_arity(),
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


def llm_ideas(count: int = typer.Option(10)) -> None:
    """Cho DeepSeek brainstorm các ý tưởng alpha."""
    from main import _setup_logging

    _setup_logging()
    from src.simulation.pre_filter import PreFilter

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)
    pf = PreFilter()
    llm_gen = _make_llm_generator(session_factory, pf)
    ideas = llm_gen.generate_ideas(count)
    for i, idea in enumerate(ideas, 1):
        console.print(f"  {i}. {idea}")
