"""Lệnh research (vòng nghiên cứu chính sinh alpha)."""

from __future__ import annotations

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository
from src.storage.db import init_db, make_engine, make_session_factory
from src.storage.repository import AlphaRepository
from src.app.cli.common import (
    _cached_symbols,
    _local_operator_arity,
    _make_client,
    _make_validated_simulator,
)
from src.app.cli.llm import _make_router, _make_llm_generator

console = Console()


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
        operator_arity=operator_arity, local_arity=_local_operator_arity(),
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
    from main import _setup_logging

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
            local_arity=_local_operator_arity(),
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
