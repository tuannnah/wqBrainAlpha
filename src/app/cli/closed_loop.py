"""Lệnh closed-loop (vòng kín local-search + Brain sim)."""

from __future__ import annotations

import random
from pathlib import Path

import typer
from rich.console import Console

from src.storage.db import init_db, make_engine, make_session_factory
from src.app.cli.common import _cached_symbols, _make_client, _portfolio_config_from_opts
from src.app.cli.research import _make_research_loop

console = Console()


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
    cfg = _portfolio_config_from_opts(local_neut, decay, truncation, delay)
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
    _catalog_fields, _catalog_ops, _, _, _ = _cached_symbols(session_factory)
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
    from main import _setup_logging

    _setup_logging()

    if not Path(market_data_dir).is_dir():
        console.print(f"[red]Không thấy thư mục MarketData: {market_data_dir}[/red]")
        raise typer.Exit(code=1)

    client = _make_client()
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
