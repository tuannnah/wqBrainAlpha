"""C1: worker đánh giá PHẦN THUẦN của cá thể GP (eval AST → danh mục → backtest → metrics)
trong process con — KHÔNG SQLite, KHÔNG pool_corr (main process lo). Windows spawn: mọi hàm
module-level; data/config/registry nạp MỘT LẦN/worker qua initializer (không pickle lại mỗi
task).

``eval_thuan`` trả đúng định dạng ``eval_cache`` của ``GPEngine`` (xem
``GPEngine._evaluate_individual``): ``("ok", daily_pnl, metrics) | ("error", [lý_do])`` — main
process chỉ cần ghi thẳng kết quả vào ``eval_cache`` theo ``canonical_hash``, vòng lặp tuần
tự sẵn có tự HIT cache mà không cần đường code riêng.

Fix OOM (review cuối C1, 2026-07-18): nhánh "ok" trả ``bt.daily_pnl`` (ndarray 1 chiều, vài
chục KB) THAY VÌ ``BacktestResult`` đầy đủ — ``weights`` (T,N) ~10-15MB/cá thể trên panel
thật pickle qua ranh giới process rất tốn, mà ``eval_cache`` lại giữ MỌI entry xuyên phiên.

Fix parity lỗi (review cuối C1): mỗi giai đoạn (eval/backtest/metrics) bọc riêng, prefix lý
do lỗi Y HỆT main process (``GPEngine._evaluate_individual``: ``"eval: ..."`` /
``"backtest: ..."`` / ``"metrics: ..."``) — trước đây worker trả lý do KHÔNG prefix, khiến
text lỗi lưu DB/avoid-list lệch tuỳ ``n_jobs`` (giống lỗi nhưng khác câu chữ tuỳ chạy song
song hay tuần tự). Lỗi ``parse()`` (chưa từng có nhánh riêng ở main vì main không parse lại
từ chuỗi) xếp vào ``"eval: ..."`` — cùng giai đoạn ý nghĩa với eval AST bên main.
"""

from __future__ import annotations

from typing import Any

_CTX: dict[str, Any] = {}  # {"data": MarketData, "config": PortfolioConfig, "registry": OperatorRegistry}


def khoi_tao_worker(data: Any, config: Any, registry: Any) -> None:
    """Initializer của ``ProcessPoolExecutor`` — chạy đúng MỘT LẦN mỗi worker process, nạp
    dữ liệu bất biến (panel/config/registry) vào ``_CTX`` module-level để mọi task
    (``eval_thuan``) sau đó dùng lại, không phải pickle lại mỗi lần submit."""
    _CTX.update(data=data, config=config, registry=registry)


def eval_thuan(expr_string: str) -> tuple[Any, ...]:
    """Đánh giá PHẦN THUẦN một biểu thức (đã serialize thành chuỗi FASTEXPR): parse lại →
    eval AST → build danh mục → backtest → metrics. Trả ``("ok", daily_pnl, metrics)`` khi
    thành công hoặc ``("error", [lý_do])`` khi bất kỳ bước nào ném exception — CHƯA từng biết
    tới pool self-corr hay SQLite (main process đảm nhiệm phần đó tuần tự theo index gốc, xem
    ``GPEngine._prefetch_parallel``/``_evaluate_individual``). ``daily_pnl`` là ``bt.daily_pnl``
    (ndarray 1 chiều) — KHÔNG trả ``BacktestResult`` đầy đủ (fix OOM, xem module docstring).

    Mỗi giai đoạn bọc ``try/except`` RIÊNG với prefix lý do lỗi Y HỆT main process
    (``GPEngine._evaluate_individual``): ``"eval: ..."`` (gồm cả lỗi ``parse()`` — main không
    có bước parse lại từ chuỗi riêng nên gộp chung giai đoạn "eval AST"), ``"backtest: ..."``,
    ``"metrics: ..."`` — DB/avoid-list học lý do lỗi qua text nên phải khớp nguyên văn bất kể
    chạy tuần tự (``n_jobs=1``) hay song song (``n_jobs>1``), nếu không hai đường sinh ra hai
    câu chữ khác nhau cho CÙNG một lỗi.

    Nhận CHUỖI expr (không nhận thẳng ``Node``): ``Node`` bản thân picklable nhưng chuỗi
    chắc chắn an toàn + rẻ hơn qua ranh giới process (không kéo theo toàn bộ cấu trúc cây
    Python object, chỉ một ``str``).

    Registry dùng để parse lại PHẢI là ``_CTX["registry"]`` (registry của worker, giống hệt
    ``GPEngine.registry`` bên main) — KHÔNG dùng ``default_registry()`` ngầm định của
    ``parse()``: registry toàn cục trong process con là một singleton MỚI (spawn), khác
    object với ``self.registry`` bên main dù cùng nội dung; nếu process con populate
    default_registry() khác cách (vd thiếu vài operator do import-order), kết quả parse có
    thể lệch giữa tuần tự và song song — vi phạm ràng buộc "song song ≡ tuần tự"."""
    from src.backtest.backtester import Backtester
    from src.backtest.metrics_local import MetricsCalculator
    from src.backtest.portfolio import PortfolioBuilder
    from src.engine.evaluator import EvalContext, Evaluator
    from src.engine.subexpr_cache import SubexprCache
    from src.lang.parser import parse

    try:
        node = parse(expr_string, registry=_CTX["registry"])
        ctx = EvalContext(data=_CTX["data"], registry=_CTX["registry"], cache=SubexprCache())
        signal = Evaluator(ctx).evaluate(node)
    except Exception as exc:  # noqa: BLE001 — worker phải sống sót mọi lỗi cây, trả về lý do
        return ("error", [f"eval: {type(exc).__name__}: {exc}"])

    try:
        weights = PortfolioBuilder().build(signal, _CTX["config"], _CTX["data"])
        bt = Backtester().run(weights, _CTX["data"])
    except Exception as exc:  # noqa: BLE001
        return ("error", [f"backtest: {type(exc).__name__}: {exc}"])

    try:
        metrics = MetricsCalculator().compute(bt, _CTX["data"])
    except Exception as exc:  # noqa: BLE001
        return ("error", [f"metrics: {type(exc).__name__}: {exc}"])

    return ("ok", bt.daily_pnl, metrics)
